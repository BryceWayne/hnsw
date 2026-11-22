// Command rag-server demonstrates a production-style, zero-dependency Go 1.27
// HTTP microservice backed by github.com/BryceWayne/hnsw.
//
// It showcases:
//   - Multi-tenant filtered search via SearchFilterInto + sync.Pool (0-alloc hot path)
//   - Per-request deadlines via WithContext and relevance cutoffs via WithMaxDistance
//   - Optional FP16 quantization (WithQuantization(QuantFloat16)) and WithInitialCapacity
//   - Periodic background compaction driven by IndexStats.TombstoneRatio()
//   - Atomic snapshot persistence on graceful SIGINT/SIGTERM shutdown via SaveFile/LoadFile
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/BryceWayne/hnsw"
)

type docMeta struct {
	Tenant    string    `json:"tenant"`
	Text      string    `json:"text"`
	UpdatedAt time.Time `json:"updated_at"`
}

type ragServer struct {
	idx              *hnsw.Index[string]
	metaMu           sync.RWMutex
	meta             map[string]docMeta
	resPool          sync.Pool
	compactThreshold float64
}

func newRAGServer(idx *hnsw.Index[string], compactThreshold float64) *ragServer {
	return &ragServer{
		idx:              idx,
		meta:             make(map[string]docMeta),
		compactThreshold: compactThreshold,
		resPool: sync.Pool{
			New: func() any {
				buf := make([]hnsw.SearchResult[string], 0, 32)
				return &buf
			},
		},
	}
}

type upsertReq struct {
	ID     string    `json:"id"`
	Tenant string    `json:"tenant"`
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

type searchReq struct {
	Tenant      string    `json:"tenant"`
	Vector      []float32 `json:"vector"`
	K           int       `json:"k"`
	Ef          int       `json:"ef,omitempty"`
	MaxDistance *float32  `json:"max_distance,omitempty"`
}

type searchHit struct {
	ID       string  `json:"id"`
	Distance float32 `json:"distance"`
	Tenant   string  `json:"tenant,omitempty"`
	Text     string  `json:"text,omitempty"`
}

func (s *ragServer) handleUpsert(w http.ResponseWriter, r *http.Request) {
	var req upsertReq
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20)).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}
	if req.ID == "" {
		http.Error(w, "id is required", http.StatusBadRequest)
		return
	}

	if err := s.idx.Insert(req.ID, req.Vector); err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, hnsw.ErrDimensionMismatch) || errors.Is(err, hnsw.ErrNaNOrInf) {
			status = http.StatusBadRequest
		}
		http.Error(w, err.Error(), status)
		return
	}

	s.metaMu.Lock()
	s.meta[req.ID] = docMeta{
		Tenant:    req.Tenant,
		Text:      req.Text,
		UpdatedAt: time.Now().UTC(),
	}
	s.metaMu.Unlock()

	writeJSON(w, map[string]any{"status": "ok", "id": req.ID})
}

func (s *ragServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchReq
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20)).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json: %v", err), http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	opts := []hnsw.SearchOption{hnsw.WithContext(r.Context())}
	if req.Ef > 0 {
		opts = append(opts, hnsw.WithSearchEf(req.Ef))
	}
	if req.MaxDistance != nil {
		opts = append(opts, hnsw.WithMaxDistance(*req.MaxDistance))
	}

	// Borrow a reusable result slice to avoid per-request slice allocations.
	bufPtr := s.resPool.Get().(*[]hnsw.SearchResult[string])
	defer s.resPool.Put(bufPtr)

	var (
		results []hnsw.SearchResult[string]
		err     error
	)
	if req.Tenant != "" {
		// Hold a read lock on metadata during filtered traversal so the predicate is race-free.
		s.metaMu.RLock()
		results, err = s.idx.SearchFilterInto(req.Vector, req.K, func(id string) bool {
			return s.meta[id].Tenant == req.Tenant
		}, *bufPtr, opts...)
		s.metaMu.RUnlock()
	} else {
		results, err = s.idx.SearchInto(req.Vector, req.K, *bufPtr, opts...)
	}
	if err != nil {
		status := http.StatusInternalServerError
		switch {
		case errors.Is(err, hnsw.ErrDimensionMismatch), errors.Is(err, hnsw.ErrNaNOrInf),
			errors.Is(err, hnsw.ErrInvalidK), errors.Is(err, hnsw.ErrInvalidConfig):
			status = http.StatusBadRequest
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			status = http.StatusRequestTimeout
		}
		http.Error(w, err.Error(), status)
		return
	}
	*bufPtr = results[:0]

	s.metaMu.RLock()
	hits := make([]searchHit, len(results))
	for i, res := range results {
		m := s.meta[res.ID]
		hits[i] = searchHit{
			ID:       res.ID,
			Distance: res.Distance,
			Tenant:   m.Tenant,
			Text:     m.Text,
		}
	}
	s.metaMu.RUnlock()

	writeJSON(w, map[string]any{"hits": hits})
}

func (s *ragServer) handleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		http.Error(w, "id is required", http.StatusBadRequest)
		return
	}
	deleted := s.idx.Delete(id)
	if deleted {
		s.metaMu.Lock()
		delete(s.meta, id)
		s.metaMu.Unlock()
	}
	writeJSON(w, map[string]any{"deleted": deleted, "id": id})
}

func (s *ragServer) handleStats(w http.ResponseWriter, _ *http.Request) {
	st := s.idx.Stats()
	writeJSON(w, map[string]any{
		"len":             st.Len,
		"deleted":         st.Deleted,
		"capacity":        st.Capacity,
		"tombstone_ratio": st.TombstoneRatio(),
		"dim":             st.Dim,
		"max_level":       st.MaxLevel,
		"metric":          st.Metric.String(),
		"quantization":    st.Quantization.String(),
		"ef_search":       st.EfSearch,
		"memory_bytes":    st.MemoryBytes,
	})
}

func (s *ragServer) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]any{"status": "healthy", "vectors": s.idx.Len()})
}

func (s *ragServer) runCompactor(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			st := s.idx.Stats()
			if st.Deleted > 0 && st.TombstoneRatio() >= s.compactThreshold {
				log.Printf("compacting index (deleted=%d, ratio=%.2f)", st.Deleted, st.TombstoneRatio())
				if err := s.idx.Compact(); err != nil {
					log.Printf("compaction error: %v", err)
				}
			}
		}
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(v)
}

func loadOrCreateIndex(path string, dim int, fp16 bool, initCap int) (*hnsw.Index[string], error) {
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			log.Printf("loading persisted index from %s", path)
			return hnsw.LoadFile[string](path)
		}
	}
	quant := hnsw.QuantNone
	if fp16 {
		quant = hnsw.QuantFloat16
	}
	return hnsw.New[string](
		dim,
		hnsw.WithM(16),
		hnsw.WithMmax(32),
		hnsw.WithEfConstruction(200),
		hnsw.WithEfSearch(64),
		hnsw.WithMetric(hnsw.MetricCosine),
		hnsw.WithQuantization(quant),
		hnsw.WithInitialCapacity(initCap),
	)
}

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	dim := flag.Int("dim", 4, "Vector dimension (used when creating a new index)")
	dataPath := flag.String("data", "", "Optional file path for SaveFile/LoadFile persistence")
	fp16 := flag.Bool("fp16", true, "Enable FP16 quantization (2x lower RAM)")
	initCap := flag.Int("initial-capacity", 1024, "Pre-allocated vector capacity")
	compactRatio := flag.Float64("compact-threshold", 0.25, "Tombstone ratio triggering background Compact")
	compactInterval := flag.Duration("compact-interval", 30*time.Second, "Background compaction check interval")
	flag.Parse()

	idx, err := loadOrCreateIndex(*dataPath, *dim, *fp16, *initCap)
	if err != nil {
		log.Fatalf("init index: %v", err)
	}

	srv := newRAGServer(idx, *compactRatio)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /upsert", srv.handleUpsert)
	mux.HandleFunc("POST /search", srv.handleSearch)
	mux.HandleFunc("DELETE /vectors/{id}", srv.handleDelete)
	mux.HandleFunc("GET /stats", srv.handleStats)
	mux.HandleFunc("GET /healthz", srv.handleHealthz)

	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go srv.runCompactor(ctx, *compactInterval)

	go func() {
		log.Printf("rag-server listening on %s (dim=%d, quant=%s, vectors=%d)",
			*addr, idx.Dim(), idx.Stats().Quantization, idx.Len())
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http server: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("shutting down gracefully...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("http shutdown: %v", err)
	}
	if *dataPath != "" {
		if err := idx.SaveFile(*dataPath); err != nil {
			log.Printf("save index failed: %v", err)
		} else {
			log.Printf("saved %d vectors atomically to %s", idx.Len(), *dataPath)
		}
	}
}
