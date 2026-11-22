// Command semantic-cache demonstrates an embedded LLM semantic prompt cache
// built with github.com/BryceWayne/hnsw using MetricCosine, FP16 quantization,
// WithMaxDistance similarity thresholds, and TTL eviction via DeleteBatch + Compact.
package main

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/BryceWayne/hnsw"
)

type cacheEntry struct {
	Response  string
	ExpiresAt time.Time
}

// SemanticCache caches LLM responses keyed by prompt embedding similarity.
type SemanticCache struct {
	idx     *hnsw.Index[uint64]
	maxDist float32

	mu      sync.RWMutex
	nextID  uint64
	entries map[uint64]cacheEntry
}

// NewSemanticCache creates a cosine-similarity FP16 cache matching prompts with
// cosine distance <= maxCosineDistance (e.g., 0.08 corresponds to >= 0.92 similarity).
func NewSemanticCache(dim int, maxCosineDistance float32, capacity int) (*SemanticCache, error) {
	idx, err := hnsw.New[uint64](
		dim,
		hnsw.WithMetric(hnsw.MetricCosine),
		hnsw.WithQuantization(hnsw.QuantFloat16),
		hnsw.WithInitialCapacity(capacity),
		hnsw.WithM(16),
		hnsw.WithEfConstruction(128),
		hnsw.WithEfSearch(32),
	)
	if err != nil {
		return nil, err
	}
	return &SemanticCache{
		idx:     idx,
		maxDist: maxCosineDistance,
		entries: make(map[uint64]cacheEntry, capacity),
	}, nil
}

// Put stores a prompt embedding and response with the given TTL.
func (c *SemanticCache) Put(embedding []float32, response string, ttl time.Duration) error {
	c.mu.Lock()
	c.nextID++
	id := c.nextID
	c.entries[id] = cacheEntry{
		Response:  response,
		ExpiresAt: time.Now().Add(ttl),
	}
	c.mu.Unlock()

	if err := c.idx.Insert(id, embedding); err != nil {
		c.mu.Lock()
		delete(c.entries, id)
		c.mu.Unlock()
		return err
	}
	return nil
}

// Lookup returns the closest non-expired cached response within maxDist.
func (c *SemanticCache) Lookup(embedding []float32) (response string, dist float32, hit bool, err error) {
	now := time.Now()
	c.mu.RLock()
	defer c.mu.RUnlock()

	var dst [1]hnsw.SearchResult[uint64]
	res, err := c.idx.SearchFilterInto(
		embedding,
		1,
		func(id uint64) bool {
			e, ok := c.entries[id]
			return ok && now.Before(e.ExpiresAt)
		},
		dst[:0],
		hnsw.WithMaxDistance(c.maxDist),
	)
	if err != nil || len(res) == 0 {
		return "", 0, false, err
	}
	return c.entries[res[0].ID].Response, res[0].Distance, true, nil
}

// EvictExpired removes expired cache entries and compacts the arena when tombstones accumulate.
func (c *SemanticCache) EvictExpired() (int, error) {
	now := time.Now()
	c.mu.Lock()
	var expired []uint64
	for id, e := range c.entries {
		if !now.Before(e.ExpiresAt) {
			expired = append(expired, id)
			delete(c.entries, id)
		}
	}
	c.mu.Unlock()

	if len(expired) == 0 {
		return 0, nil
	}
	removed := c.idx.DeleteBatch(expired)
	if c.idx.Stats().TombstoneRatio() >= 0.25 {
		if err := c.idx.Compact(); err != nil {
			return removed, err
		}
	}
	return removed, nil
}

func main() {
	cache, err := NewSemanticCache(4, 0.08, 1024)
	if err != nil {
		log.Fatal(err)
	}

	if err := cache.Put([]float32{0.95, 0.20, 0.05, 0.00}, "Use sync.Pool + SearchInto for 0 allocs/op.", time.Minute); err != nil {
		log.Fatal(err)
	}
	if err := cache.Put([]float32{0.00, 0.10, 0.95, 0.10}, "Expired weather answer", 10*time.Millisecond); err != nil {
		log.Fatal(err)
	}

	// Semantically close query -> cache HIT.
	if resp, dist, hit, err := cache.Lookup([]float32{0.96, 0.18, 0.04, 0.00}); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("close query:  hit=%v dist=%.4f resp=%q\n", hit, dist, resp)
	}

	// Unrelated query -> cache MISS (filtered by WithMaxDistance(0.08)).
	if _, _, hit, err := cache.Lookup([]float32{0.0, 1.0, 0.0, 0.0}); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("far query:    hit=%v\n", hit)
	}

	time.Sleep(20 * time.Millisecond)
	evicted, err := cache.EvictExpired()
	if err != nil {
		log.Fatal(err)
	}
	st := cache.idx.Stats()
	fmt.Printf("evicted=%d live=%d deleted=%d tombstone_ratio=%.2f quant=%s\n",
		evicted, st.Len, st.Deleted, st.TombstoneRatio(), st.Quantization)
}
