package hnsw

import (
	"fmt"
	"iter"
	"math"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

const (
	chunkBits    = 8
	chunkSize    = 1 << chunkBits // 256
	chunkMask    = chunkSize - 1
	dirL2Bits    = 9
	dirL2Size    = 1 << dirL2Bits // 512
	dirL2Mask    = dirL2Size - 1
	dirL1Size    = 512
	nodeLockSize = 4096
	nodeLockMask = nodeLockSize - 1
	readStripes  = 64
	readMask     = readStripes - 1
	maxNodeLevel = 32
	invalidID    = uint32(math.MaxUint32)
	emptyEntry   = uint64(math.MaxUint64)
)

type paddedRWMutex struct {
	mu sync.RWMutex
	_  [64 - unsafe.Sizeof(sync.RWMutex{})%64]byte
}

type dirPage[K comparable] struct {
	chunks [dirL2Size]atomic.Pointer[arenaChunk[K]]
}

type arenaChunk[K comparable] struct {
	vecs        []float32
	vecsF16     []uint16
	invNorms    []float32
	intToExt    []K
	deleted     [chunkSize / 64]atomic.Uint64
	l0Edges     []uint32
	l0Count     [chunkSize]uint16
	upperLevels [chunkSize][][]uint32
	maxLevels   [chunkSize]uint8
}

// SearchResult represents a single nearest-neighbor match returned by Search, SearchFilter, or SearchBatch.
type SearchResult[K comparable] struct {
	// ID is the external key associated with the matched vector.
	ID K
	// Distance is the distance from the query vector according to the index metric.
	Distance float32
}

// Item pairs an external key with a vector for batch insertion via InsertBatch.
type Item[K comparable] struct {
	// ID is the external key for the vector.
	ID K
	// Vector is the float32 embedding slice whose length must match the index dimension.
	Vector []float32
}

// IndexStats provides a point-in-time snapshot of index capacity, configuration, and memory usage.
type IndexStats struct {
	// Len is the number of live (non-deleted) vectors in the index.
	Len int
	// Deleted is the number of tombstoned slots awaiting Compact.
	Deleted int
	// Capacity is the total number of allocated vector slots across active arena chunks.
	Capacity int
	// Dim is the fixed vector dimensionality of the index.
	Dim int
	// MaxLevel is the current maximum HNSW layer level (0 when empty or single-layer).
	MaxLevel int
	// EntryPointSet reports whether the index currently has an active entry point.
	EntryPointSet bool
	// M is the configured target connections per upper-layer node.
	M int
	// Mmax is the configured maximum connections per layer-0 node.
	Mmax int
	// EfConstruction is the search beam width used during insertion.
	EfConstruction int
	// EfSearch is the default search beam width used during queries.
	EfSearch int
	// Metric is the configured distance metric.
	Metric Metric
	// Quantization is the configured vector storage encoding.
	Quantization Quantization
	// Heuristic reports whether Algorithm 4 neighbor selection is enabled.
	Heuristic bool
	// MemoryBytes is the estimated memory footprint of allocated arena slabs and graph structures in bytes.
	MemoryBytes int64
}

// Index is a concurrency-safe, generic Hierarchical Navigable Small World (HNSW) vector index.
type Index[K comparable] struct {
	dim            int
	m              int
	mmax           int
	efConstruction int
	efSearch       atomic.Int64
	mL             float64
	metric         Metric
	customDist     DistanceFunc
	quant          Quantization
	heuristic      bool
	seed           int64

	dir            [dirL1Size]atomic.Pointer[dirPage[K]]
	extMu          sync.RWMutex
	extToInt       map[K]uint32
	nextID         atomic.Uint32
	liveCount      atomic.Int64
	tombstoneCount atomic.Int64
	entryState     atomic.Uint64

	writeGate   sync.RWMutex
	readStripes [readStripes]paddedRWMutex
	nodeLocks   [nodeLockSize]sync.RWMutex

	rngMu       sync.Mutex
	rng         *rand.Rand
	stripeSeq   atomic.Uint32
	scratchPool sync.Pool
}

// New constructs a new HNSW [Index] for vectors of fixed dimension dim configured with opts.
//
// Returns [ErrInvalidDimension] if dim <= 0, or an error wrapping [ErrInvalidConfig]
// if any option is invalid or incompatible (such as Mmax < M).
func New[K comparable](dim int, opts ...Option) (*Index[K], error) {
	if dim <= 0 {
		return nil, ErrInvalidDimension
	}
	cfg := newDefaultConfig()
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	seed := cfg.seed
	if !cfg.hasSeed {
		seed = time.Now().UnixNano()
	}

	idx := &Index[K]{
		dim:            dim,
		m:              cfg.m,
		mmax:           cfg.mmax,
		efConstruction: cfg.efConstruction,
		mL:             1.0 / math.Log(float64(cfg.m)),
		metric:         cfg.metric,
		customDist:     cfg.customDist,
		quant:          cfg.quant,
		heuristic:      cfg.heuristic,
		seed:           seed,
		extToInt:       make(map[K]uint32),
		rng:            rand.New(rand.NewSource(seed)),
	}
	idx.efSearch.Store(int64(cfg.efSearch))
	idx.entryState.Store(emptyEntry)
	idx.initScratchPool()
	return idx, nil
}

func (idx *Index[K]) initScratchPool() {
	idx.scratchPool = sync.Pool{
		New: func() any {
			stripe := (idx.stripeSeq.Add(1) - 1) & readMask
			return newSearchScratch(stripe, idx.mmax)
		},
	}
}

func (idx *Index[K]) lockAllReadStripes() {
	for i := 0; i < readStripes; i++ {
		idx.readStripes[i].mu.Lock()
	}
}

func (idx *Index[K]) unlockAllReadStripes() {
	for i := readStripes - 1; i >= 0; i-- {
		idx.readStripes[i].mu.Unlock()
	}
}

func (idx *Index[K]) acquireExclusive() {
	idx.writeGate.Lock()
	idx.lockAllReadStripes()
}

func (idx *Index[K]) releaseExclusive() {
	idx.unlockAllReadStripes()
	idx.writeGate.Unlock()
}

func (idx *Index[K]) getChunk(id uint32) (*arenaChunk[K], int) {
	l1 := id >> (chunkBits + dirL2Bits)
	l2 := (id >> chunkBits) & dirL2Mask
	page := idx.dir[l1].Load()
	if page == nil {
		return nil, 0
	}
	return page.chunks[l2].Load(), int(id & chunkMask)
}

func (idx *Index[K]) vectorAt(chunk *arenaChunk[K], slot int) []float32 {
	off := slot * idx.dim
	return chunk.vecs[off : off+idx.dim : off+idx.dim]
}

func (idx *Index[K]) vectorF16At(chunk *arenaChunk[K], slot int) []uint16 {
	off := slot * idx.dim
	return chunk.vecsF16[off : off+idx.dim : off+idx.dim]
}

func (idx *Index[K]) extKeyAt(id uint32) K {
	chunk, slot := idx.getChunk(id)
	return chunk.intToExt[slot]
}

func isDeletedAt[K comparable](chunk *arenaChunk[K], slot int) bool {
	if chunk == nil {
		return true
	}
	word := slot >> 6
	bit := uint64(1) << (uint(slot) & 63)
	return (chunk.deleted[word].Load() & bit) != 0
}

func (idx *Index[K]) isDeleted(id uint32) bool {
	chunk, slot := idx.getChunk(id)
	return isDeletedAt(chunk, slot)
}

func (idx *Index[K]) markDeleted(id uint32) bool {
	chunk, slot := idx.getChunk(id)
	word := slot >> 6
	bit := uint64(1) << (uint(slot) & 63)
	for {
		old := chunk.deleted[word].Load()
		if (old & bit) != 0 {
			return false
		}
		if chunk.deleted[word].CompareAndSwap(old, old|bit) {
			return true
		}
	}
}

func (idx *Index[K]) maxConnForLevel(lvl int) int {
	if lvl == 0 {
		return idx.mmax
	}
	return idx.m
}

func (idx *Index[K]) randomLevel() uint8 {
	idx.rngMu.Lock()
	u := idx.rng.Float64()
	idx.rngMu.Unlock()
	if u < 1e-18 {
		u = 1e-18
	}
	lvl := int(-math.Log(u) * idx.mL)
	if lvl > maxNodeLevel {
		lvl = maxNodeLevel
	}
	if lvl < 0 {
		lvl = 0
	}
	return uint8(lvl)
}

func (idx *Index[K]) distQueryAt(chunk *arenaChunk[K], slot int, query []float32, qInvNorm float32) float32 {
	if idx.quant == QuantFloat16 {
		vec := idx.vectorF16At(chunk, slot)
		switch idx.metric {
		case MetricEuclidean, MetricSquaredEuclidean:
			return l2SquaredF16F32(vec, query)
		case MetricCosine:
			return cosineWithInvNormsF16F32(vec, query, chunk.invNorms[slot], qInvNorm)
		case MetricDotProduct:
			d := dotProductF16F32(vec, query)
			if d == 0 {
				return 0
			}
			return -d
		default:
			tmp := make([]float32, idx.dim)
			decodeFloat16Slice(tmp, vec)
			return idx.customDist(tmp, query)
		}
	}
	vec := idx.vectorAt(chunk, slot)
	switch idx.metric {
	case MetricEuclidean, MetricSquaredEuclidean:
		return l2Squared(vec, query)
	case MetricCosine:
		return cosineWithInvNorms(vec, query, chunk.invNorms[slot], qInvNorm)
	case MetricDotProduct:
		d := dotProduct(vec, query)
		if d == 0 {
			return 0
		}
		return -d
	default:
		return idx.customDist(vec, query)
	}
}

func (idx *Index[K]) distVecs(va []float32, invA float32, vb []float32, invB float32) float32 {
	switch idx.metric {
	case MetricEuclidean, MetricSquaredEuclidean:
		return l2Squared(va, vb)
	case MetricCosine:
		return cosineWithInvNorms(va, vb, invA, invB)
	case MetricDotProduct:
		d := dotProduct(va, vb)
		if d == 0 {
			return 0
		}
		return -d
	default:
		return idx.customDist(va, vb)
	}
}

func (idx *Index[K]) distVecsF16(va []uint16, invA float32, vb []uint16, invB float32, scratch *searchScratch) float32 {
	switch idx.metric {
	case MetricEuclidean, MetricSquaredEuclidean:
		return l2SquaredF16F16(va, vb)
	case MetricCosine:
		return cosineWithInvNormsF16F16(va, vb, invA, invB)
	case MetricDotProduct:
		d := dotProductF16F16(va, vb)
		if d == 0 {
			return 0
		}
		return -d
	default:
		if scratch != nil {
			if len(scratch.tmpVecA) < idx.dim {
				scratch.tmpVecA = make([]float32, idx.dim)
				scratch.tmpVecB = make([]float32, idx.dim)
			}
			decodeFloat16Slice(scratch.tmpVecA[:idx.dim], va)
			decodeFloat16Slice(scratch.tmpVecB[:idx.dim], vb)
			return idx.customDist(scratch.tmpVecA[:idx.dim], scratch.tmpVecB[:idx.dim])
		}
		fa := make([]float32, idx.dim)
		fb := make([]float32, idx.dim)
		decodeFloat16Slice(fa, va)
		decodeFloat16Slice(fb, vb)
		return idx.customDist(fa, fb)
	}
}

func (idx *Index[K]) distQuery(id uint32, query []float32, qInvNorm float32) float32 {
	chunk, slot := idx.getChunk(id)
	return idx.distQueryAt(chunk, slot, query, qInvNorm)
}

func (idx *Index[K]) distNodes(a, b uint32) float32 {
	ca, sa := idx.getChunk(a)
	cb, sb := idx.getChunk(b)
	if idx.quant == QuantFloat16 {
		va := idx.vectorF16At(ca, sa)
		vb := idx.vectorF16At(cb, sb)
		var invA, invB float32
		if idx.metric == MetricCosine {
			invA = ca.invNorms[sa]
			invB = cb.invNorms[sb]
		}
		return idx.distVecsF16(va, invA, vb, invB, nil)
	}
	va := idx.vectorAt(ca, sa)
	vb := idx.vectorAt(cb, sb)
	var invA, invB float32
	if idx.metric == MetricCosine {
		invA = ca.invNorms[sa]
		invB = cb.invNorms[sb]
	}
	return idx.distVecs(va, invA, vb, invB)
}

func (idx *Index[K]) copyNeighbors(id uint32, lvl int, dst []uint32, lockNode bool) []uint32 {
	if lockNode {
		idx.nodeLocks[id&nodeLockMask].RLock()
	}
	chunk, slot := idx.getChunk(id)
	if lvl == 0 {
		cnt := int(chunk.l0Count[slot])
		base := slot * idx.mmax
		if cap(dst) >= cnt {
			dst = dst[:cnt]
			copy(dst, chunk.l0Edges[base:base+cnt])
		} else {
			dst = append(dst[:0], chunk.l0Edges[base:base+cnt]...)
		}
	} else if lvl <= int(chunk.maxLevels[slot]) {
		src := chunk.upperLevels[slot][lvl-1]
		if cap(dst) >= len(src) {
			dst = dst[:len(src)]
			copy(dst, src)
		} else {
			dst = append(dst[:0], src...)
		}
	} else {
		dst = dst[:0]
	}
	if lockNode {
		idx.nodeLocks[id&nodeLockMask].RUnlock()
	}
	return dst
}

func (idx *Index[K]) setNeighbors(id uint32, lvl int, nbrs []uint32, lockNode bool) {
	if lockNode {
		idx.nodeLocks[id&nodeLockMask].Lock()
	}
	chunk, slot := idx.getChunk(id)
	if lvl == 0 {
		n := len(nbrs)
		if n > idx.mmax {
			n = idx.mmax
		}
		base := slot * idx.mmax
		copy(chunk.l0Edges[base:base+n], nbrs[:n])
		chunk.l0Count[slot] = uint16(n)
	} else if lvl <= int(chunk.maxLevels[slot]) {
		n := len(nbrs)
		if n > idx.m {
			n = idx.m
		}
		dst := chunk.upperLevels[slot][lvl-1]
		if cap(dst) < n {
			dst = make([]uint32, n, idx.m)
		} else {
			dst = dst[:n]
		}
		copy(dst, nbrs[:n])
		chunk.upperLevels[slot][lvl-1] = dst
	}
	if lockNode {
		idx.nodeLocks[id&nodeLockMask].Unlock()
	}
}

func (idx *Index[K]) allocSlot(extID K, vec []float32, lvl uint8) uint32 {
	newID := idx.nextID.Load()
	l1 := newID >> (chunkBits + dirL2Bits)
	l2 := (newID >> chunkBits) & dirL2Mask
	slot := int(newID & chunkMask)

	page := idx.dir[l1].Load()
	if page == nil {
		page = &dirPage[K]{}
		idx.dir[l1].Store(page)
	}
	chunk := page.chunks[l2].Load()
	if chunk == nil {
		chunk = &arenaChunk[K]{
			intToExt: make([]K, chunkSize),
			l0Edges:  make([]uint32, chunkSize*idx.mmax),
		}
		if idx.quant == QuantFloat16 {
			chunk.vecsF16 = make([]uint16, chunkSize*idx.dim)
		} else {
			chunk.vecs = make([]float32, chunkSize*idx.dim)
		}
		if idx.metric == MetricCosine {
			chunk.invNorms = make([]float32, chunkSize)
		}
		page.chunks[l2].Store(chunk)
	}

	off := slot * idx.dim
	if idx.quant == QuantFloat16 {
		dst := chunk.vecsF16[off : off+idx.dim]
		encodeFloat16Slice(dst, vec)
		if idx.metric == MetricCosine {
			chunk.invNorms[slot] = computeInvNormF16(dst)
		}
	} else {
		copy(chunk.vecs[off:off+idx.dim], vec)
		if idx.metric == MetricCosine {
			chunk.invNorms[slot] = computeInvNorm(vec)
		}
	}
	chunk.intToExt[slot] = extID
	chunk.maxLevels[slot] = lvl
	chunk.l0Count[slot] = 0
	if lvl > 0 {
		chunk.upperLevels[slot] = make([][]uint32, lvl)
	}

	idx.nextID.Store(newID + 1)
	idx.extToInt[extID] = newID
	idx.liveCount.Add(1)
	return newID
}

// Insert adds or updates the vector associated with id.
// The input slice vec is copied; mutating vec after Insert returns does not affect the index.
// If id already exists, its vector is atomically replaced without leaving stale graph links or orphans.
//
// Returns [ErrDimensionMismatch] if len(vec) != idx.Dim(), or [ErrNaNOrInf] if vec contains NaN or ±Inf.
func (idx *Index[K]) Insert(id K, vec []float32) error {
	if len(vec) != idx.dim {
		return ErrDimensionMismatch
	}
	if err := validateVectorFinite(vec); err != nil {
		return err
	}
	idx.insertValidated(id, vec)
	return nil
}

func (idx *Index[K]) tryInsertFast(id K, vec []float32, scratch *searchScratch) bool {
	// Fast path for new keys: hold writeGate.RLock + P-local readStripe.RLock.
	idx.writeGate.RLock()
	defer idx.writeGate.RUnlock()
	idx.readStripes[scratch.stripe].mu.RLock()
	defer idx.readStripes[scratch.stripe].mu.RUnlock()

	idx.extMu.RLock()
	_, exists := idx.extToInt[id]
	idx.extMu.RUnlock()
	if exists {
		return false
	}

	lvl := idx.randomLevel()
	idx.extMu.Lock()
	if _, exists = idx.extToInt[id]; exists {
		idx.extMu.Unlock()
		return false
	}
	newID := idx.allocSlot(id, vec, lvl)
	idx.extMu.Unlock()

	idx.insertNode(newID, vec, int(lvl), scratch, false)
	return true
}

func (idx *Index[K]) insertValidated(id K, vec []float32) {
	scratch := idx.scratchPool.Get().(*searchScratch)
	defer idx.scratchPool.Put(scratch)
	idx.insertValidatedWithScratch(id, vec, scratch)
}

func (idx *Index[K]) insertValidatedWithScratch(id K, vec []float32, scratch *searchScratch) {
	if idx.tryInsertFast(id, vec, scratch) {
		return
	}

	// Exclusive path for upsert (existing key).
	idx.acquireExclusive()
	defer idx.releaseExclusive()

	if oldID, existed := idx.extToInt[id]; existed {
		idx.deleteInternal(id, oldID, scratch)
	}
	lvl := idx.randomLevel()
	newID := idx.allocSlot(id, vec, lvl)
	idx.insertNode(newID, vec, int(lvl), scratch, true)
}

func (idx *Index[K]) insertNode(newID uint32, vec []float32, newLevel int, scratch *searchScratch, exclusive bool) {
	qInvNorm := float32(0)
	if idx.metric == MetricCosine {
		qInvNorm = computeInvNorm(vec)
	}

	for {
		es := idx.entryState.Load()
		if es == emptyEntry {
			newES := (uint64(newLevel) << 32) | uint64(newID)
			if idx.entryState.CompareAndSwap(emptyEntry, newES) {
				return
			}
			continue
		}

		entryID := uint32(es)
		entryLevel := int(es >> 32)

		currID := entryID
		currDist := idx.distQuery(currID, vec, qInvNorm)

		for lvl := entryLevel; lvl > newLevel; lvl-- {
			currID, currDist = idx.greedyStep(currID, currDist, vec, qInvNorm, lvl, scratch, !exclusive)
		}

		topLvl := newLevel
		if entryLevel < topLvl {
			topLvl = entryLevel
		}

		for lvl := topLvl; lvl >= 0; lvl-- {
			idx.searchLayerInsert(vec, qInvNorm, currID, currDist, lvl, idx.efConstruction, scratch, !exclusive)
			if len(scratch.results) > 0 {
				best := scratch.results[0]
				for i := 1; i < len(scratch.results); i++ {
					if scratch.results[i].dist < best.dist {
						best = scratch.results[i]
					}
				}
				currID = best.id
				currDist = best.dist
			}

			maxConn := idx.maxConnForLevel(lvl)
			chosen := scratch.chosenByLevel[lvl][:0]
			if cap(chosen) < maxConn {
				chosen = make([]uint32, 0, maxConn)
			}
			chosen = idx.selectNeighborsFromHeap(newID, scratch.results, maxConn, scratch, chosen)
			scratch.chosenByLevel[lvl] = chosen
			idx.setNeighbors(newID, lvl, chosen, !exclusive)
		}

		// Publish reverse edges only after all levels 0..topLvl on newID are populated.
		for lvl := topLvl; lvl >= 0; lvl-- {
			for _, nbrID := range scratch.chosenByLevel[lvl] {
				idx.addConnection(nbrID, newID, lvl, scratch, !exclusive)
			}
		}

		if newLevel > entryLevel {
			for {
				oldES := idx.entryState.Load()
				if oldES != emptyEntry && int(oldES>>32) >= newLevel {
					break
				}
				newES := (uint64(newLevel) << 32) | uint64(newID)
				if idx.entryState.CompareAndSwap(oldES, newES) {
					break
				}
			}
		}
		return
	}
}

func (idx *Index[K]) greedyStep(currID uint32, currDist float32, query []float32, qInvNorm float32, lvl int, scratch *searchScratch, lockNode bool) (uint32, float32) {
	isL2 := idx.metric == MetricEuclidean || idx.metric == MetricSquaredEuclidean
	isF16 := idx.quant == QuantFloat16
	for {
		improved := false
		nbrs := idx.copyNeighbors(currID, lvl, scratch.neighbors, lockNode)
		scratch.neighbors = nbrs
		for _, nbrID := range nbrs {
			var d float32
			if isL2 {
				chunk, slot := idx.getChunk(nbrID)
				if isF16 {
					d = l2SquaredF16F32(idx.vectorF16At(chunk, slot), query)
				} else {
					d = l2Squared(idx.vectorAt(chunk, slot), query)
				}
			} else {
				d = idx.distQuery(nbrID, query, qInvNorm)
			}
			if d < currDist {
				currDist = d
				currID = nbrID
				improved = true
			}
		}
		if !improved {
			return currID, currDist
		}
	}
}

func (idx *Index[K]) searchLayerInsert(query []float32, qInvNorm float32, ep uint32, epDist float32, lvl int, ef int, scratch *searchScratch, lockNode bool) {
	numNodes := idx.nextID.Load()
	scratch.begin(numNodes)
	scratch.visit(ep)
	scratch.candidates.push(distNode{id: ep, dist: epDist})
	if !idx.isDeleted(ep) {
		scratch.results.push(distNode{id: ep, dist: epDist})
	}

	isL2 := idx.metric == MetricEuclidean || idx.metric == MetricSquaredEuclidean
	isF16 := idx.quant == QuantFloat16
	for len(scratch.candidates) > 0 {
		curr := scratch.candidates.pop()
		if len(scratch.results) >= ef && curr.dist > scratch.results[0].dist {
			break
		}
		nbrs := idx.copyNeighbors(curr.id, lvl, scratch.neighbors, lockNode)
		scratch.neighbors = nbrs
		for _, nbrID := range nbrs {
			if !scratch.visit(nbrID) {
				continue
			}
			chunk, slot := idx.getChunk(nbrID)
			var d float32
			if isL2 {
				if isF16 {
					d = l2SquaredF16F32(idx.vectorF16At(chunk, slot), query)
				} else {
					d = l2Squared(idx.vectorAt(chunk, slot), query)
				}
			} else {
				d = idx.distQueryAt(chunk, slot, query, qInvNorm)
			}
			if len(scratch.results) < ef || d < scratch.results[0].dist {
				scratch.candidates.push(distNode{id: nbrID, dist: d})
				if !isDeletedAt(chunk, slot) {
					if len(scratch.results) < ef {
						scratch.results.push(distNode{id: nbrID, dist: d})
					} else if d < scratch.results[0].dist {
						scratch.results.replaceMax(distNode{id: nbrID, dist: d})
					}
				}
			}
		}
	}
}

func (idx *Index[K]) selectNeighborsFromHeap(baseID uint32, cands []distNode, maxConn int, scratch *searchScratch, out []uint32) []uint32 {
	work := scratch.heuristicWork[:0]
	for _, c := range cands {
		if c.id == baseID || idx.isDeleted(c.id) {
			continue
		}
		dup := false
		for _, existing := range work {
			if existing.id == c.id {
				dup = true
				break
			}
		}
		if !dup {
			work = append(work, c)
		}
	}
	scratch.heuristicWork = work
	return idx.selectNeighborsWork(work, maxConn, scratch, out)
}

func cmpDistNode(a, b distNode) int {
	if a.dist < b.dist {
		return -1
	}
	if a.dist > b.dist {
		return 1
	}
	if a.id < b.id {
		return -1
	}
	if a.id > b.id {
		return 1
	}
	return 0
}

func (idx *Index[K]) selectNeighborsFromIDs(baseID uint32, ids []uint32, maxConn int, scratch *searchScratch, out []uint32) []uint32 {
	work := scratch.heuristicWork[:0]
	baseChunk, baseSlot := idx.getChunk(baseID)
	var baseInvNorm float32
	if idx.metric == MetricCosine {
		baseInvNorm = baseChunk.invNorms[baseSlot]
	}
	isL2 := idx.metric == MetricEuclidean || idx.metric == MetricSquaredEuclidean

	if idx.quant == QuantFloat16 {
		baseVec := idx.vectorF16At(baseChunk, baseSlot)
		for _, cid := range ids {
			if cid == baseID {
				continue
			}
			cChunk, cSlot := idx.getChunk(cid)
			if isDeletedAt(cChunk, cSlot) {
				continue
			}
			dup := false
			for _, existing := range work {
				if existing.id == cid {
					dup = true
					break
				}
			}
			if !dup {
				var d float32
				if isL2 {
					d = l2SquaredF16F16(baseVec, idx.vectorF16At(cChunk, cSlot))
				} else {
					var cInvNorm float32
					if idx.metric == MetricCosine {
						cInvNorm = cChunk.invNorms[cSlot]
					}
					d = idx.distVecsF16(baseVec, baseInvNorm, idx.vectorF16At(cChunk, cSlot), cInvNorm, scratch)
				}
				work = append(work, distNode{id: cid, dist: d})
			}
		}
		scratch.heuristicWork = work
		return idx.selectNeighborsWork(work, maxConn, scratch, out)
	}

	baseVec := idx.vectorAt(baseChunk, baseSlot)
	for _, cid := range ids {
		if cid == baseID {
			continue
		}
		cChunk, cSlot := idx.getChunk(cid)
		if isDeletedAt(cChunk, cSlot) {
			continue
		}
		dup := false
		for _, existing := range work {
			if existing.id == cid {
				dup = true
				break
			}
		}
		if !dup {
			var d float32
			if isL2 {
				d = l2Squared(baseVec, idx.vectorAt(cChunk, cSlot))
			} else {
				var cInvNorm float32
				if idx.metric == MetricCosine {
					cInvNorm = cChunk.invNorms[cSlot]
				}
				d = idx.distVecs(baseVec, baseInvNorm, idx.vectorAt(cChunk, cSlot), cInvNorm)
			}
			work = append(work, distNode{id: cid, dist: d})
		}
	}
	scratch.heuristicWork = work
	return idx.selectNeighborsWork(work, maxConn, scratch, out)
}

func (idx *Index[K]) selectNeighborsWork(work []distNode, maxConn int, scratch *searchScratch, out []uint32) []uint32 {
	out = out[:0]
	if len(work) <= maxConn {
		for _, w := range work {
			out = append(out, w.id)
		}
		return out
	}

	slices.SortFunc(work, cmpDistNode)

	if !idx.heuristic {
		for i := 0; i < maxConn; i++ {
			out = append(out, work[i].id)
		}
		return out
	}

	selected := scratch.selected[:0]
	selectedInvNorms := scratch.selectedInvNorms[:0]
	discarded := scratch.discarded[:0]

	isL2 := idx.metric == MetricEuclidean || idx.metric == MetricSquaredEuclidean
	isCosine := idx.metric == MetricCosine

	if idx.quant == QuantFloat16 {
		selectedVecsF16 := scratch.selectedVecsF16[:0]
		for _, e := range work {
			if len(selected) >= maxConn {
				break
			}
			eChunk, eSlot := idx.getChunk(e.id)
			eVec := idx.vectorF16At(eChunk, eSlot)
			var eInvNorm float32
			if isCosine {
				eInvNorm = eChunk.invNorms[eSlot]
			}
			good := true
			if isL2 {
				for _, sVec := range selectedVecsF16 {
					if l2SquaredF16F16(eVec, sVec) < e.dist {
						good = false
						break
					}
				}
			} else {
				for i, sVec := range selectedVecsF16 {
					var sInvNorm float32
					if isCosine {
						sInvNorm = selectedInvNorms[i]
					}
					if idx.distVecsF16(eVec, eInvNorm, sVec, sInvNorm, scratch) < e.dist {
						good = false
						break
					}
				}
			}
			if good {
				selected = append(selected, e)
				selectedVecsF16 = append(selectedVecsF16, eVec)
				if isCosine {
					selectedInvNorms = append(selectedInvNorms, eInvNorm)
				}
			} else {
				discarded = append(discarded, e)
			}
		}

		for i := 0; i < len(discarded) && len(selected) < maxConn; i++ {
			selected = append(selected, discarded[i])
		}
		for _, s := range selected {
			out = append(out, s.id)
		}
		clear(selectedVecsF16)
		scratch.selected = selected
		scratch.selectedVecsF16 = selectedVecsF16[:0]
		scratch.selectedInvNorms = selectedInvNorms[:0]
		scratch.discarded = discarded
		return out
	}

	selectedVecs := scratch.selectedVecs[:0]
	for _, e := range work {
		if len(selected) >= maxConn {
			break
		}
		eChunk, eSlot := idx.getChunk(e.id)
		eVec := idx.vectorAt(eChunk, eSlot)
		var eInvNorm float32
		if isCosine {
			eInvNorm = eChunk.invNorms[eSlot]
		}
		good := true
		if isL2 {
			for _, sVec := range selectedVecs {
				if l2Squared(eVec, sVec) < e.dist {
					good = false
					break
				}
			}
		} else {
			for i, sVec := range selectedVecs {
				var sInvNorm float32
				if isCosine {
					sInvNorm = selectedInvNorms[i]
				}
				if idx.distVecs(eVec, eInvNorm, sVec, sInvNorm) < e.dist {
					good = false
					break
				}
			}
		}
		if good {
			selected = append(selected, e)
			selectedVecs = append(selectedVecs, eVec)
			if isCosine {
				selectedInvNorms = append(selectedInvNorms, eInvNorm)
			}
		} else {
			discarded = append(discarded, e)
		}
	}

	for i := 0; i < len(discarded) && len(selected) < maxConn; i++ {
		selected = append(selected, discarded[i])
	}

	for _, s := range selected {
		out = append(out, s.id)
	}
	clear(selectedVecs)
	scratch.selected = selected
	scratch.selectedVecs = selectedVecs[:0]
	scratch.selectedInvNorms = selectedInvNorms[:0]
	scratch.discarded = discarded
	return out
}

func (idx *Index[K]) addConnection(targetID, newNbr uint32, lvl int, scratch *searchScratch, lockNode bool) {
	if targetID == newNbr {
		return
	}
	if lockNode {
		idx.nodeLocks[targetID&nodeLockMask].Lock()
		defer idx.nodeLocks[targetID&nodeLockMask].Unlock()
	}

	chunk, slot := idx.getChunk(targetID)
	if isDeletedAt(chunk, slot) || lvl > int(chunk.maxLevels[slot]) {
		return
	}

	maxConn := idx.maxConnForLevel(lvl)
	cand := scratch.candIDs[:0]
	if lvl == 0 {
		cnt := int(chunk.l0Count[slot])
		base := slot * idx.mmax
		for _, nb := range chunk.l0Edges[base : base+cnt] {
			if nb == newNbr {
				scratch.candIDs = cand
				return
			}
			if !idx.isDeleted(nb) {
				cand = append(cand, nb)
			}
		}
	} else {
		for _, nb := range chunk.upperLevels[slot][lvl-1] {
			if nb == newNbr {
				scratch.candIDs = cand
				return
			}
			if !idx.isDeleted(nb) {
				cand = append(cand, nb)
			}
		}
	}
	cand = append(cand, newNbr)
	scratch.candIDs = cand

	if len(cand) <= maxConn {
		idx.setNeighbors(targetID, lvl, cand, false)
		return
	}

	chosen := idx.selectNeighborsFromIDs(targetID, cand, maxConn, scratch, scratch.chosenBuf[:0])
	scratch.chosenBuf = chosen
	idx.setNeighbors(targetID, lvl, chosen, false)
}

// addConnectionsBatch merges all candidate neighbors in newNbrs into targetID at lvl
// in a single pass, executing Algorithm 4 neighbor pruning at most once per target node.
// Caller must hold exclusive index access.
func (idx *Index[K]) addConnectionsBatch(targetID uint32, newNbrs []uint32, lvl int, scratch *searchScratch) {
	chunk, slot := idx.getChunk(targetID)
	if isDeletedAt(chunk, slot) || lvl > int(chunk.maxLevels[slot]) {
		return
	}

	maxConn := idx.maxConnForLevel(lvl)
	cand := scratch.candIDs[:0]
	if lvl == 0 {
		cnt := int(chunk.l0Count[slot])
		base := slot * idx.mmax
		for _, nb := range chunk.l0Edges[base : base+cnt] {
			if nb != targetID && !idx.isDeleted(nb) {
				cand = append(cand, nb)
			}
		}
	} else {
		for _, nb := range chunk.upperLevels[slot][lvl-1] {
			if nb != targetID && !idx.isDeleted(nb) {
				cand = append(cand, nb)
			}
		}
	}

	existingLen := len(cand)
	for _, v := range newNbrs {
		if v == targetID {
			continue
		}
		dup := false
		for _, prev := range cand {
			if prev == v {
				dup = true
				break
			}
		}
		if !dup {
			cand = append(cand, v)
		}
	}
	scratch.candIDs = cand
	if len(cand) == existingLen {
		return
	}

	if len(cand) <= maxConn {
		idx.setNeighbors(targetID, lvl, cand, false)
		return
	}

	chosen := idx.selectNeighborsFromIDs(targetID, cand, maxConn, scratch, scratch.chosenBuf[:0])
	scratch.chosenBuf = chosen
	idx.setNeighbors(targetID, lvl, chosen, false)
}

func (idx *Index[K]) removeEdge(targetID, deadID uint32, lvl int) {
	chunk, slot := idx.getChunk(targetID)
	if chunk == nil || lvl > int(chunk.maxLevels[slot]) {
		return
	}
	if lvl == 0 {
		cnt := int(chunk.l0Count[slot])
		base := slot * idx.mmax
		edges := chunk.l0Edges[base : base+cnt]
		w := 0
		for _, nb := range edges {
			if nb != deadID && !idx.isDeleted(nb) {
				edges[w] = nb
				w++
			}
		}
		chunk.l0Count[slot] = uint16(w)
		return
	}
	edges := chunk.upperLevels[slot][lvl-1]
	w := 0
	for _, nb := range edges {
		if nb != deadID && !idx.isDeleted(nb) {
			edges[w] = nb
			w++
		}
	}
	chunk.upperLevels[slot][lvl-1] = edges[:w]
}

// Delete removes id from the index, repairs 2-hop neighborhoods among its surviving
// neighbors, and re-elects the entry point if necessary. It returns true if id was present.
// Tombstoned arena slots remain allocated until [Index.Compact] or [Index.Clear] is called.
func (idx *Index[K]) Delete(id K) bool {
	scratch := idx.scratchPool.Get().(*searchScratch)
	defer idx.scratchPool.Put(scratch)

	idx.acquireExclusive()
	defer idx.releaseExclusive()

	oldID, ok := idx.extToInt[id]
	if !ok {
		return false
	}
	idx.deleteInternal(id, oldID, scratch)
	return true
}

func (idx *Index[K]) deleteInternal(extID K, oldID uint32, scratch *searchScratch) {
	delete(idx.extToInt, extID)
	if idx.markDeleted(oldID) {
		idx.liveCount.Add(-1)
		idx.tombstoneCount.Add(1)
	}

	chunk, slot := idx.getChunk(oldID)
	oldMaxLvl := int(chunk.maxLevels[slot])

	// Snapshot neighbors per level, remove oldID, and run batched 2-hop repair among live neighbors.
	nbrBuf := scratch.neighbors[:0]
	for lvl := 0; lvl <= oldMaxLvl; lvl++ {
		nbrBuf = idx.copyNeighbors(oldID, lvl, nbrBuf, false)
		w := 0
		for _, nb := range nbrBuf {
			if nb != oldID && !idx.isDeleted(nb) {
				nbrBuf[w] = nb
				w++
			}
		}
		liveNbrs := nbrBuf[:w]
		for _, u := range liveNbrs {
			idx.removeEdge(u, oldID, lvl)
		}
		if len(liveNbrs) > 1 {
			for _, u := range liveNbrs {
				idx.addConnectionsBatch(u, liveNbrs, lvl, scratch)
			}
		}
	}
	scratch.neighbors = nbrBuf[:0]

	if idx.liveCount.Load() == 0 {
		idx.entryState.Store(emptyEntry)
		return
	}
	if uint32(idx.entryState.Load()) == oldID {
		idx.electEntryPoint()
	}
}

func (idx *Index[K]) electEntryPoint() {
	total := idx.nextID.Load()
	bestID := invalidID
	bestLvl := -1
	for id := uint32(0); id < total; id++ {
		if idx.isDeleted(id) {
			continue
		}
		chunk, slot := idx.getChunk(id)
		lvl := int(chunk.maxLevels[slot])
		if lvl > bestLvl {
			bestLvl = lvl
			bestID = id
		}
	}
	if bestID == invalidID {
		idx.entryState.Store(emptyEntry)
		return
	}
	idx.entryState.Store((uint64(bestLvl) << 32) | uint64(bestID))
}

// readStripeForCall derives a cache-line-padded read stripe from the caller's
// stack address, avoiding sync.Pool traffic and shared atomic counters on fast reads.
func readStripeForCall() uint32 {
	var anchor byte
	return uint32((uintptr(unsafe.Pointer(&anchor)) >> 12) & readMask)
}

// Get returns an independent copy of the stored vector for id and true if present,
// or nil and false otherwise. For [MetricCosine], Get returns the exact unnormalized
// vector originally inserted.
func (idx *Index[K]) Get(id K) ([]float32, bool) {
	stripe := readStripeForCall()
	idx.readStripes[stripe].mu.RLock()
	defer idx.readStripes[stripe].mu.RUnlock()

	idx.extMu.RLock()
	intID, ok := idx.extToInt[id]
	idx.extMu.RUnlock()
	if !ok {
		return nil, false
	}
	chunk, slot := idx.getChunk(intID)
	if isDeletedAt(chunk, slot) {
		return nil, false
	}
	out := make([]float32, idx.dim)
	if idx.quant == QuantFloat16 {
		decodeFloat16Slice(out, idx.vectorF16At(chunk, slot))
	} else {
		copy(out, idx.vectorAt(chunk, slot))
	}
	return out, true
}

// Contains reports whether id is currently present in the index.
func (idx *Index[K]) Contains(id K) bool {
	stripe := readStripeForCall()
	idx.readStripes[stripe].mu.RLock()
	defer idx.readStripes[stripe].mu.RUnlock()

	idx.extMu.RLock()
	intID, ok := idx.extToInt[id]
	idx.extMu.RUnlock()
	return ok && !idx.isDeleted(intID)
}

// Keys returns a snapshot slice of all live external keys currently stored in the index.
// Key iteration order is unspecified. Safe for concurrent use.
func (idx *Index[K]) Keys() []K {
	stripe := readStripeForCall()
	idx.readStripes[stripe].mu.RLock()
	defer idx.readStripes[stripe].mu.RUnlock()

	idx.extMu.RLock()
	keys := make([]K, 0, len(idx.extToInt))
	for k := range idx.extToInt {
		keys = append(keys, k)
	}
	idx.extMu.RUnlock()
	return keys
}

// All returns a Go 1.23 [iter.Seq2] iterator over all live (key, vector) pairs in the index.
// Each yielded vector is an independent copy (exact unnormalized values for [MetricCosine]).
//
// All snapshots the live keys up front and does not hold index locks across yield calls,
// so callers may safely invoke [Index.Search], [Index.Insert], or [Index.Delete] inside
// the range loop without deadlocking. Keys concurrently deleted before being visited are skipped.
func (idx *Index[K]) All() iter.Seq2[K, []float32] {
	return func(yield func(K, []float32) bool) {
		keys := idx.Keys()
		for _, k := range keys {
			vec, ok := idx.Get(k)
			if !ok {
				continue
			}
			if !yield(k, vec) {
				return
			}
		}
	}
}

// SetEfSearch atomically updates the default efSearch beam width used by subsequent
// [Index.Search], [Index.SearchFilter], and [Index.SearchBatch] calls.
// Returns an error wrapping [ErrInvalidConfig] if ef <= 0.
func (idx *Index[K]) SetEfSearch(ef int) error {
	if ef <= 0 {
		return fmt.Errorf("%w: EfSearch must be positive", ErrInvalidConfig)
	}
	idx.efSearch.Store(int64(ef))
	return nil
}

// Len returns the number of live vectors currently stored in the index.
func (idx *Index[K]) Len() int {
	return int(idx.liveCount.Load())
}

// Dim returns the fixed vector dimension of the index.
func (idx *Index[K]) Dim() int {
	return idx.dim
}

// Clear removes all vectors and resets the index to an empty state while retaining configuration.
func (idx *Index[K]) Clear() {
	idx.acquireExclusive()
	defer idx.releaseExclusive()

	for i := 0; i < dirL1Size; i++ {
		idx.dir[i].Store(nil)
	}
	idx.extToInt = make(map[K]uint32)
	idx.nextID.Store(0)
	idx.liveCount.Store(0)
	idx.tombstoneCount.Store(0)
	idx.entryState.Store(emptyEntry)
}

// Compact reclaims tombstoned arena slots left by deletions and updates, densely
// remapping all live nodes and graph edges while preserving external keys and vectors.
// If no tombstoned slots exist, Compact is a no-op.
func (idx *Index[K]) Compact() error {
	idx.acquireExclusive()
	defer idx.releaseExclusive()

	if idx.tombstoneCount.Load() == 0 {
		return nil
	}
	if idx.liveCount.Load() == 0 {
		for i := 0; i < dirL1Size; i++ {
			idx.dir[i].Store(nil)
		}
		idx.extToInt = make(map[K]uint32)
		idx.nextID.Store(0)
		idx.tombstoneCount.Store(0)
		idx.entryState.Store(emptyEntry)
		return nil
	}

	oldNext := idx.nextID.Load()
	oldToNew := make([]uint32, oldNext)
	for i := range oldToNew {
		oldToNew[i] = invalidID
	}

	liveOld := make([]uint32, 0, idx.liveCount.Load())
	for id := uint32(0); id < oldNext; id++ {
		if !idx.isDeleted(id) {
			oldToNew[id] = uint32(len(liveOld))
			liveOld = append(liveOld, id)
		}
	}

	var newPages [dirL1Size]*dirPage[K]
	newExtToInt := make(map[K]uint32, len(liveOld))
	bestEntryID := uint32(0)
	bestEntryLevel := -1

	for newIdx, oldID := range liveOld {
		newID := uint32(newIdx)
		oldChunk, oldSlot := idx.getChunk(oldID)

		l1 := newID >> (chunkBits + dirL2Bits)
		l2 := (newID >> chunkBits) & dirL2Mask
		newSlot := int(newID & chunkMask)

		page := newPages[l1]
		if page == nil {
			page = &dirPage[K]{}
			newPages[l1] = page
		}
		newChunk := page.chunks[l2].Load()
		if newChunk == nil {
			newChunk = &arenaChunk[K]{
				intToExt: make([]K, chunkSize),
				l0Edges:  make([]uint32, chunkSize*idx.mmax),
			}
			if idx.quant == QuantFloat16 {
				newChunk.vecsF16 = make([]uint16, chunkSize*idx.dim)
			} else {
				newChunk.vecs = make([]float32, chunkSize*idx.dim)
			}
			if idx.metric == MetricCosine {
				newChunk.invNorms = make([]float32, chunkSize)
			}
			page.chunks[l2].Store(newChunk)
		}

		if idx.quant == QuantFloat16 {
			copy(newChunk.vecsF16[newSlot*idx.dim:(newSlot+1)*idx.dim], idx.vectorF16At(oldChunk, oldSlot))
		} else {
			copy(newChunk.vecs[newSlot*idx.dim:(newSlot+1)*idx.dim], idx.vectorAt(oldChunk, oldSlot))
		}
		if idx.metric == MetricCosine {
			newChunk.invNorms[newSlot] = oldChunk.invNorms[oldSlot]
		}
		extKey := oldChunk.intToExt[oldSlot]
		newChunk.intToExt[newSlot] = extKey
		newExtToInt[extKey] = newID

		maxLvl := oldChunk.maxLevels[oldSlot]
		newChunk.maxLevels[newSlot] = maxLvl
		if int(maxLvl) > bestEntryLevel {
			bestEntryLevel = int(maxLvl)
			bestEntryID = newID
		}

		// Remap layer 0 edges.
		oldCnt := int(oldChunk.l0Count[oldSlot])
		oldBase := oldSlot * idx.mmax
		newBase := newSlot * idx.mmax
		w := 0
		for _, oldNb := range oldChunk.l0Edges[oldBase : oldBase+oldCnt] {
			if oldNb >= oldNext {
				continue
			}
			mapped := oldToNew[oldNb]
			if mapped == invalidID || mapped == newID {
				continue
			}
			dup := false
			for j := 0; j < w; j++ {
				if newChunk.l0Edges[newBase+j] == mapped {
					dup = true
					break
				}
			}
			if !dup && w < idx.mmax {
				newChunk.l0Edges[newBase+w] = mapped
				w++
			}
		}
		newChunk.l0Count[newSlot] = uint16(w)

		// Remap upper levels.
		if maxLvl > 0 {
			newUpper := make([][]uint32, maxLvl)
			for l := 0; l < int(maxLvl); l++ {
				src := oldChunk.upperLevels[oldSlot][l]
				dst := make([]uint32, 0, len(src))
				for _, oldNb := range src {
					if oldNb >= oldNext {
						continue
					}
					mapped := oldToNew[oldNb]
					if mapped == invalidID || mapped == newID {
						continue
					}
					dup := false
					for _, prev := range dst {
						if prev == mapped {
							dup = true
							break
						}
					}
					if !dup && len(dst) < idx.m {
						dst = append(dst, mapped)
					}
				}
				newUpper[l] = dst
			}
			newChunk.upperLevels[newSlot] = newUpper
		}
	}

	for i := 0; i < dirL1Size; i++ {
		idx.dir[i].Store(newPages[i])
	}
	idx.extToInt = newExtToInt
	idx.nextID.Store(uint32(len(liveOld)))
	idx.tombstoneCount.Store(0)
	idx.entryState.Store((uint64(bestEntryLevel) << 32) | uint64(bestEntryID))
	return nil
}

// Stats returns a snapshot of current index counts, parameters, and estimated memory usage.
func (idx *Index[K]) Stats() IndexStats {
	idx.writeGate.RLock()
	defer idx.writeGate.RUnlock()

	es := idx.entryState.Load()
	entrySet := es != emptyEntry
	maxLevel := 0
	if entrySet {
		maxLevel = int(es >> 32)
	}

	totalSlots := idx.nextID.Load()
	numChunks := (int(totalSlots) + chunkSize - 1) / chunkSize
	capSlots := numChunks * chunkSize

	var memBytes int64
	memBytes += int64(unsafe.Sizeof(*idx))
	var keyZero K
	vecElemBytes := int64(4)
	if idx.quant == QuantFloat16 {
		vecElemBytes = 2
	}
	perChunkBytes := int64(chunkSize*idx.dim)*vecElemBytes +
		int64(chunkSize)*int64(unsafe.Sizeof(keyZero)) +
		int64(chunkSize*idx.mmax*4) +
		int64(unsafe.Sizeof(arenaChunk[K]{}))
	if idx.metric == MetricCosine {
		perChunkBytes += int64(chunkSize * 4)
	}
	memBytes += int64(numChunks) * perChunkBytes

	for id := uint32(0); id < totalSlots; id++ {
		chunk, slot := idx.getChunk(id)
		if chunk != nil && chunk.maxLevels[slot] > 0 {
			idx.nodeLocks[id&nodeLockMask].RLock()
			for _, lvlSlice := range chunk.upperLevels[slot] {
				memBytes += int64(cap(lvlSlice)*4) + int64(unsafe.Sizeof(lvlSlice))
			}
			idx.nodeLocks[id&nodeLockMask].RUnlock()
		}
	}

	return IndexStats{
		Len:            int(idx.liveCount.Load()),
		Deleted:        int(idx.tombstoneCount.Load()),
		Capacity:       capSlots,
		Dim:            idx.dim,
		MaxLevel:       maxLevel,
		EntryPointSet:  entrySet,
		M:              idx.m,
		Mmax:           idx.mmax,
		EfConstruction: idx.efConstruction,
		EfSearch:       int(idx.efSearch.Load()),
		Metric:         idx.metric,
		Quantization:   idx.quant,
		Heuristic:      idx.heuristic,
		MemoryBytes:    memBytes,
	}
}
