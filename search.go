package hnsw

import (
	"context"
	"math"

	"github.com/BryceWayne/hnsw/internal/simd"
)

// Search returns up to k approximate nearest neighbors to query ordered by ascending distance.
// When the index is empty, Search returns an empty non-nil slice and a nil error.
// The effective beam width is max(efSearch, k) and may be overridden per call with [WithSearchEf],
// bounded by distance with [WithMaxDistance], or canceled with [WithContext].
//
// Returns [ErrInvalidK] if k <= 0, [ErrDimensionMismatch] if len(query) != idx.Dim(),
// [ErrNaNOrInf] if query contains NaN or ±Inf, [ErrInvalidConfig] for invalid options,
// or ctx.Err() if canceled.
func (idx *Index[K]) Search(query []float32, k int, opts ...SearchOption) ([]SearchResult[K], error) {
	return idx.searchCore(query, k, nil, nil, opts...)
}

// SearchInto executes [Index.Search] and writes up to k results into dst[:0], reusing dst's
// backing array when cap(dst) >= result count to achieve zero heap allocations in steady state.
// If dst is nil or too small, SearchInto allocates a new slice.
//
// Returns the same validation and cancellation errors as [Index.Search].
func (idx *Index[K]) SearchInto(query []float32, k int, dst []SearchResult[K], opts ...SearchOption) ([]SearchResult[K], error) {
	return idx.searchCore(query, k, nil, dst, opts...)
}

// SearchFilter returns up to k approximate nearest neighbors whose keys satisfy filter,
// ordered by ascending distance. Non-matching nodes are still traversed during layer-0
// beam search so graph connectivity is preserved without post-filtering truncation.
// If filter is nil, SearchFilter behaves identically to [Index.Search].
// Callers must not mutate the index inside filter.
//
// Returns the same validation and cancellation errors as [Index.Search].
func (idx *Index[K]) SearchFilter(query []float32, k int, filter func(id K) bool, opts ...SearchOption) ([]SearchResult[K], error) {
	return idx.searchCore(query, k, filter, nil, opts...)
}

// SearchFilterInto executes [Index.SearchFilter] and writes up to k matching results into dst[:0],
// reusing dst's backing array when cap(dst) >= result count to achieve zero heap allocations in
// steady state. If dst is nil or too small, SearchFilterInto allocates a new slice.
// If filter is nil, SearchFilterInto behaves identically to [Index.SearchInto].
// Callers must not mutate the index inside filter.
//
// Returns the same validation and cancellation errors as [Index.Search].
func (idx *Index[K]) SearchFilterInto(query []float32, k int, filter func(id K) bool, dst []SearchResult[K], opts ...SearchOption) ([]SearchResult[K], error) {
	return idx.searchCore(query, k, filter, dst, opts...)
}

func (idx *Index[K]) searchCore(query []float32, k int, filter func(id K) bool, dst []SearchResult[K], opts ...SearchOption) ([]SearchResult[K], error) {
	ef := int(idx.efSearch.Load())
	var ctx context.Context
	var maxDist float32
	var maxDistSet bool
	if len(opts) > 0 {
		sc := defaultSearchConfig()
		if err := sc.apply(opts); err != nil {
			return nil, err
		}
		if sc.efSet {
			ef = sc.ef
		}
		ctx = sc.ctx
		maxDist = sc.maxDist
		maxDistSet = sc.maxDistSet
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	if len(query) != idx.dim {
		return nil, ErrDimensionMismatch
	}
	if err := validateVectorFinite(query); err != nil {
		return nil, err
	}

	scratch := idx.scratchPool.Get().(*searchScratch)
	defer idx.scratchPool.Put(scratch)
	return idx.searchValidated(ctx, query, k, ef, maxDist, maxDistSet, filter, dst, scratch)
}

func (idx *Index[K]) searchValidated(ctx context.Context, query []float32, k int, ef int, maxDist float32, maxDistSet bool, filter func(id K) bool, dst []SearchResult[K], scratch *searchScratch) ([]SearchResult[K], error) {
	stripe := scratch.stripe
	idx.readStripes[stripe].mu.RLock()
	defer idx.readStripes[stripe].mu.RUnlock()

	es := idx.entryState.Load()
	if es == emptyEntry || idx.liveCount.Load() == 0 {
		if dst != nil {
			return dst[:0], nil
		}
		return []SearchResult[K]{}, nil
	}

	entryID := uint32(es)
	entryLevel := int(es >> 32)

	qInvNorm := float32(0)
	if idx.metric == MetricCosine {
		qInvNorm = computeInvNorm(query)
	}

	currID := entryID
	currDist := idx.distQuery(currID, query, qInvNorm)
	for lvl := entryLevel; lvl >= 1; lvl-- {
		currID, currDist = idx.greedyStep(currID, currDist, query, qInvNorm, lvl, scratch, true)
	}

	if ef < k {
		ef = k
	}

	numNodes := idx.nextID.Load()
	scratch.begin(numNodes)
	scratch.visit(currID)
	scratch.candidates.push(distNode{id: currID, dist: currDist})
	epChunk, epSlot := idx.getChunk(currID)
	if !isDeletedAt(epChunk, epSlot) && (filter == nil || filter(epChunk.intToExt[epSlot])) {
		scratch.results.push(distNode{id: currID, dist: currDist})
	}

	isL2 := idx.metric == MetricEuclidean || idx.metric == MetricSquaredEuclidean
	isF16 := idx.quant == QuantFloat16
	iter := 0
	for len(scratch.candidates) > 0 {
		if iter&63 == 0 && ctx != nil {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		iter++

		curr := scratch.candidates.pop()
		if len(scratch.results) >= ef && curr.dist > scratch.results[0].dist {
			break
		}

		nbrs := idx.copyNeighbors(curr.id, 0, scratch.neighbors, true)
		scratch.neighbors = nbrs
		for _, nbrID := range nbrs {
			if !scratch.visit(nbrID) {
				continue
			}
			chunk, slot := idx.getChunk(nbrID)
			var d float32
			if isL2 {
				if isF16 {
					d = simd.L2SquaredF16F32(idx.vectorF16At(chunk, slot), query)
				} else {
					d = simd.L2Squared(idx.vectorAt(chunk, slot), query)
				}
			} else {
				d = idx.distQueryAt(chunk, slot, query, qInvNorm)
			}
			if len(scratch.results) < ef || d < scratch.results[0].dist {
				scratch.candidates.push(distNode{id: nbrID, dist: d})
				if !isDeletedAt(chunk, slot) {
					if filter == nil || filter(chunk.intToExt[slot]) {
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

	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}

	if len(scratch.results) == 0 {
		if dst != nil {
			return dst[:0], nil
		}
		return []SearchResult[K]{}, nil
	}

	for len(scratch.results) > k {
		if idx.metric == MetricEuclidean &&
			(scratch.results[0].dist < 1e-36 || math.IsInf(float64(scratch.results[0].dist), 0)) {
			return idx.rescoreEuclideanResults(scratch, query, k, maxDist, maxDistSet, dst), nil
		}
		scratch.results.pop()
	}
	if idx.metric == MetricEuclidean &&
		(scratch.results[0].dist < 1e-36 || math.IsInf(float64(scratch.results[0].dist), 0)) {
		return idx.rescoreEuclideanResults(scratch, query, k, maxDist, maxDistSet, dst), nil
	}
	count := len(scratch.results)
	var out []SearchResult[K]
	if cap(dst) >= count {
		out = dst[:count]
	} else {
		out = make([]SearchResult[K], count)
	}
	needsSort := false
	for i := count - 1; i >= 0; i-- {
		item := scratch.results.pop()
		d := item.dist
		chunk, slot := idx.getChunk(item.id)
		if idx.metric == MetricEuclidean {
			if d < 1e-36 {
				if isF16 {
					d = euclideanFallback64F16F32(idx.vectorF16At(chunk, slot), query)
				} else {
					d = euclideanFallback64(idx.vectorAt(chunk, slot), query)
				}
				if i+1 < count && d > out[i+1].Distance {
					needsSort = true
				}
			} else {
				d = float32(math.Sqrt(float64(d)))
			}
		}
		out[i] = SearchResult[K]{
			ID:       chunk.intToExt[slot],
			Distance: d,
		}
	}
	if needsSort {
		for i := 1; i < len(out); i++ {
			cur := out[i]
			j := i - 1
			for j >= 0 && out[j].Distance > cur.Distance {
				out[j+1] = out[j]
				j--
			}
			out[j+1] = cur
		}
	}
	if maxDistSet {
		keep := 0
		for keep < len(out) && out[keep].Distance <= maxDist {
			keep++
		}
		out = out[:keep]
	}

	return out, nil
}

func (idx *Index[K]) rescoreEuclideanResults(scratch *searchScratch, query []float32, k int, maxDist float32, maxDistSet bool, dst []SearchResult[K]) []SearchResult[K] {
	items := scratch.results
	isF16 := idx.quant == QuantFloat16
	for i := range items {
		chunk, slot := idx.getChunk(items[i].id)
		if isF16 {
			items[i].dist = euclideanFastF16F32(idx.vectorF16At(chunk, slot), query)
		} else {
			items[i].dist = euclideanFast(idx.vectorAt(chunk, slot), query)
		}
	}
	for i := 1; i < len(items); i++ {
		cur := items[i]
		j := i - 1
		for j >= 0 && items[j].dist > cur.dist {
			items[j+1] = items[j]
			j--
		}
		items[j+1] = cur
	}
	count := len(items)
	if count > k {
		count = k
	}
	if maxDistSet {
		keep := 0
		for keep < count && items[keep].dist <= maxDist {
			keep++
		}
		count = keep
	}
	var out []SearchResult[K]
	if cap(dst) >= count {
		out = dst[:count]
	} else {
		out = make([]SearchResult[K], count)
	}
	for i := 0; i < count; i++ {
		out[i] = SearchResult[K]{
			ID:       idx.extKeyAt(items[i].id),
			Distance: items[i].dist,
		}
	}
	return out
}
