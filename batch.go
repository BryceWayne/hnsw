package hnsw

import (
	"sync"
	"sync/atomic"
)

// InsertBatch validates and inserts all items into the index using a bounded worker pool.
// If any vector fails dimension or NaN/Inf validation, InsertBatch returns an error without mutating the index.
// When items contains duplicate IDs, last-write-wins semantics are applied deterministically.
func (idx *Index[K]) InsertBatch(items []Item[K], opts ...SearchOption) error {
	sc := defaultSearchConfig()
	if err := sc.apply(opts); err != nil {
		return err
	}
	for i := range items {
		if len(items[i].Vector) != idx.dim {
			return ErrDimensionMismatch
		}
		if err := validateVectorFinite(items[i].Vector); err != nil {
			return err
		}
	}
	if len(items) == 0 {
		return nil
	}

	// Deduplicate keeping only the last occurrence of each ID.
	lastPos := make(map[K]int, len(items))
	for i := range items {
		lastPos[items[i].ID] = i
	}
	var order []int
	if len(lastPos) < len(items) {
		order = make([]int, 0, len(lastPos))
		for i := range items {
			if lastPos[items[i].ID] == i {
				order = append(order, i)
			}
		}
	}

	total := len(lastPos)
	workers := sc.effectiveWorkers()
	if workers > total {
		workers = total
	}

	getItem := func(pos int) Item[K] {
		if order != nil {
			return items[order[pos]]
		}
		return items[pos]
	}

	if workers == 1 || total == 1 {
		for i := 0; i < total; i++ {
			if sc.ctx != nil {
				if err := sc.ctx.Err(); err != nil {
					return err
				}
			}
			it := getItem(i)
			idx.insertValidated(it.ID, it.Vector)
		}
		return nil
	}

	var next atomic.Int64
	var errOnce sync.Once
	var firstErr error
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				if sc.ctx != nil {
					if err := sc.ctx.Err(); err != nil {
						errOnce.Do(func() { firstErr = err })
						return
					}
				}
				i := int(next.Add(1) - 1)
				if i >= total {
					return
				}
				it := getItem(i)
				idx.insertValidated(it.ID, it.Vector)
			}
		}()
	}
	wg.Wait()
	if sc.ctx != nil {
		if err := sc.ctx.Err(); err != nil {
			return err
		}
	}
	return firstErr
}

// SearchBatch executes Search for each query in queries using a bounded worker pool and returns results in query order.
func (idx *Index[K]) SearchBatch(queries [][]float32, k int, opts ...SearchOption) ([][]SearchResult[K], error) {
	sc := defaultSearchConfig()
	if err := sc.apply(opts); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	for i := range queries {
		if len(queries[i]) != idx.dim {
			return nil, ErrDimensionMismatch
		}
		if err := validateVectorFinite(queries[i]); err != nil {
			return nil, err
		}
	}
	if len(queries) == 0 {
		return [][]SearchResult[K]{}, nil
	}

	results := make([][]SearchResult[K], len(queries))
	workers := sc.effectiveWorkers()
	if workers > len(queries) {
		workers = len(queries)
	}

	if workers == 1 || len(queries) == 1 {
		for i, q := range queries {
			if sc.ctx != nil {
				if err := sc.ctx.Err(); err != nil {
					return nil, err
				}
			}
			res, err := idx.searchCore(q, k, nil, opts...)
			if err != nil {
				return nil, err
			}
			results[i] = res
		}
		return results, nil
	}

	var next atomic.Int64
	var errOnce sync.Once
	var firstErr error
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				if sc.ctx != nil {
					if err := sc.ctx.Err(); err != nil {
						errOnce.Do(func() { firstErr = err })
						return
					}
				}
				i := int(next.Add(1) - 1)
				if i >= len(queries) {
					return
				}
				res, err := idx.searchCore(queries[i], k, nil, opts...)
				if err != nil {
					errOnce.Do(func() { firstErr = err })
					return
				}
				results[i] = res
			}
		}()
	}
	wg.Wait()
	if sc.ctx != nil {
		if err := sc.ctx.Err(); err != nil {
			return nil, err
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

// DeleteBatch removes all matching keys from ids in a single atomic pass and returns the number of removed entries.
func (idx *Index[K]) DeleteBatch(ids []K) int {
	if len(ids) == 0 {
		return 0
	}

	scratch := idx.scratchPool.Get().(*searchScratch)
	defer idx.scratchPool.Put(scratch)

	idx.acquireExclusive()
	defer idx.releaseExclusive()

	deadIDs := make([]uint32, 0, len(ids))
	for _, extID := range ids {
		oldID, ok := idx.extToInt[extID]
		if !ok {
			continue
		}
		delete(idx.extToInt, extID)
		if idx.markDeleted(oldID) {
			deadIDs = append(deadIDs, oldID)
		}
	}

	removed := len(deadIDs)
	if removed == 0 {
		return 0
	}
	idx.liveCount.Add(-int64(removed))
	idx.tombstoneCount.Add(int64(removed))

	if idx.liveCount.Load() == 0 {
		idx.entryState.Store(emptyEntry)
		return removed
	}

	// Unlink deleted IDs from surviving nodes.
	if removed <= 8 {
		nbrBuf := make([]uint32, 0, idx.mmax)
		for _, oldID := range deadIDs {
			chunk, slot := idx.getChunk(oldID)
			maxLvl := int(chunk.maxLevels[slot])
			for lvl := 0; lvl <= maxLvl; lvl++ {
				nbrBuf = idx.copyNeighbors(oldID, lvl, nbrBuf, false)
				for _, u := range nbrBuf {
					if !idx.isDeleted(u) {
						idx.removeEdge(u, oldID, lvl)
					}
				}
			}
		}
	} else {
		total := idx.nextID.Load()
		for id := uint32(0); id < total; id++ {
			if idx.isDeleted(id) {
				continue
			}
			chunk, slot := idx.getChunk(id)
			cnt := int(chunk.l0Count[slot])
			base := slot * idx.mmax
			edges := chunk.l0Edges[base : base+cnt]
			w := 0
			for _, nb := range edges {
				if !idx.isDeleted(nb) {
					edges[w] = nb
					w++
				}
			}
			chunk.l0Count[slot] = uint16(w)

			for l := 0; l < int(chunk.maxLevels[slot]); l++ {
				ue := chunk.upperLevels[slot][l]
				uw := 0
				for _, nb := range ue {
					if !idx.isDeleted(nb) {
						ue[uw] = nb
						uw++
					}
				}
				chunk.upperLevels[slot][l] = ue[:uw]
			}
		}
	}

	// 2-hop repair among live neighbors of deleted nodes.
	nbrBuf := make([]uint32, 0, idx.mmax)
	for _, oldID := range deadIDs {
		chunk, slot := idx.getChunk(oldID)
		maxLvl := int(chunk.maxLevels[slot])
		for lvl := 0; lvl <= maxLvl; lvl++ {
			nbrBuf = idx.copyNeighbors(oldID, lvl, nbrBuf, false)
			w := 0
			for _, nb := range nbrBuf {
				if !idx.isDeleted(nb) {
					nbrBuf[w] = nb
					w++
				}
			}
			liveNbrs := nbrBuf[:w]
			for i, u := range liveNbrs {
				for j, v := range liveNbrs {
					if i != j {
						idx.addConnection(u, v, lvl, scratch, false)
					}
				}
			}
		}
	}

	ep := uint32(idx.entryState.Load())
	if idx.isDeleted(ep) {
		idx.electEntryPoint()
	}
	return removed
}
