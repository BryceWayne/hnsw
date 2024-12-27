// batch.go
package hnsw

import (
	"runtime"
	"sync"
	"sync/atomic"
)

type BatchInserter struct {
	hnsw        *HNSW
	batchSize   int
	workerCount int
	queue       chan batchTask
	wg          sync.WaitGroup
	inserted    atomic.Int64
}

type batchTask struct {
	id  int
	vec Vector
}

func NewBatchInserter(hnsw *HNSW, batchSize, workerCount int) *BatchInserter {
	if workerCount <= 0 {
		workerCount = runtime.NumCPU()
	}
	if batchSize <= 0 {
		batchSize = 100
	}
	return &BatchInserter{
		hnsw:        hnsw,
		batchSize:   batchSize,
		workerCount: workerCount,
		queue:       make(chan batchTask, batchSize*2),
	}
}

func (bi *BatchInserter) worker() {
	for task := range bi.queue {
		bi.hnsw.Insert(task.id, task.vec)
		bi.inserted.Add(1)
		bi.wg.Done()
	}
}

func (bi *BatchInserter) Start() {
	for i := 0; i < bi.workerCount; i++ {
		go bi.worker()
	}
}

func (bi *BatchInserter) Add(id int, vec Vector) {
	bi.wg.Add(1)
	bi.queue <- batchTask{id, vec}
}

func (bi *BatchInserter) Stop() {
	close(bi.queue)
	bi.wg.Wait()
}

func (bi *BatchInserter) Inserted() int64 {
	return bi.inserted.Load()
}

// BatchInsert adds multiple vectors efficiently
func (h *HNSW) BatchInsert(vectors map[int]Vector) {
	batchSize := 100
	if len(vectors) < batchSize {
		batchSize = len(vectors)
	}

	inserter := NewBatchInserter(h, batchSize, runtime.NumCPU())
	inserter.Start()

	for id, vec := range vectors {
		inserter.Add(id, vec)
	}

	inserter.Stop()
}

// BatchSearch performs parallel searches for multiple queries
func (h *HNSW) BatchSearch(queries []Vector, k int, config SearchConfig) [][]int {
	results := make([][]int, len(queries))
	var wg sync.WaitGroup

	workerCount := config.WorkerCount
	if workerCount <= 0 {
		workerCount = runtime.NumCPU()
	}

	taskChan := make(chan int, len(queries))

	for w := 0; w < workerCount; w++ {
		go func() {
			for i := range taskChan {
				results[i] = h.SearchWithConfig(queries[i], k, config)
				wg.Done()
			}
		}()
	}

	wg.Add(len(queries))
	for i := range queries {
		taskChan <- i
	}
	close(taskChan)

	wg.Wait()
	return results
}

// BatchDelete removes multiple vectors efficiently
func (h *HNSW) BatchDelete(ids []int) {
	if len(ids) == 0 {
		return
	}

	var wg sync.WaitGroup
	workerCount := runtime.NumCPU()

	// Calculate items per worker
	itemsPerWorker := (len(ids) + workerCount - 1) / workerCount

	for w := 0; w < workerCount; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * itemsPerWorker
			if start >= len(ids) {
				return
			}
			end := start + itemsPerWorker
			if end > len(ids) {
				end = len(ids)
			}
			for _, id := range ids[start:end] {
				h.Delete(id)
			}
		}(w)
	}

	wg.Wait()
}
