package hnsw

type distNode struct {
	id   uint32
	dist float32
}

type minHeap []distNode

func (h *minHeap) reset() {
	*h = (*h)[:0]
}

func (h *minHeap) push(x distNode) {
	*h = append(*h, x)
	h.siftUp(len(*h) - 1)
}

func (h *minHeap) pop() distNode {
	s := *h
	top := s[0]
	n := len(s) - 1
	s[0] = s[n]
	*h = s[:n]
	if n > 0 {
		h.siftDown()
	}
	return top
}

func (h minHeap) siftUp(i int) {
	for i > 0 {
		p := (i - 1) >> 1
		if h[p].dist <= h[i].dist {
			break
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
}

func (h minHeap) siftDown() {
	n := len(h)
	i := 0
	for {
		left := (i << 1) + 1
		if left >= n {
			break
		}
		smallest := left
		if right := left + 1; right < n && h[right].dist < h[left].dist {
			smallest = right
		}
		if h[i].dist <= h[smallest].dist {
			break
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
}

type maxHeap []distNode

func (h *maxHeap) reset() {
	*h = (*h)[:0]
}

func (h *maxHeap) push(x distNode) {
	*h = append(*h, x)
	h.siftUp(len(*h) - 1)
}

func (h *maxHeap) pop() distNode {
	s := *h
	top := s[0]
	n := len(s) - 1
	s[0] = s[n]
	*h = s[:n]
	if n > 0 {
		h.siftDown()
	}
	return top
}

func (h maxHeap) replaceMax(x distNode) {
	h[0] = x
	h.siftDown()
}

func (h maxHeap) siftUp(i int) {
	for i > 0 {
		p := (i - 1) >> 1
		if h[p].dist >= h[i].dist {
			break
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
}

func (h maxHeap) siftDown() {
	n := len(h)
	i := 0
	for {
		left := (i << 1) + 1
		if left >= n {
			break
		}
		largest := left
		if right := left + 1; right < n && h[right].dist > h[left].dist {
			largest = right
		}
		if h[i].dist >= h[largest].dist {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}
