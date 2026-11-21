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
	val := h[i]
	for i > 0 {
		p := (i - 1) >> 1
		if h[p].dist <= val.dist {
			break
		}
		h[i] = h[p]
		i = p
	}
	h[i] = val
}

func (h minHeap) siftDown() {
	n := len(h)
	i := 0
	val := h[0]
	for {
		left := (i << 1) + 1
		if left >= n {
			break
		}
		smallest := left
		if right := left + 1; right < n && h[right].dist < h[left].dist {
			smallest = right
		}
		if val.dist <= h[smallest].dist {
			break
		}
		h[i] = h[smallest]
		i = smallest
	}
	h[i] = val
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
	val := h[i]
	for i > 0 {
		p := (i - 1) >> 1
		if h[p].dist >= val.dist {
			break
		}
		h[i] = h[p]
		i = p
	}
	h[i] = val
}

func (h maxHeap) siftDown() {
	n := len(h)
	i := 0
	val := h[0]
	for {
		left := (i << 1) + 1
		if left >= n {
			break
		}
		largest := left
		if right := left + 1; right < n && h[right].dist > h[left].dist {
			largest = right
		}
		if val.dist >= h[largest].dist {
			break
		}
		h[i] = h[largest]
		i = largest
	}
	h[i] = val
}
