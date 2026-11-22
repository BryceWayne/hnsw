package hnsw

type searchScratch struct {
	visited          []uint32
	epoch            uint32
	candidates       minHeap
	results          maxHeap
	stripe           uint32
	neighbors        []uint32
	candIDs          []uint32
	chosenBuf        []uint32
	heuristicWork    []distNode
	selected         []distNode
	selectedVecs     [][]float32
	selectedVecsF16  [][]uint16
	selectedInvNorms []float32
	discarded        []distNode
	chosenByLevel    [maxNodeLevel + 1][]uint32
	tmpVecA          []float32
	tmpVecB          []float32
}

func newSearchScratch(stripe uint32, mmax int) *searchScratch {
	s := &searchScratch{
		visited:          make([]uint32, 64),
		candidates:       make(minHeap, 0, 64),
		results:          make(maxHeap, 0, 64),
		stripe:           stripe,
		neighbors:        make([]uint32, 0, mmax),
		candIDs:          make([]uint32, 0, mmax+1),
		chosenBuf:        make([]uint32, 0, mmax),
		heuristicWork:    make([]distNode, 0, mmax+1),
		selected:         make([]distNode, 0, mmax),
		selectedVecs:     make([][]float32, 0, mmax),
		selectedVecsF16:  make([][]uint16, 0, mmax),
		selectedInvNorms: make([]float32, 0, mmax),
		discarded:        make([]distNode, 0, mmax),
	}
	s.chosenByLevel[0] = make([]uint32, 0, mmax)
	return s
}

func (s *searchScratch) begin(numNodes uint32) {
	if int(numNodes) > len(s.visited) {
		newCap := len(s.visited) * 2
		if newCap < int(numNodes) {
			newCap = int(numNodes)
		}
		if newCap < 64 {
			newCap = 64
		}
		s.visited = make([]uint32, newCap)
	}
	s.epoch++
	if s.epoch == 0 {
		clear(s.visited)
		s.epoch = 1
	}
	s.candidates.reset()
	s.results.reset()
}

func (s *searchScratch) growVisited(id uint32) {
	newCap := len(s.visited) * 2
	if newCap <= int(id) {
		newCap = int(id) + 1
	}
	grown := make([]uint32, newCap)
	copy(grown, s.visited)
	s.visited = grown
}

func (s *searchScratch) visit(id uint32) bool {
	if int(id) >= len(s.visited) {
		s.growVisited(id)
	}
	if s.visited[id] == s.epoch {
		return false
	}
	s.visited[id] = s.epoch
	return true
}
