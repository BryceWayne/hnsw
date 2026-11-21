package hnsw

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
)

const (
	serVersion         uint32 = 1
	maxSerDim          uint32 = 1 << 20
	maxSerKeyTypeLen   uint16 = 4096
	maxSerStringKeyLen uint32 = 16 << 20
	maxSerGobLen       uint64 = 256 << 20
	serBufSize                = 64 * 1024
)

var serMagic = [4]byte{'H', 'N', 'S', 'W'}

const (
	keyKindInt     uint8 = 1
	keyKindInt64   uint8 = 2
	keyKindInt32   uint8 = 3
	keyKindUint64  uint8 = 4
	keyKindUint32  uint8 = 5
	keyKindString  uint8 = 6
	keyKindUint    uint8 = 7
	keyKindInt16   uint8 = 8
	keyKindUint16  uint8 = 9
	keyKindInt8    uint8 = 10
	keyKindUint8   uint8 = 11
	keyKindBytes16 uint8 = 12
	keyKindGob     uint8 = 255
)

type countWriter struct {
	w io.Writer
	n int64
}

func (cw *countWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

func keyCodecFor[K comparable]() (uint8, string) {
	t := reflect.TypeFor[K]()
	typeName := t.String()
	var zero K
	switch any(zero).(type) {
	case int:
		return keyKindInt, typeName
	case int64:
		return keyKindInt64, typeName
	case int32:
		return keyKindInt32, typeName
	case uint64:
		return keyKindUint64, typeName
	case uint32:
		return keyKindUint32, typeName
	case string:
		return keyKindString, typeName
	case uint:
		return keyKindUint, typeName
	case int16:
		return keyKindInt16, typeName
	case uint16:
		return keyKindUint16, typeName
	case int8:
		return keyKindInt8, typeName
	case uint8:
		return keyKindUint8, typeName
	case [16]byte:
		return keyKindBytes16, typeName
	default:
		return keyKindGob, typeName
	}
}

func wrapCorrupt(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrCorruptedIndex) {
		return err
	}
	return fmt.Errorf("%w: %v", ErrCorruptedIndex, err)
}

func readBoundedBytes(r io.Reader, total int) ([]byte, error) {
	if total == 0 {
		return nil, nil
	}
	capHint := total
	if capHint > serBufSize {
		capHint = serBufSize
	}
	out := make([]byte, 0, capHint)
	var chunk [4096]byte
	rem := total
	for rem > 0 {
		n := rem
		if n > len(chunk) {
			n = len(chunk)
		}
		if _, err := io.ReadFull(r, chunk[:n]); err != nil {
			return nil, wrapCorrupt(err)
		}
		out = append(out, chunk[:n]...)
		rem -= n
	}
	return out, nil
}

// WriteTo serializes a tombstone-free snapshot of the index in a versioned, CRC32-checksummed
// binary format to w, implementing [io.WriterTo]. Primitive integer types, string, and [16]byte
// keys use compact binary codecs; other comparable key types use encoding/gob.
func (idx *Index[K]) WriteTo(w io.Writer) (int64, error) {
	idx.writeGate.Lock()
	defer idx.writeGate.Unlock()

	totalSlots := idx.nextID.Load()
	oldToNew := make([]uint32, totalSlots)
	for i := range oldToNew {
		oldToNew[i] = invalidID
	}
	liveIDs := make([]uint32, 0, idx.liveCount.Load())
	for id := uint32(0); id < totalSlots; id++ {
		if !idx.isDeleted(id) {
			oldToNew[id] = uint32(len(liveIDs))
			liveIDs = append(liveIDs, id)
		}
	}

	entryPt := invalidID
	var maxLvl uint32
	if len(liveIDs) > 0 {
		es := idx.entryState.Load()
		oldEP := uint32(es)
		if es != emptyEntry && oldEP < totalSlots && oldToNew[oldEP] != invalidID {
			entryPt = oldToNew[oldEP]
			maxLvl = uint32(es >> 32)
		} else {
			bestLvl := -1
			for newIdx, oldID := range liveIDs {
				chunk, slot := idx.getChunk(oldID)
				lvl := int(chunk.maxLevels[slot])
				if lvl > bestLvl {
					bestLvl = lvl
					entryPt = uint32(newIdx)
				}
			}
			maxLvl = uint32(bestLvl)
		}
	}

	cw := &countWriter{w: w}
	bw := bufio.NewWriterSize(cw, serBufSize)
	crc := crc32.NewIEEE()
	mw := io.MultiWriter(bw, crc)

	keyKind, keyTypeName := keyCodecFor[K]()
	if len(keyTypeName) > int(maxSerKeyTypeLen) {
		return cw.n, fmt.Errorf("%w: key type name too long", ErrCorruptedIndex)
	}

	// Header
	var hdr [48]byte
	copy(hdr[0:4], serMagic[:])
	binary.LittleEndian.PutUint32(hdr[4:8], serVersion)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(idx.dim))
	hdr[12] = uint8(idx.metric)
	if idx.heuristic {
		hdr[13] = 1
	}
	binary.LittleEndian.PutUint16(hdr[14:16], 0)
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(idx.m))
	binary.LittleEndian.PutUint32(hdr[20:24], uint32(idx.mmax))
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(idx.efConstruction))
	binary.LittleEndian.PutUint32(hdr[28:32], uint32(idx.efSearch))
	binary.LittleEndian.PutUint64(hdr[32:40], uint64(idx.seed))
	binary.LittleEndian.PutUint32(hdr[40:44], uint32(len(liveIDs)))
	binary.LittleEndian.PutUint32(hdr[44:48], maxLvl)
	if _, err := mw.Write(hdr[:]); err != nil {
		return cw.n, err
	}

	var hdrTail [7]byte
	binary.LittleEndian.PutUint32(hdrTail[0:4], entryPt)
	hdrTail[4] = keyKind
	binary.LittleEndian.PutUint16(hdrTail[5:7], uint16(len(keyTypeName)))
	if _, err := mw.Write(hdrTail[:]); err != nil {
		return cw.n, err
	}
	if _, err := io.WriteString(mw, keyTypeName); err != nil {
		return cw.n, err
	}

	// Keys
	if err := idx.writeKeys(mw, liveIDs, keyKind); err != nil {
		return cw.n, err
	}

	// Vectors
	if len(liveIDs) > 0 {
		chunkFloats := idx.dim
		if chunkFloats > 4096 {
			chunkFloats = 4096
		}
		vecBuf := make([]byte, chunkFloats*4)
		for _, oldID := range liveIDs {
			chunk, slot := idx.getChunk(oldID)
			vec := idx.vectorAt(chunk, slot)
			for off := 0; off < len(vec); off += chunkFloats {
				end := off + chunkFloats
				if end > len(vec) {
					end = len(vec)
				}
				sub := vec[off:end]
				buf := vecBuf[:len(sub)*4]
				for j, v := range sub {
					binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(v))
				}
				if _, err := mw.Write(buf); err != nil {
					return cw.n, err
				}
			}
		}
	}

	// Graph adjacency
	var lbuf [1]byte
	var u32buf [4]byte
	nbrScratch := make([]uint32, 0, idx.mmax)
	var adjBuf []byte
	if len(liveIDs) > 0 {
		adjBuf = make([]byte, (1+idx.mmax)*4)
	}
	for newIdx, oldID := range liveIDs {
		selfNew := uint32(newIdx)
		chunk, slot := idx.getChunk(oldID)
		numLevels := chunk.maxLevels[slot] + 1
		lbuf[0] = numLevels
		if _, err := mw.Write(lbuf[:]); err != nil {
			return cw.n, err
		}
		for lvl := 0; lvl < int(numLevels); lvl++ {
			nbrScratch = nbrScratch[:0]
			var raw []uint32
			if lvl == 0 {
				cnt := int(chunk.l0Count[slot])
				base := slot * idx.mmax
				raw = chunk.l0Edges[base : base+cnt]
			} else {
				raw = chunk.upperLevels[slot][lvl-1]
			}
			for _, oldNb := range raw {
				if oldNb >= totalSlots {
					continue
				}
				mapped := oldToNew[oldNb]
				if mapped == invalidID || mapped == selfNew {
					continue
				}
				dup := false
				for _, prev := range nbrScratch {
					if prev == mapped {
						dup = true
						break
					}
				}
				if !dup {
					nbrScratch = append(nbrScratch, mapped)
				}
			}
			need := (1 + len(nbrScratch)) * 4
			buf := adjBuf[:need]
			binary.LittleEndian.PutUint32(buf[0:4], uint32(len(nbrScratch)))
			for j, nb := range nbrScratch {
				binary.LittleEndian.PutUint32(buf[(j+1)*4:], nb)
			}
			if _, err := mw.Write(buf); err != nil {
				return cw.n, err
			}
		}
	}

	// Trailer CRC32
	binary.LittleEndian.PutUint32(u32buf[:], crc.Sum32())
	if _, err := bw.Write(u32buf[:]); err != nil {
		return cw.n, err
	}
	if err := bw.Flush(); err != nil {
		return cw.n, err
	}
	return cw.n, nil
}

func (idx *Index[K]) writeKeys(w io.Writer, liveIDs []uint32, keyKind uint8) error {
	if len(liveIDs) == 0 {
		return nil
	}
	if keyKind == keyKindGob {
		keys := make([]K, len(liveIDs))
		for i, oldID := range liveIDs {
			keys[i] = idx.extKeyAt(oldID)
		}
		var buf bytes.Buffer
		if err := gob.NewEncoder(&buf).Encode(keys); err != nil {
			return err
		}
		var lenBuf [8]byte
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(buf.Len()))
		if _, err := w.Write(lenBuf[:]); err != nil {
			return err
		}
		_, err := w.Write(buf.Bytes())
		return err
	}

	var buf [8]byte
	for _, oldID := range liveIDs {
		k := idx.extKeyAt(oldID)
		switch v := any(k).(type) {
		case int:
			binary.LittleEndian.PutUint64(buf[:], uint64(int64(v)))
			if _, err := w.Write(buf[:8]); err != nil {
				return err
			}
		case int64:
			binary.LittleEndian.PutUint64(buf[:], uint64(v))
			if _, err := w.Write(buf[:8]); err != nil {
				return err
			}
		case int32:
			binary.LittleEndian.PutUint32(buf[:4], uint32(v))
			if _, err := w.Write(buf[:4]); err != nil {
				return err
			}
		case uint64:
			binary.LittleEndian.PutUint64(buf[:], v)
			if _, err := w.Write(buf[:8]); err != nil {
				return err
			}
		case uint32:
			binary.LittleEndian.PutUint32(buf[:4], v)
			if _, err := w.Write(buf[:4]); err != nil {
				return err
			}
		case string:
			binary.LittleEndian.PutUint32(buf[:4], uint32(len(v)))
			if _, err := w.Write(buf[:4]); err != nil {
				return err
			}
			if _, err := io.WriteString(w, v); err != nil {
				return err
			}
		case uint:
			binary.LittleEndian.PutUint64(buf[:], uint64(v))
			if _, err := w.Write(buf[:8]); err != nil {
				return err
			}
		case int16:
			binary.LittleEndian.PutUint16(buf[:2], uint16(v))
			if _, err := w.Write(buf[:2]); err != nil {
				return err
			}
		case uint16:
			binary.LittleEndian.PutUint16(buf[:2], v)
			if _, err := w.Write(buf[:2]); err != nil {
				return err
			}
		case int8:
			buf[0] = byte(v)
			if _, err := w.Write(buf[:1]); err != nil {
				return err
			}
		case uint8:
			buf[0] = v
			if _, err := w.Write(buf[:1]); err != nil {
				return err
			}
		case [16]byte:
			if _, err := w.Write(v[:]); err != nil {
				return err
			}
		}
	}
	return nil
}

// SaveFile atomically persists a tombstone-free snapshot of the index to path by writing
// to a temporary file in the target directory and renaming it into place. On error, any
// temporary file is removed before returning.
func (idx *Index[K]) SaveFile(path string) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".hnsw-tmp-*")
	if err != nil {
		return err
	}
	tmpPath := f.Name()
	defer func() {
		if tmpPath != "" {
			_ = os.Remove(tmpPath)
		}
	}()

	_, writeErr := idx.WriteTo(f)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	tmpPath = ""
	return nil
}

// ReadFrom deserializes a versioned, CRC32-checksummed [Index] from r using OOM-safe
// incremental reads, validating header bounds, key type, vector finiteness, graph structure,
// and trailing checksum.
//
// Indexes serialized with a custom distance function require passing [WithCustomDistance] in opts.
// Callers may also pass [WithEfSearch] or [WithSeed] to override those runtime parameters on load.
// Format, key-type, structural, or checksum failures return an error wrapping [ErrCorruptedIndex];
// conflicting metric options return an error wrapping [ErrInvalidConfig].
func ReadFrom[K comparable](r io.Reader, opts ...Option) (*Index[K], error) {
	optCfg := newDefaultConfig()
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(&optCfg); err != nil {
			return nil, err
		}
	}

	br := bufio.NewReaderSize(r, serBufSize)
	crc := crc32.NewIEEE()
	tr := io.TeeReader(br, crc)

	var hdr [48]byte
	if _, err := io.ReadFull(tr, hdr[:]); err != nil {
		return nil, wrapCorrupt(err)
	}
	if !bytes.Equal(hdr[0:4], serMagic[:]) {
		return nil, fmt.Errorf("%w: invalid magic header", ErrCorruptedIndex)
	}
	version := binary.LittleEndian.Uint32(hdr[4:8])
	if version != serVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCorruptedIndex, version)
	}
	dim := binary.LittleEndian.Uint32(hdr[8:12])
	if dim == 0 || dim > maxSerDim {
		return nil, fmt.Errorf("%w: invalid dimension %d", ErrCorruptedIndex, dim)
	}
	metric := Metric(hdr[12])
	if metric > MetricDotProduct && metric != metricCustom {
		return nil, fmt.Errorf("%w: invalid metric %d", ErrCorruptedIndex, metric)
	}
	heurByte := hdr[13]
	if heurByte > 1 {
		return nil, fmt.Errorf("%w: invalid heuristic flag %d", ErrCorruptedIndex, heurByte)
	}
	reserved := binary.LittleEndian.Uint16(hdr[14:16])
	if reserved != 0 {
		return nil, fmt.Errorf("%w: non-zero reserved field", ErrCorruptedIndex)
	}
	m := binary.LittleEndian.Uint32(hdr[16:20])
	mmax := binary.LittleEndian.Uint32(hdr[20:24])
	efConst := binary.LittleEndian.Uint32(hdr[24:28])
	efSearch := binary.LittleEndian.Uint32(hdr[28:32])
	if m < 2 || m > math.MaxUint16 || mmax < m || mmax > math.MaxUint16 || efConst == 0 || efSearch == 0 {
		return nil, fmt.Errorf("%w: invalid graph parameters", ErrCorruptedIndex)
	}
	seed := int64(binary.LittleEndian.Uint64(hdr[32:40]))
	liveCount := binary.LittleEndian.Uint32(hdr[40:44])
	maxLvl := binary.LittleEndian.Uint32(hdr[44:48])

	if uint64(liveCount) > uint64(dirL1Size)*uint64(dirL2Size)*uint64(chunkSize) {
		return nil, wrapCorrupt(fmt.Errorf("live count %d exceeds arena limit", liveCount))
	}
	if uint64(liveCount)*uint64(dim) > math.MaxInt32 {
		return nil, fmt.Errorf("%w: payload size exceeds limits", ErrCorruptedIndex)
	}

	var hdrTail [7]byte
	if _, err := io.ReadFull(tr, hdrTail[:]); err != nil {
		return nil, wrapCorrupt(err)
	}
	entryPt := binary.LittleEndian.Uint32(hdrTail[0:4])
	keyKind := hdrTail[4]
	keyTypeLen := binary.LittleEndian.Uint16(hdrTail[5:7])

	if liveCount == 0 {
		if maxLvl != 0 || entryPt != invalidID {
			return nil, fmt.Errorf("%w: invalid empty index header", ErrCorruptedIndex)
		}
	} else {
		if maxLvl > maxNodeLevel || entryPt >= liveCount {
			return nil, fmt.Errorf("%w: invalid entry point or max level", ErrCorruptedIndex)
		}
	}

	expKind, expTypeName := keyCodecFor[K]()
	if keyKind != expKind || keyTypeLen > maxSerKeyTypeLen {
		return nil, fmt.Errorf("%w: key type mismatch", ErrCorruptedIndex)
	}
	typeBytes, err := readBoundedBytes(tr, int(keyTypeLen))
	if err != nil {
		return nil, err
	}
	if string(typeBytes) != expTypeName {
		return nil, fmt.Errorf("%w: key type %q does not match %q", ErrCorruptedIndex, string(typeBytes), expTypeName)
	}

	// Validate metric compatibility with options.
	var customDist DistanceFunc
	if metric == metricCustom {
		if optCfg.customDist == nil {
			return nil, fmt.Errorf("%w: custom distance index requires WithCustomDistance", ErrInvalidConfig)
		}
		customDist = optCfg.customDist
	} else {
		if optCfg.customDistSet {
			return nil, fmt.Errorf("%w: WithCustomDistance conflicts with serialized metric %s", ErrInvalidConfig, metric)
		}
		if optCfg.metricSet && optCfg.metric != metric {
			return nil, fmt.Errorf("%w: WithMetric(%s) conflicts with serialized metric %s", ErrInvalidConfig, optCfg.metric, metric)
		}
	}

	effEfSearch := int(efSearch)
	if optCfg.efSearchSet {
		effEfSearch = optCfg.efSearch
	}
	effSeed := seed
	if optCfg.hasSeed {
		effSeed = optCfg.seed
	}

	idx := &Index[K]{
		dim:            int(dim),
		m:              int(m),
		mmax:           int(mmax),
		efConstruction: int(efConst),
		efSearch:       effEfSearch,
		mL:             1.0 / math.Log(float64(m)),
		metric:         metric,
		customDist:     customDist,
		heuristic:      heurByte == 1,
		seed:           effSeed,
		extToInt:       make(map[K]uint32),
		rng:            rand.New(rand.NewSource(effSeed)),
	}
	idx.initScratchPool()

	// Read keys incrementally (OOM-safe for corrupt LiveCount).
	keys, err := readKeys[K](tr, int(liveCount), keyKind)
	if err != nil {
		return nil, err
	}
	if len(keys) > 0 {
		idx.extToInt = make(map[K]uint32, len(keys))
	}

	// Read vectors chunk by chunk directly into arenaChunks (OOM-safe).
	dimInt := int(dim)
	mmaxInt := int(mmax)
	mInt := int(m)
	chunkFloats := dimInt
	if chunkFloats > 4096 {
		chunkFloats = 4096
	}
	var vecBuf []byte
	if len(keys) > 0 {
		vecBuf = make([]byte, chunkFloats*4)
	}

	for id := 0; id < len(keys); id++ {
		uid := uint32(id)
		l1 := uid >> (chunkBits + dirL2Bits)
		l2 := (uid >> chunkBits) & dirL2Mask
		slot := int(uid & chunkMask)

		page := idx.dir[l1].Load()
		if page == nil {
			page = &dirPage[K]{}
			idx.dir[l1].Store(page)
		}
		chunk := page.chunks[l2].Load()
		if chunk == nil {
			chunk = &arenaChunk[K]{
				vecs:     make([]float32, chunkSize*dimInt),
				intToExt: make([]K, chunkSize),
				l0Edges:  make([]uint32, chunkSize*mmaxInt),
			}
			if metric == MetricCosine {
				chunk.invNorms = make([]float32, chunkSize)
			}
			page.chunks[l2].Store(chunk)
		}

		extKey := keys[id]
		if _, dup := idx.extToInt[extKey]; dup {
			return nil, fmt.Errorf("%w: duplicate key in serialized stream", ErrCorruptedIndex)
		}
		idx.extToInt[extKey] = uid
		chunk.intToExt[slot] = extKey

		off := slot * dimInt
		vecSlice := chunk.vecs[off : off+dimInt]
		for vOff := 0; vOff < dimInt; vOff += chunkFloats {
			end := vOff + chunkFloats
			if end > dimInt {
				end = dimInt
			}
			sub := vecSlice[vOff:end]
			buf := vecBuf[:len(sub)*4]
			if _, err := io.ReadFull(tr, buf); err != nil {
				return nil, wrapCorrupt(err)
			}
			for j := range sub {
				v := math.Float32frombits(binary.LittleEndian.Uint32(buf[j*4:]))
				if v != v || math.IsInf(float64(v), 0) {
					return nil, fmt.Errorf("%w: non-finite float in serialized vector", ErrCorruptedIndex)
				}
				sub[j] = v
			}
		}
		if metric == MetricCosine {
			chunk.invNorms[slot] = computeInvNorm(vecSlice)
		}
	}

	// Read graph adjacency.
	var lbuf [1]byte
	var u32buf [4]byte
	var adjBuf []byte
	for id := 0; id < len(keys); id++ {
		uid := uint32(id)
		chunk, slot := idx.getChunk(uid)

		if _, err := io.ReadFull(tr, lbuf[:]); err != nil {
			return nil, wrapCorrupt(err)
		}
		numLevels := lbuf[0]
		if numLevels == 0 || uint32(numLevels-1) > maxLvl {
			return nil, fmt.Errorf("%w: invalid node level count %d", ErrCorruptedIndex, numLevels)
		}
		chunk.maxLevels[slot] = numLevels - 1
		if numLevels > 1 {
			chunk.upperLevels[slot] = make([][]uint32, numLevels-1)
		}

		for lvl := 0; lvl < int(numLevels); lvl++ {
			if _, err := io.ReadFull(tr, u32buf[:]); err != nil {
				return nil, wrapCorrupt(err)
			}
			deg := binary.LittleEndian.Uint32(u32buf[:])
			if deg >= liveCount {
				return nil, fmt.Errorf("%w: neighbor degree %d >= live count %d", ErrCorruptedIndex, deg, liveCount)
			}
			if (lvl == 0 && int(deg) > mmaxInt) || (lvl > 0 && int(deg) > mInt) {
				return nil, fmt.Errorf("%w: neighbor degree %d exceeds level bound", ErrCorruptedIndex, deg)
			}

			need := int(deg) * 4
			if len(adjBuf) < need {
				adjBuf = make([]byte, need)
			}
			buf := adjBuf[:need]
			if need > 0 {
				if _, err := io.ReadFull(tr, buf); err != nil {
					return nil, wrapCorrupt(err)
				}
			}

			if lvl == 0 {
				base := slot * mmaxInt
				for j := 0; j < int(deg); j++ {
					nb := binary.LittleEndian.Uint32(buf[j*4:])
					if nb >= liveCount || nb == uid {
						return nil, fmt.Errorf("%w: invalid neighbor id %d", ErrCorruptedIndex, nb)
					}
					chunk.l0Edges[base+j] = nb
				}
				chunk.l0Count[slot] = uint16(deg)
			} else {
				edges := make([]uint32, deg)
				for j := 0; j < int(deg); j++ {
					nb := binary.LittleEndian.Uint32(buf[j*4:])
					if nb >= liveCount || nb == uid {
						return nil, fmt.Errorf("%w: invalid neighbor id %d", ErrCorruptedIndex, nb)
					}
					edges[j] = nb
				}
				chunk.upperLevels[slot][lvl-1] = edges
			}
		}
	}

	// Verify CRC32 trailer from underlying br (not tr).
	if _, err := io.ReadFull(br, u32buf[:]); err != nil {
		return nil, wrapCorrupt(err)
	}
	gotCRC := binary.LittleEndian.Uint32(u32buf[:])
	if gotCRC != crc.Sum32() {
		return nil, fmt.Errorf("%w: crc32 mismatch", ErrCorruptedIndex)
	}

	idx.nextID.Store(liveCount)
	idx.liveCount.Store(int64(liveCount))
	if liveCount == 0 {
		idx.entryState.Store(emptyEntry)
	} else {
		epChunk, epSlot := idx.getChunk(entryPt)
		if uint32(epChunk.maxLevels[epSlot]) != maxLvl {
			return nil, fmt.Errorf("%w: entry point level mismatch", ErrCorruptedIndex)
		}
		idx.entryState.Store((uint64(maxLvl) << 32) | uint64(entryPt))
	}

	return idx, nil
}

func readKeys[K comparable](r io.Reader, count int, keyKind uint8) ([]K, error) {
	if count == 0 {
		return nil, nil
	}
	if keyKind == keyKindGob {
		var lenBuf [8]byte
		if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
			return nil, wrapCorrupt(err)
		}
		payloadLen := binary.LittleEndian.Uint64(lenBuf[:])
		if payloadLen > maxSerGobLen {
			return nil, fmt.Errorf("%w: gob key payload exceeds max size", ErrCorruptedIndex)
		}
		payload, err := readBoundedBytes(r, int(payloadLen))
		if err != nil {
			return nil, err
		}
		var keys []K
		if err := gob.NewDecoder(bytes.NewReader(payload)).Decode(&keys); err != nil {
			return nil, wrapCorrupt(err)
		}
		if len(keys) != count {
			return nil, fmt.Errorf("%w: gob key count %d != live count %d", ErrCorruptedIndex, len(keys), count)
		}
		return keys, nil
	}

	capHint := count
	if capHint > 4096 {
		capHint = 4096
	}
	keys := make([]K, 0, capHint)
	var buf [16]byte
	var zero K
	for i := 0; i < count; i++ {
		var k K
		switch any(zero).(type) {
		case int:
			if _, err := io.ReadFull(r, buf[:8]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := int(int64(binary.LittleEndian.Uint64(buf[:8])))
			k = any(v).(K)
		case int64:
			if _, err := io.ReadFull(r, buf[:8]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := int64(binary.LittleEndian.Uint64(buf[:8]))
			k = any(v).(K)
		case int32:
			if _, err := io.ReadFull(r, buf[:4]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := int32(binary.LittleEndian.Uint32(buf[:4]))
			k = any(v).(K)
		case uint64:
			if _, err := io.ReadFull(r, buf[:8]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := binary.LittleEndian.Uint64(buf[:8])
			k = any(v).(K)
		case uint32:
			if _, err := io.ReadFull(r, buf[:4]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := binary.LittleEndian.Uint32(buf[:4])
			k = any(v).(K)
		case string:
			if _, err := io.ReadFull(r, buf[:4]); err != nil {
				return nil, wrapCorrupt(err)
			}
			slen := binary.LittleEndian.Uint32(buf[:4])
			if slen > maxSerStringKeyLen {
				return nil, fmt.Errorf("%w: string key length %d exceeds limit", ErrCorruptedIndex, slen)
			}
			sb, err := readBoundedBytes(r, int(slen))
			if err != nil {
				return nil, err
			}
			k = any(string(sb)).(K)
		case uint:
			if _, err := io.ReadFull(r, buf[:8]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := uint(binary.LittleEndian.Uint64(buf[:8]))
			k = any(v).(K)
		case int16:
			if _, err := io.ReadFull(r, buf[:2]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := int16(binary.LittleEndian.Uint16(buf[:2]))
			k = any(v).(K)
		case uint16:
			if _, err := io.ReadFull(r, buf[:2]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := binary.LittleEndian.Uint16(buf[:2])
			k = any(v).(K)
		case int8:
			if _, err := io.ReadFull(r, buf[:1]); err != nil {
				return nil, wrapCorrupt(err)
			}
			v := int8(buf[0])
			k = any(v).(K)
		case uint8:
			if _, err := io.ReadFull(r, buf[:1]); err != nil {
				return nil, wrapCorrupt(err)
			}
			k = any(buf[0]).(K)
		case [16]byte:
			if _, err := io.ReadFull(r, buf[:16]); err != nil {
				return nil, wrapCorrupt(err)
			}
			var arr [16]byte
			copy(arr[:], buf[:16])
			k = any(arr).(K)
		}
		keys = append(keys, k)
	}
	return keys, nil
}

// LoadFile opens path and deserializes an [Index] using [ReadFrom].
// File open errors are returned directly; stream format or checksum errors wrap [ErrCorruptedIndex].
func LoadFile[K comparable](path string, opts ...Option) (*Index[K], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return ReadFrom[K](f, opts...)
}
