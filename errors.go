package hnsw

import "errors"

var (
	// ErrInvalidDimension indicates a non-positive dimension was supplied to [New]
	// or an empty vector was passed to a standalone distance helper.
	ErrInvalidDimension = errors.New("hnsw: invalid dimension")

	// ErrDimensionMismatch indicates a vector or query length does not match the
	// index dimension or the paired vector length.
	ErrDimensionMismatch = errors.New("hnsw: vector dimension mismatch")

	// ErrInvalidConfig indicates an invalid [Option] or [SearchOption] value,
	// or a conflicting metric option passed to [ReadFrom] or [LoadFile].
	ErrInvalidConfig = errors.New("hnsw: invalid configuration")

	// ErrInvalidK indicates a non-positive result count k was requested in
	// [Index.Search], [Index.SearchFilter], or [Index.SearchBatch].
	ErrInvalidK = errors.New("hnsw: k must be positive")

	// ErrNaNOrInf indicates a vector contains NaN or ±Inf float32 values.
	ErrNaNOrInf = errors.New("hnsw: vector contains NaN or Inf")

	// ErrNotFound indicates a requested key was not present in the index.
	ErrNotFound = errors.New("hnsw: id not found")

	// ErrCorruptedIndex indicates serialized index data passed to [ReadFrom] or
	// [LoadFile] failed header, key-type, structural, or CRC32 validation.
	ErrCorruptedIndex = errors.New("hnsw: corrupted or incompatible index data")
)
