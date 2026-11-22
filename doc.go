// Package hnsw implements a concurrency-safe, embeddable Hierarchical Navigable
// Small World (HNSW) vector index for approximate nearest neighbor (ANN) search.
//
// # Architecture
//
// An [Index] maps comparable external keys K to fixed-dimension float32 vectors
// stored in a flat, chunked memory arena with dense uint32 internal node IDs.
// Graph construction uses Algorithm 4 diversity-aware neighbor selection
// (configurable via [WithHeuristic]) and level multiplier mL = 1/ln(M).
//
// # Distance Metrics and SIMD Acceleration
//
// Built-in metrics ([MetricEuclidean], [MetricSquaredEuclidean],
// [MetricCosine], and [MetricDotProduct]) are selected with [WithMetric], or
// callers may supply a custom [DistanceFunc] via [WithCustomDistance]. Note
// that [MetricDotProduct] ranks candidates by -dot(a, b) so higher inner
// products appear first, while the standalone [DotProduct] helper returns the
// raw inner product. [MetricCosine] stores inverse norms separately so [Index.Get]
// returns exact unnormalized vectors.
//
// On amd64 CPUs with AVX2 and FMA (plus F16C for [QuantFloat16]), distance
// kernels dispatch automatically at runtime via Plan 9 assembly. On arm64 CPUs
// (Apple Silicon, AWS Graviton), distance kernels use 4-accumulator NEON Plan 9
// assembly. Other architectures and builds with the "purego" build tag use
// portable pure-Go fallbacks.
//
// Optional IEEE-754 binary16 vector quantization ([WithQuantization] with
// [QuantFloat16]) halves vector memory and disk footprint while accelerating
// high-dimensional search via hardware F16C conversion on amd64.
//
// # Concurrency and Zero-Allocation Search
//
// All [Index] operations are safe for concurrent use. Concurrent reads
// ([Index.Search], [Index.SearchInto], [Index.SearchFilter], [Index.SearchBatch],
// [Index.Get], [Index.Contains], [Index.Keys], [Index.All], [Index.Len],
// [Index.Stats]) and new-key [Index.Insert] calls coordinate through striped
// read/write locks, per-node locks, and atomic entry-point and efSearch state.
// Pooled search scratch buffers (epoch-stamped visited slices, hole-sifting
// value-typed heaps, and reusable Algorithm 4 pruning buffers) eliminate
// per-query heap allocations in steady state—[Index.SearchInto] achieves 0
// allocations/op when reusing a caller buffer—and batch workers reuse one
// scratch buffer per worker across the entire batch.
//
// # Updates, Deletion, and Compaction
//
// Calling [Index.Insert] with an existing key atomically replaces its vector
// without leaving stale graph links. [Index.Delete] and [Index.DeleteBatch]
// tombstone nodes, repair 2-hop neighborhoods, and re-elect the entry point if
// necessary. [Index.Compact] reclaims tombstoned arena slots and densely remaps
// live nodes and edges.
//
// # Persistence
//
// [Index.WriteTo] and [ReadFrom] (along with atomic file helpers
// [Index.SaveFile] and [LoadFile]) stream a versioned, CRC32-checksummed,
// tombstone-free binary representation using chunked bulk little-endian I/O
// buffers for high throughput and OOM-safe incremental deserialization. Indexes
// configured with [WithCustomDistance] require passing [WithCustomDistance]
// again on load.
package hnsw
