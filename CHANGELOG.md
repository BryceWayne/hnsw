# Changelog

All notable changes to `github.com/BryceWayne/hnsw` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [v0.3.0] - 2026-11-22

### Added
- **ARM64 NEON FP16 Assembly Kernels (`linux/arm64`, `darwin/arm64`)**: Hardware `FCVTL`/`FCVTL2` + 4-accumulator `VFMLA` Plan 9 assembly kernels (`l2SquaredF16F32NEON`, `dotProductF16F32NEON`, `l2SquaredF16F16NEON`, `dotProductF16F16NEON`) bringing full FP16 SIMD acceleration to Apple Silicon and AWS Graviton, verified on native `ubuntu-24.04-arm` CI.
- **`WithInitialCapacity(n int)`**: Pre-allocates the key map and contiguous arena chunks ahead of bulk startup ingestion.
- **`Index.SearchFilterInto`**: Zero-allocation (`0 B/op, 0 allocs/op`) predicate-filtered search reusing caller-supplied `[]SearchResult[K]` buffers.
- **`IndexStats.TombstoneRatio()`**: Helper returning `Deleted / (Len + Deleted)` to drive periodic auto-compaction policies.
- **Isolated Head-to-Head Benchmark Suite (`benchmarks/`)**: Reproducible Go sub-module comparing `BryceWayne/hnsw` (F32 & FP16) against `coder/hnsw`, `fogfish/hnsw`, and exact brute-force across Recall@10, QPS, p50/p99 latency, allocs/op, mixed 80/20 concurrent reads/writes, build speed, and total heap usage.
- **Production Blueprints**: Zero-dependency Go 1.23 RAG HTTP microservice (`examples/rag-server`) and LLM semantic cache (`examples/semantic-cache`).

### Changed
- Relaxed `go.mod` minimum directive from `go 1.23.3` to `go 1.23.0` for frictionless adoption across all Go 1.23 patch releases.
- Expanded `README.md` with a Go library comparison matrix, verified head-to-head benchmark table, and "When to Use `hnsw`" architectural guide.

## [v0.2.1] - 2026-11-22

### Changed
- Moved architecture-specific SIMD and FP16 distance kernels into `internal/simd` for a cleaner root module layout (100% backwards-compatible) (#21).
- Organized test files by topic (`quantization_test.go`, `edge_cases_test.go`) (#21).
- Overhauled all six `README.md` `Algorithm Overview` SVG diagrams (`hnsw.svg`, `search-process.svg`, `insertion-process.svg`, `pruning-process.svg`, `deletion-process.svg`, `ef-impact.svg`) with unified `hnsw` Go branding and WCAG AA light/dark theme contrast (#22).

### Added
- Added `CHANGELOG.md` and GitHub issue/PR templates (#21).

## [v0.2.0] - 2026-11-21

### Added
- **ARM64 NEON SIMD Assembly** (`linux/arm64`, `darwin/arm64`): 4-accumulator NEON kernels (`l2SquaredNEON`, `dotProductNEON`) achieving `3.44x–4.58x` speedup over pure Go on ARM64 hardware, verified by a native `ubuntu-24.04-arm` CI job.
- **FP16 Scalar Quantization (`WithQuantization(QuantFloat16)`)**: IEEE-754 binary16 vector storage with runtime `CPUID` `F16C` + AVX2 + FMA assembly kernels on `amd64` (`1.40x–1.75x` faster high-dim search, `~48%` lower RAM and file size, `~1.95x` faster `ReadFrom`) and backward-compatible `serVersionV2 = 2` persistence.
- **Zero-Allocation `Index.SearchInto`**: Reuses caller-provided `[]SearchResult[K]` buffers for `0 B/op, 0 allocs/op` steady-state search.
- **`WithMaxDistance`**: Distance threshold filtering across `Search`, `SearchInto`, `SearchFilter`, and `SearchBatch`.
- **CI Fuzz Gate**: 5-second `FuzzReadFrom` smoke run on every pull request.

## [v0.1.0] - 2026-11-21

### Added
- Initial tagged release of `github.com/BryceWayne/hnsw`: concurrency-safe generic `Index[K comparable]`, chunked flat arena, AVX2+FMA assembly kernels, Algorithm 4 neighbor pruning, `SearchFilter`, bounded `InsertBatch`/`SearchBatch`/`DeleteBatch`, `Compact`, `Keys()`, Go 1.23 `All()`, `SetEfSearch()`, chunked CRC32 binary persistence (`WriteTo`/`ReadFrom`, `SaveFile`/`LoadFile`), `FuzzReadFrom`, `CONTRIBUTING.md`, `SECURITY.md`, and `CODEOWNERS`.
- Hot-path optimizations: generic `slices.SortFunc` pruning (-99% insert allocs), hole-sifting heaps, batched 2-hop `Delete` repair (`11.2x` faster, `0 allocs/op`), pool-free `Get`/`Contains`, and bulk binary I/O (`1.43–1.62 GB/s`).
