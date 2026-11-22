# Changelog

All notable changes to `github.com/BryceWayne/hnsw` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [v1.0.0] - 2026-11-22

First stable release of `github.com/BryceWayne/hnsw`, committing to Semantic Versioning API stability for the public `Index[K comparable]` API, functional options, sentinel errors, and versioned CRC32 binary persistence format (`v1`/`v2`).

### Changed
- **Minimum Go Version Bumped to Go 1.27 (`go 1.27.0`)**: Unlocks Go 1.24–1.27 runtime and compiler improvements (Swiss Tables maps, Green Tea GC, size-specialized `<80B` allocator, LICM/known-bits/lookup-table compiler passes, faster `encoding/json` v2 backend), updates `golang.org/x/sys` to `v0.50.0`, and updates CI to Go `1.27` + `golangci-lint` `v2.15`.
- **16-Way Sharded External Key Map (`hash/maphash.Comparable`)**: Replaced single-lock `extMu` contention with 16 cache-line-padded shards using Go's zero-alloc `hash/maphash.Comparable[K]`, speeding up parallel `Contains` by **`3.6×`** (`59.3 ns → 16.4 ns`, `0 allocs/op`) and parallel `Get` by **`-13.8%`** (`91.4 ns → 78.8 ns`).
- **Inlined `internal/simd/*Fast` Dispatchers**: Eliminated double-call wrapper overhead on `amd64` AVX2+FMA+F16C and `arm64` NEON hot paths, delivering **`-4.2%` to `-13.7%`** faster search (`10K×128` 12T: **`391,696 q/s`**; `1K×32` 12T: **`1,071,352 q/s`**), **`-18.9%` to `-28.5%`** faster `WriteTo` (`1,887 MB/s`), and **`-6.7%` to `-14.5%`** faster FP16 `ReadFrom` vs Go 1.23.0.

## [v0.3.0] - 2026-11-22

### Added
- **ARM64 NEON FP16 Assembly Kernels (`linux/arm64`, `darwin/arm64`)**: Hardware `FCVTL`/`FCVTL2` + 4-accumulator `VFMLA` Plan 9 assembly kernels (`l2SquaredF16F32NEON`, `dotProductF16F32NEON`, `l2SquaredF16F16NEON`, `dotProductF16F16NEON`) delivering **`2.95x–3.40x`** speedup over pure Go on ARM64 hardware (Apple Silicon / AWS Graviton), verified on native `ubuntu-24.04-arm` CI.
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
