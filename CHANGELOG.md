# Changelog

All notable changes to `github.com/BryceWayne/hnsw` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
