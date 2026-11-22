# Go HNSW Head-to-Head Benchmarks

This isolated Go sub-module (`benchmarks/go.mod`) benchmarks [`github.com/BryceWayne/hnsw`](..) against popular pure-Go HNSW libraries (`github.com/coder/hnsw`, `github.com/fogfish/hnsw`) and an exact brute-force baseline **without adding any dependencies to the root module**.

## Methodology

- **Dataset**: Deterministic clustered Gaussian embeddings (`50` clusters, L2-normalized, seed `42`) across realistic embedding dimensions: `10K × dim=128`, `10K × dim=768` (MiniLM / BGE / BERT), and `5K × dim=1536` (OpenAI `text-embedding-3-small`).
- **Parameters**: `MetricCosine`, `M = 16`, `Mmax = 32`, `EfConstruction = 200`, `EfSearch = 64`, `k = 10`.
- **Ground Truth**: Exact brute-force Cosine top-10 computed for every query to measure real `Recall@10` alongside latency (`ns/op`, `p50_us`, `p99_us`), `B/op`, and `allocs/op`.
- **Mixed Production Workload (`BenchmarkCompareMixedReadWrite`)**: 80% concurrent reads / 20% concurrent inserts across all CPUs. `BryceWayne/hnsw` uses its built-in striped concurrency control; libraries without built-in concurrent write safety are guarded with `sync.RWMutex`.

## Reproduce

```bash
cd benchmarks

# 1. Verify recall sanity across all libraries
go test -v -run=TestComparisonSanity .

# 2. Measure build throughput & live heap footprint (5K x dim=768)
go test -v -run=TestBuildAndMemoryReport .

# 3. Run single-threaded & parallel search benchmarks (Recall@10, p50/p99, allocs/op)
go test -run=^$ -bench=BenchmarkCompareSearch -benchmem -benchtime=100ms .

# 4. Run concurrent 80% read / 20% write production contention benchmark
go test -run=^$ -bench=BenchmarkCompareMixedReadWrite -benchmem -benchtime=100ms .
```
