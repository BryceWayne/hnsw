# Production RAG HTTP Microservice Example

A zero-dependency Go 1.23 `net/http` vector search microservice built directly on `github.com/BryceWayne/hnsw`.

## What It Demonstrates

- **Zero-allocation filtered search**: `SearchFilterInto` with a `sync.Pool` of result slices for multi-tenant isolation (`tenant` filter).
- **Per-request deadlines & relevance cutoffs**: `WithContext(r.Context())` and `WithMaxDistance(threshold)`.
- **Memory optimization**: `WithQuantization(hnsw.QuantFloat16)` + `WithInitialCapacity`.
- **Operational hygiene**: Background compaction when `Stats().TombstoneRatio() >= 0.25`, `/stats` JSON telemetry, and atomic `SaveFile` / `LoadFile` on `SIGINT`/`SIGTERM`.

## Run It

```bash
go run ./examples/rag-server -addr :8080 -dim 4 -data /tmp/rag.hnsw -fp16=true
```

### Upsert Documents

```bash
curl -s -X POST http://localhost:8080/upsert \
  -H 'Content-Type: application/json' \
  -d '{"id":"doc-1","tenant":"acme","text":"Go HNSW vector index","vector":[1,0.1,0,0]}'

curl -s -X POST http://localhost:8080/upsert \
  -H 'Content-Type: application/json' \
  -d '{"id":"doc-2","tenant":"beta","text":"Other tenant doc","vector":[1,0.05,0,0]}'
```

### Multi-Tenant Filtered Search with Relevance Cutoff

```bash
curl -s -X POST http://localhost:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"tenant":"acme","vector":[1,0.1,0,0],"k":5,"max_distance":0.2}'
```

### Inspect Telemetry & Delete

```bash
curl -s http://localhost:8080/stats
curl -s -X DELETE http://localhost:8080/vectors/doc-1
```
