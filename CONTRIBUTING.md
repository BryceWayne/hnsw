# Contributing to `hnsw`

Thank you for your interest in contributing to `hnsw`!

## Pull Request Workflow

The `root` branch is protected and requires passing CI checks plus maintainer (`@BryceWayne`) approval before merging.

1. Fork the repository and create a topic branch from `root`.
2. Keep changes focused, well-tested, and formatted with `gofmt -s`.
3. Update or add unit tests, GoDoc comments, and runnable `Example*` tests for any new or modified behavior.
4. Open a pull request against `root` describing the change and (for performance changes) before/after benchmark numbers.

## Local Verification

Run the same checks enforced by GitHub Actions CI before opening a PR:

```bash
gofmt -s -l .
go vet ./...
go doc -all .
go test -race ./...
go test -tags purego -race ./...
go build -o /dev/null ./examples
```

### Optional Additional Checks

- **Linter**: `golangci-lint run --timeout=5m` (configured in [`.golangci.yml`](./.golangci.yml))
- **Fuzzing**: `go test -fuzz=FuzzReadFrom -fuzztime=15s .`
- **Cross-platform builds**:
  ```bash
  GOOS=linux GOARCH=arm64 go build ./...
  GOOS=darwin GOARCH=arm64 go build ./...
  GOOS=windows GOARCH=amd64 go build ./...
  ```
- **Benchmarks**:
  ```bash
  go test -run=^$ -bench=. -benchmem -benchtime=100ms -count=5 .
  ```

## Performance & Safety Guidelines

- Maintain zero extra allocations on the steady-state `Search` hot path (`TestSearchZeroAllocs` must continue reporting `1 allocs/op`).
- Keep all `Index` operations concurrency-safe under `go test -race ./...`.
- Preserve wire compatibility and OOM-safe bounded reads in `serialize.go`.
- Maintain parity between AVX2+FMA assembly (`distance_amd64.s`) and portable pure-Go fallbacks (`distance.go`, `-tags purego`).
