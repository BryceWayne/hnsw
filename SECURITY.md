# Security Policy

## Supported Versions

Security updates are applied to the latest commit on the `root` branch and the most recent tagged release.

## Reporting a Vulnerability

If you discover a security issue (such as a deserialization crash, OOM/DoS vector in `ReadFrom`, memory-safety issue in SIMD assembly, or race condition), please report it privately rather than opening a public issue:

- Use GitHub's **Private Vulnerability Reporting** ("Report a vulnerability" under the repository's **Security** tab), or contact the repository maintainer (`@BryceWayne`) directly.

Please include a minimal reproduction test case or input payload when possible.

## Security Design Notes

- **Bounded Deserialization**: `ReadFrom` / `LoadFile` validate magic bytes, version, dimension bounds, key-type metadata, string/gob payload caps, float32 finiteness (`NaN`/`±Inf` rejection), adjacency bounds, and trailing IEEE CRC32 checksums incrementally to prevent OOM allocation bombs on untrusted streams.
- **Input Validation**: All public vector ingestion and distance helpers reject empty/mismatched dimensions and non-finite `float32` values (`ErrInvalidDimension`, `ErrDimensionMismatch`, `ErrNaNOrInf`).
- **Automated Checks**: CI runs `govulncheck ./...`, `go test -race ./...`, `go test -tags purego -race ./...`, and `golangci-lint` on every pull request.
