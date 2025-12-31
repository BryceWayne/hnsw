module github.com/BryceWayne/hnsw/examples/gcp

go 1.23.3

require (
	cloud.google.com/go/storage v1.38.0
	github.com/BryceWayne/hnsw v0.0.0
)

replace github.com/BryceWayne/hnsw => ../..
