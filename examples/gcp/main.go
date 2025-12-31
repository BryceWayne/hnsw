package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"cloud.google.com/go/storage"
	"github.com/BryceWayne/hnsw"
)

func main() {
	// Set up Google Cloud Storage client
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// Just demonstrate usage if no credentials
		if os.Getenv("GOOGLE_APPLICATION_CREDENTIALS") == "" {
			fmt.Println("No GCP credentials found, skipping actual cloud operations.")
			fmt.Println("This example shows how to use HNSW with GCS:")
			fmt.Println()
			fmt.Println("  // Write to GCS")
			fmt.Println("  wc := bucket.Object(\"index.gob\").NewWriter(ctx)")
			fmt.Println("  if err := index.SaveToWriter(wc); err != nil { ... }")
			fmt.Println("  wc.Close()")
			fmt.Println()
			fmt.Println("  // Read from GCS")
			fmt.Println("  rc, err := bucket.Object(\"index.gob\").NewReader(ctx)")
			fmt.Println("  index, err := hnsw.LoadFromReader(rc, hnsw.Euclidean)")
			return
		}
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	bucketName := os.Getenv("BUCKET_NAME")
	if bucketName == "" {
		log.Fatal("BUCKET_NAME environment variable must be set")
	}

	// Create a sample index
	fmt.Println("Creating index...")
	index := hnsw.New(128, 16, 32, 100, hnsw.Euclidean)

	// Add some dummy data
	vec := make(hnsw.Vector, 128)
	for i := range vec {
		vec[i] = float64(i)
	}
	index.Insert(1, vec)

	// Save to GCS
	fmt.Printf("Saving index to gs://%s/index.gob...\n", bucketName)
	bucket := client.Bucket(bucketName)
	wc := bucket.Object("index.gob").NewWriter(ctx)

	if err := index.SaveToWriter(wc); err != nil {
		log.Fatalf("Failed to save to GCS: %v", err)
	}
	if err := wc.Close(); err != nil {
		log.Fatalf("Failed to close GCS writer: %v", err)
	}
	fmt.Println("Save complete.")

	// Load from GCS
	fmt.Println("Loading index back from GCS...")
	rc, err := bucket.Object("index.gob").NewReader(ctx)
	if err != nil {
		log.Fatalf("Failed to open GCS reader: %v", err)
	}
	defer rc.Close()

	loadedIndex, err := hnsw.LoadFromReader(rc, hnsw.Euclidean)
	if err != nil {
		log.Fatalf("Failed to load from GCS: %v", err)
	}

	// Verify
	results := loadedIndex.Search(vec, 1)
	if len(results) > 0 && results[0] == 1 {
		fmt.Println("Success! Loaded index and found inserted vector.")
	} else {
		fmt.Printf("Search failed. Results: %v\n", results)
	}
}
