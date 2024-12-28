# DiskANN Implementation in Rust

[![Rust](https://github.com/lukaesch/diskann-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/lukaesch/diskann-rs/actions/workflows/rust.yml)

A Rust implementation of DiskANN (Disk-based Approximate Nearest Neighbor search) featuring a 3-layer index architecture and parallel query processing. This project provides an efficient and scalable solution for large-scale vector similarity search with single-file storage.

## Overview

This implementation provides a memory-efficient approach to similarity search by:
- Using a 3-layer hierarchical index structure for faster search
- Storing all data in a single file using memory mapping
- Supporting both Euclidean distance and Cosine similarity
- Managing adjacency lists for graph-based search
- Implementing parallel query processing
- Supporting large-scale datasets that don't fit in RAM

## Features

- Three-layer hierarchical index structure:
  - Top layer (L0): Smallest, most selective layer
  - Middle layer (L1): Intermediate connectivity
  - Base layer (L2): Complete dataset
- Single-file storage format for simplified deployment
- Choice of distance metrics:
  - Euclidean distance
  - Cosine similarity
- Cluster-based graph construction for meaningful adjacency
- Parallel query processing using rayon
- Memory-mapped file access for handling large datasets
- Comprehensive error handling with custom error types

## Usage

### Building a New Index

```rust
use diskannrs::{SingleFileDiskANN, DistanceMetric};

let index = SingleFileDiskANN::build_index_singlefile(
    1_000_000,        // number of vectors
    128,              // dimension
    32,               // max neighbors per node
    0.01,             // fraction of vectors in top layer
    0.1,              // fraction of vectors in middle layer
    DistanceMetric::Euclidean,  // or DistanceMetric::Cosine
    "index.db"        // single file to store everything
)?;
```

### Opening an Existing Index

```rust
let index = SingleFileDiskANN::open_index_singlefile("index.db")?;
```

### Searching the Index

```rust
// Prepare your query vector
let query = vec![0.1, 0.2, ...; 128];  // must match index dimension

// Search for nearest neighbors
let k = 10;  // number of neighbors to return
let beam_width = 64;  // search beam width
let neighbors = index.search(&query, k, beam_width);
```

### Parallel Search

```rust
use rayon::prelude::*;
use std::sync::Arc;

// Create shared index reference
let index = Arc::new(index);

// Perform parallel queries
let results: Vec<Vec<u32>> = query_batch
    .par_iter()
    .map(|query| index.search(query, k, beam_width))
    .collect();
```

## Performance Characteristics

- Memory Usage: O(1) for vector storage due to memory mapping
- Disk Space: Single file containing:
  - Vectors: num_vectors * dimension * 4 bytes
  - Adjacency Lists: Varies by layer size and max_degree
  - Metadata: Small overhead
- Search Time: Logarithmic due to hierarchical structure
- Parallel Processing: Scales with available CPU cores

## Building and Testing

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run with example
cargo run --release --example simple_search
```

## Current Status

This implementation features:
- [x] Single-file storage format
- [x] 3-layer hierarchical index structure
- [x] Multiple distance metrics support
- [x] Cluster-based graph construction
- [x] Parallel query processing
- [x] Memory-mapped I/O
- [x] Comprehensive test suite

## Future Improvements

1. Add more distance metrics
2. Implement dynamic index updates
3. Add parameter auto-tuning
4. Expand benchmarking suite
5. Add more examples
6. Improve documentation

## Contributing

Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests
- Submit PRs for improvements
- Share ideas for optimization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Original DiskANN paper: [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/)
