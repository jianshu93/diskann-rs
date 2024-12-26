# DiskANN Implementation in Rust

[![Rust](https://github.com/lukaesch/diskann-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/lukaesch/diskann-rs/actions/workflows/rust.yml)

A Rust implementation of DiskANN (Disk-based Approximate Nearest Neighbor search) featuring a 3-layer index architecture and parallel query processing. This project provides an efficient and scalable solution for large-scale vector similarity search.

## Overview

This implementation provides a memory-efficient approach to similarity search by:
- Using a 3-layer hierarchical index structure for faster search
- Storing vectors on disk using memory mapping
- Managing adjacency lists for graph-based search
- Supporting parallel query processing
- Implementing cluster-based graph construction
- Supporting large-scale datasets that don't fit in RAM

## Features

- Three-layer hierarchical index structure:
  - Top layer (L0): Smallest, most selective layer
  - Middle layer (L1): Intermediate connectivity
  - Base layer (L2): Complete dataset
- Cluster-based graph construction for meaningful adjacency
- Parallel query processing using rayon
- Memory-mapped vector storage for handling large datasets
- Serialized adjacency lists for graph structure
- Support for euclidean distance metric
- Comprehensive error handling with custom error types

## Usage

### Building a 3-Layer Index

```rust
let index = MultiLayerDiskANN::build_index_on_disk(
    1_000_000,        // number of vectors
    128,              // dimension
    32,               // max neighbors per node
    0.01,             // fraction of vectors in top layer
    0.1,              // fraction of vectors in middle layer
    "vectors.bin",    // path to store vectors
    "adjacency.bin",  // path to store adjacency lists
    "metadata.bin"    // path to store metadata
)?;
```

### Opening an Existing Index

```rust
let index = MultiLayerDiskANN::open_index(
    "vectors.bin",
    "adjacency.bin",
    "metadata.bin"
)?;
```

### Parallel Search

```rust
use rayon::prelude::*;

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
- Disk Space: 
  - Vectors: num_vectors * dimension * 4 bytes
  - Adjacency Lists: Varies by layer size and max_degree
- Search Time: Logarithmic due to hierarchical structure
- Parallel Processing: Scales with available CPU cores

## Current Status

This implementation features:
- [x] 3-layer hierarchical index structure
- [x] Cluster-based graph construction
- [x] Parallel query processing
- [x] Memory-mapped I/O
- [x] Efficient disk-based storage
- [x] Beam search implementation

## Building and Testing

```bash
# Build in release mode
cargo build --release

# Run the demo with parallel queries
cargo run --release
```

## Future Improvements

1. Implement more sophisticated graph construction algorithms
2. Add support for multiple distance metrics
3. Further optimize search algorithm
4. Add dynamic index updates
5. Implement advanced parameter tuning
6. Expand benchmarking suite

## Contributing

This is a learning project, and contributions or suggestions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit PRs for improvements
- Share ideas for optimization

## References

- Original DiskANN paper: [[link](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/)]
