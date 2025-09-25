# DiskANN Implementation in Rust

[![Rust](https://github.com/lukaesch/diskann-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/lukaesch/diskann-rs/actions/workflows/rust.yml)

A Rust implementation of DiskANN (Disk-based Approximate Nearest Neighbor search) using the Vamana graph algorithm. This project provides an efficient and scalable solution for large-scale vector similarity search with minimal memory footprint.

## Overview

This implementation follows the DiskANN paper's approach by:
- Using the Vamana graph algorithm for index construction
- Memory-mapping the index file for efficient disk-based access
- Implementing beam search with medoid entry points
- Supporting both Euclidean distance and Cosine similarity
- Maintaining minimal memory footprint during search operations

## Key Features

- **Single-file storage**: All index data stored in one memory-mapped file
- **Vamana graph construction**: Efficient graph building with α-pruning
- **Memory-efficient search**: Uses beam search that visits < 1% of vectors
- **Distance metrics**: Support for both Euclidean and Cosine similarity
- **Medoid-based entry points**: Smart starting points for search
- **Parallel query processing**: Using rayon for concurrent searches
- **Minimal memory footprint**: ~330MB RAM for 2GB index (16% of file size)

## Usage

### Building a New Index

```rust
use anndists::dist::{DistL2, DistCosine}; // or your own Distance types
use diskann_rs::{DiskANN, DiskAnnParams};

// Your vectors to index (all rows must share the same dimension)
let vectors: Vec<Vec<f32>> = vec![
    vec![0.1, 0.2, 0.3, /* ... */],
    vec![0.4, 0.5, 0.6, /* ... */],
    // ...
];

// Construction parameters
let params = DiskAnnParams {
    dim: 128,              // vector dimension
    max_degree: 32,        // max neighbors per node
    build_beam_width: 128, // construction beam width
    alpha: 1.2,            // α for pruning
};

// Choose a distance (here: L2). Use DistCosine for cosine, etc.
let index = DiskANN::<DistL2>::build_index(&vectors, &params, "index.dann")?;
```

### Opening an Existing Index

```rust
use anndists::dist::DistL2;
use diskann_rs::DiskANN;

// The distance type must match what you used at build time
let index = DiskANN::<DistL2>::open("index.dann")?;
```

### Searching the Index

```rust
use diskann_rs::SearchParams;

let query: Vec<f32> = vec![0.1, 0.2, /* ... */]; // length must equal params.dim
let k = 10;

// Search parameters (tune for recall vs speed)
let sp = SearchParams { beam_width: 64 };

let neighbors: Vec<u32> = index.search(&query, k, &sp);
// neighbors contains the IDs of the k nearest vectors
```

### Parallel Search

```rust
use rayon::prelude::*;
use std::sync::Arc;
use diskann_rs::SearchParams;

let index = Arc::new(index);
let sp = Arc::new(SearchParams { beam_width: 64 });

// Suppose you have a batch of queries
let query_batch: Vec<Vec<f32>> = /* ... */;

let results: Vec<Vec<u32>> = query_batch
    .par_iter()
    .map(|q| index.search(q, 10, &sp))
    .collect();
```

## Algorithm Details

### Vamana Graph Construction
1. Initialize with random graph connectivity
2. Iteratively improve edges using greedy search
3. Apply α-pruning to maintain diversity in neighbors
4. Ensure graph remains undirected

### Beam Search Algorithm
1. Start from medoid (vector closest to dataset centroid)
2. Maintain beam of promising candidates
3. Explore neighbors of best candidates
4. Terminate when no improvement found
5. Return top-k results

### Memory Management
- **Vectors**: Memory-mapped, loaded on-demand during distance calculations
- **Graph structure**: Adjacency lists stored contiguously in file
- **Search memory**: Only beam_width vectors in memory at once
- **Typical usage**: 10-100MB RAM for billion-scale indices

## Performance Characteristics

- **Index Build Time**: O(n * max_degree * beam_width)
- **Search Time**: O(beam_width * log n) - typically visits < 1% of dataset
- **Memory Usage**: O(beam_width) during search
- **Disk Space**: n * (dimension * 4 + max_degree * 4) bytes
- **Query Throughput**: Scales linearly with CPU cores

## Parameters Tuning

### Build Parameters
- `max_degree`: 32-64 for most datasets
- `build_beam_width`: 128-256 for good graph quality
- `alpha`: 1.2-2.0 (higher = more diverse neighbors)

### Search Parameters
- `beam_width`: 32-128 (trade-off between speed and recall)
- Higher beam_width = better recall but slower search

## Building and Testing

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run demo
cargo run --release --example demo

# Run performance test
cargo run --release --example perf_test
```

## Examples

See the `examples/` directory for:
- `demo.rs`: Demo with 100k vectors  
- `perf_test.rs`: Performance benchmarking with 1M vectors

## Current Implementation

✅ **Completed Features**:
- Single-file storage format with memory mapping
- Vamana graph construction with α-pruning
- Beam search with medoid entry points
- Multiple distance metrics (Euclidean, Cosine)
- Parallel query processing
- Comprehensive test suite
- Memory-efficient design (< 100MB for large indices)

## Future Improvements

1. Support for incremental index updates
2. Additional distance metrics (Manhattan, Hamming)
3. Compressed vector storage
4. Distributed index support
5. GPU acceleration for distance calculations
6. Auto-tuning of parameters

## Contributing

Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests
- Submit PRs for improvements
- Share performance benchmarks
- Suggest optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/)
- [Microsoft DiskANN Repository](https://github.com/Microsoft/DiskANN)

## Acknowledgments

This implementation is based on the DiskANN paper and the official Microsoft implementation, adapted for Rust with focus on simplicity and memory efficiency.