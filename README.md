# DiskANN Implementation in Rust

[![Latest Version](https://img.shields.io/crates/v/diskann_rs?style=for-the-badge&color=mediumpurple&logo=rust)](https://crates.io/crates/diskann_rs)
[![docs.rs](https://img.shields.io/docsrs/diskann_rs?style=for-the-badge&logo=docs.rs&color=mediumseagreen)](https://docs.rs/diskann_rs/latest/diskann_rs/)


A Rust implementation of [DiskANN]() (Disk-based Approximate Nearest Neighbor search) using the Vamana graph algorithm. This project provides an efficient and scalable solution for large-scale vector similarity search with minimal memory footprint, as an alternative to the widely used in-memory [HNSW](https://crates.io/crates/hnsw_rs) algorithm. 

## Overview

This implementation follows the DiskANN paper's approach by:
- Using the Vamana graph algorithm for index construction
- Memory-mapping the index file for efficient disk-based access
- Implementing beam search with medoid entry points
- Supporting Euclidean, Cosine, Hamming and other distance metrics via a generic distance trait
- Maintaining minimal memory footprint during search operations

## Key Features

- **Single-file storage**: All index data stored in one memory-mapped file
- **Vamana graph construction**: Efficient graph building with α-pruning, with rayon for concurrent and parallel construction
- **Memory-efficient search**: Uses beam search that visits < 1% of vectors
- **Distance metrics**: Support for Euclidean, Cosine and Hamming similarity et.al. via [anndists](https://crates.io/crates/anndists). A generic distance trait that can be extended to other distance metrics
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
- `beam_width`: 128 or larger (trade-off between speed and recall)
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

# test MNIST fashion dataset
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
cargo run --release --example diskann_mnist

# test SIFT dataset
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
cargo run --release --example diskann_sift
```

## Examples

See the `examples/` directory for:
- `demo.rs`: Demo with 100k vectors  
- `perf_test.rs`: Performance benchmarking with 1M vectors
- `diskann_mnist.rs`: Performance benchmarking with MNIST fashion dataset
- `diskann_sift.rs`: Performance benchmarking with SIFT 1M dataset

## Benchmark against HNSW ([hnsw_rs](https://crates.io/crates/hnsw_rs) crate)
```bash
### MNIST fashion, diskann, M4 Max
Building DiskANN index: n=60000, dim=784, max_degree=48, build_beam=256, alpha=1.2
Build complete. CPU time: 1726.199372s, wall time: 111.145414s
Searching 10000 queries with k=10, beam_width=384 …
 mean fraction nb returned by search 1.0

 last distances ratio 1.0031366
 recall rate for "./fashion-mnist-784-euclidean.hdf5" is 0.98838 , nb req /s 18067.664

 total cpu time for search requests 8.520862s , system time 553.475ms


### MNIST fashion, hnsw_rs, M4 Max
parallel insertion

 hnsw data insertion cpu time  111.169283s  system time Ok(7.256291s) 
 debug dump of PointIndexation
 layer 0 : length : 59999 
 layer 1 : length : 1 
 debug dump of PointIndexation end
 hnsw data nb point inserted 60000

 searching with ef : 24
 
 parallel search
total cpu time for search requests 3838.7310ms , system time 263.571ms 

 mean fraction nb returned by search 1.0 

 last distances ratio 1.0003573 

 recall rate for "./fashion-mnist-784-euclidean.hdf5" is 0.99054 , nb req /s 37940.44

```

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