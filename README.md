# DiskANN Implementation in Rust

A Rust implementation of DiskANN (Disk-based Approximate Nearest Neighbor search) focused on learning, testing, and optimization. This project aims to create an easy-to-use library for large-scale vector similarity search that can be dropped into other projects.

## Overview

This implementation provides a memory-efficient approach to similarity search by:
- Storing vectors on disk using memory mapping
- Managing adjacency lists for graph-based search
- Supporting large-scale datasets that don't fit in RAM
- Implementing approximate nearest neighbor search using graph traversal

## Features

- Memory-mapped vector storage for handling large datasets
- Serialized adjacency lists for graph structure
- Random index generation for testing
- Basic BFS-like search implementation
- Support for euclidean distance metric
- Error handling with custom error types

## Usage

### Building an Index

```rust
let index = LargeScaleDiskANN::build_index_on_disk(
    1_000_000,    // number of vectors
    128,          // dimension
    64,           // max neighbors per node
    "vectors.bin", // path to store vectors
    "adjacency.bin" // path to store adjacency lists
)?;
```

### Opening an Existing Index

```rust
let index = LargeScaleDiskANN::open_index(
    "vectors.bin",
    "adjacency.bin",
    128  // dimension
)?;
```

### Searching

```rust
let query: Vec<f32> = vec![/* your query vector */];
let k = 10; // number of nearest neighbors to find
let neighbors = index.search(&query, k);
```

## Performance Characteristics

- Memory Usage: O(1) for vector storage due to memory mapping
- Disk Space: 
  - Vectors: num_vectors * dimension * 4 bytes
  - Adjacency Lists: Depends on max_degree setting
- Search Time: Depends on graph structure and dataset size

## Current Status

This is a learning implementation focused on understanding DiskANN's core concepts. Current areas of focus:
- [ ] Optimizing graph construction
- [ ] Implementing better search strategies
- [ ] Benchmarking and profiling
- [ ] Adding more distance metrics
- [ ] Improving memory efficiency

## Building and Testing

```bash
# Build in release mode
cargo build --release

# Run the demo
cargo run --release
```

## Future Improvements

1. Implement proper graph construction using proximity information
2. Add support for multiple distance metrics
3. Optimize search algorithm
4. Add batch processing capabilities
5. Implement proper parameter tuning
6. Add comprehensive benchmarking suite

## Contributing

This is a learning project, and contributions or suggestions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit PRs for improvements
- Share ideas for optimization

## References

- Original DiskANN paper: [[link](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/?msockid=1b74891036f76220363b98c3379c6384)]
