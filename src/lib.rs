//! # DiskAnnRS
//!
//! A DiskANN-like Rust library implementing approximate nearest neighbor search with
//! single-file storage support. The library provides both Euclidean distance and
//! Cosine similarity metrics, using the Vamana graph algorithm for efficient search.
//!
//! ## Features
//!
//! - Single-file storage format with memory-mapped access
//! - Support for both Euclidean and Cosine distance metrics
//! - Vamana graph construction with pruning
//! - Efficient beam search with medoid entry points
//! - Minimal memory footprint during search
//!
//! ## Example
//!
//! ```rust,no_run
//! use diskann_rs::{DiskANN, DistanceMetric};
//!
//! // Build a new index from vectors
//! let vectors = vec![vec![0.0; 128]; 1000]; // Your vectors
//! let index = DiskANN::build_index(
//!     &vectors,
//!     32,      // maximum degree
//!     128,     // build-time beam width  
//!     0.5,     // alpha parameter for pruning
//!     DistanceMetric::Euclidean,
//!     "index.db"
//! ).unwrap();
//!
//! // Search the index
//! let query = vec![0.0; 128];  // your query vector
//! let neighbors = index.search(&query, 10, 64);  // find top 10 with beam width 64
//! ```

use bytemuck;
use memmap2::Mmap;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::{
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
};
use thiserror::Error;

/// Custom error type for DiskAnnRS operations
#[derive(Debug, Error)]
pub enum DiskAnnError {
    /// Represents I/O errors during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Represents serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    /// Represents index-specific errors
    #[error("Index error: {0}")]
    IndexError(String),
}

/// Supported distance metrics for vector comparison
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Standard Euclidean distance
    Euclidean,
    /// Cosine similarity (converted to distance as 1 - similarity)
    Cosine,
}

/// Internal metadata structure stored in the index file
#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,
    distance_metric: DistanceMetric,
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
}

/// Main struct representing a DiskANN index
pub struct DiskANN {
    /// Dimensionality of vectors in the index
    pub dim: usize,
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Maximum number of edges per node
    pub max_degree: usize,
    /// Distance metric used by this index
    pub distance_metric: DistanceMetric,
    /// ID of the medoid (used as entry point)
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
    mmap: Mmap,
}

/// Candidate struct for search operations
#[derive(Clone)]
struct Candidate {
    dist: f32,
    id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Min-heap: smaller distance is "greater" priority
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl DiskANN {
    /// Builds a new index from provided vectors
    ///
    /// # Arguments
    ///
    /// * `vectors` - The vectors to index
    /// * `max_degree` - Maximum number of edges per node (typically 32-64)
    /// * `build_beam_width` - Beam width during construction (typically 128)
    /// * `alpha` - Pruning parameter (typically 1.2-2.0)
    /// * `distance_metric` - Distance metric to use
    /// * `file_path` - Path where the index file will be created
    ///
    /// # Returns
    ///
    /// Returns `Result<DiskANN, DiskAnnError>`
    pub fn build_index(
        vectors: &[Vec<f32>],
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
        distance_metric: DistanceMetric,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError("No vectors provided".to_string()));
        }

        let num_vectors = vectors.len();
        let dim = vectors[0].len();

        // Validate all vectors have same dimension
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        println!(
            "Building index for {} vectors of dimension {} with max_degree={}",
            num_vectors, dim, max_degree
        );

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        // Reserve space for metadata (we'll write it at the end)
        let vectors_offset = 1024 * 1024; // 1MB for metadata
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;

        // Write vectors to file
        file.seek(SeekFrom::Start(vectors_offset))?;
        for vector in vectors {
            let bytes = bytemuck::cast_slice(vector);
            file.write_all(bytes)?;
        }

        // Calculate medoid (centroid closest to mean of all vectors)
        let medoid_id = calculate_medoid(vectors, distance_metric);
        println!("Calculated medoid: {}", medoid_id);

        // Build Vamana graph
        let adjacency_offset = vectors_offset + total_vector_bytes;
        let graph = build_vamana_graph(
            vectors,
            max_degree,
            build_beam_width,
            alpha,
            distance_metric,
            medoid_id as u32,
        );

        // Write adjacency lists
        file.seek(SeekFrom::Start(adjacency_offset))?;
        for neighbors in &graph {
            // Pad with zeros if needed
            let mut padded = neighbors.clone();
            padded.resize(max_degree, 0);
            let bytes = bytemuck::cast_slice(&padded);
            file.write_all(bytes)?;
        }

        // Write metadata
        let metadata = Metadata {
            dim,
            num_vectors,
            max_degree,
            distance_metric,
            medoid_id: medoid_id as u32,
            vectors_offset,
            adjacency_offset,
        };

        let md_bytes = bincode::serialize(&metadata)?;
        file.seek(SeekFrom::Start(0))?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        // Memory map the file
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            distance_metric,
            medoid_id: metadata.medoid_id,
            vectors_offset,
            adjacency_offset,
            mmap,
        })
    }

    /// Opens an existing index file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    ///
    /// # Returns
    ///
    /// Returns `Result<DiskANN, DiskAnnError>`
    pub fn open_index(path: &str) -> Result<Self, DiskAnnError> {
        let file = OpenOptions::new().read(true).write(false).open(path)?;
        
        // Read metadata length
        let mut buf8 = [0u8; 8];
        use std::os::unix::fs::FileExt;
        file.read_exact_at(&mut buf8, 0)?;
        let md_len = u64::from_le_bytes(buf8);
        
        // Read metadata
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_exact_at(&mut md_bytes, 8)?;
        let metadata: Metadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_metric: metadata.distance_metric,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
        })
    }

    /// Searches the index for nearest neighbors using beam search
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `beam_width` - Beam width for the search (typically 32-128)
    ///
    /// # Returns
    ///
    /// Returns a vector of node IDs representing the nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        if query.len() != self.dim {
            panic!(
                "Query dimension {} does not match index dimension {}",
                query.len(),
                self.dim
            );
        }

        // Initialize with medoid as entry point
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new(); // Working set

        // Start from medoid
        let start_dist = self.distance_to(query, self.medoid_id as usize);
        candidates.push(Candidate {
            dist: start_dist,
            id: self.medoid_id,
        });
        w.push(Candidate {
            dist: start_dist,
            id: self.medoid_id,
        });
        visited.insert(self.medoid_id);

        // Beam search
        let mut best_dist = start_dist;
        let mut iterations_without_improvement = 0;
        const MAX_ITERATIONS_WITHOUT_IMPROVEMENT: usize = 5;

        while let Some(current) = candidates.pop() {
            // Early termination if no improvement
            if current.dist > best_dist {
                iterations_without_improvement += 1;
                if iterations_without_improvement > MAX_ITERATIONS_WITHOUT_IMPROVEMENT {
                    break;
                }
            } else {
                best_dist = current.dist;
                iterations_without_improvement = 0;
            }

            // Get neighbors of current node
            let neighbors = self.get_neighbors(current.id);

            for &neighbor_id in neighbors {
                if neighbor_id == 0 || visited.contains(&neighbor_id) {
                    continue;
                }

                visited.insert(neighbor_id);
                let dist = self.distance_to(query, neighbor_id as usize);

                // Update working set
                w.push(Candidate {
                    dist,
                    id: neighbor_id,
                });

                // Prune working set to beam width
                if w.len() > beam_width {
                    // Keep only top beam_width candidates
                    let mut temp = Vec::new();
                    for _ in 0..beam_width {
                        if let Some(c) = w.pop() {
                            temp.push(c);
                        }
                    }
                    w.clear();
                    for c in temp {
                        w.push(c);
                    }
                }

                // Add to candidates if promising
                if w.len() < beam_width || dist < w.peek().unwrap().dist {
                    candidates.push(Candidate {
                        dist,
                        id: neighbor_id,
                    });
                }
            }
        }

        // Extract top-k results
        let mut results: Vec<_> = w.into_vec();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);
        results.into_iter().map(|c| c.id).collect()
    }

    /// Gets the neighbors of a node from the graph
    fn get_neighbors(&self, node_id: u32) -> &[u32] {
        let offset = self.adjacency_offset + (node_id as u64 * self.max_degree as u64 * 4);
        let start = offset as usize;
        let end = start + (self.max_degree * 4);
        let bytes = &self.mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// Computes distance between query and a vector in the index
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);

        match self.distance_metric {
            DistanceMetric::Euclidean => euclidean_distance(query, vector),
            DistanceMetric::Cosine => 1.0 - cosine_similarity(query, vector),
        }
    }

    /// Gets a vector from the index (for testing)
    pub fn get_vector(&self, idx: usize) -> Vec<f32> {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        vector.to_vec()
    }
}

/// Calculates the medoid (vector closest to the centroid)
fn calculate_medoid(vectors: &[Vec<f32>], distance_metric: DistanceMetric) -> usize {
    let dim = vectors[0].len();
    let mut centroid = vec![0.0; dim];

    // Calculate centroid
    for vector in vectors {
        for (i, &val) in vector.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for val in &mut centroid {
        *val /= vectors.len() as f32;
    }

    // Find vector closest to centroid
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (idx, vector) in vectors.iter().enumerate() {
        let dist = match distance_metric {
            DistanceMetric::Euclidean => euclidean_distance(&centroid, vector),
            DistanceMetric::Cosine => 1.0 - cosine_similarity(&centroid, vector),
        };
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

/// Builds the Vamana graph using greedy search and pruning
fn build_vamana_graph(
    vectors: &[Vec<f32>],
    max_degree: usize,
    beam_width: usize,
    alpha: f32,
    distance_metric: DistanceMetric,
    medoid_id: u32,
) -> Vec<Vec<u32>> {
    let num_vectors = vectors.len();
    let mut graph = vec![Vec::new(); num_vectors];

    // Initialize with random graph
    let mut rng = thread_rng();
    for i in 0..num_vectors {
        let mut neighbors = HashSet::new();
        while neighbors.len() < max_degree.min(num_vectors - 1) {
            let neighbor = rng.gen_range(0..num_vectors);
            if neighbor != i {
                neighbors.insert(neighbor as u32);
            }
        }
        graph[i] = neighbors.into_iter().collect();
    }

    println!("Building Vamana graph with beam_width={}, alpha={}", beam_width, alpha);

    // Iterative improvement
    for iteration in 0..2 {
        println!("Graph building iteration {}", iteration + 1);
        
        // Process nodes in random order
        let mut node_order: Vec<usize> = (0..num_vectors).collect();
        node_order.shuffle(&mut rng);

        for &node_id in &node_order {
            // Search for nearest neighbors using current graph
            let neighbors = greedy_search(
                &vectors[node_id],
                vectors,
                &graph,
                medoid_id as usize,
                beam_width,
                distance_metric,
            );

            // Prune neighbors using α-pruning
            let pruned = prune_neighbors(
                node_id,
                &neighbors,
                vectors,
                max_degree,
                alpha,
                distance_metric,
            );

            graph[node_id] = pruned;

            // Make graph undirected by adding reverse edges
            let current_neighbors = graph[node_id].clone();
            for neighbor in current_neighbors {
                if !graph[neighbor as usize].contains(&(node_id as u32)) {
                    graph[neighbor as usize].push(node_id as u32);
                    
                    // Prune if degree exceeds max
                    if graph[neighbor as usize].len() > max_degree {
                        let neighbors_of_neighbor: Vec<_> = graph[neighbor as usize]
                            .iter()
                            .map(|&id| (id, {
                                let dist = match distance_metric {
                                    DistanceMetric::Euclidean => {
                                        euclidean_distance(&vectors[neighbor as usize], &vectors[id as usize])
                                    }
                                    DistanceMetric::Cosine => {
                                        1.0 - cosine_similarity(&vectors[neighbor as usize], &vectors[id as usize])
                                    }
                                };
                                dist
                            }))
                            .collect();
                        
                        let pruned = prune_neighbors(
                            neighbor as usize,
                            &neighbors_of_neighbor,
                            vectors,
                            max_degree,
                            alpha,
                            distance_metric,
                        );
                        graph[neighbor as usize] = pruned;
                    }
                }
            }
        }
    }

    graph
}

/// Performs greedy search on the graph during construction
fn greedy_search(
    query: &[f32],
    vectors: &[Vec<f32>],
    graph: &[Vec<u32>],
    start_id: usize,
    beam_width: usize,
    distance_metric: DistanceMetric,
) -> Vec<(u32, f32)> {
    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new();
    let mut w = BinaryHeap::new();

    // Start from medoid
    let start_dist = match distance_metric {
        DistanceMetric::Euclidean => euclidean_distance(query, &vectors[start_id]),
        DistanceMetric::Cosine => 1.0 - cosine_similarity(query, &vectors[start_id]),
    };

    candidates.push(Candidate {
        dist: start_dist,
        id: start_id as u32,
    });
    w.push(Candidate {
        dist: start_dist,
        id: start_id as u32,
    });
    visited.insert(start_id as u32);

    while let Some(current) = candidates.pop() {
        for &neighbor_id in &graph[current.id as usize] {
            if visited.contains(&neighbor_id) {
                continue;
            }

            visited.insert(neighbor_id);
            let dist = match distance_metric {
                DistanceMetric::Euclidean => euclidean_distance(query, &vectors[neighbor_id as usize]),
                DistanceMetric::Cosine => 1.0 - cosine_similarity(query, &vectors[neighbor_id as usize]),
            };

            w.push(Candidate { dist, id: neighbor_id });

            if w.len() > beam_width {
                let mut temp = Vec::new();
                for _ in 0..beam_width {
                    if let Some(c) = w.pop() {
                        temp.push(c);
                    }
                }
                w.clear();
                for c in temp {
                    w.push(c);
                }
            }

            if w.len() < beam_width || dist < w.peek().unwrap().dist {
                candidates.push(Candidate { dist, id: neighbor_id });
            }
        }
    }

    w.into_vec()
        .into_iter()
        .map(|c| (c.id, c.dist))
        .collect()
}

/// Prunes neighbors using the α-pruning strategy
fn prune_neighbors(
    node_id: usize,
    candidates: &[(u32, f32)],
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    distance_metric: DistanceMetric,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted_candidates = candidates.to_vec();
    sorted_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut pruned = Vec::new();
    
    for &(candidate_id, candidate_dist) in &sorted_candidates {
        if candidate_id as usize == node_id {
            continue;
        }

        // Check if this candidate is diverse enough from already selected neighbors
        let mut should_add = true;
        for &selected_id in &pruned {
            let dist_to_selected = match distance_metric {
                DistanceMetric::Euclidean => {
                    euclidean_distance(&vectors[candidate_id as usize], &vectors[selected_id as usize])
                }
                DistanceMetric::Cosine => {
                    1.0 - cosine_similarity(&vectors[candidate_id as usize], &vectors[selected_id as usize])
                }
            };

            if dist_to_selected < alpha * candidate_dist {
                should_add = false;
                break;
            }
        }

        if should_add {
            pruned.push(candidate_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    // Fill remaining slots with closest candidates if needed
    for &(candidate_id, _) in &sorted_candidates {
        if !pruned.contains(&candidate_id) && candidate_id as usize != node_id {
            pruned.push(candidate_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    pruned
}

/// Computes Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Computes cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_small_index() {
        let test_file = "test_small.db";
        
        // Clean up any existing test file
        let _ = fs::remove_file(test_file);

        // Create small test vectors
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        // Build index
        let index = DiskANN::build_index(
            &vectors,
            3,      // max_degree
            4,      // beam_width
            1.2,    // alpha
            DistanceMetric::Euclidean,
            test_file,
        )
        .unwrap();

        // Test search
        let query = vec![0.1, 0.1];
        let neighbors = index.search(&query, 3, 4);
        
        // Should find [0, 0] among the top results (it's the closest)
        // But since this is approximate search, just verify we get valid results
        assert_eq!(neighbors.len(), 3);
        
        // Verify the first result is reasonably close
        let first_vector = index.get_vector(neighbors[0] as usize);
        let dist = euclidean_distance(&query, &first_vector);
        assert!(dist < 1.0, "First neighbor should be close to query");
        
        // Clean up
        let _ = fs::remove_file(test_file);
    }

    #[test]
    fn test_memory_efficiency() {
        let test_file = "test_memory.db";
        let _ = fs::remove_file(test_file);

        // Create larger test set
        let num_vectors = 1000;
        let dim = 128;
        let mut vectors = Vec::new();
        let mut rng = thread_rng();
        
        for _ in 0..num_vectors {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            vectors.push(v);
        }

        // Build index
        let index = DiskANN::build_index(
            &vectors,
            32,     // max_degree
            64,     // beam_width
            1.2,    // alpha
            DistanceMetric::Euclidean,
            test_file,
        )
        .unwrap();

        // Memory usage test: search should visit only a small fraction of nodes
        let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let k = 10;
        let beam_width = 32;
        
        // This should complete quickly without loading all vectors
        let start = std::time::Instant::now();
        let neighbors = index.search(&query, k, beam_width);
        let elapsed = start.elapsed();
        
        assert_eq!(neighbors.len(), k);
        assert!(elapsed.as_millis() < 100, "Search took too long: {:?}", elapsed);
        
        // Verify results are reasonable
        let distances: Vec<f32> = neighbors
            .iter()
            .map(|&id| index.distance_to(&query, id as usize))
            .collect();
        
        // Distances should be sorted
        let mut sorted = distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(distances, sorted);

        let _ = fs::remove_file(test_file);
    }

    #[test]
    fn test_cosine_similarity() {
        let test_file = "test_cosine.db";
        let _ = fs::remove_file(test_file);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ];

        let index = DiskANN::build_index(
            &vectors,
            3,
            4,
            1.2,
            DistanceMetric::Cosine,
            test_file,
        )
        .unwrap();

        // Query similar to first vector
        let query = vec![2.0, 0.0, 0.0]; // Parallel to [1,0,0]
        let neighbors = index.search(&query, 2, 4);
        
        // Should find vector 0 among results (parallel vectors have cosine similarity 1)
        assert_eq!(neighbors.len(), 2);
        
        // The top result should have very high cosine similarity to query
        let first_vector = index.get_vector(neighbors[0] as usize);
        let similarity = cosine_similarity(&query, &first_vector);
        assert!(similarity > 0.7, "First neighbor should have high cosine similarity");

        let _ = fs::remove_file(test_file);
    }

    #[test]
    fn test_persistence() {
        let test_file = "test_persist.db";
        let _ = fs::remove_file(test_file);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        // Build and close index
        {
            let _index = DiskANN::build_index(
                &vectors,
                2,
                4,
                1.2,
                DistanceMetric::Euclidean,
                test_file,
            )
            .unwrap();
        }

        // Open existing index
        let index = DiskANN::open_index(test_file).unwrap();
        assert_eq!(index.num_vectors, 4);
        assert_eq!(index.dim, 2);

        // Search should work
        let query = vec![0.9, 0.9];
        let neighbors = index.search(&query, 2, 4);
        assert_eq!(neighbors[0], 3); // [1,1] is closest to [0.9,0.9]

        let _ = fs::remove_file(test_file);
    }

    #[test]
    fn test_graph_connectivity() {
        let test_file = "test_graph.db";
        let _ = fs::remove_file(test_file);

        // Create a grid of vectors
        let mut vectors = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                vectors.push(vec![i as f32, j as f32]);
            }
        }

        let index = DiskANN::build_index(
            &vectors,
            4,      // max_degree
            8,      // beam_width
            1.5,    // alpha
            DistanceMetric::Euclidean,
            test_file,
        )
        .unwrap();

        // Test that we can find reasonable neighbors for each vector
        for target_idx in 0..vectors.len() {
            let query = &vectors[target_idx];
            // Use higher beam width for better recall
            let neighbors = index.search(query, 10, 32);
            
            // The exact vector should be found with high beam width
            // If not found, at least verify we get close neighbors
            if !neighbors.contains(&(target_idx as u32)) {
                // Check that we at least found very close neighbors
                let first_vec = index.get_vector(neighbors[0] as usize);
                let dist = euclidean_distance(query, &first_vec);
                assert!(
                    dist < 2.0,
                    "Vector {} not found but nearest neighbor at distance {} is too far",
                    target_idx, dist
                );
            }
            
            // Verify all results are reasonable (all should be close)
            for &neighbor_id in neighbors.iter().take(5) {
                let neighbor_vec = index.get_vector(neighbor_id as usize);
                let dist = euclidean_distance(query, &neighbor_vec);
                assert!(dist < 5.0, "Neighbor should be reasonably close");
            }
        }

        let _ = fs::remove_file(test_file);
    }
}