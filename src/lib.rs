//! # DiskAnnRS (generic over `anndists::Distance<f32>`)
//!
//! A minimal, in-memory DiskANN-like library that:
//! - Builds a Vamana-style graph (greedy + α-pruning) in memory
//! - Writes vectors + fixed-degree adjacency to a single file
//! - Memory-maps the file for low-overhead reads
//! - Is **generic over any Distance<f32>** from `anndists` (L2, Cosine, Dot, …)
//!
//!
//! ## Example
//! ```no_run
//! use anndists::dist::{DistL2, DistCosine};
//! use diskann_rs::{DiskANN, DiskAnnParams};
//!
//! // Build a new index from vectors, using L2 and default params
//! let vectors = vec![vec![0.0; 128]; 1000];
//! let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2{}, "index.db").unwrap();
//!
//! // Or with custom params
//! let index2 = DiskANN::<DistCosine>::build_index_with_params(
//!     &vectors,
//!     DistCosine{},
//!     "index_cos.db",
//!     DiskAnnParams { max_degree: 48, ..Default::default() },
//! ).unwrap();
//!
//! // Search the index
//! let query = vec![0.0; 128];
//! let neighbors = index.search(&query, 10, 64);
//!
//! // Open later (provide the same distance type)
//! let reopened = DiskANN::<DistL2>::open_index_default_metric("index.db").unwrap();
//! ```
//!
//! ## File Layout (simple, in-memory oriented)
//! [ metadata_len:u64 ][ metadata (bincode) ][ padding up to vectors_offset ]
//! [ vectors (num * dim * f32) ][ adjacency (num * max_degree * u32) ]
//!
//! `vectors_offset` is a fixed 1 MiB gap by default.

use anndists::prelude::Distance;
use bytemuck;
use memmap2::Mmap;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Mutex;
use thiserror::Error;

/// Padding sentinel for adjacency slots (avoid colliding with node 0).
const PAD_U32: u32 = u32::MAX;

/// Sane defaults for in-memory DiskANN builds.
pub const DISKANN_DEFAULT_MAX_DEGREE: usize = 32;
pub const DISKANN_DEFAULT_BUILD_BEAM: usize = 128;
pub const DISKANN_DEFAULT_ALPHA: f32 = 1.2;

/// Optional bag of knobs if you want to override just a few.
#[derive(Clone, Copy, Debug)]
pub struct DiskAnnParams {
    pub max_degree: usize,
    pub build_beam_width: usize,
    pub alpha: f32,
}
impl Default for DiskAnnParams {
    fn default() -> Self {
        Self {
            max_degree: DISKANN_DEFAULT_MAX_DEGREE,
            build_beam_width: DISKANN_DEFAULT_BUILD_BEAM,
            alpha: DISKANN_DEFAULT_ALPHA,
        }
    }
}

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

/// Internal metadata structure stored in the index file
#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
    distance_name: String,
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
        // Make BinaryHeap a min-heap on distance by reversing comparison.
        other.dist.partial_cmp(&self.dist)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Main struct representing a DiskANN index (generic over distance)
pub struct DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Dimensionality of vectors in the index
    pub dim: usize,
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Maximum number of edges per node
    pub max_degree: usize,
    /// Informational: type name of the distance (from metadata)
    pub distance_name: String,

    /// ID of the medoid (used as entry point)
    medoid_id: u32,
    // Legacy offsets kept for clarity
    vectors_offset: u64,
    adjacency_offset: u64,

    /// Memory-mapped file
    mmap: Mmap,

    /// The distance strategy
    dist: D,
}

// constructors

impl<D> DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default parameters: (M=32, beam=128, alpha=1.2).
    pub fn build_index_default(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            DISKANN_DEFAULT_MAX_DEGREE,
            DISKANN_DEFAULT_BUILD_BEAM,
            DISKANN_DEFAULT_ALPHA,
            dist,
            file_path,
        )
    }

    /// Build with a `DiskAnnParams` bundle.
    pub fn build_index_with_params(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        p: DiskAnnParams,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(vectors, p.max_degree, p.build_beam_width, p.alpha, dist, file_path)
    }
}

/// Extra sugar when your distance type implements `Default` (most unit-struct metrics do).
impl<D> DiskANN<D>
where
    D: Distance<f32> + Default + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default params **and** `D::default()` metric.
    pub fn build_index_default_metric(
        vectors: &[Vec<f32>],
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index_default(vectors, D::default(), file_path)
    }

    /// Open an index using `D::default()` as the distance (matches what you built with).
    pub fn open_index_default_metric(path: &str) -> Result<Self, DiskAnnError> {
        Self::open_index_with(path, D::default())
    }
}

impl<D> DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Builds a new index from provided vectors
    pub fn build_index(
        vectors: &[Vec<f32>],
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError("No vectors provided".to_string()));
        }

        let num_vectors = vectors.len();
        let dim = vectors[0].len();
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

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        // Reserve space for metadata (we'll write it after data)
        let vectors_offset = 1024 * 1024;
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;

        // Write vectors contiguous (sequential I/O is typically fastest)
        file.seek(SeekFrom::Start(vectors_offset))?;
        for vector in vectors {
            let bytes = bytemuck::cast_slice(vector);
            file.write_all(bytes)?;
        }

        // Compute medoid using provided distance (parallelized)
        let medoid_id = calculate_medoid(vectors, dist);

        // Build Vamana-like graph (parallelized)
        let adjacency_offset = vectors_offset as u64 + total_vector_bytes;
        let graph = build_vamana_graph(
            vectors,
            max_degree,
            build_beam_width,
            alpha,
            dist,
            medoid_id as u32,
        );

        // Write adjacency lists (fixed max_degree, pad with PAD_U32) - sequential I/O
        file.seek(SeekFrom::Start(adjacency_offset))?;
        for neighbors in &graph {
            let mut padded = neighbors.clone();
            padded.resize(max_degree, PAD_U32);
            let bytes = bytemuck::cast_slice(&padded);
            file.write_all(bytes)?;
        }

        // Write metadata
        let metadata = Metadata {
            dim,
            num_vectors,
            max_degree,
            medoid_id: medoid_id as u32,
            vectors_offset: vectors_offset as u64,
            adjacency_offset,
            distance_name: std::any::type_name::<D>().to_string(),
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
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
            dist,
        })
    }

    /// Opens an existing index file, supplying the distance strategy explicitly.
    pub fn open_index_with(path: &str, dist: D) -> Result<Self, DiskAnnError> {
        let mut file = OpenOptions::new().read(true).write(false).open(path)?;

        // Read metadata length
        let mut buf8 = [0u8; 8];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut buf8)?;
        let md_len = u64::from_le_bytes(buf8);

        // Read metadata
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_exact(&mut md_bytes)?;
        let metadata: Metadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Optional sanity/logging: warn if type differs from recorded name
        let expected = std::any::type_name::<D>();
        if metadata.distance_name != expected {
            eprintln!(
                "Warning: index recorded distance `{}` but you opened with `{}`",
                metadata.distance_name, expected
            );
        }

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
            dist,
        })
    }

    /// Searches the index for nearest neighbors using a simple beam search
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // frontier
        let mut w = BinaryHeap::new(); // working set

        // Start from medoid
        let start_dist = self.distance_to(query, self.medoid_id as usize);
        let start = Candidate {
            dist: start_dist,
            id: self.medoid_id,
        };
        candidates.push(start.clone());
        w.push(start);
        visited.insert(self.medoid_id);

        // Beam search with a mild early-stop heuristic
        let mut best_dist = start_dist;
        let mut iterations_without_improvement = 0usize;
        const MAX_NO_IMPROVE: usize = 5;

        while let Some(current) = candidates.pop() {
            if current.dist > best_dist {
                iterations_without_improvement += 1;
                if iterations_without_improvement > MAX_NO_IMPROVE {
                    break;
                }
            } else {
                best_dist = current.dist;
                iterations_without_improvement = 0;
            }

            let neighbors = self.get_neighbors(current.id).to_owned();

            for &neighbor_id in neighbors.iter() {
                if neighbor_id == PAD_U32 {
                    continue;
                }
                if !visited.insert(neighbor_id) {
                    continue;
                }

                let d = self.distance_to(query, neighbor_id as usize);
                w.push(Candidate { dist: d, id: neighbor_id });

                // Cap working set to beam_width (keep best `beam_width`)
                if w.len() > beam_width {
                    let mut temp = Vec::with_capacity(beam_width);
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

                // Add to frontier if promising
                if w.len() < beam_width || d < w.peek().unwrap().dist {
                    candidates.push(Candidate { dist: d, id: neighbor_id });
                }
            }
        }

        // Extract top-k by distance
        let mut results: Vec<_> = w.into_vec();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);
        results.into_iter().map(|c| c.id).collect()
    }

    /// Gets the neighbors of a node from the (fixed-degree) adjacency region
    fn get_neighbors(&self, node_id: u32) -> &[u32] {
        let offset = self.adjacency_offset + (node_id as u64 * self.max_degree as u64 * 4);
        let start = offset as usize;
        let end = start + (self.max_degree * 4);
        let bytes = &self.mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// Computes distance between `query` and vector `idx`
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        self.dist.eval(query, vector)
    }

    /// Gets a vector from the index (useful for tests)
    pub fn get_vector(&self, idx: usize) -> Vec<f32> {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        vector.to_vec()
    }
}

/// Calculates the medoid (vector closest to the centroid) using distance `D`
/// Parallelized with Rayon; `D: Send + Sync` so it can be shared across threads.
fn calculate_medoid<D: Distance<f32> + Copy + Send + Sync>(vectors: &[Vec<f32>], dist: D) -> usize {
    let n = vectors.len();
    let dim = vectors[0].len();

    // Parallel sum across vectors → centroid
    let mut centroid: Vec<f32> = vectors
        .par_iter()
        .map(|v| v.clone())
        .reduce(|| vec![0.0f32; dim], |mut a, b| {
            for j in 0..dim {
                a[j] += b[j];
            }
            a
        });
    for x in &mut centroid {
        *x /= n as f32;
    }

    // Parallel argmin distance to centroid
    vectors
        .par_iter()
        .enumerate()
        .map(|(idx, v)| (idx, dist.eval(&centroid, v)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Builds a Vamana-like graph using greedy search + α-pruning (parallelized)
fn build_vamana_graph<D: Distance<f32> + Copy + Send + Sync>(
    vectors: &[Vec<f32>],
    max_degree: usize,
    beam_width: usize,
    alpha: f32,
    dist: D,
    medoid_id: u32,
) -> Vec<Vec<u32>> {
    let n = vectors.len();

    // 1) Initialize with random neighbors (per-node parallel)
    let mut graph: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = thread_rng();
            let target = max_degree.min(n.saturating_sub(1));
            if target == 0 {
                return Vec::new();
            }
            let mut s = HashSet::with_capacity(target);
            while s.len() < target {
                let nb = rng.gen_range(0..n);
                if nb != i {
                    s.insert(nb as u32);
                }
            }
            s.into_iter().collect()
        })
        .collect();

    // 2) Improve graph with a few parallel passes
    let passes = 2usize;
    for _ in 0..passes {
        // 2a) Per-node parallel refinement: greedy search → α-prune (read-only view of `graph`)
        let next_graph: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|u| {
                let cands = greedy_search(
                    &vectors[u],
                    vectors,
                    &graph,
                    medoid_id as usize,
                    beam_width,
                    dist,
                );
                prune_neighbors(u, &cands, vectors, max_degree, alpha, dist)
            })
            .collect();

        // 2b) Enforce symmetry using a mutex-protected accumulator (parallel)
        let sym_lists: Vec<Mutex<Vec<u32>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();

        // Seed with current (u -> nb)
        (0..n).into_par_iter().for_each(|u| {
            let mut guard = sym_lists[u].lock().unwrap();
            guard.extend_from_slice(&next_graph[u]);
        });

        // Add back-edges (nb -> u)
        (0..n).into_par_iter().for_each(|u| {
            for &nb in &next_graph[u] {
                let mut g = sym_lists[nb as usize].lock().unwrap();
                g.push(u as u32);
            }
        });

        // Collect and dedup
        let sym_graph: Vec<Vec<u32>> = sym_lists
            .into_iter()
            .map(|m| {
                let mut v = m.into_inner().unwrap();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        // 2c) Final per-node prune to cap degree and maintain diversity (parallel)
        graph = (0..n)
            .into_par_iter()
            .map(|u| {
                if sym_graph[u].is_empty() {
                    return Vec::new();
                }
                let candidates: Vec<(u32, f32)> = sym_graph[u]
                    .iter()
                    .map(|&id| (id, dist.eval(&vectors[u], &vectors[id as usize])))
                    .collect();
                prune_neighbors(u, &candidates, vectors, max_degree, alpha, dist)
            })
            .collect();
    }

    graph
}

/// Greedy search used during construction (per-node, sequential)
fn greedy_search<D: Distance<f32> + Copy>(
    query: &[f32],
    vectors: &[Vec<f32>],
    graph: &[Vec<u32>],
    start_id: usize,
    beam_width: usize,
    dist: D,
) -> Vec<(u32, f32)> {
    let mut visited = HashSet::new();
    let mut frontier = BinaryHeap::new();
    let mut w = BinaryHeap::new();

    let start_dist = dist.eval(query, &vectors[start_id]);
    let start = Candidate {
        dist: start_dist,
        id: start_id as u32,
    };
    frontier.push(start.clone());
    w.push(start);
    visited.insert(start_id as u32);

    while let Some(cur) = frontier.pop() {
        for &nb in &graph[cur.id as usize] {
            if visited.contains(&nb) {
                continue;
            }
            visited.insert(nb);
            let d = dist.eval(query, &vectors[nb as usize]);
            w.push(Candidate { dist: d, id: nb });

            if w.len() > beam_width {
                let mut tmp = Vec::with_capacity(beam_width);
                for _ in 0..beam_width {
                    if let Some(c) = w.pop() {
                        tmp.push(c);
                    }
                }
                w.clear();
                for c in tmp {
                    w.push(c);
                }
            }

            if w.len() < beam_width || d < w.peek().unwrap().dist {
                frontier.push(Candidate { dist: d, id: nb });
            }
        }
    }

    w.into_vec().into_iter().map(|c| (c.id, c.dist)).collect()
}

/// α-pruning from DiskANN/Vamana (sequential per node)
fn prune_neighbors<D: Distance<f32> + Copy>(
    node_id: usize,
    candidates: &[(u32, f32)],
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    dist: D,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut pruned = Vec::<u32>::new();

    for &(cand_id, cand_dist) in &sorted {
        if cand_id as usize == node_id {
            continue;
        }
        let mut ok = true;
        for &sel in &pruned {
            let d = dist.eval(&vectors[cand_id as usize], &vectors[sel as usize]);
            if d < alpha * cand_dist {
                ok = false;
                break;
            }
        }
        if ok {
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    // fill with closest if still not full
    for &(cand_id, _) in &sorted {
        if cand_id as usize == node_id {
            continue;
        }
        if !pruned.contains(&cand_id) {
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    pruned
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::{DistCosine, DistL2};
    use rand::Rng;
    use std::fs;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    }

    #[test]
    fn test_small_index_l2() {
        let path = "test_small_l2.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let index =
            DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();

        let q = vec![0.1, 0.1];
        let nns = index.search(&q, 3, 8);
        assert_eq!(nns.len(), 3);

        // Verify the first neighbor is quite close in L2
        let v = index.get_vector(nns[0] as usize);
        assert!(euclid(&q, &v) < 1.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cosine() {
        let path = "test_cosine.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ];

        let index =
            DiskANN::<DistCosine>::build_index_default(&vectors, DistCosine {}, path).unwrap();

        let q = vec![2.0, 0.0, 0.0]; // parallel to [1,0,0]
        let nns = index.search(&q, 2, 8);
        assert_eq!(nns.len(), 2);

        // Top neighbor should have high cosine similarity (close direction)
        let v = index.get_vector(nns[0] as usize);
        let dot = v.iter().zip(&q).map(|(a,b)| a*b).sum::<f32>();
        let n1 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
        let n2 = q.iter().map(|x| x*x).sum::<f32>().sqrt();
        let cos = dot / (n1 * n2);
        assert!(cos > 0.7);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_persistence_and_open() {
        let path = "test_persist.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        {
            let _idx = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        }

        let idx2 = DiskANN::<DistL2>::open_index_default_metric(path).unwrap();
        assert_eq!(idx2.num_vectors, 4);
        assert_eq!(idx2.dim, 2);

        let q = vec![0.9, 0.9];
        let res = idx2.search(&q, 2, 8);
        // [1,1] should be best
        assert_eq!(res[0], 3);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_grid_connectivity() {
        let path = "test_grid.db";
        let _ = fs::remove_file(path);

        // 5x5 grid
        let mut vectors = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                vectors.push(vec![i as f32, j as f32]);
            }
        }

        let index =
            DiskANN::<DistL2>::build_index_with_params(
                &vectors,
                DistL2 {},
                path,
                DiskAnnParams { max_degree: 4, build_beam_width: 16, alpha: 1.5 }
            ).unwrap();

        for target in 0..vectors.len() {
            let q = &vectors[target];
            let nns = index.search(q, 10, 32);
            if !nns.contains(&(target as u32)) {
                let v = index.get_vector(nns[0] as usize);
                assert!(euclid(q, &v) < 2.0);
            }
            for &nb in nns.iter().take(5) {
                let v = index.get_vector(nb as usize);
                assert!(euclid(q, &v) < 5.0);
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_medium_random() {
        let path = "test_medium.db";
        let _ = fs::remove_file(path);

        let n = 200usize;
        let d = 32usize;
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.r#gen::<f32>()).collect())
            .collect();

        let index =
            DiskANN::<DistL2>::build_index_with_params(
                &vectors,
                DistL2 {},
                path,
                DiskAnnParams { max_degree: 32, build_beam_width: 64, alpha: 1.2 }
            ).unwrap();

        let q: Vec<f32> = (0..d).map(|_| rng.r#gen::<f32>()).collect();
        let res = index.search(&q, 10, 32);
        assert_eq!(res.len(), 10);

        // Ensure distances are nondecreasing
        let dists: Vec<f32> = res.iter().map(|&id| {
            let v = index.get_vector(id as usize);
            euclid(&q, &v)
        }).collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dists, sorted);

        let _ = fs::remove_file(path);
    }
}