//! diskannrs - A DiskANN-like Rust library with single-file storage
//! including Euclidean or Cosine distance and basic tests.

use bytemuck;
use memmap2::Mmap;
use rand::prelude::{Rng, SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    os::unix::fs::FileExt, // for read_at / write_at
    time::Instant,
};
use thiserror::Error;

/// Custom error type for our library
#[derive(Debug, Error)]
pub enum DiskAnnError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Index error: {0}")]
    IndexError(String),
}

/// DistanceMetric allows either Euclidean distance or Cosine similarity.
/// We'll interpret "distance" for Cosine as `1 - cos(...)` so smaller is "closer."
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
}

/// We store metadata in the single file
#[derive(Serialize, Deserialize, Debug)]
struct SingleFileMetadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,

    fraction_top: f64,
    fraction_mid: f64,
    distance_metric: DistanceMetric, // <--- store which metric

    layer0_ids: Vec<u32>,
    layer1_ids: Vec<u32>,

    vectors_offset: u64,
    adjacency_offset: u64,

    offset_layer0: usize,
    offset_layer1: usize,
    offset_layer2: usize,
}

/// Main struct
pub struct SingleFileDiskANN {
    pub dim: usize,
    pub num_vectors: usize,
    pub max_degree: usize,

    pub fraction_top: f64,
    pub fraction_mid: f64,
    pub distance_metric: DistanceMetric,

    layer0_ids: Vec<u32>,
    layer1_ids: Vec<u32>,

    offset_layer0: usize,
    offset_layer1: usize,
    offset_layer2: usize,

    vectors_offset: u64,
    adjacency_offset: u64,

    mmap: Mmap,
}

impl SingleFileDiskANN {
    /// Build a single-file index with Euclidean or Cosine distance.
    pub fn build_index_singlefile(
        num_vectors: usize,
        dim: usize,
        max_degree: usize,
        fraction_top: f64,
        fraction_mid: f64,
        distance_metric: DistanceMetric,
        singlefile_path: &str,
    ) -> Result<Self, DiskAnnError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(singlefile_path)?;

        let vectors_offset = 1024 * 1024;
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;

        // 1) generate random vectors
        println!(
            "Generating {num_vectors} random vectors of dim {dim} => ~{:.2} GB on disk...",
            (num_vectors as f64 * dim as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0)
        );
        let chunk_size = 100_000;
        let mut rng = rand::thread_rng();
        let mut buffer = Vec::with_capacity(chunk_size * dim);

        let gen_start = Instant::now();
        let mut written = 0usize;
        while written < num_vectors {
            buffer.clear();
            let remaining = num_vectors - written;
            let batch = remaining.min(chunk_size);
            for _ in 0..batch {
                for _ in 0..dim {
                    let val: f32 = rng.gen();
                    buffer.push(val);
                }
            }
            let bytes = bytemuck::cast_slice(&buffer);
            let offset = vectors_offset + (written * dim * 4) as u64;
            file.write_at(bytes, offset)?;
            written += batch;
        }
        let gen_time = gen_start.elapsed().as_secs_f32();
        println!("Vector generation took {gen_time:.2} s");

        // 2) pick subsets
        let mut all_ids: Vec<u32> = (0..num_vectors as u32).collect();
        all_ids.shuffle(&mut rng);
        let size_l0 = (num_vectors as f64 * fraction_top).ceil() as usize;
        let size_l1 = (num_vectors as f64 * fraction_mid).ceil() as usize;
        let size_l1 = size_l1.max(size_l0);

        let l0slice = &all_ids[..size_l0];
        let l1slice = &all_ids[..size_l1];
        let mut layer0_ids = l0slice.to_vec();
        let mut layer1_ids = l1slice.to_vec();
        layer0_ids.sort_unstable();
        layer1_ids.sort_unstable();

        let bytes_per_node = max_degree * 4;
        let offset_layer0 = 0;
        let offset_layer1 = size_l0 * bytes_per_node;
        let offset_layer2 = offset_layer1 + (size_l1 * bytes_per_node);
        let total_adj_bytes = offset_layer2 + (num_vectors * bytes_per_node);

        let adjacency_offset = vectors_offset + total_vector_bytes;
        let adjacency_end = adjacency_offset + total_adj_bytes as u64;
        file.set_len(adjacency_end)?;

        let cluster_count = 20;
        let centroids =
            pick_random_centroids(cluster_count, &file, vectors_offset, dim, num_vectors)?;

        // 3) parallel adjacency
        let build_start = Instant::now();
        build_layer_adjacency_parallel(
            &file,
            adjacency_offset,
            offset_layer0,
            &layer0_ids,
            dim,
            max_degree,
            vectors_offset,
            &centroids,
            distance_metric,
        )?;
        build_layer_adjacency_parallel(
            &file,
            adjacency_offset,
            offset_layer1,
            &layer1_ids,
            dim,
            max_degree,
            vectors_offset,
            &centroids,
            distance_metric,
        )?;
        let base_ids: Vec<u32> = (0..num_vectors as u32).collect();
        build_layer_adjacency_parallel(
            &file,
            adjacency_offset,
            offset_layer2,
            &base_ids,
            dim,
            max_degree,
            vectors_offset,
            &centroids,
            distance_metric,
        )?;
        let build_time = build_start.elapsed().as_secs_f32();
        println!("Parallel adjacency build took {build_time:.2} s");

        // 4) write metadata
        let metadata = SingleFileMetadata {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            distance_metric,
            layer0_ids,
            layer1_ids,
            vectors_offset,
            adjacency_offset,
            offset_layer0,
            offset_layer1,
            offset_layer2,
        };
        let md_bytes = bincode::serialize(&metadata)?;
        file.seek(SeekFrom::Start(0))?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            distance_metric: metadata.distance_metric,
            layer0_ids: metadata.layer0_ids,
            layer1_ids: metadata.layer1_ids,
            offset_layer0: metadata.offset_layer0,
            offset_layer1: metadata.offset_layer1,
            offset_layer2: metadata.offset_layer2,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
        })
    }

    /// Open an existing single-file index
    pub fn open_index_singlefile(path: &str) -> Result<Self, DiskAnnError> {
        let file = OpenOptions::new().read(true).write(false).open(path)?;
        let mut buf8 = [0u8; 8];
        file.read_at(&mut buf8, 0)?;
        let md_len = u64::from_le_bytes(buf8);
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_at(&mut md_bytes, 8)?;
        let metadata: SingleFileMetadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            fraction_top: metadata.fraction_top,
            fraction_mid: metadata.fraction_mid,
            distance_metric: metadata.distance_metric,
            layer0_ids: metadata.layer0_ids,
            layer1_ids: metadata.layer1_ids,
            offset_layer0: metadata.offset_layer0,
            offset_layer1: metadata.offset_layer1,
            offset_layer2: metadata.offset_layer2,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
        })
    }

    /// 3-layer BFS/beam search
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        let _l0 = self.search_layer(query, &self.layer0_ids, self.offset_layer0, beam_width, 1);
        let _l1 = self.search_layer(query, &self.layer1_ids, self.offset_layer1, beam_width, 1);
        let base_ids: Vec<u32> = (0..self.num_vectors as u32).collect();
        self.search_layer(query, &base_ids, self.offset_layer2, beam_width, k)
    }

    fn search_layer(
        &self,
        query: &[f32],
        layer_ids: &[u32],
        layer_offset: usize,
        beam_width: usize,
        k: usize,
    ) -> Vec<u32> {
        if layer_ids.is_empty() {
            return vec![];
        }
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Clone)]
        struct Candidate {
            dist: f32,
            node_id: u32,
        }
        impl PartialEq for Candidate {
            fn eq(&self, other: &Self) -> bool {
                self.dist == other.dist
            }
        }
        impl Eq for Candidate {}
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.dist.partial_cmp(&self.dist)
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        // pick the node in layer_ids closest to the query
        let mut best_id = layer_ids[0];
        let mut best_dist = self.distance_to(query, best_id as usize);
        for &candidate_id in layer_ids.iter().skip(1) {
            let d = self.distance_to(query, candidate_id as usize);
            if d < best_dist {
                best_dist = d;
                best_id = candidate_id;
            }
        }
        let start_id = best_id;

        let mut visited = vec![false; layer_ids.len()];
        let id_to_idx = layer_ids
            .iter()
            .enumerate()
            .map(|(i, &nid)| (nid, i))
            .collect::<std::collections::HashMap<u32, usize>>();

        let start_dist = self.distance_to(query, start_id as usize);
        let mut frontier = vec![Candidate {
            dist: start_dist,
            node_id: start_id,
        }];
        if let Some(&idx) = id_to_idx.get(&start_id) {
            visited[idx] = true;
        }

        let mut best = BinaryHeap::new();
        best.push(frontier[0].clone());

        while let Some(current) = frontier.pop() {
            let neighbors = self.get_layer_neighbors(current.node_id, layer_offset);
            for &nbr in neighbors {
                if nbr == 0 {
                    continue;
                }
                if let Some(&nbr_idx) = id_to_idx.get(&nbr) {
                    if !visited[nbr_idx] {
                        visited[nbr_idx] = true;
                        let d = self.distance_to(query, nbr as usize);
                        let cand = Candidate {
                            dist: d,
                            node_id: nbr,
                        };
                        frontier.push(cand.clone());
                        best.push(cand);
                        if best.len() > beam_width {
                            best.pop();
                        }
                    }
                }
            }
        }

        let mut final_vec = best.into_vec();
        final_vec.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        final_vec.truncate(k);
        final_vec.into_iter().map(|c| c.node_id).collect()
    }

    fn get_layer_neighbors(&self, node_id: u32, layer_offset: usize) -> &[u32] {
        let node_off = layer_offset + (node_id as usize * self.max_degree * 4);
        let start = (self.adjacency_offset as usize) + node_off;
        let end = start + (self.max_degree * 4);
        let bytes = &self.mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// distance_to calls either Euclidean or Cosine, depending on self.distance_metric.
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let vector_offset = self.vectors_offset + (idx * self.dim * 4) as u64;
        let start = vector_offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vecf: &[f32] = bytemuck::cast_slice(bytes);

        match self.distance_metric {
            DistanceMetric::Euclidean => euclidean_distance(query, vecf),
            DistanceMetric::Cosine => {
                // We'll interpret "distance" = 1 - cos
                1.0 - cosine_similarity(query, vecf)
            }
        }
    }
}

// Parallel adjacency build
fn build_layer_adjacency_parallel(
    file: &File,
    adjacency_offset: u64,
    layer_offset: usize,
    layer_ids: &[u32],
    dim: usize,
    max_degree: usize,
    vectors_offset: u64,
    centroids: &[(usize, Vec<f32>)],
    distance_metric: DistanceMetric,
) -> Result<(), DiskAnnError> {
    if layer_ids.is_empty() {
        return Ok(());
    }

    // 1) parallel assignment
    let node_assignments: Vec<(usize, u32)> = layer_ids
        .par_iter()
        .map(|&nid| {
            let nv = read_vector(file, vectors_offset, dim, nid as usize).unwrap();
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (cidx, (_, cvec)) in centroids.iter().enumerate() {
                let d = match distance_metric {
                    DistanceMetric::Euclidean => euclidean_distance(&nv, cvec),
                    DistanceMetric::Cosine => 1.0 - cosine_similarity(&nv, cvec),
                };
                if d < best_d {
                    best_d = d;
                    best_c = cidx;
                }
            }
            (best_c, nid)
        })
        .collect();

    // group them
    let cluster_count = centroids.len();
    let mut buckets = vec![Vec::new(); cluster_count];
    for (cidx, nid) in node_assignments {
        buckets[cidx].push(nid);
    }

    // 2) parallel over buckets
    buckets.into_par_iter().for_each(|bucket| {
        if bucket.len() <= 1 {
            return;
        }
        let mut rng = rand::thread_rng();
        let sample_size = 256.min(bucket.len());
        let mut sample_ids = bucket.clone();
        sample_ids.shuffle(&mut rng);
        sample_ids.truncate(sample_size);

        let sample_vecs: Vec<(u32, Vec<f32>)> = sample_ids
            .iter()
            .map(|&sid| {
                let v = read_vector(file, vectors_offset, dim, sid as usize).unwrap();
                (sid, v)
            })
            .collect();

        bucket.par_iter().for_each(|&nid| {
            let nv = read_vector(file, vectors_offset, dim, nid as usize).unwrap();
            let mut dists = Vec::with_capacity(sample_vecs.len());
            for (sid, sv) in &sample_vecs {
                if *sid != nid {
                    let d = match distance_metric {
                        DistanceMetric::Euclidean => euclidean_distance(&nv, sv),
                        DistanceMetric::Cosine => 1.0 - cosine_similarity(&nv, sv),
                    };
                    dists.push((*sid, d));
                }
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(max_degree);

            let mut nbrs: Vec<u32> = dists.iter().map(|(id, _)| *id).collect();
            while nbrs.len() < max_degree {
                nbrs.push(0);
            }
            let node_off = layer_offset + (nid as usize * max_degree * 4);
            let off = adjacency_offset + node_off as u64;
            let bytes = bytemuck::cast_slice(&nbrs);
            file.write_at(bytes, off).unwrap();
        });
    });

    Ok(())
}

fn pick_random_centroids(
    cluster_count: usize,
    file: &File,
    vectors_offset: u64,
    dim: usize,
    num_vectors: usize,
) -> Result<Vec<(usize, Vec<f32>)>, DiskAnnError> {
    let mut rng = rand::thread_rng();
    let mut cents = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let id = rng.gen_range(0..num_vectors);
        let vec = read_vector(file, vectors_offset, dim, id)?;
        cents.push((id, vec));
    }
    Ok(cents)
}

fn read_vector(
    file: &File,
    vectors_offset: u64,
    dim: usize,
    idx: usize,
) -> Result<Vec<f32>, DiskAnnError> {
    let off = vectors_offset + (idx * dim * 4) as u64;
    let mut buf = vec![0u8; dim * 4];
    file.read_at(&mut buf, off)?;
    let floats: &[f32] = bytemuck::cast_slice(&buf);
    Ok(floats.to_vec())
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity = (a·b) / (||a|| * ||b||)
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
        return 0.0; // handle degenerate case
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Basic unit tests
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // We'll define a small set of 5 2D vectors in a static array.
    // This is purely deterministic.
    const TEST_VECTORS_2D: &[[f32; 2]] =
        &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];

    /// A special builder that does not generate random vectors, but writes
    /// the `TEST_VECTORS_2D` to the file, uses cluster_count=1 or 2,
    /// and no partial sampling for the adjacency build.
    fn build_index_singlefile_for_test(
        dim: usize,
        distance_metric: DistanceMetric,
        file_path: &str,
    ) -> Result<SingleFileDiskANN, DiskAnnError> {
        // 1) Create/truncate the file
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        let num_vectors = TEST_VECTORS_2D.len();
        let vectors_offset = 1024 * 1024;
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;
        file.set_len(vectors_offset + total_vector_bytes)?;

        // 2) Write the fixed vectors into the file
        // We'll assume dim=2 matches TEST_VECTORS_2D
        for (i, vec2) in TEST_VECTORS_2D.iter().enumerate() {
            let offset = vectors_offset + (i * dim * 4) as u64;
            let bytes = bytemuck::cast_slice(vec2);
            file.write_at(bytes, offset)?;
        }

        // 3) We'll place all vectors in a single "cluster" => cluster_count=1
        // So adjacency is effectively complete (we won't skip anything).
        // Or if you want 2 clusters, do cluster_count=2 but it usually won't matter
        let cluster_count = 1;

        // We'll store IDs in [0..num_vectors)
        let layer0_ids: Vec<u32> = (0..num_vectors as u32).collect();
        let layer1_ids = layer0_ids.clone();
        let fraction_top = 0.2;
        let fraction_mid = 0.4;
        let max_degree = 2; // for the small test
        let bytes_per_node = max_degree * 4;

        let offset_layer0 = 0;
        let offset_layer1 = layer0_ids.len() * bytes_per_node;
        let offset_layer2 = offset_layer1 + (layer1_ids.len() * bytes_per_node);

        let adjacency_offset = vectors_offset + total_vector_bytes;
        let total_adj_bytes = offset_layer2 + (num_vectors * bytes_per_node);
        file.set_len(adjacency_offset + total_adj_bytes as u64)?;

        // 4) Build adjacency *without partial sampling*
        //    We'll treat the entire bucket. That ensures a "complete" adjacency for a small set.
        let centroids =
            pick_predefined_centroids(cluster_count, &file, vectors_offset, dim, distance_metric)?;

        // Build adjacency for layer0 => all vectors
        build_layer_adjacency_test(
            &file,
            adjacency_offset,
            offset_layer0,
            &layer0_ids,
            dim,
            max_degree,
            vectors_offset,
            &centroids,
            distance_metric,
            /* no partial sampling = entire bucket */
        )?;

        // same for layer1 => same IDs
        build_layer_adjacency_test(
            &file,
            adjacency_offset,
            offset_layer1,
            &layer1_ids,
            dim,
            max_degree,
            vectors_offset,
            &centroids,
            distance_metric,
            /* no partial sampling = entire bucket */
        )?;

        // layer2 => also same set
        let base_ids: Vec<u32> = (0..num_vectors as u32).collect();
        build_layer_adjacency_test(
            &file,
            adjacency_offset,
            offset_layer2,
            &base_ids,
            dim,
            max_degree,
            vectors_offset,
            &centroids,
            distance_metric,
            /* no partial sampling */
        )?;

        // 5) Write metadata
        let md = SingleFileMetadata {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            distance_metric,
            layer0_ids: layer0_ids.clone(),
            layer1_ids: layer1_ids.clone(),
            vectors_offset,
            adjacency_offset,
            offset_layer0,
            offset_layer1,
            offset_layer2,
        };
        let md_bytes = bincode::serialize(&md)?;
        file.seek(std::io::SeekFrom::Start(0))?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Return the struct
        Ok(SingleFileDiskANN {
            dim,
            num_vectors,
            max_degree: md.max_degree,
            fraction_top: md.fraction_top,
            fraction_mid: md.fraction_mid,
            distance_metric: md.distance_metric,
            layer0_ids: md.layer0_ids,
            layer1_ids: md.layer1_ids,
            offset_layer0: md.offset_layer0,
            offset_layer1: md.offset_layer1,
            offset_layer2: md.offset_layer2,
            vectors_offset: md.vectors_offset,
            adjacency_offset: md.adjacency_offset,
            mmap,
        })
    }

    /// We define a small "centroid" for cluster_count=1 => we just pick e.g. the first vector
    fn pick_predefined_centroids(
        cluster_count: usize,
        file: &std::fs::File,
        vectors_offset: u64,
        dim: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Vec<(usize, Vec<f32>)>, DiskAnnError> {
        // If cluster_count=1, let's just pick the first vector
        let mut out = Vec::new();
        for i in 0..cluster_count {
            // We'll pick i-th
            let v = read_vector(file, vectors_offset, dim, i)?;
            out.push((i, v));
        }
        Ok(out)
    }

    /// A specialized adjacency builder that uses the entire bucket for small tests
    fn build_layer_adjacency_test(
        file: &std::fs::File,
        adjacency_offset: u64,
        layer_offset: usize,
        layer_ids: &[u32],
        dim: usize,
        max_degree: usize,
        vectors_offset: u64,
        centroids: &[(usize, Vec<f32>)],
        distance_metric: DistanceMetric,
        // no partial sampling => entire bucket
    ) -> Result<(), DiskAnnError> {
        if layer_ids.is_empty() {
            return Ok(());
        }
        // single cluster approach => everything in 1 bucket
        let mut bucket = Vec::new();
        for &nid in layer_ids {
            bucket.push(nid);
        }

        // sample = entire bucket
        let sample_vecs: Vec<(u32, Vec<f32>)> = bucket
            .iter()
            .map(|&sid| {
                let v = read_vector(file, vectors_offset, dim, sid as usize).unwrap();
                (sid, v)
            })
            .collect();

        // adjacency for each node => top max_degree from entire bucket minus itself
        for &nid in bucket.iter() {
            let nv = read_vector(file, vectors_offset, dim, nid as usize).unwrap();
            // compute distance to all other nodes in bucket
            let mut dists = Vec::with_capacity(sample_vecs.len());
            for (sid, sv) in &sample_vecs {
                if *sid != nid {
                    let d = match distance_metric {
                        DistanceMetric::Euclidean => euclidean_distance(&nv, sv),
                        DistanceMetric::Cosine => 1.0 - cosine_similarity(&nv, sv),
                    };
                    dists.push((*sid, d));
                }
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(max_degree);

            let mut nbrs: Vec<u32> = dists.iter().map(|(id, _)| *id).collect();
            while nbrs.len() < max_degree {
                nbrs.push(0);
            }

            let node_off = layer_offset + (nid as usize * max_degree * 4);
            let off = adjacency_offset + node_off as u64;
            let bytes = bytemuck::cast_slice(&nbrs);
            file.write_at(bytes, off).unwrap();
        }

        Ok(())
    }

    // test_small_euclidean using the above "hard-coded" approach
    #[test]
    fn test_small_euclidean() -> Result<(), DiskAnnError> {
        let tmpfile = "test_small_euclid.db";
        if std::path::Path::new(tmpfile).exists() {
            std::fs::remove_file(tmpfile).unwrap();
        }

        // build w/ no random + single cluster => adjacency is effectively "complete"
        let index = build_index_singlefile_for_test(
            2, // dim
            DistanceMetric::Euclidean,
            tmpfile,
        )?;

        // We'll pick vector 0 as the query
        let query = index.get_vector(0)?;
        let k = 2;
        let beam_width = 4;
        let neighbors = index.search(&query, k, beam_width);

        // Manually compute actual top-2 by Euclidean among our TEST_VECTORS_2D
        // We know them from e.g. [0,0], [1,0], [0,1], [1,1], [0.5,0.5]
        // In this approach, let's do it generically:
        let n = index.num_vectors;
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = index.get_vector(i)?;
                Ok((i, euclidean_distance(&query, &v)))
            })
            .collect::<Result<Vec<_>, DiskAnnError>>()?;

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let correct_ids: Vec<usize> = dists[..k].iter().map(|(i, _)| *i).collect();

        let set1: std::collections::HashSet<_> = correct_ids.into_iter().collect();
        let set2: std::collections::HashSet<_> = neighbors.iter().map(|&x| x as usize).collect();
        assert_eq!(set1, set2);

        Ok(())
    }

    #[test]
    fn test_small_cosine() -> Result<(), DiskAnnError> {
        let tmpfile = "test_small_cosine.db";
        if std::path::Path::new(tmpfile).exists() {
            std::fs::remove_file(tmpfile).unwrap();
        }

        let index = build_index_singlefile_for_test(
            2, // dim
            DistanceMetric::Cosine,
            tmpfile,
        )?;

        // Use vector[1] = [1.0, 0.0] as query instead of vector[0]
        let query = index.get_vector(1)?;
        let k = 2;
        let beam_width = 4;
        let neighbors = index.search(&query, k, beam_width);

        // compute actual top-2 by (1 - cos)
        let n = index.num_vectors;
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = index.get_vector(i)?;
                let sim = cosine_similarity(&query, &v);
                Ok((i, 1.0 - sim)) // interpret distance = 1 - cos
            })
            .collect::<Result<Vec<_>, DiskAnnError>>()?;

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let correct_ids: Vec<usize> = dists[..k].iter().map(|(i, _)| *i).collect();

        let set1: HashSet<_> = setify(correct_ids);
        let set2: HashSet<_> = setify(neighbors.iter().map(|&x| x as usize));
        assert_eq!(set1, set2);

        Ok(())
    }

    // small convenience fn to build a hashset from an iterator
    fn setify<I>(iter: I) -> HashSet<<I as IntoIterator>::Item>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Eq + Hash,
    {
        iter.into_iter().collect()
    }

    // reuses your get_vector for the test
    impl SingleFileDiskANN {
        pub fn get_vector(&self, idx: usize) -> Result<Vec<f32>, DiskAnnError> {
            let vector_offset = self.vectors_offset + (idx * self.dim * 4) as u64;
            let start = vector_offset as usize;
            let end = start + (self.dim * 4);
            let bytes = &self.mmap[start..end];
            let vecf: &[f32] = bytemuck::cast_slice(bytes);
            Ok(vecf.to_vec())
        }
    }
}
