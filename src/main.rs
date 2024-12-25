use rand::prelude::{Rng, SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Write, Seek, SeekFrom}, // IMPORTANT: needed for file.seek
    path::Path,
    time::Instant,
};
use thiserror::Error;
use memmap2::Mmap;

/// Custom error type
#[derive(Debug, Error)]
pub enum DiskAnnTestError {
    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization Error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Index Error: {0}")]
    IndexError(String),
}

/// A toy multi-layer index that stores:
///  - The dimension of vectors
///  - Two layers: 
///      layer 0 (coarse) has a subset of vector IDs,
///      layer 1 (base) has all vectors.
///  - Both layers' adjacency is stored in a single flat binary file (memory-mapped).
///  - The vectors themselves are also memory-mapped from a .bin file.
pub struct MultiLayerDiskANN {
    dim: usize,
    num_vectors: usize,

    /// The fraction of vectors we keep in the top (coarse) layer
    top_layer_fraction: f64,

    /// File + mmap for vectors
    #[allow(dead_code)]
    vectors_file: File,
    vectors_mmap: Mmap,

    /// File + mmap for adjacency
    #[allow(dead_code)]
    adjacency_file: File,
    adjacency_mmap: Mmap,

    /// IDs in the top layer (subset) 
    top_layer_ids: Vec<u32>,

    /// Offsets in the adjacency file for each layer
    layer0_offset: usize,
    layer1_offset: usize,

    /// The max_degree used
    max_degree: usize,
}

#[derive(Serialize, Deserialize)]
struct MultiLayerMetadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,
    top_layer_fraction: f64,
    top_layer_ids: Vec<u32>,
    layer0_offset: usize,
    layer1_offset: usize,
}

impl MultiLayerDiskANN {
    /// Build a 2-layer index on disk:
    pub fn build_index_on_disk(
        num_vectors: usize,
        dim: usize,
        max_degree: usize,
        top_layer_fraction: f64,
        vectors_path: &str,
        adjacency_path: &str,
        metadata_path: &str,
    ) -> Result<Self, DiskAnnTestError> 
    {
        // 1) Generate random vectors
        println!(
            "Generating {} random vectors of dim {} => ~{:.2} GB of floats on disk...",
            num_vectors,
            dim,
            (num_vectors as f64 * dim as f64 * 4.0) / (1024.0*1024.0*1024.0)
        );
        let mut vfile = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(vectors_path)?;

        let total_bytes = num_vectors * dim * 4;
        vfile.set_len(total_bytes as u64)?;

        let mut rng = rand::thread_rng();
        let chunk_size = 100_000;
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
            vfile.write_all(bytes)?;
            written += batch;
        }
        vfile.sync_all()?;
        let gen_time = gen_start.elapsed().as_secs_f32();
        println!("Vector generation took {:.2} s", gen_time);

        // 2) Sample top_layer_fraction of the IDs for layer 0
        let mut all_ids: Vec<u32> = (0..num_vectors as u32).collect();
        // Shuffle them
        all_ids.shuffle(&mut rng);

        let top_size = (num_vectors as f64 * top_layer_fraction).ceil() as usize;
        let mut top_layer_ids = all_ids[..top_size].to_vec();
        top_layer_ids.sort();  // optional: keep sorted

        println!(
            "Top layer fraction = {}, so top layer size = {} out of {}",
            top_layer_fraction, top_size, num_vectors
        );

        // 3) Build adjacency in a single flat file:
        //    We'll create adjacency for layer 0 (top) and layer 1 (base).
        //    For each node in each layer, we store max_degree neighbors (u32) contiguously.

        let mut afile = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(adjacency_path)?;

        // offset for layer 0 adjacency
        let layer0_offset = 0;
        let bytes_per_node = max_degree * 4; // each neighbor is u32 => 4 bytes
        let total_top_adj_bytes = top_size * bytes_per_node;

        // offset for layer 1 adjacency
        let layer1_offset = total_top_adj_bytes; 
        let total_base_adj_bytes = num_vectors * bytes_per_node;

        let total_adj_bytes = layer1_offset + total_base_adj_bytes;
        afile.set_len(total_adj_bytes as u64)?;

        // We'll fill adjacency in place:
        let chunk_adj_size = 100_000;
        let mut adj_buffer = Vec::with_capacity(chunk_adj_size * max_degree);

        println!(
            "Generating adjacency for layer0={} nodes + layer1={} nodes. Writing to disk...",
            top_size, num_vectors
        );
        let adj_start = Instant::now();

        // helper to fill adjacency for a set of node IDs
        let mut write_layer_adjacency = | 
            node_ids: &[u32], 
            file: &mut File, 
            offset: usize,
            max_degree: usize,
            total_count: usize,
            rng: &mut rand::prelude::ThreadRng
        | -> Result<(), DiskAnnTestError> {
            let mut written_nodes = 0;
            while written_nodes < node_ids.len() {
                let batch = (node_ids.len() - written_nodes).min(chunk_adj_size);
                adj_buffer.clear();

                for i_idx in 0..batch {
                    let node_id = node_ids[written_nodes + i_idx];
                    let mut nbrs = Vec::new();
                    // pick distinct random neighbors 
                    while nbrs.len() < max_degree && nbrs.len() < total_count - 1 {
                        let n = rng.gen_range(0..total_count as u32);
                        if n != node_id && !nbrs.contains(&n) {
                            nbrs.push(n);
                        }
                    }
                    adj_buffer.extend_from_slice(&nbrs);
                }

                let start_of_batch = offset + (written_nodes * max_degree * 4);
                file.seek(SeekFrom::Start(start_of_batch as u64))?;
                let bytes_slice = bytemuck::cast_slice(&adj_buffer);
                file.write_all(bytes_slice)?;
                written_nodes += batch;
            }
            Ok(())
        };

        // layer 0 adjacency
        write_layer_adjacency(
            &top_layer_ids,
            &mut afile,
            layer0_offset,
            max_degree,
            num_vectors,
            &mut rng,
        )?;

        // layer 1 adjacency (for all nodes 0..num_vectors)
        let all_ids_slice: Vec<u32> = (0..num_vectors as u32).collect();
        write_layer_adjacency(
            &all_ids_slice,
            &mut afile,
            layer1_offset,
            max_degree,
            num_vectors,
            &mut rng,
        )?;

        afile.sync_all()?;
        let adj_time = adj_start.elapsed().as_secs_f32();
        println!("Adjacency generation+writing took {:.2} s", adj_time);

        // 4) Write metadata
        let metadata = MultiLayerMetadata {
            dim,
            num_vectors,
            max_degree,
            top_layer_fraction,
            top_layer_ids,
            layer0_offset,
            layer1_offset,
        };
        let mdata = bincode::serialize(&metadata)?;
        std::fs::write(metadata_path, &mdata)?;

        println!("Done building multi-layer index.");

        // 5) Open files and memory-map for returning the struct
        let vectors_file = OpenOptions::new().read(true).write(false).open(vectors_path)?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file)? };

        let adjacency_file = OpenOptions::new().read(true).write(false).open(adjacency_path)?;
        let adjacency_mmap = unsafe { Mmap::map(&adjacency_file)? };

        // Reload metadata to get top_layer_ids etc.
        let mdata = std::fs::read(metadata_path)?;
        let meta: MultiLayerMetadata = bincode::deserialize(&mdata)?;
        let top_layer_ids = meta.top_layer_ids;

        Ok(Self {
            dim: meta.dim,
            num_vectors: meta.num_vectors,
            top_layer_fraction: meta.top_layer_fraction,
            vectors_file,
            vectors_mmap,
            adjacency_file,
            adjacency_mmap,
            top_layer_ids,
            layer0_offset: meta.layer0_offset,
            layer1_offset: meta.layer1_offset,
            max_degree: meta.max_degree,
        })
    }

    /// Open existing index from disk (vectors + adjacency + metadata).
    pub fn open_index(
        vectors_path: &str,
        adjacency_path: &str,
        metadata_path: &str,
    ) -> Result<Self, DiskAnnTestError> {
        // read metadata
        let mdata = std::fs::read(metadata_path)?;
        let meta: MultiLayerMetadata = bincode::deserialize(&mdata)?;

        // memory-map
        let vectors_file = OpenOptions::new().read(true).write(false).open(vectors_path)?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file)? };

        let adjacency_file = OpenOptions::new().read(true).write(false).open(adjacency_path)?;
        let adjacency_mmap = unsafe { Mmap::map(&adjacency_file)? };

        Ok(Self {
            dim: meta.dim,
            num_vectors: meta.num_vectors,
            top_layer_fraction: meta.top_layer_fraction,
            vectors_file,
            vectors_mmap,
            adjacency_file,
            adjacency_mmap,
            top_layer_ids: meta.top_layer_ids,
            layer0_offset: meta.layer0_offset,
            layer1_offset: meta.layer1_offset,
            max_degree: meta.max_degree,
        })
    }

    /// Multi-layer search
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        if query.len() != self.dim {
            panic!("Query dim != index dim");
        }
        let entry_id = self.search_layer_coarse(query, beam_width);
        self.search_layer_base(query, entry_id, k, beam_width)
    }

    /// Search top layer using beam search
    fn search_layer_coarse(&self, query: &[f32], beam_width: usize) -> u32 {
        if self.top_layer_ids.is_empty() {
            return 0;
        }
        let mut rng = rand::thread_rng();
        let start_idx = rng.gen_range(0..self.top_layer_ids.len());
        let start_id = self.top_layer_ids[start_idx];

        let mut visited = vec![false; self.top_layer_ids.len()];

        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(Clone)]
        struct Candidate {
            dist: f32,
            idx_in_layer: usize,
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

        let start_dist = self.distance_to(query, start_id as usize);
        visited[start_idx] = true;
        let start_cand = Candidate { dist: start_dist, idx_in_layer: start_idx };

        let mut frontier = vec![start_cand.clone()];
        let mut best_candidates = BinaryHeap::new();
        best_candidates.push(start_cand);

        while let Some(current) = frontier.pop() {
            let node_id = self.top_layer_ids[current.idx_in_layer];
            let neighbors = self.get_layer0_neighbors(current.idx_in_layer);
            for &nbr_id in neighbors {
                // find its index via binary_search
                if let Ok(nbr_layer_idx) = self.top_layer_ids.binary_search(&nbr_id) {
                    if !visited[nbr_layer_idx] {
                        visited[nbr_layer_idx] = true;
                        let d = self.distance_to(query, nbr_id as usize);
                        let cand = Candidate { dist: d, idx_in_layer: nbr_layer_idx };
                        frontier.push(cand.clone());
                        best_candidates.push(cand);
                        if best_candidates.len() > beam_width {
                            best_candidates.pop();
                        }
                    }
                }
            }
        }

        let mut final_vec = best_candidates.into_vec();
        final_vec.sort_by(|a,b| a.dist.partial_cmp(&b.dist).unwrap());
        if final_vec.is_empty() {
            self.top_layer_ids[0]
        } else {
            self.top_layer_ids[final_vec[0].idx_in_layer]
        }
    }

    /// Search base layer with beam search
    fn search_layer_base(&self, query: &[f32], entry_id: u32, k: usize, beam_width: usize) -> Vec<u32> {
        let mut visited = vec![false; self.num_vectors];
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

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

        let start_dist = self.distance_to(query, entry_id as usize);
        visited[entry_id as usize] = true;
        let start_cand = Candidate { dist: start_dist, node_id: entry_id };
        let mut frontier = vec![start_cand.clone()];
        let mut best_candidates = BinaryHeap::new();
        best_candidates.push(start_cand);

        while let Some(current) = frontier.pop() {
            let neighbors = self.get_layer1_neighbors(current.node_id);
            for &nbr_id in neighbors {
                if !visited[nbr_id as usize] {
                    visited[nbr_id as usize] = true;
                    let d = self.distance_to(query, nbr_id as usize);
                    let cand = Candidate { dist: d, node_id: nbr_id };
                    frontier.push(cand.clone());
                    best_candidates.push(cand);
                    if best_candidates.len() > beam_width {
                        best_candidates.pop();
                    }
                }
            }
        }

        let mut final_vec = best_candidates.into_vec();
        final_vec.sort_by(|a,b| a.dist.partial_cmp(&b.dist).unwrap());
        final_vec.truncate(k);

        final_vec.into_iter().map(|c| c.node_id).collect()
    }

    /// get adjacency for top-layer node at index `idx_in_layer0`
    fn get_layer0_neighbors(&self, idx_in_layer0: usize) -> &[u32] {
        let offset = self.layer0_offset + (idx_in_layer0 * self.max_degree * 4);
        let end = offset + (self.max_degree * 4);
        let bytes = &self.adjacency_mmap[offset..end];
        bytemuck::cast_slice(bytes)
    }

    /// get adjacency for base-layer node_id
    fn get_layer1_neighbors(&self, node_id: u32) -> &[u32] {
        let offset = self.layer1_offset + (node_id as usize * self.max_degree * 4);
        let end = offset + (self.max_degree * 4);
        let bytes = &self.adjacency_mmap[offset..end];
        bytemuck::cast_slice(bytes)
    }

    /// distance to vector #idx in the vectors file
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim * 4;
        let end = start + (self.dim * 4);
        let bytes = &self.vectors_mmap[start..end];
        let vec_f32: &[f32] = bytemuck::cast_slice(bytes);
        euclidean_distance(query, vec_f32)
    }
}

/// Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Demo main
fn main() -> Result<(), DiskAnnTestError> {
    // Adjust for your environment
    const NUM_VECTORS: usize = 1_000_000;   // smaller for quick test
    const DIM: usize = 1536;
    const MAX_DEGREE: usize = 32;
    const TOP_LAYER_FRACTION: f64 = 0.01; // 1% in top layer

    let vectors_path = "vectors.bin";
    let adjacency_path = "adjacency.bin";
    let metadata_path = "metadata.bin";

    // 1) Build if needed
    if !Path::new(vectors_path).exists() {
        println!("Building multi-layer index on disk...");
        let build_start = Instant::now();
        MultiLayerDiskANN::build_index_on_disk(
            NUM_VECTORS,
            DIM,
            MAX_DEGREE,
            TOP_LAYER_FRACTION,
            vectors_path,
            adjacency_path,
            metadata_path
        )?;
        let build_time = build_start.elapsed().as_secs_f32();
        println!("Done building. Elapsed = {:.2} s", build_time);
    } else {
        println!("Index files already exist, skipping build.");
    }

    // 2) Open
    let open_start = Instant::now();
    let index = MultiLayerDiskANN::open_index(vectors_path, adjacency_path, metadata_path)?;
    let open_time = open_start.elapsed().as_secs_f32();
    println!(
        "Opened multi-layer index with {} vectors, dim={}, top fraction={} in {:.2} s",
        index.num_vectors, index.dim, index.top_layer_fraction, open_time
    );

    // 3) Queries
    let queries = 5;
    let k = 10;
    let beam_width = 64;
    let mut rng = rand::thread_rng();

    let search_start = Instant::now();
    for i in 0..queries {
        let query: Vec<f32> = (0..index.dim).map(|_| rng.gen()).collect();
        let neighbors = index.search(&query, k, beam_width);
        println!("Query {i} => top-{k} neighbors: {neighbors:?}");
    }
    let search_time = search_start.elapsed().as_secs_f32();
    println!(
        "Performed {queries} queries in {search_time:.2} s (~{:.2} s/query)",
        search_time / queries as f32
    );

    Ok(())
}

