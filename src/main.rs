use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    os::unix::fs::FileExt, // for read_at / write_at
    path::Path,
    sync::Arc,
    time::Instant,
};

use bytemuck;
use memmap2::Mmap;
use rand::prelude::{Rng, SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// =============================
//         ERROR TYPE
// =============================
#[derive(Debug, Error)]
pub enum DiskAnnError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Index error: {0}")]
    IndexError(String),
}

// =============================
//     SINGLE-FILE METADATA
// =============================
#[derive(Serialize, Deserialize, Debug)]
struct SingleFileMetadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,

    fraction_top: f64,
    fraction_mid: f64,

    layer0_ids: Vec<u32>,
    layer1_ids: Vec<u32>,

    // Where the vector data starts
    vectors_offset: u64,
    // Where adjacency data starts
    adjacency_offset: u64,

    offset_layer0: usize,
    offset_layer1: usize,
    offset_layer2: usize,
}

// =============================
//     SINGLE-FILE DISKANN
// =============================
pub struct SingleFileDiskANN {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,

    fraction_top: f64,
    fraction_mid: f64,

    layer0_ids: Vec<u32>,
    layer1_ids: Vec<u32>,

    offset_layer0: usize,
    offset_layer1: usize,
    offset_layer2: usize,

    vectors_offset: u64,
    adjacency_offset: u64,

    file: File,
    mmap: Mmap,
}

impl SingleFileDiskANN {
    /// Build everything in a single file (header + vectors + adjacency)
    /// with parallel adjacency building.
    pub fn build_index_singlefile(
        num_vectors: usize,
        dim: usize,
        max_degree: usize,
        fraction_top: f64,
        fraction_mid: f64,
        singlefile_path: &str,
    ) -> Result<Self, DiskAnnError> {
        // 1) Create the file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(singlefile_path)?;

        let vectors_offset = 1024 * 1024; // 1MB offset for the header

        // total bytes for vectors
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;
        // generate random vectors
        println!(
            "Generating {} random vectors of dim {} => ~{:.2}GB on disk...",
            num_vectors,
            dim,
            (num_vectors as f64 * dim as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0)
        );

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
            let offset = vectors_offset + (written * dim * 4) as u64;
            file.write_at(bytes, offset)?;
            written += batch;
        }
        let gen_time = gen_start.elapsed().as_secs_f32();
        println!("Vector generation took {:.2} s", gen_time);

        // pick layer subsets
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

        println!(
            "Layer0 size={}, Layer1 size={}, total={}",
            size_l0, size_l1, num_vectors
        );

        // adjacency offsets
        let bytes_per_node = max_degree * 4;
        let offset_layer0 = 0;
        let offset_layer1 = size_l0 * bytes_per_node;
        let offset_layer2 = offset_layer1 + (size_l1 * bytes_per_node);
        let total_adj_bytes = offset_layer2 + (num_vectors * bytes_per_node);

        let adjacency_offset = vectors_offset + total_vector_bytes;
        let adjacency_end = adjacency_offset + (total_adj_bytes as u64);
        file.set_len(adjacency_end)?;

        // pick random centroids
        let cluster_count = 20;
        let centroids =
            pick_random_centroids(cluster_count, &file, vectors_offset, dim, num_vectors)?;

        // 2) Build adjacency in parallel
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
        )?;
        // base layer => all IDs
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
        )?;
        let build_time = build_start.elapsed().as_secs_f32();
        println!("Parallel adjacency build took {:.2} s", build_time);

        // 3) Write metadata at offset 0
        let metadata = SingleFileMetadata {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            layer0_ids,
            layer1_ids,
            vectors_offset,
            adjacency_offset,
            offset_layer0,
            offset_layer1,
            offset_layer2,
        };
        let md_bytes = bincode::serialize(&metadata)?;
        // [u64: md_len][md_bytes...]
        file.seek(SeekFrom::Start(0))?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        // 4) memory-map entire file
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            layer0_ids: metadata.layer0_ids,
            layer1_ids: metadata.layer1_ids,
            offset_layer0: metadata.offset_layer0,
            offset_layer1: metadata.offset_layer1,
            offset_layer2: metadata.offset_layer2,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            file,
            mmap,
        })
    }

    /// Open existing single-file
    pub fn open_index_singlefile(path: &str) -> Result<Self, DiskAnnError> {
        let file = OpenOptions::new().read(true).write(false).open(path)?;
        let mut buf8 = [0u8; 8];
        file.read_at(&mut buf8, 0)?;
        let md_len = u64::from_le_bytes(buf8);
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_at(&mut md_bytes, 8)?;
        let metadata: SingleFileMetadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            fraction_top: metadata.fraction_top,
            fraction_mid: metadata.fraction_mid,
            layer0_ids: metadata.layer0_ids,
            layer1_ids: metadata.layer1_ids,
            offset_layer0: metadata.offset_layer0,
            offset_layer1: metadata.offset_layer1,
            offset_layer2: metadata.offset_layer2,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            file,
            mmap,
        })
    }

    /// 3-layer search
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

        let mut rng = rand::thread_rng();
        let start_idx = rng.gen_range(0..layer_ids.len());
        let start_id = layer_ids[start_idx];

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
        if let Some(&layer_idx) = id_to_idx.get(&start_id) {
            visited[layer_idx] = true;
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

    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let vector_offset = self.vectors_offset + (idx * self.dim * 4) as u64;
        let start = vector_offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vecf: &[f32] = bytemuck::cast_slice(bytes);
        euclidean_distance(query, vecf)
    }
}

// =============================
//     PARALLEL BUILD LOGIC
// =============================

/// Build adjacency using your naive cluster-based approach, **in parallel**.
fn build_layer_adjacency_parallel(
    file: &File,
    adjacency_offset: u64,
    layer_offset: usize,
    layer_ids: &[u32],
    dim: usize,
    max_degree: usize,
    vectors_offset: u64,
    centroids: &[(usize, Vec<f32>)],
) -> Result<(), DiskAnnError> {
    if layer_ids.is_empty() {
        return Ok(());
    }
    // 1) Parallel assignment of node -> centroid
    let node_assignments: Vec<(usize, u32)> = layer_ids
        .par_iter()
        .map(|&nid| {
            let nv = read_vector(file, vectors_offset, dim, nid as usize).unwrap();
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (cidx, (_, cvec)) in centroids.iter().enumerate() {
                let d = euclidean_distance(&nv, cvec);
                if d < best_d {
                    best_d = d;
                    best_c = cidx;
                }
            }
            (best_c, nid)
        })
        .collect();

    // 2) Now place them into buckets
    let cluster_count = centroids.len();
    let mut buckets = vec![Vec::new(); cluster_count];
    for (cidx, nid) in node_assignments {
        buckets[cidx].push(nid);
    }

    // 3) Parallel build adjacency per bucket
    buckets.into_par_iter().for_each(|bucket| {
        if bucket.len() <= 1 {
            return;
        }
        let mut rng = rand::thread_rng();
        let sample_size = 256.min(bucket.len());
        let mut sample_ids = bucket.clone();
        sample_ids.shuffle(&mut rng);
        sample_ids.truncate(sample_size);

        // read vectors for sample
        let sample_vecs: Vec<(u32, Vec<f32>)> = sample_ids
            .iter()
            .map(|&sid| {
                let v = read_vector(file, vectors_offset, dim, sid as usize).unwrap();
                (sid, v)
            })
            .collect();

        // Now build adjacency for each node in the bucket, but do it in parallel
        // because each node writes to a unique offset in the file => no overlap
        bucket.par_iter().for_each(|&nid| {
            let nv = read_vector(file, vectors_offset, dim, nid as usize).unwrap();
            let mut dists = Vec::with_capacity(sample_vecs.len());
            for (sid, sv) in &sample_vecs {
                if *sid != nid {
                    let d = euclidean_distance(&nv, sv);
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
            // safe to write at unique offset from parallel threads
            file.write_at(bytes, off).unwrap();
        });
    });

    Ok(())
}

// =============================
//   READ & DISTANCE HELPERS
// =============================
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

// =============================
//         DEMO MAIN
// =============================
fn main() -> Result<(), DiskAnnError> {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 1536;
    const MAX_DEGREE: usize = 32;
    const FRACTION_TOP: f64 = 0.01;
    const FRACTION_MID: f64 = 0.10;

    let singlefile_path = "diskann_parallel.db";

    // 1) Build if missing
    if !Path::new(singlefile_path).exists() {
        println!("Building single-file index at {singlefile_path} with parallel adjacency...");
        let start = Instant::now();
        let index = SingleFileDiskANN::build_index_singlefile(
            NUM_VECTORS,
            DIM,
            MAX_DEGREE,
            FRACTION_TOP,
            FRACTION_MID,
            singlefile_path,
        )?;
        let build_time = start.elapsed().as_secs_f32();
        println!("Done building index in {build_time:.2} s");
    } else {
        println!("Index file {singlefile_path} already exists, skipping build.");
    }

    // 2) Open
    let open_start = Instant::now();
    let index = Arc::new(SingleFileDiskANN::open_index_singlefile(singlefile_path)?);
    let open_time = open_start.elapsed().as_secs_f32();
    println!(
        "Opened index with {} vectors, dim={} in {:.2} s",
        index.num_vectors, index.dim, open_time
    );

    // 3) Queries
    let queries = 5;
    let k = 10;
    let beam_width = 64;
    let mut rng = rand::thread_rng();

    let search_start = Instant::now();
    let mut qvecs = Vec::new();
    for _ in 0..queries {
        let q: Vec<f32> = (0..index.dim).map(|_| rng.gen()).collect();
        qvecs.push(q);
    }
    // parallel queries
    qvecs.par_iter().enumerate().for_each(|(i, query)| {
        let knn = index.search(query, k, beam_width);
        println!("Query {i} => top-{k} neighbors = {knn:?}");
    });
    let search_time = search_start.elapsed().as_secs_f32();
    println!("Performed {queries} queries in {search_time:.2} s");

    Ok(())
}
