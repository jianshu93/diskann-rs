use bytemuck;
use memmap2::Mmap;
use rand::prelude::{Rng, SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    os::unix::fs::FileExt, // <-- for read_at / write_at
    path::Path,
    sync::Arc,
    time::Instant,
};
use thiserror::Error;

/// =============================
///         ERROR TYPE
/// =============================
#[derive(Debug, Error)]
pub enum DiskAnnTestError {
    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization Error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Index Error: {0}")]
    IndexError(String),
}

/// =============================
///      METADATA STRUCTS
/// =============================

/// We’ll build THREE layers:
///   L0 (top)    => fraction_top
///   L1 (middle) => fraction_mid
///   L2 (base)   => 100%
/// We store adjacency for each layer in a single file (separate offsets).
#[derive(Serialize, Deserialize)]
struct MultiLayerMetadata {
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
}

/// =============================
///    DISKANN-LIKE STRUCT
/// =============================
pub struct MultiLayerDiskANN {
    dim: usize,
    num_vectors: usize,

    max_degree: usize,

    fraction_top: f64,
    fraction_mid: f64,

    #[allow(dead_code)]
    vectors_file: File, // read-only
    vectors_mmap: Mmap,

    #[allow(dead_code)]
    adjacency_file: File, // read-only
    adjacency_mmap: Mmap,

    layer0_ids: Vec<u32>,
    layer1_ids: Vec<u32>,

    offset_layer0: usize,
    offset_layer1: usize,
    offset_layer2: usize,
}

/// =============================
///      BUILDING THE INDEX
/// =============================
impl MultiLayerDiskANN {
    /// Build a 3-layer index on disk with a naive "cluster-based" adjacency.
    pub fn build_index_on_disk(
        num_vectors: usize,
        dim: usize,
        max_degree: usize,
        fraction_top: f64,
        fraction_mid: f64,
        vectors_path: &str,
        adjacency_path: &str,
        metadata_path: &str,
    ) -> Result<Self, DiskAnnTestError> {
        // 1) Generate random vectors on disk
        println!(
            "Generating {} random vectors of dim {} => ~{:.2} GB on disk...",
            num_vectors,
            dim,
            (num_vectors as f64 * dim as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0)
        );
        let vfile = OpenOptions::new()
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
            // write at offset = written * dim * 4
            let offset = (written * dim * 4) as u64;
            vfile.write_at(bytes, offset)?; // <-- no seek, just write_at
            written += batch;
        }
        // done
        let gen_time = gen_start.elapsed().as_secs_f32();
        println!("Vector generation took {:.2} s", gen_time);

        // 2) Choose L0 and L1 IDs
        let mut all_ids: Vec<u32> = (0..num_vectors as u32).collect();
        all_ids.shuffle(&mut rng);

        let size_l0 = (num_vectors as f64 * fraction_top).ceil() as usize;
        let size_l1 = (num_vectors as f64 * fraction_mid).ceil() as usize;
        let size_l1 = size_l1.max(size_l0);

        let layer0_ids = &all_ids[..size_l0];
        let layer1_ids = &all_ids[..size_l1];
        let mut l0 = layer0_ids.to_vec();
        let mut l1 = layer1_ids.to_vec();

        l0.sort_unstable();
        l1.sort_unstable();

        println!(
            "Layer0 size={}, Layer1 size={}, total={}",
            size_l0, size_l1, num_vectors
        );

        // 3) Build adjacency (cluster-based)
        let cluster_count = 20;
        let centroids = pick_random_centroids(cluster_count, &vfile, dim, num_vectors)?;

        let bytes_per_node = max_degree * 4;
        let offset_layer0 = 0;
        let offset_layer1 = size_l0 * bytes_per_node;
        let offset_layer2 = offset_layer1 + (size_l1 * bytes_per_node);

        let total_adj_bytes = offset_layer2 + (num_vectors * bytes_per_node);

        let afile = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(adjacency_path)?;
        afile.set_len(total_adj_bytes as u64)?;

        // build adjacency for L0
        build_layer_adjacency(
            &afile,
            offset_layer0,
            &l0,
            dim,
            max_degree,
            &vfile,
            &centroids,
        )?;

        // build adjacency for L1
        build_layer_adjacency(
            &afile,
            offset_layer1,
            &l1,
            dim,
            max_degree,
            &vfile,
            &centroids,
        )?;

        // build adjacency for all (base)
        let base_ids: Vec<u32> = (0..num_vectors as u32).collect();
        build_layer_adjacency(
            &afile,
            offset_layer2,
            &base_ids,
            dim,
            max_degree,
            &vfile,
            &centroids,
        )?;

        println!("Done building 3-layer index.");

        // 4) Write metadata
        let meta = MultiLayerMetadata {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            layer0_ids: l0,
            layer1_ids: l1,
            offset_layer0,
            offset_layer1,
            offset_layer2,
        };
        let mbytes = bincode::serialize(&meta)?;
        std::fs::write(metadata_path, &mbytes)?;

        println!("Index build complete.");

        // 5) Memory-map
        let vectors_file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(vectors_path)?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file)? };

        let adjacency_file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(adjacency_path)?;
        let adjacency_mmap = unsafe { Mmap::map(&adjacency_file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            fraction_top,
            fraction_mid,
            vectors_file,
            vectors_mmap,
            adjacency_file,
            adjacency_mmap,
            layer0_ids: meta.layer0_ids,
            layer1_ids: meta.layer1_ids,
            offset_layer0: meta.offset_layer0,
            offset_layer1: meta.offset_layer1,
            offset_layer2: meta.offset_layer2,
        })
    }

    /// Open existing index
    pub fn open_index(
        vectors_path: &str,
        adjacency_path: &str,
        metadata_path: &str,
    ) -> Result<Self, DiskAnnTestError> {
        let mbytes = std::fs::read(metadata_path)?;
        let meta: MultiLayerMetadata = bincode::deserialize(&mbytes)?;

        let vectors_file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(vectors_path)?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file)? };

        let adjacency_file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(adjacency_path)?;
        let adjacency_mmap = unsafe { Mmap::map(&adjacency_file)? };

        Ok(Self {
            dim: meta.dim,
            num_vectors: meta.num_vectors,
            max_degree: meta.max_degree,
            fraction_top: meta.fraction_top,
            fraction_mid: meta.fraction_mid,
            vectors_file,
            vectors_mmap,
            adjacency_file,
            adjacency_mmap,
            layer0_ids: meta.layer0_ids,
            layer1_ids: meta.layer1_ids,
            offset_layer0: meta.offset_layer0,
            offset_layer1: meta.offset_layer1,
            offset_layer2: meta.offset_layer2,
        })
    }
}

/// =============================
///   1) Meaningful Adjacency
/// =============================

/// We'll do a naive "cluster-based" approach:
///   - pick `cluster_count` random centroids
///   - for each node, find the centroid it's closest to
///   - among that centroid's sample, pick up to max_degree nearest neighbors
fn pick_random_centroids(
    cluster_count: usize,
    file: &File,
    dim: usize,
    num_vectors: usize,
) -> Result<Vec<(usize, Vec<f32>)>, DiskAnnTestError> {
    let mut rng = rand::thread_rng();
    let mut cents = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let id = rng.gen_range(0..num_vectors);
        let vec = read_vector(file, dim, id)?;
        cents.push((id, vec));
    }
    Ok(cents)
}

/// Builds adjacency for a given layer's node IDs, storing the result in `afile` at `offset_start`.
fn build_layer_adjacency(
    afile: &File, // <-- read_at, write_at
    offset_start: usize,
    layer_ids: &[u32],
    dim: usize,
    max_degree: usize,
    vectors_file: &File,
    centroids: &Vec<(usize, Vec<f32>)>,
) -> Result<(), DiskAnnTestError> {
    let cluster_count = centroids.len();
    let mut buckets = vec![Vec::new(); cluster_count];

    // 1) assign each node to its nearest centroid
    for &nid in layer_ids {
        let nvec = read_vector(vectors_file, dim, nid as usize)?;
        let mut best_c = 0;
        let mut best_d = f32::MAX;
        for (cidx, (_, cvec)) in centroids.iter().enumerate() {
            let d = euclidean_distance(&nvec, cvec);
            if d < best_d {
                best_d = d;
                best_c = cidx;
            }
        }
        buckets[best_c].push(nid);
    }

    // 2) For each bucket, pick up to max_degree neighbors from a partial sample
    for bucket in &buckets {
        let bucket_size = bucket.len();
        if bucket_size <= 1 {
            continue;
        }

        let sample_size = 256.min(bucket_size);
        let mut rng = rand::thread_rng();
        let mut sample_indices = bucket.clone();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(sample_size);

        // read each sample vector
        let mut sample_vecs = Vec::with_capacity(sample_size);
        for &sid in &sample_indices {
            let v = read_vector(vectors_file, dim, sid as usize)?;
            sample_vecs.push((sid, v));
        }

        // compute adjacency for each node in bucket
        for &nid in bucket {
            let node_off = offset_start + (nid as usize * max_degree * 4);

            let nv = read_vector(vectors_file, dim, nid as usize)?;
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

            let bytes_slice = bytemuck::cast_slice(&nbrs);
            afile.write_at(bytes_slice, node_off as u64)?; // no seek, just write_at
        }
    }

    Ok(())
}

/// =============================
///       LAYERED SEARCH
/// =============================
impl MultiLayerDiskANN {
    /// 3-layer search: L0 -> L1 -> L2. We only use the final L2 search to get top-k.
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        // we won't use these, so let's prefix them with `_`
        let _entry_l0 =
            self.search_layer(query, &self.layer0_ids, self.offset_layer0, beam_width, 1);
        let _entry_l1 =
            self.search_layer(query, &self.layer1_ids, self.offset_layer1, beam_width, 1);

        // final top-k from base
        self.search_layer(
            query,
            &(0..self.num_vectors as u32).collect::<Vec<u32>>(),
            self.offset_layer2,
            beam_width,
            k,
        )
    }

    /// best-first beam search with up to `k` final neighbors
    fn search_layer(
        &self,
        query: &[f32],
        layer_ids: &[u32],
        layer_offset: usize,
        beam_width: usize,
        k: usize,
    ) -> Vec<u32> {
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
                // max-heap by distance
                other.dist.partial_cmp(&self.dist)
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        if layer_ids.is_empty() {
            return vec![];
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
        if let Some(&idx) = id_to_idx.get(&start_id) {
            visited[idx] = true;
        }

        let mut best_candidates = BinaryHeap::new();
        best_candidates.push(frontier[0].clone());

        while let Some(current) = frontier.pop() {
            let neighbors = self.get_layer_neighbors(current.node_id, layer_offset);
            // expand
            for &nbr in neighbors {
                if nbr == 0 {
                    // ignore dummy
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
                        best_candidates.push(cand);
                        if best_candidates.len() > beam_width {
                            best_candidates.pop();
                        }
                    }
                }
            }
        }

        let mut final_vec = best_candidates.into_vec();
        final_vec.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        final_vec.truncate(k);

        final_vec.into_iter().map(|c| c.node_id).collect()
    }

    /// get adjacency for a node in the given layer
    fn get_layer_neighbors(&self, node_id: u32, layer_offset: usize) -> &[u32] {
        let start = layer_offset + (node_id as usize * self.max_degree * 4);
        let end = start + (self.max_degree * 4);
        let bytes = &self.adjacency_mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// distance to vector #idx
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim * 4;
        let end = start + (self.dim * 4);
        let bytes = &self.vectors_mmap[start..end];
        let vec_f32: &[f32] = bytemuck::cast_slice(bytes);
        euclidean_distance(query, vec_f32)
    }
}

/// =============================
///   HELPER: read a vector
/// =============================

/// We use `read_at` so we don't need a mutable `File`.
fn read_vector(file: &File, dim: usize, idx: usize) -> Result<Vec<f32>, DiskAnnTestError> {
    let offset = (idx * dim * 4) as u64;
    let mut vbytes = vec![0u8; dim * 4];
    file.read_at(&mut vbytes, offset)?;
    let floats: &[f32] = bytemuck::cast_slice(&vbytes);
    Ok(floats.to_vec())
}

/// Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// =============================
///       MAIN DEMO
/// =============================
fn main() -> Result<(), DiskAnnTestError> {
    // Large scale same as BFS version
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 1536;
    const MAX_DEGREE: usize = 32;
    const FRACTION_TOP: f64 = 0.01;
    const FRACTION_MID: f64 = 0.1;

    let vectors_path = "vectors.bin";
    let adjacency_path = "adjacency.bin";
    let metadata_path = "metadata.bin";

    // Build if not exists
    if !Path::new(vectors_path).exists() {
        println!("Building 3-layer index on disk...");
        let start = Instant::now();
        MultiLayerDiskANN::build_index_on_disk(
            NUM_VECTORS,
            DIM,
            MAX_DEGREE,
            FRACTION_TOP,
            FRACTION_MID,
            vectors_path,
            adjacency_path,
            metadata_path,
        )?;
        let elapsed = start.elapsed().as_secs_f32();
        println!("Done building. Elapsed = {:.2} s", elapsed);
    } else {
        println!("Index already exists, skipping build.");
    }

    // Open
    let open_start = Instant::now();
    let index = Arc::new(MultiLayerDiskANN::open_index(
        vectors_path,
        adjacency_path,
        metadata_path,
    )?);
    let open_time = open_start.elapsed().as_secs_f32();
    println!(
        "Opened 3-layer index with {} vectors, dim={}, top fraction={}, mid fraction={} in {:.2} s",
        index.num_vectors, index.dim, index.fraction_top, index.fraction_mid, open_time
    );

    // We'll do concurrency for queries with rayon
    let queries = 5;
    let k = 10;
    let beam_width = 64;

    let search_start = Instant::now();

    // generate queries in a vector
    let mut rng = rand::thread_rng();
    let mut query_batch = Vec::new();
    for _ in 0..queries {
        let q: Vec<f32> = (0..index.dim).map(|_| rng.gen()).collect();
        query_batch.push(q);
    }

    // parallelize the queries with rayon
    let _results: Vec<Vec<u32>> = query_batch
        .par_iter()
        .enumerate()
        .map(|(i, query)| {
            let knn = index.search(query, k, beam_width);
            println!("Query {i} => top-{k} neighbors = {:?}", knn);
            knn
        })
        .collect();

    let search_time = search_start.elapsed().as_secs_f32();
    println!(
        "Performed {queries} queries (in parallel) in {search_time:.2} s (~{:.2} s/query avg)",
        search_time / queries as f32
    );

    Ok(())
}
