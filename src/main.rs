use memmap2::{Mmap, MmapMut};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};
use thiserror::Error;

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

/// A simple adjacency list for demonstration
#[derive(Debug, Serialize, Deserialize)]
struct AdjacencyList {
    neighbors: Vec<u32>,
}

/// A struct that holds big data on disk: vectors + adjacency
pub struct LargeScaleDiskANN {
    dim: usize,
    num_vectors: usize,

    // Memory-mapped vectors file
    file: File,
    mmap: Mmap,

    // Memory-mapped adjacency (one adjacency list per vector)
    adjacency: Vec<AdjacencyList>,
}

/// ----- BUILDING THE DATA AND INDEX -----

impl LargeScaleDiskANN {
    /// Generate random vectors on disk, build random adjacency, then map them.
    ///
    /// # Parameters
    /// - `num_vectors`: how many vectors to generate.
    /// - `dim`: dimension of each vector.
    /// - `max_degree`: how many neighbors each vector will have.
    /// - `vectors_path`: file path to store vectors.
    /// - `adjacency_path`: file path to store adjacency (serialized).
    ///
    /// # Example
    /// ```no_run
    /// LargeScaleDiskANN::build_index_on_disk(
    ///     1_000_000, // 1 million
    ///     128,       // dimension
    ///     64,        // max neighbors
    ///     "vectors.bin",
    ///     "adjacency.bin",
    /// );
    /// ```
    pub fn build_index_on_disk(
        num_vectors: usize,
        dim: usize,
        max_degree: usize,
        vectors_path: &str,
        adjacency_path: &str,
    ) -> Result<Self, DiskAnnTestError> {
        // 1) Generate big vector file on disk
        println!(
            "Generating {} random vectors of dim {} => ~{:.2} GB of floats on disk...",
            num_vectors,
            dim,
            (num_vectors as f64 * dim as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0)
        );

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(vectors_path)?;

        let total_bytes = num_vectors * dim * 4; // f32 => 4 bytes
        file.set_len(total_bytes as u64)?;

        let mut rng = rand::thread_rng();
        // We'll generate in chunks to avoid storing all in memory at once
        let chunk_size = 100_000; // generate 100k vectors at a time
        let mut buffer = Vec::with_capacity(chunk_size * dim);

        let gen_start = Instant::now();
        let mut written = 0usize;
        while written < num_vectors {
            buffer.clear();
            let remaining = num_vectors - written;
            let batch = remaining.min(chunk_size);
            for _ in 0..batch {
                for _ in 0..dim {
                    let val: f32 = rng.gen(); // random in [0,1)
                    buffer.push(val);
                }
            }
            // Now write to disk
            let bytes = bytemuck::cast_slice(&buffer);
            file.write_all(bytes)?;
            written += batch;
        }
        file.sync_all()?;
        let gen_time = gen_start.elapsed().as_secs_f32();
        println!("Vector generation+writing took {:.2} s", gen_time);

        // 2) Create random adjacency
        println!(
            "Generating adjacency: each node has up to {} random neighbors...",
            max_degree
        );
        let adj_start = Instant::now();
        let mut adjacency = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            // pick distinct neighbors
            let mut nbrs = Vec::new();
            while nbrs.len() < max_degree && nbrs.len() < (num_vectors - 1) {
                let n = rng.gen_range(0..num_vectors) as u32;
                if n != i as u32 && !nbrs.contains(&n) {
                    nbrs.push(n);
                }
            }
            adjacency.push(AdjacencyList { neighbors: nbrs });
        }
        let adj_time = adj_start.elapsed().as_secs_f32();
        println!("Adjacency generation took {:.2} s", adj_time);

        println!("Serializing adjacency to {adjacency_path}...");
        let adj_ser_start = Instant::now();
        let adj_data = bincode::serialize(&adjacency)?;
        std::fs::write(adjacency_path, &adj_data)?;
        let adj_ser_time = adj_ser_start.elapsed().as_secs_f32();
        println!("Adjacency serialization took {:.2} s", adj_ser_time);

        // 3) Memory-map the vectors
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(vectors_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // 4) Return the struct
        Ok(Self {
            dim,
            num_vectors,
            file,
            mmap,
            adjacency,
        })
    }

    /// Open an existing index from disk (vectors + adjacency).
    pub fn open_index(
        vectors_path: &str,
        adjacency_path: &str,
        dim: usize,
    ) -> Result<Self, DiskAnnTestError> {
        // Memory-map the vectors
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(vectors_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Determine how many vectors are in the file
        let meta = file.metadata()?;
        let total_bytes = meta.len() as usize;
        if total_bytes % (dim * 4) != 0 {
            return Err(DiskAnnTestError::IndexError(
                "File size not multiple of dim*4".to_string(),
            ));
        }
        let num_vectors = total_bytes / (dim * 4);

        // Load adjacency
        let adj_data = std::fs::read(adjacency_path)?;
        let adjacency: Vec<AdjacencyList> = bincode::deserialize(&adj_data)?;
        if adjacency.len() != num_vectors {
            return Err(DiskAnnTestError::IndexError(format!(
                "Adjacency len {} != number of vectors {}",
                adjacency.len(),
                num_vectors
            )));
        }

        Ok(Self {
            dim,
            num_vectors,
            file,
            mmap,
            adjacency,
        })
    }
}

/// ----- SEARCH & DISTANCE -----

impl LargeScaleDiskANN {
    /// Basic BFS-like search for demonstration:
    /// 1. Start from a random node
    /// 2. Keep track of visited
    /// 3. Expand neighbors
    /// 4. Keep top-k by distance
    pub fn search(&self, query: &[f32], k: usize) -> Vec<u32> {
        if query.len() != self.dim {
            panic!("Query dimension != index dimension");
        }

        let mut rng = rand::thread_rng();
        // pick a random start
        let start = rng.gen_range(0..self.num_vectors) as u32;

        let mut visited = vec![false; self.num_vectors];
        let mut frontier = vec![start];
        visited[start as usize] = true;

        // We'll keep a small list of best candidates
        let mut best: Vec<(u32, f32)> = Vec::new();

        while let Some(current) = frontier.pop() {
            // compute distance
            let dist = self.distance_to(query, current as usize);
            // push to best
            if best.len() < k {
                best.push((current, dist));
            } else {
                // maybe replace the worst
                let (worst_idx, &(_, worst_val)) = best
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();

                if dist < worst_val {
                    best[worst_idx] = (current, dist);
                }
            }

            // expand adjacency
            let neighbors = &self.adjacency[current as usize].neighbors;
            for &nbr in neighbors {
                if !visited[nbr as usize] {
                    visited[nbr as usize] = true;
                    frontier.push(nbr);
                }
            }
        }

        // sort best ascending
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        best.into_iter().map(|(id, _)| id).collect()
    }

    /// Compute distance to the vector at index `idx`
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim * 4;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vec_f32: &[f32] = bytemuck::cast_slice(bytes);
        euclidean_distance(query, vec_f32)
    }
}

/// Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// ----- A DEMO MAIN FOR LARGE-SCALE TESTING -----
///
/// Adjust NUM_VECTORS, DIM, and MAX_DEGREE to taste. For example:
/// - 1_000_000 vectors * 128 dim * 4 bytes = 512 MB
/// - 10_000_000 vectors * 128 dim * 4 bytes = 5 GB
/// (Plus adjacency storage)
///
/// # Usage
/// 1. cargo run --release
/// 2. Watch the output, measure time or memory with external tools.
fn main() -> Result<(), DiskAnnTestError> {
    // Tweak for your environment
    const NUM_VECTORS: usize = 1_000_000; // 1 million
    const DIM: usize = 128;
    const MAX_DEGREE: usize = 32;

    let vectors_path = "vectors.bin";
    let adjacency_path = "adjacency.bin";

    // 1) Build index on disk (comment out if already built)
    if !Path::new(vectors_path).exists() {
        println!("Building index on disk...");
        let build_start = Instant::now();
        LargeScaleDiskANN::build_index_on_disk(
            NUM_VECTORS,
            DIM,
            MAX_DEGREE,
            vectors_path,
            adjacency_path,
        )?;
        let build_time = build_start.elapsed().as_secs_f32();
        println!("Done building. Elapsed = {:.2} s", build_time);
    } else {
        println!("Index files already exist, skipping build.");
    }

    // 2) Open existing index
    println!("Opening index from disk...");
    let index_open_start = Instant::now();
    let index = LargeScaleDiskANN::open_index(vectors_path, adjacency_path, DIM)?;
    let open_time = index_open_start.elapsed().as_secs_f32();
    println!(
        "Opened index with {} vectors, dim={} in {:.2} s",
        index.num_vectors, index.dim, open_time
    );

    // 3) Search a few queries
    let queries = 5;
    let k = 10;
    let mut rng = rand::thread_rng();
    let search_start = Instant::now();
    for i in 0..queries {
        // create a random query
        let query: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
        let neighbors = index.search(&query, k);
        println!("Query {i} => top-{} neighbors: {:?}", k, &neighbors[..]);
    }
    let search_time = search_start.elapsed().as_secs_f32();
    println!(
        "Performed {} queries with BFS-like search in {:.2} s (~{:.2} s/query)",
        queries,
        search_time,
        search_time / queries as f32
    );

    Ok(())
}

