use diskann_rs::{DiskAnnError, DistanceMetric, SingleFileDiskANN};
use rand::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), DiskAnnError> {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 1536;
    const MAX_DEGREE: usize = 32;
    const FRACTION_TOP: f64 = 0.01;
    const FRACTION_MID: f64 = 0.1;
    let distance_metric = DistanceMetric::Cosine;

    let singlefile_path = "diskann_parallel.db";

    // Build if missing
    if !std::path::Path::new(singlefile_path).exists() {
        println!(
            "Building single-file index with parallel adjacency + distance={:?}",
            distance_metric
        );
        let start = Instant::now();
        let _index = SingleFileDiskANN::build_index_singlefile(
            NUM_VECTORS,
            DIM,
            MAX_DEGREE,
            FRACTION_TOP,
            FRACTION_MID,
            distance_metric,
            singlefile_path,
        )?;
        let elapsed = start.elapsed().as_secs_f32();
        println!("Done building index in {:.2} s", elapsed);
    } else {
        println!(
            "Index file {} already exists, skipping build.",
            singlefile_path
        );
    }

    // open
    let open_start = Instant::now();
    let index = Arc::new(SingleFileDiskANN::open_index_singlefile(singlefile_path)?);
    let open_time = open_start.elapsed().as_secs_f32();
    println!(
        "Opened index with {} vectors, dim={}, metric={:?} in {:.2} s",
        index.num_vectors, index.dim, index.distance_metric, open_time
    );

    // Create queries
    let queries = 5;
    let k = 10;
    let beam_width = 64;

    // Generate all queries in a batch
    let mut rng = rand::thread_rng();
    let mut query_batch: Vec<Vec<f32>> = Vec::with_capacity(queries);
    for _ in 0..queries {
        let q: Vec<f32> = (0..index.dim).map(|_| rng.gen()).collect();
        query_batch.push(q);
    }

    // Now run queries in parallel
    let search_start = Instant::now();
    query_batch.par_iter().enumerate().for_each(|(i, query)| {
        let neighbors = index.search(query, k, beam_width);
        println!("Query {i} => top-{k} neighbors = {:?}", neighbors);
    });
    let search_time = search_start.elapsed().as_secs_f32();
    println!("Performed {queries} queries in {:.2} s", search_time);

    Ok(())
}
