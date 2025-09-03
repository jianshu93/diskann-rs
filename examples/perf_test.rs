use diskann_rs::{DiskANN, DiskAnnError, DistanceMetric};
use rand::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), DiskAnnError> {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 1536;
    const MAX_DEGREE: usize = 32;
    const BUILD_BEAM_WIDTH: usize = 128;
    const ALPHA: f32 = 1.2;
    let distance_metric = DistanceMetric::Cosine;

    let singlefile_path = "diskann_large.db";

    // Build if missing
    if !std::path::Path::new(singlefile_path).exists() {
        println!(
            "Building DiskANN index with {} vectors, dim={}, distance={:?}",
            NUM_VECTORS, DIM, distance_metric
        );
        
        // Generate vectors
        println!("Generating vectors...");
        let mut rng = thread_rng();
        let mut vectors = Vec::new();
        for i in 0..NUM_VECTORS {
            if i % 100_000 == 0 {
                println!("  Generated {} vectors...", i);
            }
            let v: Vec<f32> = (0..DIM).map(|_| rng.gen()).collect();
            vectors.push(v);
        }
        
        println!("Starting index build...");
        let start = Instant::now();
        let _index = DiskANN::build_index(
            &vectors,
            MAX_DEGREE,
            BUILD_BEAM_WIDTH,
            ALPHA,
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

    // Open index
    let open_start = Instant::now();
    let index = Arc::new(DiskANN::open_index(singlefile_path)?);
    let open_time = open_start.elapsed().as_secs_f32();
    println!(
        "Opened index with {} vectors, dim={}, metric={:?} in {:.2} s",
        index.num_vectors, index.dim, index.distance_metric, open_time
    );

    // Test memory efficiency with queries
    let num_queries = 100;
    let k = 10;
    let beam_width = 64;

    // Generate query batch
    println!("\nGenerating {} query vectors...", num_queries);
    let mut rng = thread_rng();
    let mut query_batch: Vec<Vec<f32>> = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let q: Vec<f32> = (0..index.dim).map(|_| rng.gen()).collect();
        query_batch.push(q);
    }

    // Sequential queries to measure individual performance
    println!("\nRunning sequential queries to measure performance...");
    let mut times = Vec::new();
    for (i, query) in query_batch.iter().take(10).enumerate() {
        let start = Instant::now();
        let neighbors = index.search(query, k, beam_width);
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros());
        println!("Query {}: found {} neighbors in {:?}", i, neighbors.len(), elapsed);
    }
    
    let avg_time = times.iter().sum::<u128>() as f64 / times.len() as f64;
    println!("Average query time: {:.2} µs", avg_time);

    // Parallel queries to test throughput
    println!("\nRunning {} queries in parallel...", num_queries);
    let search_start = Instant::now();
    let results: Vec<Vec<u32>> = query_batch
        .par_iter()
        .map(|query| index.search(query, k, beam_width))
        .collect();
    let search_time = search_start.elapsed().as_secs_f32();
    
    println!("Performed {} queries in {:.2} s", num_queries, search_time);
    println!("Throughput: {:.2} queries/sec", num_queries as f32 / search_time);
    
    // Verify all queries returned results
    let all_valid = results.iter().all(|r| r.len() == k.min(index.num_vectors));
    println!("All queries returned valid results: {}", all_valid);

    // Memory footprint check
    println!("\nMemory-mapped index ready. The process should have minimal memory footprint.");
    println!("You can check memory usage with 'ps aux | grep perf_test' in another terminal.");
    println!("Press Enter to exit...");
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}