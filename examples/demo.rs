// examples/demo.rs
use diskann_rs::{DiskANN, DiskAnnError, DistanceMetric};
use rand::prelude::*;
use std::sync::Arc;

fn main() -> Result<(), DiskAnnError> {
    let singlefile_path = "diskann.db";
    let num_vectors = 100_000;
    let dim = 128;
    let max_degree = 32;
    let build_beam_width = 128;
    let alpha = 1.2;
    let distance_metric = DistanceMetric::Cosine;

    // Build if missing
    if !std::path::Path::new(singlefile_path).exists() {
        println!("Building DiskANN index at {singlefile_path}...");
        
        // Generate sample vectors (in real usage, you'd load your own data)
        println!("Generating {} sample vectors of dimension {}...", num_vectors, dim);
        let mut rng = thread_rng();
        let mut vectors = Vec::new();
        for _ in 0..num_vectors {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            vectors.push(v);
        }
        
        let index = DiskANN::build_index(
            &vectors,
            max_degree,
            build_beam_width,
            alpha,
            distance_metric,
            singlefile_path,
        )?;
        println!("Build done. Index contains {} vectors", index.num_vectors);
    } else {
        println!("Index file {singlefile_path} already exists, skipping build.");
    }

    // Open the index
    let index = Arc::new(DiskANN::open_index(singlefile_path)?);
    println!(
        "Opened index: {} vectors, dimension={}, max_degree={}",
        index.num_vectors, index.dim, index.max_degree
    );

    // Perform a sample query
    let mut rng = thread_rng();
    let query: Vec<f32> = (0..index.dim).map(|_| rng.gen()).collect();
    let k = 10;
    let search_beam_width = 64;
    
    println!("\nSearching for {} nearest neighbors with beam_width={}...", k, search_beam_width);
    let start = std::time::Instant::now();
    let neighbors = index.search(&query, k, search_beam_width);
    let elapsed = start.elapsed();
    
    println!("Search completed in {:?}", elapsed);
    println!("Found {} neighbors:", neighbors.len());
    for (i, &id) in neighbors.iter().enumerate() {
        println!("  {}: node {}", i + 1, id);
    }

    Ok(())
}