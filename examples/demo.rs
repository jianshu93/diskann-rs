// examples/demo.rs
use diskannrs::{DiskAnnError, SingleFileDiskANN};
use std::sync::Arc;

fn main() -> Result<(), DiskAnnError> {
    let singlefile_path = "diskann.db";
    let num_vectors = 100_000;
    let dim = 128;
    let max_degree = 32;
    let fraction_top = 0.01;
    let fraction_mid = 0.1;
    let distance_metric = diskannrs::DistanceMetric::Cosine;

    // Build if missing
    if !std::path::Path::new(singlefile_path).exists() {
        println!("Building single-file diskann at {singlefile_path}...");
        let index = SingleFileDiskANN::build_index_singlefile(
            num_vectors,
            dim,
            max_degree,
            fraction_top,
            fraction_mid,
            distance_metric,
            singlefile_path,
        )?;
        println!("Build done. Index dimension = {}", index.dim);
    } else {
        println!("Index file {singlefile_path} already exists, skipping build.");
    }

    // Open
    let index = Arc::new(SingleFileDiskANN::open_index_singlefile(singlefile_path)?);

    // Query
    let query = vec![0.1, 0.2, 0.3 /* ... up to dim */];
    let k = 10;
    let beam_width = 64;
    let neighbors = index.search(&query, k, beam_width);
    println!("Neighbors for the sample query = {:?}", neighbors);

    Ok(())
}
