/// Compute total number of chunks for a given shape and chunk dimensions.
pub(crate) fn compute_chunk_count(shape: &[u64], chunk_dims: &[u64]) -> usize {
    let mut total = 1usize;
    for (s, c) in shape.iter().zip(chunk_dims.iter()) {
        total *= (*s).div_ceil(*c) as usize;
    }
    total
}

/// Enumerate all chunk starting coordinates in row-major order.
pub(crate) fn enumerate_chunks(shape: &[u64], chunk_dims: &[u64]) -> Vec<Vec<u64>> {
    let ndims = shape.len();
    let mut chunks_per_dim: Vec<u64> = Vec::with_capacity(ndims);
    for i in 0..ndims {
        chunks_per_dim.push(shape[i].div_ceil(chunk_dims[i]));
    }
    let total: usize = chunks_per_dim.iter().map(|&c| c as usize).product();
    let mut result = Vec::with_capacity(total);
    let mut coord = vec![0u64; ndims];
    for _ in 0..total {
        result.push(coord.to_vec());
        for d in (0..ndims).rev() {
            coord[d] += 1;
            if coord[d] < chunks_per_dim[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    result
}

/// Extract the data for a specific chunk from the full dataset array.
///
/// Returns a full chunk-sized block (chunk_dims product * element_size bytes).
/// Positions beyond the dataset edge are zero-padded. The layout within the
/// block is row-major based on chunk_dims, matching what the HDF5 reader expects.
pub(crate) fn extract_chunk_data(
    data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    chunk_coords: &[u64],
    element_size: usize,
) -> Vec<u8> {
    let ndims = shape.len();
    let chunk_elements: u64 = chunk_dims.iter().product();
    let mut result = vec![0u8; chunk_elements as usize * element_size];

    let mut local_coord = vec![0u64; ndims];
    for flat_idx in 0..chunk_elements as usize {
        let mut in_bounds = true;
        for d in 0..ndims {
            let global_d = chunk_coords[d] * chunk_dims[d] + local_coord[d];
            if global_d >= shape[d] {
                in_bounds = false;
                break;
            }
        }

        if in_bounds {
            let mut global_idx = 0u64;
            let mut stride = 1u64;
            for d in (0..ndims).rev() {
                let global_d = chunk_coords[d] * chunk_dims[d] + local_coord[d];
                global_idx += global_d * stride;
                stride *= shape[d];
            }
            let src_start = global_idx as usize * element_size;
            let dst_start = flat_idx * element_size;
            result[dst_start..dst_start + element_size]
                .copy_from_slice(&data[src_start..src_start + element_size]);
        }

        for d in (0..ndims).rev() {
            local_coord[d] += 1;
            if local_coord[d] < chunk_dims[d] {
                break;
            }
            local_coord[d] = 0;
        }
    }

    result
}
