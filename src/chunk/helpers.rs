use crate::error::Error;
use crate::io::ReadAt;

/// Convert a linear chunk index to scaled (per-dimension) chunk coordinates.
pub(crate) fn linear_to_scaled(mut linear: u64, chunks_per_dim: &[u64]) -> Vec<u64> {
    let ndims = chunks_per_dim.len();
    let mut scaled = vec![0u64; ndims];
    // Row-major order: last dimension varies fastest
    for i in (0..ndims).rev() {
        scaled[i] = linear % chunks_per_dim[i];
        linear /= chunks_per_dim[i];
    }
    scaled
}

/// Copy a chunk's decompressed data into the correct position in the output buffer.
///
/// Handles edge chunks that may be smaller than a full chunk
/// (when dataset dims aren't evenly divisible by chunk dims).
pub(crate) fn copy_chunk_to_output(
    chunk_data: &[u8],
    scaled: &[u64],
    chunk_dims: &[u32],
    dataset_dims: &[u64],
    element_size: usize,
    output: &mut [u8],
) {
    let ndims = dataset_dims.len();

    if ndims == 0 {
        // Scalar — shouldn't happen for chunked, but handle gracefully
        let len = chunk_data.len().min(output.len());
        output[..len].copy_from_slice(&chunk_data[..len]);
        return;
    }

    // Compute the actual element counts for this chunk (handle edge chunks)
    let mut actual_dims = vec![0u64; ndims];
    for i in 0..ndims {
        let start = scaled[i] * chunk_dims[i] as u64;
        let end = (start + chunk_dims[i] as u64).min(dataset_dims[i]);
        actual_dims[i] = end - start;
    }

    // For 1D: simple memcpy
    if ndims == 1 {
        let dst_start = scaled[0] as usize * chunk_dims[0] as usize * element_size;
        let copy_len = actual_dims[0] as usize * element_size;
        let src_len = copy_len.min(chunk_data.len());
        if dst_start + src_len <= output.len() {
            output[dst_start..dst_start + src_len].copy_from_slice(&chunk_data[..src_len]);
        }
        return;
    }

    // For N-D: copy row by row (innermost dimension is contiguous)
    // We iterate over all rows in the chunk and copy each one.
    let inner_len = actual_dims[ndims - 1] as usize * element_size;
    let _chunk_inner_stride = chunk_dims[ndims - 1] as usize * element_size;

    // Compute strides for the dataset (row-major)
    let mut ds_strides = vec![element_size; ndims];
    for i in (0..ndims - 1).rev() {
        ds_strides[i] = ds_strides[i + 1] * dataset_dims[i + 1] as usize;
    }
    // Compute strides for the chunk
    let mut ch_strides = vec![element_size; ndims];
    for i in (0..ndims - 1).rev() {
        ch_strides[i] = ch_strides[i + 1] * chunk_dims[i + 1] as usize;
    }

    // Number of "rows" to copy (product of all dims except the innermost)
    let nrows: usize = actual_dims[..ndims - 1]
        .iter()
        .map(|&d| d as usize)
        .product();

    for row in 0..nrows {
        // Convert row index to per-dimension indices (excluding innermost)
        let mut remaining = row;
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;

        for i in 0..ndims - 1 {
            let _dim_count = actual_dims[i] as usize;
            let rows_below: usize = actual_dims[i + 1..ndims - 1]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let idx = remaining / rows_below;
            remaining %= rows_below;

            src_offset += idx * ch_strides[i];
            dst_offset += (scaled[i] as usize * chunk_dims[i] as usize + idx) * ds_strides[i];
        }
        // Add the innermost dimension's base offset
        dst_offset += scaled[ndims - 1] as usize * chunk_dims[ndims - 1] as usize * element_size;

        if src_offset + inner_len <= chunk_data.len() && dst_offset + inner_len <= output.len() {
            output[dst_offset..dst_offset + inner_len]
                .copy_from_slice(&chunk_data[src_offset..src_offset + inner_len]);
        }
    }
}

/// Read a variable-width little-endian integer from a reader.
pub(crate) fn read_var_le<R: ReadAt + ?Sized>(
    reader: &R,
    offset: u64,
    size: usize,
) -> crate::error::Result<u64> {
    let mut buf = [0u8; 8];
    let n = size.min(8);
    reader
        .read_exact_at(offset, &mut buf[..n])
        .map_err(Error::Io)?;
    let mut result = 0u64;
    for (i, &byte) in buf.iter().enumerate().take(n) {
        result |= (byte as u64) << (i * 8);
    }
    Ok(result)
}
