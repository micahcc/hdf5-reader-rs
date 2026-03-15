/// A single chunk's location and metadata.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// File address of the (possibly filtered) chunk data.
    pub address: u64,
    /// On-disk size in bytes (after filtering/compression).
    pub filtered_size: u64,
    /// Filter mask — bit N set means filter N was NOT applied.
    pub filter_mask: u32,
    /// Scaled chunk coordinates (chunk_offset / chunk_dim for each axis).
    pub scaled: Vec<u64>,
}
