/// Read a variable-width little-endian offset from a byte slice.
pub(crate) fn read_offset_from_slice(data: &[u8], offset: usize, size: u8) -> u64 {
    match size {
        4 => u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as u64,
        8 => u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]),
        _ => 0,
    }
}
