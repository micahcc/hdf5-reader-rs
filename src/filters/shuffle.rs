/// Reverse the HDF5 shuffle filter.
///
/// Shuffle interleaves bytes by element position:
/// Input:  [A0 A1 A2 A3 B0 B1 B2 B3] (two 4-byte elements)
/// Shuffled: [A0 B0 A1 B1 A2 B2 A3 B3] (all byte-0s, then byte-1s, etc.)
///
/// We need to un-shuffle: given the shuffled form, reconstruct the original.
pub(crate) fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let num_elements = data.len() / element_size;
    let mut output = vec![0u8; data.len()];

    for byte_idx in 0..element_size {
        let src_start = byte_idx * num_elements;
        for elem in 0..num_elements {
            output[elem * element_size + byte_idx] = data[src_start + elem];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_identity_for_size_1() {
        let data = vec![1, 2, 3, 4];
        assert_eq!(unshuffle(&data, 1), data);
    }

    #[test]
    fn unshuffle_reverses_shuffle() {
        // Two 4-byte elements: [0x01020304, 0x05060708]
        // Shuffled (byte-0s first, etc.): [01, 05, 02, 06, 03, 07, 04, 08]
        let shuffled = vec![0x01, 0x05, 0x02, 0x06, 0x03, 0x07, 0x04, 0x08];
        let expected = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(unshuffle(&shuffled, 4), expected);
    }
}
