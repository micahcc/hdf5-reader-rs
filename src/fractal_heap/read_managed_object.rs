use crate::error::Error;
use crate::error::Result;
use crate::fractal_heap::header::FHDB_MAGIC;
use crate::fractal_heap::header::FHIB_MAGIC;
use crate::fractal_heap::header::FractalHeapHeader;
use crate::io::Le;
use crate::io::ReadAt;

/// Read a managed object from the fractal heap given a heap ID.
///
/// For "managed" objects (the common case), the heap ID encodes:
/// - Version/type (bits 6-7 of byte 0): 0 = managed, 1 = tiny, 2 = huge
/// - Offset into the heap (variable bits)
/// - Length of the object (variable bits)
///
/// This function handles the "managed" type by locating the direct block
/// and reading the object bytes.
pub fn read_managed_object<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    if heap_id.is_empty() {
        return Err(Error::InvalidFractalHeap {
            msg: "empty heap ID".into(),
        });
    }

    let id_type = (heap_id[0] >> 4) & 0x03;
    match id_type {
        0 => read_managed_object_inner(reader, header, heap_id, size_of_offsets, size_of_lengths),
        1 => read_tiny_object(header, heap_id),
        2 => read_huge_object(reader, header, heap_id, size_of_offsets, size_of_lengths),
        _ => Err(Error::InvalidFractalHeap {
            msg: format!("unknown heap ID type {}", id_type),
        }),
    }
}

fn read_huge_object<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    let o = size_of_offsets as usize;
    let l = size_of_lengths as usize;

    // `huge_ids_direct` is NOT a flag bit — it's derived from whether the
    // heap ID is large enough to hold address + length inline.
    // H5HFhuge.c:166-219: sizeof_addr + sizeof_size [+ 4 + sizeof_size if filtered]
    let has_filters = header.io_filter_encoded_length > 0;
    let needed = if has_filters { o + l + 4 + l } else { o + l };
    let directly_accessed = needed <= (header.heap_id_length as usize).saturating_sub(1);

    if directly_accessed {
        if has_filters {
            // Direct access with filters: address + filtered_length + filter_mask + mem_length
            if heap_id.len() < 1 + o + l + 4 + l {
                return Err(Error::InvalidFractalHeap {
                    msg: "huge object heap ID too short for filtered direct access".into(),
                });
            }
            let address = read_var_le(&heap_id[1..], o);
            let filtered_length = read_var_le(&heap_id[1 + o..], l);
            let filter_mask = u32::from_le_bytes([
                heap_id[1 + o + l],
                heap_id[1 + o + l + 1],
                heap_id[1 + o + l + 2],
                heap_id[1 + o + l + 3],
            ]);
            let _mem_length = read_var_le(&heap_id[1 + o + l + 4..], l);

            let mut compressed = vec![0u8; filtered_length as usize];
            reader
                .read_exact_at(address, &mut compressed)
                .map_err(Error::Io)?;

            if let Some(pipeline) = &header.filter_pipeline {
                if filter_mask == 0 {
                    compressed = pipeline.decompress(compressed)?;
                }
            }
            Ok(compressed)
        } else {
            // Direct access without filters: address + length
            if heap_id.len() < 1 + o + l {
                return Err(Error::InvalidFractalHeap {
                    msg: "huge object heap ID too short for direct access".into(),
                });
            }
            let address = read_var_le(&heap_id[1..], o);
            let length = read_var_le(&heap_id[1 + o..], l);

            let mut data = vec![0u8; length as usize];
            reader
                .read_exact_at(address, &mut data)
                .map_err(Error::Io)?;
            Ok(data)
        }
    } else {
        // Indirectly accessed: heap ID contains a unique ID, look up in B-tree v2
        let id_bytes = (header.heap_id_length as usize).saturating_sub(1);
        if heap_id.len() < 1 + id_bytes {
            return Err(Error::InvalidFractalHeap {
                msg: "huge object heap ID too short for indirect access".into(),
            });
        }
        let obj_id = read_var_le(&heap_id[1..], id_bytes);

        if header.huge_bt2_address == u64::MAX {
            return Err(Error::InvalidFractalHeap {
                msg: "huge B-tree v2 address is undefined".into(),
            });
        }

        let bt2 = crate::btree2::BTree2Header::parse(
            reader,
            header.huge_bt2_address,
            size_of_offsets,
            size_of_lengths,
        )?;

        // Search B-tree for matching ID
        // Type 2 record: address(O) + length(L) + id(L)
        let mut found: Option<(u64, u64)> = None;
        crate::btree2::iterate_records(reader, &bt2, size_of_offsets, |record| {
            if record.data.len() >= o + l + l {
                let addr = read_var_le(&record.data, o);
                let len = read_var_le(&record.data[o..], l);
                let id = read_var_le(&record.data[o + l..], l);
                if id == obj_id {
                    found = Some((addr, len));
                }
            }
            Ok(())
        })?;

        match found {
            Some((address, length)) => {
                let mut data = vec![0u8; length as usize];
                reader
                    .read_exact_at(address, &mut data)
                    .map_err(Error::Io)?;
                Ok(data)
            }
            None => Err(Error::InvalidFractalHeap {
                msg: format!("huge object ID {} not found in B-tree", obj_id),
            }),
        }
    }
}

fn read_managed_object_inner<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    // Managed heap ID layout (after the type/version nibble):
    //   offset: variable bits (max_heap_size_bits wide)
    //   length: remaining bits
    // The offset is the byte position within the managed heap space.
    // We need to decode it from the heap ID bytes.

    let offset_bits = header.max_heap_size_bits as usize;
    // The length field uses the remaining bytes of the heap ID after the offset
    let id_len = header.heap_id_length as usize;

    // Read the offset and length from the heap ID (skip first nibble of byte 0)
    // The encoding packs offset and length into the heap_id bytes starting at bit 4 of byte 0.
    let (heap_offset, obj_length) = decode_managed_heap_id(heap_id, offset_bits, id_len)?;

    // Now we need to find which direct block contains this offset.
    // Walk the fractal heap structure to find the right direct block.
    let root_addr = header.root_block_address;
    if root_addr == u64::MAX {
        return Err(Error::InvalidFractalHeap {
            msg: "root block address is undefined".into(),
        });
    }

    if header.root_is_direct() {
        // Root is a direct block — may be filtered
        let filtered_size = header.filtered_root_direct_block_size;
        let filter_mask = header.io_filter_mask.unwrap_or(0);
        let block_size = header.starting_block_size;
        read_from_direct_block(
            reader,
            header,
            root_addr,
            heap_offset,
            obj_length,
            size_of_offsets,
            filtered_size,
            filter_mask,
            block_size,
        )
    } else {
        // Root is an indirect block — need to traverse
        read_from_indirect_block(
            reader,
            header,
            root_addr,
            header.current_root_rows as u32,
            heap_offset,
            obj_length,
            size_of_offsets,
            size_of_lengths,
        )
    }
}

fn read_tiny_object(_header: &FractalHeapHeader, heap_id: &[u8]) -> Result<Vec<u8>> {
    // Tiny objects are stored directly in the heap ID.
    // The lower 4 bits of byte 0 encode the length (minus 1).
    let len = ((heap_id[0] & 0x0F) + 1) as usize;
    if heap_id.len() < 1 + len {
        return Err(Error::InvalidFractalHeap {
            msg: "tiny object extends past heap ID".into(),
        });
    }
    Ok(heap_id[1..1 + len].to_vec())
}

/// Decode a managed heap ID into (offset, length).
fn decode_managed_heap_id(heap_id: &[u8], offset_bits: usize, id_len: usize) -> Result<(u64, u64)> {
    // The heap ID starts with a flags byte. Bits 4-5 encode the type (0 = managed).
    // Bits 0-3 are reserved. The actual offset and length are packed into
    // bytes 1..id_len.
    //
    // For managed objects:
    //   Bytes 1..: offset (ceil(offset_bits/8) bytes, LE) followed by length (remaining bytes, LE)

    let offset_bytes = offset_bits.div_ceil(8);
    let length_bytes = id_len - 1 - offset_bytes;

    if heap_id.len() < 1 + offset_bytes + length_bytes {
        return Err(Error::InvalidFractalHeap {
            msg: "heap ID too short for managed object".into(),
        });
    }

    let offset = read_var_le(&heap_id[1..], offset_bytes);
    let length = read_var_le(&heap_id[1 + offset_bytes..], length_bytes);

    Ok((offset, length))
}

/// Read variable-length little-endian unsigned integer.
fn read_var_le(data: &[u8], len: usize) -> u64 {
    let mut result = 0u64;
    for (i, &byte) in data.iter().enumerate().take(len.min(8)) {
        result |= (byte as u64) << (i * 8);
    }
    result
}

/// Read an object from a direct block at the given heap offset.
///
/// If `filtered_size` is `Some`, the block is compressed: read `filtered_size` bytes,
/// decompress through the heap's filter pipeline, then extract the object.
fn read_from_direct_block<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    block_addr: u64,
    heap_offset: u64,
    obj_length: u64,
    _size_of_offsets: u8,
    filtered_size: Option<u64>,
    _filter_mask: u32,
    block_size: u64,
) -> Result<Vec<u8>> {
    // Direct block layout:
    //   Signature "FHDB" (4)
    //   Version (1)
    //   Heap header address (size_of_offsets)
    //   Block offset within heap (block_offset_byte_size)
    //   [optional checksum if heap flags indicate it]
    //   Data...
    //
    // The object is at (block_data_start + heap_offset_within_block).

    if let (Some(fsize), Some(pipeline)) = (filtered_size, &header.filter_pipeline) {
        // Filtered direct block: read the on-disk (compressed) data and decompress
        let read_size = fsize as usize;
        let mut compressed = vec![0u8; read_size];
        reader
            .read_exact_at(block_addr, &mut compressed)
            .map_err(Error::Io)?;

        // Decompress through the filter pipeline
        let decompressed = pipeline.decompress(compressed)?;

        // Verify we got the expected block size
        if decompressed.len() != block_size as usize {
            return Err(Error::InvalidFractalHeap {
                msg: format!(
                    "filtered direct block decompressed to {} bytes, expected {}",
                    decompressed.len(),
                    block_size
                ),
            });
        }

        // Verify FHDB magic in decompressed data
        if decompressed.len() < 4 || decompressed[..4] != FHDB_MAGIC {
            return Err(Error::InvalidFractalHeap {
                msg: format!(
                    "expected FHDB magic in decompressed block at {:#x}",
                    block_addr
                ),
            });
        }

        // Extract the object from the decompressed block
        let off = heap_offset as usize;
        let end = off + obj_length as usize;
        if end > decompressed.len() {
            return Err(Error::InvalidFractalHeap {
                msg: format!(
                    "object at offset {} length {} exceeds decompressed block size {}",
                    off,
                    obj_length,
                    decompressed.len()
                ),
            });
        }
        Ok(decompressed[off..end].to_vec())
    } else {
        // Unfiltered direct block: read directly from file
        let mut magic = [0u8; 4];
        reader
            .read_exact_at(block_addr, &mut magic)
            .map_err(Error::Io)?;
        if magic != FHDB_MAGIC {
            return Err(Error::InvalidFractalHeap {
                msg: format!("expected FHDB magic at {:#x}", block_addr),
            });
        }

        // The heap offset in the managed object ID includes the direct block overhead.
        // Objects are addressed relative to the start of the direct block.
        let mut obj_data = vec![0u8; obj_length as usize];
        reader
            .read_exact_at(block_addr + heap_offset, &mut obj_data)
            .map_err(Error::Io)?;
        Ok(obj_data)
    }
}

/// Walk an indirect block to find the direct block containing `heap_offset`.
fn read_from_indirect_block<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    iblock_addr: u64,
    nrows: u32,
    heap_offset: u64,
    obj_length: u64,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    // Indirect block layout:
    //   Signature "FHIB" (4)
    //   Version (1)
    //   Heap header address (size_of_offsets)
    //   Block offset within heap (block_offset_byte_size bytes)
    //   Child block entries:
    //     For direct block rows: address (O) [+ filtered_size (L) + filter_mask (4) if filtered]
    //     For indirect block rows: address (O)
    //   Checksum (4)

    let mut magic = [0u8; 4];
    reader
        .read_exact_at(iblock_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != FHIB_MAGIC {
        return Err(Error::InvalidFractalHeap {
            msg: format!("expected FHIB magic at {:#x}", iblock_addr),
        });
    }

    let block_offset_size = header.block_offset_byte_size();
    let overhead = 5 + size_of_offsets as u64 + block_offset_size as u64;
    let mut pos = iblock_addr + overhead;

    let width = header.table_width as u64;
    let has_filters = header.io_filter_encoded_length > 0;

    // Determine which rows are direct vs indirect.
    let max_direct_rows = header.max_direct_rows();

    // Walk through entries to find the one containing our heap_offset
    let mut current_heap_offset = 0u64;

    for row in 0..nrows {
        let block_size = header.block_size_for_row(row);
        let is_direct = row < max_direct_rows;

        for _col in 0..width {
            let child_addr = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
            pos += size_of_offsets as u64;

            // For filtered direct blocks, read filtered_size and filter_mask
            let (entry_filtered_size, entry_filter_mask) = if is_direct && has_filters {
                let fsize = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
                pos += size_of_lengths as u64;
                let fmask = Le::read_u32(reader, pos).map_err(Error::Io)?;
                pos += 4;
                (Some(fsize), fmask)
            } else {
                (None, 0)
            };

            let next_offset = current_heap_offset + block_size;

            if heap_offset >= current_heap_offset && heap_offset < next_offset {
                // Found the block
                if child_addr == u64::MAX {
                    return Err(Error::InvalidFractalHeap {
                        msg: "child block address is undefined".into(),
                    });
                }
                let offset_in_block = heap_offset - current_heap_offset;
                if is_direct {
                    return read_from_direct_block(
                        reader,
                        header,
                        child_addr,
                        offset_in_block,
                        obj_length,
                        size_of_offsets,
                        entry_filtered_size,
                        entry_filter_mask,
                        block_size,
                    );
                } else {
                    // Recurse into child indirect block
                    let child_nrows = header.child_indirect_nrows(row);
                    return read_from_indirect_block(
                        reader,
                        header,
                        child_addr,
                        child_nrows,
                        offset_in_block,
                        obj_length,
                        size_of_offsets,
                        size_of_lengths,
                    );
                }
            }

            current_heap_offset = next_offset;
        }
    }

    Err(Error::InvalidFractalHeap {
        msg: format!(
            "heap offset {} not found in indirect block at {:#x}",
            heap_offset, iblock_addr
        ),
    })
}
