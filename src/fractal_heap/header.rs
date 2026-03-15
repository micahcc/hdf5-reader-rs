use crate::checksum;
use crate::error::Error;
use crate::error::Result;
use crate::filters::FilterPipeline;
use crate::io::Le;
use crate::io::ReadAt;

/// Fractal heap header magic: `FRHP`
pub const FRHP_MAGIC: [u8; 4] = *b"FRHP";
/// Fractal heap direct block magic: `FHDB`
pub const FHDB_MAGIC: [u8; 4] = *b"FHDB";
/// Fractal heap indirect block magic: `FHIB`
pub const FHIB_MAGIC: [u8; 4] = *b"FHIB";

/// Parsed fractal heap header.
///
/// ## On-disk layout
///
/// ```text
/// Byte 0-3:   Signature ("FRHP")
/// Byte 4:     Version (0)
/// Byte 5-6:   Heap ID length (u16)
/// Byte 7-8:   I/O filter pipeline encoded length (u16)
/// Byte 9:     Flags
/// Byte 10-13: Max managed object size (u32)
/// Byte 14+L:  Next huge object ID (size_of_lengths)
/// +O:         B-tree v2 address for huge objects (size_of_offsets)
/// +L:         Free space in managed objects (size_of_lengths)
/// +O:         Free space manager address (size_of_offsets)
/// +L:         Managed space total (size_of_lengths)
/// +L:         Managed space allocated (size_of_lengths)
/// +L:         Managed alloc iterator offset (size_of_lengths)
/// +L:         Managed objects count (size_of_lengths)
/// +L:         Huge objects total size (size_of_lengths)
/// +L:         Huge objects count (size_of_lengths)
/// +L:         Tiny objects total size (size_of_lengths)
/// +L:         Tiny objects count (size_of_lengths)
/// +2:         Table width (u16)
/// +L:         Starting block size (size_of_lengths)
/// +L:         Max direct block size (size_of_lengths)
/// +2:         Max heap size (u16, in bits — e.g. 32 means max offset = 2^32)
/// +2:         Starting # of rows in root indirect block (u16)
/// +O:         Root block address (size_of_offsets)
/// +2:         Current # of rows in root indirect block (u16)
/// [optional I/O filter info if encoded length > 0]
/// +4:         Checksum
/// ```
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    pub heap_id_length: u16,
    pub io_filter_encoded_length: u16,
    pub flags: u8,
    pub max_managed_object_size: u32,
    pub next_huge_object_id: u64,
    pub huge_bt2_address: u64,
    pub free_space_in_managed: u64,
    pub free_space_manager_address: u64,
    pub managed_space_total: u64,
    pub managed_space_allocated: u64,
    pub managed_alloc_iterator_offset: u64,
    pub managed_objects_count: u64,
    pub huge_objects_total_size: u64,
    pub huge_objects_count: u64,
    pub tiny_objects_total_size: u64,
    pub tiny_objects_count: u64,
    pub table_width: u16,
    pub starting_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size_bits: u16,
    pub starting_root_rows: u16,
    pub root_block_address: u64,
    pub current_root_rows: u16,
    /// If I/O filters are present, the filtered root direct block size.
    pub filtered_root_direct_block_size: Option<u64>,
    /// If I/O filters are present, the I/O filter mask.
    pub io_filter_mask: Option<u32>,
    /// If I/O filters are present, the parsed filter pipeline.
    pub filter_pipeline: Option<FilterPipeline>,
}

impl FractalHeapHeader {
    pub fn parse<R: ReadAt + ?Sized>(
        reader: &R,
        addr: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
        if magic != FRHP_MAGIC {
            return Err(Error::InvalidFractalHeap {
                msg: format!("expected FRHP magic at {:#x}, got {:?}", addr, magic),
            });
        }

        let version = Le::read_u8(reader, addr + 4).map_err(Error::Io)?;
        if version != 0 {
            return Err(Error::InvalidFractalHeap {
                msg: format!("expected fractal heap version 0, got {}", version),
            });
        }

        let heap_id_length = Le::read_u16(reader, addr + 5).map_err(Error::Io)?;
        let io_filter_encoded_length = Le::read_u16(reader, addr + 7).map_err(Error::Io)?;
        let flags = Le::read_u8(reader, addr + 9).map_err(Error::Io)?;
        let max_managed_object_size = Le::read_u32(reader, addr + 10).map_err(Error::Io)?;

        let o = size_of_offsets as u64;
        let l = size_of_lengths as u64;
        let mut pos = addr + 14;

        let next_huge_object_id =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_bt2_address = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let free_space_in_managed =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let free_space_manager_address =
            Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let managed_space_total =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_space_allocated =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_alloc_iterator_offset =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_objects_count =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_objects_total_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_objects_count =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let tiny_objects_total_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let tiny_objects_count =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;

        let table_width = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let starting_block_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let max_direct_block_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let max_heap_size_bits = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let starting_root_rows = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let root_block_address =
            Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let current_root_rows = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;

        // Optional I/O filter info
        let (filtered_root_direct_block_size, io_filter_mask, filter_pipeline) =
            if io_filter_encoded_length > 0 {
                let fsize = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
                pos += l;
                let mask = Le::read_u32(reader, pos).map_err(Error::Io)?;
                pos += 4;
                // Read and parse the encoded filter pipeline message
                let fpl = io_filter_encoded_length as usize;
                let mut filter_data = vec![0u8; fpl];
                reader
                    .read_exact_at(pos, &mut filter_data)
                    .map_err(Error::Io)?;
                let pipeline = FilterPipeline::parse(&filter_data)?;
                pos += fpl as u64;
                (Some(fsize), Some(mask), Some(pipeline))
            } else {
                (None, None, None)
            };

        // Checksum
        let stored_checksum = Le::read_u32(reader, pos).map_err(Error::Io)?;
        let check_len = (pos - addr) as usize;
        let mut check_data = vec![0u8; check_len];
        reader
            .read_exact_at(addr, &mut check_data)
            .map_err(Error::Io)?;
        let computed = checksum::lookup3(&check_data);
        if computed != stored_checksum {
            return Err(Error::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed,
            });
        }

        Ok(FractalHeapHeader {
            heap_id_length,
            io_filter_encoded_length,
            flags,
            max_managed_object_size,
            next_huge_object_id,
            huge_bt2_address,
            free_space_in_managed,
            free_space_manager_address,
            managed_space_total,
            managed_space_allocated,
            managed_alloc_iterator_offset,
            managed_objects_count,
            huge_objects_total_size,
            huge_objects_count,
            tiny_objects_total_size,
            tiny_objects_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size_bits,
            starting_root_rows,
            root_block_address,
            current_root_rows,
            filtered_root_direct_block_size,
            io_filter_mask,
            filter_pipeline,
        })
    }

    /// Compute the block size for a given row number.
    ///
    /// Row 0 and row 1 both have `starting_block_size`.
    /// Row 2 = 2 * starting_block_size, row 3 = 4 * starting_block_size, etc.
    ///
    /// Formula: row_block_size[0] = starting_block_size
    ///          row_block_size[u] = starting_block_size * 2^(u-1) for u >= 1
    /// Compute the block size for a given row number.
    ///
    /// Row 0 and row 1 both have `starting_block_size`.
    /// Row 2 = 2 * starting_block_size, row 3 = 4 * starting_block_size, etc.
    /// Does NOT cap at max_direct_block_size — indirect rows continue doubling
    /// to represent the total managed space of their sub-trees (H5HFdtable.c:110-119).
    pub fn block_size_for_row(&self, row: u32) -> u64 {
        if row <= 1 {
            self.starting_block_size
        } else {
            self.starting_block_size * (1u64 << (row - 1))
        }
    }

    /// Number of rows that contain direct blocks.
    ///
    /// C: `max_dblock_rows = (max_direct_bits - start_bits) + 2` (H5HFdtable.c:94)
    pub fn max_direct_rows(&self) -> u32 {
        if self.max_direct_block_size == 0 || self.starting_block_size == 0 {
            return 0;
        }
        (self.max_direct_block_size / self.starting_block_size).trailing_zeros() + 2
    }

    /// Number of rows in a child indirect block at a given row.
    ///
    /// C: `nrows = log2(row_block_size) - first_row_bits + 1` (H5HFdtable.c:237-251)
    pub fn child_indirect_nrows(&self, row: u32) -> u32 {
        let first_row_bits = (self.starting_block_size * self.table_width as u64).trailing_zeros();
        let log2_bs = self.block_size_for_row(row).trailing_zeros();
        log2_bs - first_row_bits + 1
    }

    /// Number of bytes needed to encode a block offset within the heap.
    /// This is ceil(max_heap_size_bits / 8).
    pub fn block_offset_byte_size(&self) -> usize {
        (self.max_heap_size_bits as usize).div_ceil(8)
    }

    /// True if the root block is a direct block (current_root_rows == 0).
    pub fn root_is_direct(&self) -> bool {
        self.current_root_rows == 0
    }
}
