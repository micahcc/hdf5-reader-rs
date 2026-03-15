use crate::btree2::btree2_type::BTree2Type;
use crate::checksum;
use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

/// B-tree v2 header magic: `BTHD`
pub const BTHD_MAGIC: [u8; 4] = *b"BTHD";
/// B-tree v2 internal node magic: `BTIN`
pub const BTIN_MAGIC: [u8; 4] = *b"BTIN";
/// B-tree v2 leaf node magic: `BTLF`
pub const BTLF_MAGIC: [u8; 4] = *b"BTLF";

/// Parsed B-tree v2 header.
///
/// ## On-disk layout
///
/// ```text
/// Byte 0-3:  Signature ("BTHD")
/// Byte 4:    Version (0)
/// Byte 5:    Type (BTree2Type)
/// Byte 6-9:  Node size (u32)
/// Byte 10-11: Record size (u16)
/// Byte 12-13: Depth (u16)
/// Byte 14:   Split percent
/// Byte 15:   Merge percent
/// Byte 16+O: Root node address (O = size_of_offsets)
/// +2:        Number of records in root node (u16)
/// +O:        Total number of records in tree (size_of_lengths bytes)
/// +4:        Checksum
/// ```
#[derive(Debug, Clone)]
pub struct BTree2Header {
    pub tree_type: BTree2Type,
    pub node_size: u32,
    pub record_size: u16,
    pub depth: u16,
    pub split_percent: u8,
    pub merge_percent: u8,
    pub root_node_address: u64,
    pub root_num_records: u16,
    pub total_records: u64,
}

impl BTree2Header {
    /// Parse a B-tree v2 header from `addr`.
    pub fn parse<R: ReadAt + ?Sized>(
        reader: &R,
        addr: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
        if magic != BTHD_MAGIC {
            return Err(Error::InvalidBTreeV2 {
                msg: format!("expected BTHD magic at {:#x}, got {:?}", addr, magic),
            });
        }

        let version = Le::read_u8(reader, addr + 4).map_err(Error::Io)?;
        if version != 0 {
            return Err(Error::InvalidBTreeV2 {
                msg: format!("expected B-tree v2 header version 0, got {}", version),
            });
        }

        let type_id = Le::read_u8(reader, addr + 5).map_err(Error::Io)?;
        let tree_type = BTree2Type::from_u8(type_id)?;
        let node_size = Le::read_u32(reader, addr + 6).map_err(Error::Io)?;
        let record_size = Le::read_u16(reader, addr + 10).map_err(Error::Io)?;
        let depth = Le::read_u16(reader, addr + 12).map_err(Error::Io)?;
        let split_percent = Le::read_u8(reader, addr + 14).map_err(Error::Io)?;
        let merge_percent = Le::read_u8(reader, addr + 15).map_err(Error::Io)?;

        let o = size_of_offsets as u64;
        let l = size_of_lengths as u64;

        let root_addr_off = addr + 16;
        let root_node_address =
            Le::read_offset(reader, root_addr_off, size_of_offsets).map_err(Error::Io)?;
        let root_num_records = Le::read_u16(reader, root_addr_off + o).map_err(Error::Io)?;
        let total_records =
            Le::read_length(reader, root_addr_off + o + 2, size_of_lengths).map_err(Error::Io)?;

        let checksum_off = root_addr_off + o + 2 + l;
        let stored_checksum = Le::read_u32(reader, checksum_off).map_err(Error::Io)?;

        // Verify checksum
        let header_len = (checksum_off - addr) as usize;
        let mut check_data = vec![0u8; header_len];
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

        Ok(BTree2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_node_address,
            root_num_records,
            total_records,
        })
    }
}
