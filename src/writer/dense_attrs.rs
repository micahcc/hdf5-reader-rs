//! Encoder for dense attribute storage (fractal heap + B-tree v2).
//!
//! When a group's attribute count exceeds the compact threshold, attributes
//! are stored in a fractal heap (FRHP + FHDB direct block) with a B-tree v2
//! type-8 index for lookup by name hash. A free-space manager (FSHD + FSSE)
//! tracks unused space in the heap.

use crate::checksum;
use crate::error::Result;
use crate::superblock::UNDEF_ADDR;
use crate::writer::encode::encode_attribute;
use crate::writer::types::AttrData;

/// Parameters for dense attribute storage structures.
pub(crate) struct DenseAttrStorage {
    /// Serialized attribute message bodies.
    pub attr_bodies: Vec<Vec<u8>>,
    /// Name hashes for each attribute (same order as attr_bodies).
    pub name_hashes: Vec<u32>,
    /// Direct block size (power of 2, at least large enough for all attributes).
    pub dblk_size: usize,
    /// B-tree v2 node size.
    pub btree_node_size: usize,
}

/// Sizes of each dense attribute sub-structure.
pub(crate) struct DenseAttrSizes {
    pub frhp_size: usize,
    pub bthd_size: usize,
    pub fshd_size: usize,
    pub btlf_size: usize,
    pub fsse_size: usize,
    pub fhdb_size: usize,
}

impl DenseAttrSizes {
    pub fn total(&self) -> usize {
        self.frhp_size
            + self.bthd_size
            + self.fshd_size
            + self.btlf_size
            + self.fsse_size
            + self.fhdb_size
    }
}

/// Compute the dense attribute storage parameters from a set of attributes.
pub(crate) fn compute_dense_attr_storage(attrs: &[AttrData]) -> Result<DenseAttrStorage> {
    let mut attr_bodies = Vec::with_capacity(attrs.len());
    let mut name_hashes = Vec::with_capacity(attrs.len());

    for attr in attrs {
        let body = encode_attribute(attr)?;
        let hash = checksum::lookup3(attr.name.as_bytes());
        attr_bodies.push(body);
        name_hashes.push(hash);
    }

    // Direct block size: C library default starting_block_size = 1024
    let dblk_size = 1024;
    let btree_node_size = 512;

    Ok(DenseAttrStorage {
        attr_bodies,
        name_hashes,
        dblk_size,
        btree_node_size,
    })
}

/// Compute the sizes of all dense attribute sub-structures.
pub(crate) fn compute_dense_attr_sizes(storage: &DenseAttrStorage) -> DenseAttrSizes {
    // FRHP header: sig(4) + ver(1) + heap_id_len(2) + io_filter_len(2) + flags(1) +
    //   max_managed(4) + next_huge_id(8) + btree_huge(8) + free_space(8) + fsm_addr(8) +
    //   managed_space(8) + alloc_managed(8) + iter_offset(8) + num_managed(8) +
    //   huge_size(8) + huge_num(8) + tiny_size(8) + tiny_num(8) +
    //   table_width(2) + start_blk_size(8) + max_dblk_size(8) + max_heap_size(2) +
    //   start_nrows(2) + root_blk_addr(8) + curr_nrows(2) + cksum(4)
    let frhp_size = 146;

    // BTHD: sig(4) + ver(1) + type(1) + node_size(4) + rec_size(2) + depth(2) +
    //   split(1) + merge(1) + root_addr(8) + root_nrec(2) + total_rec(8) + cksum(4)
    let bthd_size = 38;

    // FSHD: sig(4) + ver(1) + client_id(1) + total_space(8) + total_sections(8) +
    //   serial_sections(8) + ghost_sections(8) + n_classes(2) + shrink(2) + expand(2) +
    //   addr_size(2) + max_section_size(8) + serial_list_addr(8) +
    //   serial_list_used(8) + alloc_size(8) + cksum(4)
    let fshd_size = 82;

    // BTLF (leaf node): node_size bytes (padded with zeros)
    let btlf_size = storage.btree_node_size;

    // FSSE: sig(4) + ver(1) + fshd_addr(8) + section_data + cksum(4)
    // section_data: count(varint 1 byte) + size(sect_off_size) + type(1) + offset(sect_off_size)
    // sect_off_size = limit_enc_size(free_space) — for typical sizes, 2 bytes
    // Plus zero padding to fill the allocated size
    // The C library allocates 27 bytes total for the FSSE in this case.
    let fsse_size = 27;

    // FHDB: dblk_size bytes
    let fhdb_size = storage.dblk_size;

    DenseAttrSizes {
        frhp_size,
        bthd_size,
        fshd_size,
        btlf_size,
        fsse_size,
        fhdb_size,
    }
}

/// Encode all dense attribute structures given their base address.
///
/// Returns the concatenated bytes for FRHP + BTHD + FSHD + BTLF + FSSE + FHDB.
pub(crate) fn encode_dense_attr_structures(
    storage: &DenseAttrStorage,
    base_addr: u64,
) -> Result<Vec<u8>> {
    let sizes = compute_dense_attr_sizes(storage);
    let n_attrs = storage.attr_bodies.len();

    // Compute addresses of each sub-structure.
    let frhp_addr = base_addr;
    let bthd_addr = frhp_addr + sizes.frhp_size as u64;
    let fshd_addr = bthd_addr + sizes.bthd_size as u64;
    let btlf_addr = fshd_addr + sizes.fshd_size as u64;
    let fsse_addr = btlf_addr + sizes.btlf_size as u64;
    let fhdb_addr = fsse_addr + sizes.fsse_size as u64;

    // --- Direct block (FHDB) ---
    // Header: sig(4) + ver(1) + heap_addr(8) + block_offset(5) + cksum(4) = 22 bytes
    // The header checksum covers the first 18 bytes. No block-end checksum.
    let max_heap_size_bits: u16 = 40; // C library default for attribute heap
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8); // 5
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 22

    // Place attributes sequentially after the direct block header.
    let mut attr_offsets = Vec::with_capacity(n_attrs);
    let mut offset = dblk_header_size;
    for body in &storage.attr_bodies {
        attr_offsets.push(offset);
        offset += body.len();
    }

    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;

    // --- Build heap IDs ---
    // Managed heap ID: byte0 = ver(4 bits) | type(4 bits) = 0x00,
    // then offset(max_heap_size_bits=40 bits=5 bytes) + length(remaining bits)
    // heap_id_len = 8, so after byte0: 7 bytes
    // offset uses 40 bits = 5 bytes, length uses remaining 56-40=16 bits = 2 bytes
    let heap_id_len: u16 = 8;

    let mut heap_ids = Vec::with_capacity(n_attrs);
    for i in 0..n_attrs {
        let off = attr_offsets[i] as u64;
        let len = storage.attr_bodies[i].len() as u64;
        let mut hid = [0u8; 8];
        hid[0] = 0x00; // version 0, type 0 (managed)
        // Pack offset (40 bits) + length (16 bits) into 7 bytes LE
        let packed = off | (len << 40);
        hid[1..].copy_from_slice(&packed.to_le_bytes()[..7]);
        heap_ids.push(hid);
    }

    // --- B-tree v2 records ---
    // Type 8 record: heap_id(8) + msg_flags(1) + creation_order(4) + name_hash(4) = 17
    let rec_size: u16 = 17;
    let mut records: Vec<(u32, usize)> = storage
        .name_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();
    records.sort_by_key(|&(hash, _)| hash);

    // --- Encode FHDB ---
    let fhdb = encode_fhdb(
        frhp_addr,
        max_heap_size_bits,
        storage.dblk_size,
        &storage.attr_bodies,
        dblk_header_size,
    );

    // --- Encode BTLF (leaf node) ---
    let btlf = encode_btlf(&records, &heap_ids, rec_size, storage.btree_node_size);

    // --- Encode BTHD ---
    let bthd = encode_bthd(
        btlf_addr,
        n_attrs as u16,
        n_attrs as u64,
        rec_size,
        storage.btree_node_size as u32,
    );

    // sect_off_size for free space encoding
    let sect_off_size = limit_enc_size_u64(free_size as u64);

    // --- Encode FSSE ---
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size; // count + size + type + offset
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4; // sig + ver + addr + data + cksum
    // The C library allocates a fixed 27 bytes; pad if smaller.
    let fsse_alloc = 27.max(fsse_total);
    let fsse = encode_fsse(fshd_addr, free_offset, free_size, sect_off_size, fsse_alloc);

    // --- Encode FSHD ---
    let fshd = encode_fshd(
        free_size as u64,
        fsse_addr,
        fsse_alloc as u64,
        max_heap_size_bits,
    );

    // --- Encode FRHP ---
    let total_attr_bytes: usize = storage.attr_bodies.iter().map(|b| b.len()).sum();
    let frhp = encode_frhp(
        heap_id_len,
        max_heap_size_bits,
        n_attrs as u64,
        total_attr_bytes as u64,
        storage.dblk_size as u64,
        free_size as u64,
        fshd_addr,
        fhdb_addr,
    );

    // Concatenate all structures.
    let mut buf = Vec::with_capacity(sizes.total());
    buf.extend_from_slice(&frhp);
    buf.extend_from_slice(&bthd);
    buf.extend_from_slice(&fshd);
    buf.extend_from_slice(&btlf);
    buf.extend_from_slice(&fsse);
    buf.extend_from_slice(&fhdb);

    debug_assert_eq!(buf.len(), sizes.total());
    Ok(buf)
}

fn encode_frhp(
    heap_id_len: u16,
    max_heap_size_bits: u16,
    num_managed_objs: u64,
    _total_attr_bytes: u64,
    dblk_size: u64,
    free_space: u64,
    fsm_addr: u64,
    root_block_addr: u64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(146);

    buf.extend_from_slice(b"FRHP");
    buf.push(0); // version
    buf.extend_from_slice(&heap_id_len.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
    buf.push(0x02); // flags: bit 1 = checksum direct blocks

    // max_managed_object_size
    buf.extend_from_slice(&4096u32.to_le_bytes());

    // next_huge_object_id (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // btree_huge_addr (8 bytes)
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    // free_space_in_managed (8 bytes)
    buf.extend_from_slice(&free_space.to_le_bytes());
    // free_space_manager_address (8 bytes)
    buf.extend_from_slice(&fsm_addr.to_le_bytes());
    // managed_space_total (8 bytes)
    buf.extend_from_slice(&dblk_size.to_le_bytes());
    // managed_space_allocated (8 bytes)
    buf.extend_from_slice(&dblk_size.to_le_bytes());
    // managed_alloc_iterator_offset (8 bytes) — 0 when all objects fit in first block
    buf.extend_from_slice(&0u64.to_le_bytes());
    // managed_objects_count (8 bytes)
    buf.extend_from_slice(&num_managed_objs.to_le_bytes());

    // huge_objects_total_size (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // huge_objects_count (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // tiny_objects_total_size (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // tiny_objects_count (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());

    // table_width (2 bytes) — C library default: 4
    buf.extend_from_slice(&4u16.to_le_bytes());
    // starting_block_size (8 bytes)
    buf.extend_from_slice(&dblk_size.to_le_bytes());
    // max_direct_block_size (8 bytes) — C library default: 65536
    buf.extend_from_slice(&65536u64.to_le_bytes());
    // max_heap_size (2 bytes, in bits)
    buf.extend_from_slice(&max_heap_size_bits.to_le_bytes());
    // starting_num_rows_root (2 bytes)
    buf.extend_from_slice(&1u16.to_le_bytes());
    // root_block_address (8 bytes)
    buf.extend_from_slice(&root_block_addr.to_le_bytes());
    // current_num_rows_root (2 bytes) — 0 = root is a direct block
    buf.extend_from_slice(&0u16.to_le_bytes());

    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    debug_assert_eq!(buf.len(), 146);
    buf
}

fn encode_bthd(
    root_addr: u64,
    root_nrecords: u16,
    total_records: u64,
    rec_size: u16,
    node_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(38);

    buf.extend_from_slice(b"BTHD");
    buf.push(0); // version
    buf.push(8); // type 8 = attribute name index
    buf.extend_from_slice(&node_size.to_le_bytes());
    buf.extend_from_slice(&rec_size.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // depth = 0 (leaf only)
    buf.push(100); // split_percent
    buf.push(40); // merge_percent
    buf.extend_from_slice(&root_addr.to_le_bytes());
    buf.extend_from_slice(&root_nrecords.to_le_bytes());
    buf.extend_from_slice(&total_records.to_le_bytes());

    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    debug_assert_eq!(buf.len(), 38);
    buf
}

fn encode_btlf(
    records: &[(u32, usize)], // (name_hash, attr_index) sorted by hash
    heap_ids: &[[u8; 8]],
    _rec_size: u16,
    node_size: usize,
) -> Vec<u8> {
    let mut buf = vec![0u8; node_size];

    buf[0..4].copy_from_slice(b"BTLF");
    buf[4] = 0; // version
    buf[5] = 8; // type 8

    let mut off = 6;
    for &(name_hash, attr_idx) in records {
        // Record: heap_id(8) + msg_flags(1) + creation_order(4) + name_hash(4)
        buf[off..off + 8].copy_from_slice(&heap_ids[attr_idx]);
        off += 8;
        buf[off] = 0x00; // message flags
        off += 1;
        // creation_order = 0xFFFF (not tracked)
        buf[off..off + 4].copy_from_slice(&0xFFFFu32.to_le_bytes());
        off += 4;
        buf[off..off + 4].copy_from_slice(&name_hash.to_le_bytes());
        off += 4;
    }

    // Checksum immediately after records (not at end of node).
    let cksum = checksum::lookup3(&buf[..off]);
    buf[off..off + 4].copy_from_slice(&cksum.to_le_bytes());

    buf
}

fn encode_fshd(
    total_space: u64,
    serial_list_addr: u64,
    serial_list_alloc: u64,
    max_heap_size_bits: u16,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(82);

    buf.extend_from_slice(b"FSHD");
    buf.push(0); // version
    buf.push(0); // client_id = 0 (fractal heap)

    // total_space (8 bytes)
    buf.extend_from_slice(&total_space.to_le_bytes());
    // total_sections (8 bytes)
    buf.extend_from_slice(&1u64.to_le_bytes());
    // serial_sections (8 bytes)
    buf.extend_from_slice(&1u64.to_le_bytes());
    // ghost_sections (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());

    // n_section_classes (2 bytes) — 4 classes for fractal heap
    buf.extend_from_slice(&4u16.to_le_bytes());
    // shrink_percent (2 bytes) — C default: 80
    buf.extend_from_slice(&80u16.to_le_bytes());
    // expand_percent (2 bytes) — C default: 120
    buf.extend_from_slice(&120u16.to_le_bytes());
    // address_size_encoding (2 bytes, in bits)
    buf.extend_from_slice(&max_heap_size_bits.to_le_bytes());
    // max_section_size (8 bytes) — C default: 65536
    buf.extend_from_slice(&65536u64.to_le_bytes());

    // serial_list_addr (8 bytes)
    buf.extend_from_slice(&serial_list_addr.to_le_bytes());
    // serial_list_used (8 bytes)
    buf.extend_from_slice(&serial_list_alloc.to_le_bytes());
    // alloc_section_size (8 bytes)
    buf.extend_from_slice(&serial_list_alloc.to_le_bytes());

    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    debug_assert_eq!(buf.len(), 82);
    buf
}

fn encode_fsse(
    fshd_addr: u64,
    free_offset: usize,
    free_size: usize,
    sect_off_size: usize,
    alloc_size: usize,
) -> Vec<u8> {
    let header_size = 4 + 1 + 8; // sig + ver + fshd_addr
    let mut buf = vec![0u8; alloc_size];

    buf[0..4].copy_from_slice(b"FSSE");
    buf[4] = 0; // version
    buf[5..13].copy_from_slice(&fshd_addr.to_le_bytes());

    let mut off = header_size;

    // Section data: count(varint) + size(sect_off_size LE) + type(1) + offset(sect_off_size LE)
    // count = 1
    buf[off] = 1;
    off += 1;

    // size (sect_off_size bytes LE)
    let size_bytes = (free_size as u64).to_le_bytes();
    buf[off..off + sect_off_size].copy_from_slice(&size_bytes[..sect_off_size]);
    off += sect_off_size;

    // section type = 0 (FHEAP_SECT_SINGLE)
    buf[off] = 0;
    off += 1;

    // offset (sect_off_size bytes LE)
    let off_bytes = (free_offset as u64).to_le_bytes();
    buf[off..off + sect_off_size].copy_from_slice(&off_bytes[..sect_off_size]);

    // Checksum at end.
    let cksum = checksum::lookup3(&buf[..alloc_size - 4]);
    buf[alloc_size - 4..alloc_size].copy_from_slice(&cksum.to_le_bytes());

    buf
}

fn encode_fhdb(
    frhp_addr: u64,
    max_heap_size_bits: u16,
    dblk_size: usize,
    attr_bodies: &[Vec<u8>],
    header_size: usize,
) -> Vec<u8> {
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let mut buf = vec![0u8; dblk_size];

    // Header: sig(4) + ver(1) + heap_addr(8) + block_offset(block_offset_bytes) + cksum(4)
    buf[0..4].copy_from_slice(b"FHDB");
    buf[4] = 0; // version
    buf[5..13].copy_from_slice(&frhp_addr.to_le_bytes());
    // block_offset = 0 (first and only direct block) — already zero.

    // Place attributes after header (checksum field at offset 18 is zero for now).
    let mut off = header_size;
    for body in attr_bodies {
        buf[off..off + body.len()].copy_from_slice(body);
        off += body.len();
    }

    // Checksum: computed over the entire block with the checksum field zeroed.
    // The C library zeros the 4-byte checksum field, computes lookup3 over the
    // full block, then writes the checksum at offset 18.
    let cksum_off = 4 + 1 + 8 + block_offset_bytes; // 18
    // buf[cksum_off..cksum_off+4] is already zero from initialization.
    let block_cksum = checksum::lookup3(&buf);
    buf[cksum_off..cksum_off + 4].copy_from_slice(&block_cksum.to_le_bytes());

    buf
}

/// Encode dense attribute metadata structures (FRHP + BTHD + FSHD + BTLF + FSSE)
/// with FHDB placed at a separate address (in the data region).
///
/// `base_addr` is where FRHP starts (right after the OHDR).
/// `fhdb_addr` is where the FHDB will be placed (in the data region).
pub(crate) fn encode_dense_attr_structures_split(
    storage: &DenseAttrStorage,
    base_addr: u64,
    fhdb_addr: u64,
) -> Result<Vec<u8>> {
    let sizes = compute_dense_attr_sizes(storage);
    let n_attrs = storage.attr_bodies.len();

    // Compute addresses of each sub-structure (metadata only).
    let frhp_addr = base_addr;
    let bthd_addr = frhp_addr + sizes.frhp_size as u64;
    let fshd_addr = bthd_addr + sizes.bthd_size as u64;
    let btlf_addr = fshd_addr + sizes.fshd_size as u64;
    let fsse_addr = btlf_addr + sizes.btlf_size as u64;

    let max_heap_size_bits: u16 = 40;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 22

    // Place attributes sequentially after the direct block header.
    let mut attr_offsets = Vec::with_capacity(n_attrs);
    let mut offset = dblk_header_size;
    for body in &storage.attr_bodies {
        attr_offsets.push(offset);
        offset += body.len();
    }

    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;

    // Build heap IDs.
    let heap_id_len: u16 = 8;
    let mut heap_ids = Vec::with_capacity(n_attrs);
    for i in 0..n_attrs {
        let off = attr_offsets[i] as u64;
        let len = storage.attr_bodies[i].len() as u64;
        let mut hid = [0u8; 8];
        hid[0] = 0x00;
        let packed = off | (len << 40);
        hid[1..].copy_from_slice(&packed.to_le_bytes()[..7]);
        heap_ids.push(hid);
    }

    // B-tree v2 records.
    let rec_size: u16 = 17;
    let mut records: Vec<(u32, usize)> = storage
        .name_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();
    records.sort_by_key(|&(hash, _)| hash);

    // Encode BTLF.
    let btlf = encode_btlf(&records, &heap_ids, rec_size, storage.btree_node_size);

    // Encode BTHD.
    let bthd = encode_bthd(
        btlf_addr,
        n_attrs as u16,
        n_attrs as u64,
        rec_size,
        storage.btree_node_size as u32,
    );

    let sect_off_size = limit_enc_size_u64(free_size as u64);

    // Encode FSSE.
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size;
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4;
    let fsse_alloc = 27.max(fsse_total);
    let fsse = encode_fsse(fshd_addr, free_offset, free_size, sect_off_size, fsse_alloc);

    // Encode FSHD.
    let fshd = encode_fshd(
        free_size as u64,
        fsse_addr,
        fsse_alloc as u64,
        max_heap_size_bits,
    );

    // Encode FRHP — root_block_addr points to FHDB in data region.
    let total_attr_bytes: usize = storage.attr_bodies.iter().map(|b| b.len()).sum();
    let frhp = encode_frhp(
        heap_id_len,
        max_heap_size_bits,
        n_attrs as u64,
        total_attr_bytes as u64,
        storage.dblk_size as u64,
        free_size as u64,
        fshd_addr,
        fhdb_addr,
    );

    // Concatenate metadata structures (no FHDB).
    let meta_size =
        sizes.frhp_size + sizes.bthd_size + sizes.fshd_size + sizes.btlf_size + sizes.fsse_size;
    let mut buf = Vec::with_capacity(meta_size);
    buf.extend_from_slice(&frhp);
    buf.extend_from_slice(&bthd);
    buf.extend_from_slice(&fshd);
    buf.extend_from_slice(&btlf);
    buf.extend_from_slice(&fsse);

    debug_assert_eq!(buf.len(), meta_size);
    Ok(buf)
}

/// Encode just the FHDB (fractal heap direct block) as a standalone blob.
pub(crate) fn encode_fhdb_standalone(storage: &DenseAttrStorage, frhp_addr: u64) -> Vec<u8> {
    let max_heap_size_bits: u16 = 40;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 22
    encode_fhdb(
        frhp_addr,
        max_heap_size_bits,
        storage.dblk_size,
        &storage.attr_bodies,
        dblk_header_size,
    )
}

// ─── Dense link storage ───────────────────────────────────────────────

/// Parameters for dense link storage with creation order.
pub(crate) struct DenseLinkStorage {
    /// Serialized link message bodies (with creation order).
    pub link_bodies: Vec<Vec<u8>>,
    /// Name hashes for each link.
    pub name_hashes: Vec<u32>,
    /// Direct block size (512 for links).
    pub dblk_size: usize,
    /// B-tree v2 node size.
    pub btree_node_size: usize,
}

/// Sizes of dense link sub-structures.
pub(crate) struct DenseLinkSizes {
    pub frhp_size: usize,
    pub bthd_name_size: usize,
    pub bthd_crt_size: usize,
    pub fshd_size: usize,
    pub btlf_name_size: usize,
    pub btlf_crt_size: usize,
    pub fsse_size: usize,
    pub fhdb_size: usize,
}

impl DenseLinkSizes {
    /// Metadata total (FRHP + BTHD×2 + FSHD + BTLF×2 + FSSE). Excludes FHDB.
    pub fn meta_total(&self) -> usize {
        self.frhp_size
            + self.bthd_name_size
            + self.bthd_crt_size
            + self.fshd_size
            + self.btlf_name_size
            + self.btlf_crt_size
            + self.fsse_size
    }
}

/// Compute dense link storage from link names, addresses, and creation orders.
pub(crate) fn compute_dense_link_storage(names: &[String], addrs: &[u64]) -> DenseLinkStorage {
    use crate::writer::encode::encode_link_body_crt_order;

    let mut link_bodies = Vec::with_capacity(names.len());
    let mut name_hashes = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let body = encode_link_body_crt_order(name, addrs[i], i as u64);
        let hash = crate::checksum::lookup3(name.as_bytes());
        link_bodies.push(body);
        name_hashes.push(hash);
    }
    DenseLinkStorage {
        link_bodies,
        name_hashes,
        dblk_size: 512,
        btree_node_size: 512,
    }
}

pub(crate) fn compute_dense_link_sizes(storage: &DenseLinkStorage) -> DenseLinkSizes {
    DenseLinkSizes {
        frhp_size: 146,
        bthd_name_size: 38,
        bthd_crt_size: 38,
        fshd_size: 82,
        btlf_name_size: storage.btree_node_size,
        btlf_crt_size: storage.btree_node_size,
        fsse_size: 26, // typical for link heaps
        fhdb_size: storage.dblk_size,
    }
}

/// Encode dense link structures (metadata only: FRHP + BTHD_name + BTHD_crt + FSHD + BTLF_name + BTLF_crt).
/// FSSE is encoded separately. FHDB goes in sdata.
pub(crate) fn encode_dense_link_meta(
    storage: &DenseLinkStorage,
    base_addr: u64,
    fhdb_addr: u64,
    fsse_addr_override: u64,
) -> Result<Vec<u8>> {
    let sizes = compute_dense_link_sizes(storage);
    let n = storage.link_bodies.len();

    let frhp_addr = base_addr;
    let bthd_name_addr = frhp_addr + sizes.frhp_size as u64;
    let bthd_crt_addr = bthd_name_addr + sizes.bthd_name_size as u64;
    let fshd_addr = bthd_crt_addr + sizes.bthd_crt_size as u64;
    let btlf_name_addr = fshd_addr + sizes.fshd_size as u64;
    let btlf_crt_addr = btlf_name_addr + sizes.btlf_name_size as u64;

    // Link heap: heap_id_len=7, max_heap_size=32 bits, block_offset_bytes=4
    let max_heap_size_bits: u16 = 32;
    let block_offset_bytes = 4usize;
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 21

    let mut link_offsets = Vec::with_capacity(n);
    let mut offset = dblk_header_size;
    for body in &storage.link_bodies {
        link_offsets.push(offset);
        offset += body.len();
    }
    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;

    // Build heap IDs (7 bytes: byte0=0, then offset(32bit) + length(16bit) packed LE)
    let heap_id_len: u16 = 7;
    let mut heap_ids: Vec<[u8; 7]> = Vec::with_capacity(n);
    for i in 0..n {
        let off = link_offsets[i] as u64;
        let len = storage.link_bodies[i].len() as u64;
        let mut hid = [0u8; 7];
        hid[0] = 0x00;
        let packed = off | (len << 32);
        hid[1..].copy_from_slice(&packed.to_le_bytes()[..6]);
        heap_ids.push(hid);
    }

    // B-tree type 5 records: sorted by name_hash. Record = hash(4) + heap_id(7).
    let rec_size_name: u16 = 4 + heap_id_len;
    let mut name_records: Vec<(u32, usize)> = storage
        .name_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();
    name_records.sort_by_key(|&(hash, _)| hash);

    // B-tree type 6 records: sorted by creation_order. Record = crt_order(8) + heap_id(7).
    let rec_size_crt: u16 = 8 + heap_id_len;

    // Encode BTLF type 5 (name)
    let btlf_name = encode_btlf_type5(&name_records, &heap_ids, storage.btree_node_size);
    // Encode BTLF type 6 (creation order) — records in order 0,1,2,...
    let btlf_crt = encode_btlf_type6(&heap_ids, storage.btree_node_size);

    // Encode BTHDs
    let bthd_name = encode_bthd(
        btlf_name_addr,
        n as u16,
        n as u64,
        rec_size_name,
        storage.btree_node_size as u32,
    );
    // Override type to 5
    let mut bthd_name = bthd_name;
    bthd_name[5] = 5;

    let bthd_crt = {
        let mut b = encode_bthd(
            btlf_crt_addr,
            n as u16,
            n as u64,
            rec_size_crt,
            storage.btree_node_size as u32,
        );
        b[5] = 6; // type 6
        // Recompute checksum since we changed the type byte
        let cksum = crate::checksum::lookup3(&b[..b.len() - 4]);
        let len = b.len();
        b[len - 4..].copy_from_slice(&cksum.to_le_bytes());
        b
    };

    // Fix BTHD name checksum too
    {
        let cksum = crate::checksum::lookup3(&bthd_name[..bthd_name.len() - 4]);
        let len = bthd_name.len();
        bthd_name[len - 4..].copy_from_slice(&cksum.to_le_bytes());
    }

    // FSSE for links
    let sect_off_size = limit_enc_size_u64(free_size as u64);
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size;
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4;
    let fsse_alloc = 26.max(fsse_total);

    let _fsse = encode_fsse(fshd_addr, free_offset, free_size, sect_off_size, fsse_alloc);

    // FSHD — uses externally-provided FSSE address for interleaved layouts.
    let fshd = encode_fshd(
        free_size as u64,
        fsse_addr_override,
        fsse_alloc as u64,
        max_heap_size_bits,
    );

    // FRHP
    let total_link_bytes: usize = storage.link_bodies.iter().map(|b| b.len()).sum();
    let frhp = encode_frhp_link(
        heap_id_len,
        max_heap_size_bits,
        n as u64,
        total_link_bytes as u64,
        storage.dblk_size as u64,
        free_size as u64,
        fshd_addr,
        fhdb_addr,
    );

    let meta_size = sizes.frhp_size
        + sizes.bthd_name_size
        + sizes.bthd_crt_size
        + sizes.fshd_size
        + sizes.btlf_name_size
        + sizes.btlf_crt_size;
    let mut buf = Vec::with_capacity(meta_size);
    buf.extend_from_slice(&frhp);
    buf.extend_from_slice(&bthd_name);
    buf.extend_from_slice(&bthd_crt);
    buf.extend_from_slice(&fshd);
    buf.extend_from_slice(&btlf_name);
    buf.extend_from_slice(&btlf_crt);

    Ok(buf)
}

/// Encode FSSE for link heap (standalone).
pub(crate) fn encode_link_fsse(storage: &DenseLinkStorage, fshd_addr: u64) -> Vec<u8> {
    let max_heap_size_bits: u16 = 32;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4;
    let mut offset = dblk_header_size;
    for body in &storage.link_bodies {
        offset += body.len();
    }
    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;
    let sect_off_size = limit_enc_size_u64(free_size as u64);
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size;
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4;
    let fsse_alloc = 26.max(fsse_total);
    encode_fsse(fshd_addr, free_offset, free_size, sect_off_size, fsse_alloc)
}

/// Encode FHDB for link heap (standalone).
pub(crate) fn encode_link_fhdb(storage: &DenseLinkStorage, frhp_addr: u64) -> Vec<u8> {
    let max_heap_size_bits: u16 = 32;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 21
    encode_fhdb(
        frhp_addr,
        max_heap_size_bits,
        storage.dblk_size,
        &storage.link_bodies,
        dblk_header_size,
    )
}

fn encode_frhp_link(
    heap_id_len: u16,
    max_heap_size_bits: u16,
    num_managed_objs: u64,
    _total_bytes: u64,
    dblk_size: u64,
    free_space: u64,
    fsm_addr: u64,
    root_block_addr: u64,
) -> Vec<u8> {
    // Same structure as attr FRHP but with link-specific heap_id_len and max_heap_size
    encode_frhp(
        heap_id_len,
        max_heap_size_bits,
        num_managed_objs,
        _total_bytes,
        dblk_size,
        free_space,
        fsm_addr,
        root_block_addr,
    )
}

fn encode_btlf_type5(
    name_records: &[(u32, usize)], // (hash, link_index) sorted by hash
    heap_ids: &[[u8; 7]],
    node_size: usize,
) -> Vec<u8> {
    let mut buf = vec![0u8; node_size];
    buf[0..4].copy_from_slice(b"BTLF");
    buf[4] = 0; // version
    buf[5] = 5; // type 5 = link name

    let mut off = 6;
    for &(hash, idx) in name_records {
        buf[off..off + 4].copy_from_slice(&hash.to_le_bytes());
        off += 4;
        buf[off..off + 7].copy_from_slice(&heap_ids[idx]);
        off += 7;
    }

    let cksum = crate::checksum::lookup3(&buf[..off]);
    buf[off..off + 4].copy_from_slice(&cksum.to_le_bytes());
    buf
}

fn encode_btlf_type6(heap_ids: &[[u8; 7]], node_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; node_size];
    buf[0..4].copy_from_slice(b"BTLF");
    buf[4] = 0; // version
    buf[5] = 6; // type 6 = link creation order

    let mut off = 6;
    for (i, hid) in heap_ids.iter().enumerate() {
        // Record: creation_order(8) + heap_id(7)
        buf[off..off + 8].copy_from_slice(&(i as u64).to_le_bytes());
        off += 8;
        buf[off..off + 7].copy_from_slice(hid);
        off += 7;
    }

    let cksum = crate::checksum::lookup3(&buf[..off]);
    buf[off..off + 4].copy_from_slice(&cksum.to_le_bytes());
    buf
}

// ─── Dense attrs with creation order ──────────────────────────────────

/// Sizes of dense attr sub-structures with creation order (adds BTLF for type 9).
pub(crate) struct DenseAttrCrtSizes {
    pub frhp_size: usize,
    pub bthd_name_size: usize,
    pub bthd_crt_size: usize,
    pub fshd_size: usize,
    pub btlf_name_size: usize,
    pub btlf_crt_size: usize,
    pub fsse_size: usize,
    pub fhdb_size: usize,
}

impl DenseAttrCrtSizes {
    pub fn meta_total(&self) -> usize {
        self.frhp_size
            + self.bthd_name_size
            + self.bthd_crt_size
            + self.fshd_size
            + self.btlf_name_size
            + self.btlf_crt_size
            + self.fsse_size
    }
}

pub(crate) fn compute_dense_attr_crt_sizes(storage: &DenseAttrStorage) -> DenseAttrCrtSizes {
    DenseAttrCrtSizes {
        frhp_size: 146,
        bthd_name_size: 38,
        bthd_crt_size: 38,
        fshd_size: 82,
        btlf_name_size: storage.btree_node_size,
        btlf_crt_size: storage.btree_node_size,
        fsse_size: 27,
        fhdb_size: storage.dblk_size,
    }
}

/// Encode dense attr metadata with creation order (FRHP + BTHD_name + BTHD_crt).
/// This is part 1 — FSHD + BTLFs are separate (part 2) due to C library allocation order.
pub(crate) fn encode_dense_attr_crt_part1(
    storage: &DenseAttrStorage,
    frhp_addr: u64,
    fhdb_addr: u64,
    fshd_addr: u64,
    fsse_addr: u64,
) -> Result<Vec<u8>> {
    let sizes = compute_dense_attr_crt_sizes(storage);
    let n = storage.attr_bodies.len();
    let bthd_name_addr = frhp_addr + sizes.frhp_size as u64;
    let bthd_crt_addr = bthd_name_addr + sizes.bthd_name_size as u64;

    let max_heap_size_bits: u16 = 40;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 22

    let mut attr_offsets = Vec::with_capacity(n);
    let mut offset = dblk_header_size;
    for body in &storage.attr_bodies {
        attr_offsets.push(offset);
        offset += body.len();
    }
    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;

    // Build heap IDs (8 bytes)
    let heap_id_len: u16 = 8;
    let mut heap_ids = Vec::with_capacity(n);
    for i in 0..n {
        let off = attr_offsets[i] as u64;
        let len = storage.attr_bodies[i].len() as u64;
        let mut hid = [0u8; 8];
        hid[0] = 0x00;
        let packed = off | (len << 40);
        hid[1..].copy_from_slice(&packed.to_le_bytes()[..7]);
        heap_ids.push(hid);
    }

    // Type 8 records (name): sorted by hash
    let rec_size_name: u16 = 17;
    let mut name_records: Vec<(u32, usize)> = storage
        .name_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();
    name_records.sort_by_key(|&(hash, _)| hash);

    // Type 9 records (creation order): sorted by creation order (= index)
    let rec_size_crt: u16 = 13;

    // FSSE
    let sect_off_size = limit_enc_size_u64(free_size as u64);
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size;
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4;
    let fsse_alloc = 27.max(fsse_total);

    // BTHD name (type 8)
    let btlf_name_addr = fshd_addr + sizes.fshd_size as u64;
    let bthd_name = encode_bthd(
        btlf_name_addr,
        n as u16,
        n as u64,
        rec_size_name,
        storage.btree_node_size as u32,
    );

    // BTHD crt (type 9)
    let btlf_crt_addr = btlf_name_addr + sizes.btlf_name_size as u64;
    let mut bthd_crt = encode_bthd(
        btlf_crt_addr,
        n as u16,
        n as u64,
        rec_size_crt,
        storage.btree_node_size as u32,
    );
    bthd_crt[5] = 9; // override type to 9
    let cksum = crate::checksum::lookup3(&bthd_crt[..bthd_crt.len() - 4]);
    let len = bthd_crt.len();
    bthd_crt[len - 4..].copy_from_slice(&cksum.to_le_bytes());

    // FRHP
    let total_attr_bytes: usize = storage.attr_bodies.iter().map(|b| b.len()).sum();
    let frhp = encode_frhp(
        heap_id_len,
        max_heap_size_bits,
        n as u64,
        total_attr_bytes as u64,
        storage.dblk_size as u64,
        free_size as u64,
        fshd_addr,
        fhdb_addr,
    );

    // Part 1: FRHP + BTHD_name + BTHD_crt
    let part1_size = sizes.frhp_size + sizes.bthd_name_size + sizes.bthd_crt_size;
    let mut buf = Vec::with_capacity(part1_size);
    buf.extend_from_slice(&frhp);
    buf.extend_from_slice(&bthd_name);
    buf.extend_from_slice(&bthd_crt);
    Ok(buf)
}

/// Encode dense attr metadata part 2 (FSHD + BTLF_name + BTLF_crt).
pub(crate) fn encode_dense_attr_crt_part2(
    storage: &DenseAttrStorage,
    fshd_addr: u64,
    fsse_addr: u64,
) -> Result<Vec<u8>> {
    let sizes = compute_dense_attr_crt_sizes(storage);
    let n = storage.attr_bodies.len();

    let max_heap_size_bits: u16 = 40;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4;

    let mut attr_offsets = Vec::with_capacity(n);
    let mut offset = dblk_header_size;
    for body in &storage.attr_bodies {
        attr_offsets.push(offset);
        offset += body.len();
    }
    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;

    let heap_id_len: u16 = 8;
    let mut heap_ids = Vec::with_capacity(n);
    for i in 0..n {
        let off = attr_offsets[i] as u64;
        let len = storage.attr_bodies[i].len() as u64;
        let mut hid = [0u8; 8];
        hid[0] = 0x00;
        let packed = off | (len << 40);
        hid[1..].copy_from_slice(&packed.to_le_bytes()[..7]);
        heap_ids.push(hid);
    }

    let mut name_records: Vec<(u32, usize)> = storage
        .name_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();
    name_records.sort_by_key(|&(hash, _)| hash);

    let sect_off_size = limit_enc_size_u64(free_size as u64);
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size;
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4;
    let fsse_alloc = 27.max(fsse_total);

    // FSHD
    let fshd = encode_fshd(
        free_size as u64,
        fsse_addr,
        fsse_alloc as u64,
        max_heap_size_bits,
    );

    // BTLF type 8 (name)
    let btlf_name = encode_btlf_type8_crt(&name_records, &heap_ids, storage.btree_node_size);

    // BTLF type 9 (creation order)
    let btlf_crt = encode_btlf_type9(&heap_ids, storage.btree_node_size);

    let part2_size = sizes.fshd_size + sizes.btlf_name_size + sizes.btlf_crt_size;
    let mut buf = Vec::with_capacity(part2_size);
    buf.extend_from_slice(&fshd);
    buf.extend_from_slice(&btlf_name);
    buf.extend_from_slice(&btlf_crt);
    Ok(buf)
}

/// Encode FSSE for attr heap with creation order (standalone).
pub(crate) fn encode_attr_crt_fsse(storage: &DenseAttrStorage, fshd_addr: u64) -> Vec<u8> {
    let max_heap_size_bits: u16 = 40;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4;
    let mut offset = dblk_header_size;
    for body in &storage.attr_bodies {
        offset += body.len();
    }
    let free_offset = offset;
    let free_size = storage.dblk_size - free_offset;
    let sect_off_size = limit_enc_size_u64(free_size as u64);
    let fsse_data_size = 1 + sect_off_size + 1 + sect_off_size;
    let fsse_total = 4 + 1 + 8 + fsse_data_size + 4;
    let fsse_alloc = 27.max(fsse_total);
    encode_fsse(fshd_addr, free_offset, free_size, sect_off_size, fsse_alloc)
}

fn encode_btlf_type8_crt(
    name_records: &[(u32, usize)],
    heap_ids: &[[u8; 8]],
    node_size: usize,
) -> Vec<u8> {
    let mut buf = vec![0u8; node_size];
    buf[0..4].copy_from_slice(b"BTLF");
    buf[4] = 0;
    buf[5] = 8; // type 8

    let mut off = 6;
    for &(hash, idx) in name_records {
        buf[off..off + 8].copy_from_slice(&heap_ids[idx]);
        off += 8;
        buf[off] = 0x00; // msg_flags
        off += 1;
        buf[off..off + 4].copy_from_slice(&(idx as u32).to_le_bytes()); // creation_order = index
        off += 4;
        buf[off..off + 4].copy_from_slice(&hash.to_le_bytes());
        off += 4;
    }

    let cksum = crate::checksum::lookup3(&buf[..off]);
    buf[off..off + 4].copy_from_slice(&cksum.to_le_bytes());
    buf
}

fn encode_btlf_type9(heap_ids: &[[u8; 8]], node_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; node_size];
    buf[0..4].copy_from_slice(b"BTLF");
    buf[4] = 0;
    buf[5] = 9; // type 9

    let mut off = 6;
    for (i, hid) in heap_ids.iter().enumerate() {
        buf[off..off + 8].copy_from_slice(hid);
        off += 8;
        buf[off] = 0x00; // msg_flags
        off += 1;
        buf[off..off + 4].copy_from_slice(&(i as u32).to_le_bytes()); // creation_order
        off += 4;
    }

    let cksum = crate::checksum::lookup3(&buf[..off]);
    buf[off..off + 4].copy_from_slice(&cksum.to_le_bytes());
    buf
}

fn limit_enc_size_u64(val: u64) -> usize {
    if val <= 0xFF {
        1
    } else if val <= 0xFFFF {
        2
    } else if val <= 0xFFFFFFFF {
        4
    } else {
        8
    }
}
