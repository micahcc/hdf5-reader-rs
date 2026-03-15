use crate::checksum;
use crate::datatype::ByteOrder;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::datatype::StringPadding;
use crate::error::Error;
use crate::error::Result;
use crate::filters;
use crate::superblock::HDF5_SIGNATURE;
use crate::superblock::UNDEF_ADDR;

use super::file_writer::WriteOptions;
use super::types::{AttrData, ChunkFilter};

pub(crate) const SIZE_OF_OFFSETS: u8 = 8;
pub(crate) const SIZE_OF_LENGTHS: u8 = 8;
pub(crate) const SUPERBLOCK_SIZE: usize = 48;

/// Total object header size given the sum of all message (header+body) bytes.
pub(crate) fn ohdr_overhead(total_msg_bytes: usize, opts: &WriteOptions) -> usize {
    let (prefix_size, _) = chunk_size_encoding(total_msg_bytes, opts);
    prefix_size + total_msg_bytes + 4
}

fn chunk_size_encoding(total_msg_bytes: usize, opts: &WriteOptions) -> (usize, u8) {
    let ts_extra = if opts.timestamps.is_some() { 16 } else { 0 };
    let base_flags: u8 = if opts.timestamps.is_some() {
        0x20
    } else {
        0
    };
    if total_msg_bytes <= 0xFF {
        (7 + ts_extra, base_flags)
    } else if total_msg_bytes <= 0xFFFF {
        (8 + ts_extra, base_flags | 0x01)
    } else {
        (10 + ts_extra, base_flags | 0x02)
    }
}

pub(crate) fn encode_object_header(
    messages: &[(u8, Vec<u8>)],
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    let total_msg_bytes: usize = messages.iter().map(|(_, b)| 4 + b.len()).sum();
    let (prefix_size, flags) = chunk_size_encoding(total_msg_bytes, opts);

    let mut buf = Vec::with_capacity(prefix_size + total_msg_bytes + 4);

    buf.extend_from_slice(b"OHDR");
    buf.push(2);
    buf.push(flags);

    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    match flags & 0x03 {
        0x00 => buf.push(total_msg_bytes as u8),
        0x01 => buf.extend_from_slice(&(total_msg_bytes as u16).to_le_bytes()),
        0x02 => buf.extend_from_slice(&(total_msg_bytes as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    for (type_id, body) in messages {
        buf.push(*type_id);
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(0);
        buf.extend_from_slice(body);
    }

    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    Ok(buf)
}

pub(crate) fn encode_superblock(root_group_addr: u64, eof: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(SUPERBLOCK_SIZE);
    buf.extend_from_slice(&HDF5_SIGNATURE);
    buf.push(2);
    buf.push(SIZE_OF_OFFSETS);
    buf.push(SIZE_OF_LENGTHS);
    buf.push(0);
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    buf.extend_from_slice(&eof.to_le_bytes());
    buf.extend_from_slice(&root_group_addr.to_le_bytes());
    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());
    debug_assert_eq!(buf.len(), SUPERBLOCK_SIZE);
    buf
}

pub(crate) fn encode_datatype(dt: &Datatype) -> Result<Vec<u8>> {
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            let mut buf = Vec::with_capacity(12);
            buf.push(0x10);
            let mut f0 = 0u8;
            if *byte_order == ByteOrder::BigEndian {
                f0 |= 0x01;
            }
            if *signed {
                f0 |= 0x08;
            }
            buf.push(f0);
            buf.push(0);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&bit_offset.to_le_bytes());
            buf.extend_from_slice(&bit_precision.to_le_bytes());
            Ok(buf)
        }
        Datatype::FloatingPoint {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } => {
            let mut buf = Vec::with_capacity(20);
            buf.push(0x11);
            let mut f0 = 0x20u8;
            if *byte_order == ByteOrder::BigEndian {
                f0 |= 0x01;
            }
            buf.push(f0);
            let sign_bit_pos = exponent_location + exponent_size;
            buf.push(sign_bit_pos);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&bit_offset.to_le_bytes());
            buf.extend_from_slice(&bit_precision.to_le_bytes());
            buf.push(*exponent_location);
            buf.push(*exponent_size);
            buf.push(*mantissa_location);
            buf.push(*mantissa_size);
            buf.extend_from_slice(&exponent_bias.to_le_bytes());
            Ok(buf)
        }
        Datatype::String {
            size,
            padding,
            char_set,
        } => {
            let mut buf = Vec::with_capacity(8);
            buf.push(0x13);
            let pad = match padding {
                StringPadding::NullTerminate => 0u8,
                StringPadding::NullPad => 1,
                StringPadding::SpacePad => 2,
            };
            let cs = match char_set {
                CharacterSet::Ascii => 0u8,
                CharacterSet::Utf8 => 1,
            };
            buf.push(pad | (cs << 4));
            buf.push(0);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            Ok(buf)
        }
        Datatype::Compound { size, members } => {
            let nmembers = members.len() as u16;
            let class_version_byte = (6u8) | (3u8 << 4);
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.extend_from_slice(&nmembers.to_le_bytes());
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            let off_size = limit_enc_size(*size);
            for m in members {
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
                match off_size {
                    1 => buf.push(m.byte_offset as u8),
                    2 => buf.extend_from_slice(&(m.byte_offset as u16).to_le_bytes()),
                    3 => {
                        buf.push(m.byte_offset as u8);
                        buf.push((m.byte_offset >> 8) as u8);
                        buf.push((m.byte_offset >> 16) as u8);
                    }
                    _ => buf.extend_from_slice(&m.byte_offset.to_le_bytes()),
                }
                buf.extend_from_slice(&encode_datatype(&m.datatype)?);
            }
            Ok(buf)
        }
        Datatype::Enum { base, members } => {
            let nmembers = members.len() as u16;
            let class_version_byte = (8u8) | (3u8 << 4);
            let base_size = base.element_size();
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.extend_from_slice(&nmembers.to_le_bytes());
            buf.push(0);
            buf.extend_from_slice(&base_size.to_le_bytes());
            buf.extend_from_slice(&encode_datatype(base)?);
            for m in members {
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
            }
            for m in members {
                buf.extend_from_slice(&m.value);
            }
            Ok(buf)
        }
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            let class_version_byte = (10u8) | (3u8 << 4);
            let ndims = dimensions.len() as u8;
            let total_elements: u32 = dimensions.iter().product();
            let elem_size = total_elements * element_type.element_size();
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.push(0);
            buf.push(0);
            buf.push(0);
            buf.extend_from_slice(&elem_size.to_le_bytes());
            buf.push(ndims);
            for d in dimensions {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&encode_datatype(element_type)?);
            Ok(buf)
        }
        Datatype::Complex { size, base } => {
            let class_version_byte = (11u8) | (5u8 << 4);
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.push(0x01);
            buf.push(0);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&encode_datatype(base)?);
            Ok(buf)
        }
        Datatype::VarLen {
            element_type,
            is_string,
            padding,
            char_set,
        } => {
            let class_version_byte = (9u8) | (4u8 << 4);
            let mut class_bits_lo = 0u8;
            if *is_string {
                class_bits_lo |= 0x01;
                if let Some(p) = padding {
                    let pad_val = match p {
                        StringPadding::NullTerminate => 0u8,
                        StringPadding::NullPad => 1,
                        StringPadding::SpacePad => 2,
                    };
                    class_bits_lo |= pad_val << 4;
                }
            }
            let mut class_bits_hi = 0u8;
            if *is_string {
                if let Some(cs) = char_set {
                    class_bits_hi = match cs {
                        CharacterSet::Ascii => 0,
                        CharacterSet::Utf8 => 1,
                    };
                }
            }
            let vlen_element_size: u32 = 4 + SIZE_OF_OFFSETS as u32 + 4;
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.push(class_bits_lo);
            buf.push(class_bits_hi);
            buf.push(0);
            buf.extend_from_slice(&vlen_element_size.to_le_bytes());
            buf.extend_from_slice(&encode_datatype(element_type)?);
            Ok(buf)
        }
        _ => Err(Error::Other {
            msg: format!(
                "encoding not yet supported for datatype: {:?}",
                std::mem::discriminant(dt)
            ),
        }),
    }
}

fn limit_enc_size(size: u32) -> usize {
    if size <= 0xFF {
        1
    } else if size <= 0xFFFF {
        2
    } else if size <= 0xFFFFFF {
        3
    } else {
        4
    }
}

pub(crate) fn encode_dataspace(shape: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    if shape.is_empty() {
        return vec![2, 0, 0, 0];
    }
    let has_max = max_dims.is_some();
    let mut buf = Vec::with_capacity(4 + shape.len() * 8 * if has_max { 2 } else { 1 });
    buf.push(2);
    buf.push(shape.len() as u8);
    buf.push(if has_max { 0x01 } else { 0x00 });
    buf.push(1);
    for &dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    if let Some(md) = max_dims {
        for &dim in md {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
    }
    buf
}

pub(crate) fn encode_link_info() -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    buf
}

pub(crate) fn encode_group_info() -> Vec<u8> {
    vec![0, 0]
}

pub(crate) fn encode_link(name: &str, target_addr: u64) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    let (name_enc_bits, name_enc_size) = if name_len <= 0xFF {
        (0u8, 1usize)
    } else if name_len <= 0xFFFF {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };

    let mut buf = Vec::with_capacity(2 + name_enc_size + name_len + 8);
    buf.push(1);
    buf.push(name_enc_bits);

    match name_enc_size {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&target_addr.to_le_bytes());

    buf
}

pub(crate) fn encode_fill_value_msg(fill_data: &Option<Vec<u8>>) -> Vec<u8> {
    match fill_data {
        Some(data) => {
            let mut buf = Vec::with_capacity(6 + data.len());
            buf.push(3);
            buf.push(0x29);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
            buf
        }
        None => {
            vec![3, 0x09]
        }
    }
}

pub(crate) fn encode_contiguous_layout(data_addr: u64, data_size: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(3);
    buf.push(1);
    buf.extend_from_slice(&data_addr.to_le_bytes());
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf
}

pub(crate) fn encode_compact_layout(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + data.len());
    buf.push(3);
    buf.push(0);
    buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
    buf.extend_from_slice(data);
    buf
}

pub(crate) fn encode_chunked_layout(
    chunk_dims: &[u64],
    index_type_id: u8,
    chunk_index_addr: u64,
    single_filtered_size: Option<u64>,
) -> Vec<u8> {
    let ndims = chunk_dims.len() as u8;

    let max_dim = chunk_dims.iter().copied().max().unwrap_or(1);
    let enc_bytes = if max_dim <= 0xFF {
        1u8
    } else if max_dim <= 0xFFFF {
        2
    } else if max_dim <= 0xFFFFFFFF {
        4
    } else {
        8
    };

    let mut flags = 0u8;
    if single_filtered_size.is_some() {
        flags |= 0x02;
    }

    let mut buf = Vec::new();
    buf.push(4);
    buf.push(2);
    buf.push(flags);
    buf.push(ndims);
    buf.push(enc_bytes);

    for &dim in chunk_dims {
        match enc_bytes {
            1 => buf.push(dim as u8),
            2 => buf.extend_from_slice(&(dim as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(dim as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&dim.to_le_bytes()),
            _ => unreachable!(),
        }
    }

    buf.push(index_type_id);

    match index_type_id {
        1 => {
            if let Some(filtered_size) = single_filtered_size {
                buf.extend_from_slice(&filtered_size.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        2 => {}
        3 => {
            buf.push(0);
        }
        4 => {
            buf.push(32);
            buf.push(1);
            buf.push(0);
            buf.push(0);
            buf.push(0);
        }
        5 => {
            buf.extend_from_slice(&4096u32.to_le_bytes());
            buf.push(98);
            buf.push(40);
        }
        _ => {}
    }

    buf.extend_from_slice(&chunk_index_addr.to_le_bytes());

    buf
}

pub(crate) fn encode_filter_pipeline(
    chunk_filters: &[ChunkFilter],
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(2);
    buf.push(chunk_filters.len() as u8);

    for filter in chunk_filters {
        match filter {
            ChunkFilter::Deflate(level) => {
                buf.extend_from_slice(&filters::FILTER_DEFLATE.to_le_bytes());
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&1u16.to_le_bytes());
                buf.extend_from_slice(&level.to_le_bytes());
            }
            ChunkFilter::Shuffle => {
                buf.extend_from_slice(&filters::FILTER_SHUFFLE.to_le_bytes());
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&1u16.to_le_bytes());
                buf.extend_from_slice(&element_size.to_le_bytes());
            }
            ChunkFilter::Fletcher32 => {
                buf.extend_from_slice(&filters::FILTER_FLETCHER32.to_le_bytes());
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&0u16.to_le_bytes());
            }
        }
    }

    buf
}

pub(crate) fn encode_attribute(attr: &AttrData) -> Result<Vec<u8>> {
    let dt_enc = encode_datatype(&attr.datatype)?;
    let ds_enc = encode_dataspace(&attr.shape, None);
    let name_with_nul = attr.name.len() + 1;

    let mut buf = Vec::new();
    buf.push(3);
    buf.push(0);
    buf.extend_from_slice(&(name_with_nul as u16).to_le_bytes());
    buf.extend_from_slice(&(dt_enc.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(ds_enc.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(attr.name.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&dt_enc);
    buf.extend_from_slice(&ds_enc);
    buf.extend_from_slice(&attr.value);

    Ok(buf)
}

pub(crate) fn limit_enc_size_u64(size: u64) -> usize {
    if size <= 0xFF {
        1
    } else if size <= 0xFFFF {
        2
    } else if size <= 0xFFFFFFFF {
        4
    } else {
        8
    }
}
