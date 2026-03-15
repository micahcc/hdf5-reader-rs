use crate::error::{Error, Result};
use crate::object_header::messages::MessageType;
use crate::superblock::UNDEF_ADDR;

use super::chunk_index::{
    write_btree_v1_chunk_index, write_btree_v2_chunk_index, write_extensible_array_index,
    write_fixed_array_index,
};
use super::chunk_util::{compute_chunk_count, enumerate_chunks, extract_chunk_data};
use super::dataset_node::DatasetNode;
use super::encode::{
    SIZE_OF_OFFSETS, encode_attribute, encode_chunked_layout, encode_compact_layout,
    encode_contiguous_layout, encode_dataspace, encode_datatype, encode_fill_value_msg,
    encode_filter_pipeline, encode_group_info, encode_link, encode_link_info,
    encode_object_header, ohdr_overhead,
};
use super::file_writer::WriteOptions;
use super::gcol::{build_global_heap_collection, build_vlen_heap_ids};
use super::group_node::GroupNode;
use super::types::{ChildNode, StorageLayout};
use super::write_filters::apply_filters_forward;

pub(crate) fn write_group(
    group: &GroupNode,
    buf: &mut Vec<u8>,
    opts: &WriteOptions,
) -> Result<u64> {
    let mut child_addrs: Vec<(&str, u64)> = Vec::new();
    for (name, child) in &group.children {
        let addr = match child {
            ChildNode::Group(g) => write_group(g, buf, opts)?,
            ChildNode::Dataset(d) => write_dataset(d, buf, opts)?,
        };
        child_addrs.push((name, addr));
    }

    let ohdr_addr = buf.len() as u64;

    let mut messages: Vec<(u8, Vec<u8>)> = Vec::new();
    messages.push((MessageType::LinkInfo.as_u8(), encode_link_info()));
    messages.push((MessageType::GroupInfo.as_u8(), encode_group_info()));
    for (name, addr) in &child_addrs {
        messages.push((MessageType::Link.as_u8(), encode_link(name, *addr)));
    }
    for attr in &group.attributes {
        messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?));
    }

    let ohdr = encode_object_header(&messages, opts)?;
    buf.extend_from_slice(&ohdr);

    Ok(ohdr_addr)
}

fn write_dataset(ds: &DatasetNode, buf: &mut Vec<u8>, opts: &WriteOptions) -> Result<u64> {
    validate_dataset(ds)?;

    let ohdr_addr = buf.len() as u64;

    match ds.layout {
        StorageLayout::Contiguous => {
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;
            let layout_body_size = 18usize;
            let total_msg_size: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(total_msg_size, opts);

            if let Some(ref vlen_elems) = ds.vlen_elements {
                let gcol_addr = ohdr_addr + ohdr_size as u64;
                let gcol_bytes = build_global_heap_collection(vlen_elems);
                let heap_id_data_addr = gcol_addr + gcol_bytes.len() as u64;
                let heap_id_data = build_vlen_heap_ids(vlen_elems, gcol_addr);

                let layout_body = encode_contiguous_layout(
                    heap_id_data_addr,
                    heap_id_data.len() as u64,
                );
                debug_assert_eq!(layout_body.len(), layout_body_size);

                let mut messages: Vec<(u8, Vec<u8>)> = vec![
                    (MessageType::Datatype.as_u8(), dt_body),
                    (MessageType::Dataspace.as_u8(), ds_body),
                    (MessageType::FillValue.as_u8(), fv_body),
                    (MessageType::DataLayout.as_u8(), layout_body),
                ];
                for body in attr_bodies {
                    messages.push((MessageType::Attribute.as_u8(), body));
                }

                let ohdr = encode_object_header(&messages, opts)?;
                debug_assert_eq!(ohdr.len(), ohdr_size);

                buf.extend_from_slice(&ohdr);
                buf.extend_from_slice(&gcol_bytes);
                buf.extend_from_slice(&heap_id_data);
            } else {
                let data_addr = if ds.data.is_empty() {
                    UNDEF_ADDR
                } else {
                    ohdr_addr + ohdr_size as u64
                };

                let layout_body = encode_contiguous_layout(data_addr, ds.data.len() as u64);
                debug_assert_eq!(layout_body.len(), layout_body_size);

                let mut messages: Vec<(u8, Vec<u8>)> = vec![
                    (MessageType::Datatype.as_u8(), dt_body),
                    (MessageType::Dataspace.as_u8(), ds_body),
                    (MessageType::FillValue.as_u8(), fv_body),
                    (MessageType::DataLayout.as_u8(), layout_body),
                ];
                for body in attr_bodies {
                    messages.push((MessageType::Attribute.as_u8(), body));
                }

                let ohdr = encode_object_header(&messages, opts)?;
                debug_assert_eq!(ohdr.len(), ohdr_size);

                buf.extend_from_slice(&ohdr);
                buf.extend_from_slice(&ds.data);
            }
        }
        StorageLayout::Compact => {
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;
            let layout_body = encode_compact_layout(&ds.data);

            let mut messages: Vec<(u8, Vec<u8>)> = vec![
                (MessageType::Datatype.as_u8(), dt_body),
                (MessageType::Dataspace.as_u8(), ds_body),
                (MessageType::FillValue.as_u8(), fv_body),
                (MessageType::DataLayout.as_u8(), layout_body),
            ];
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body));
            }

            let ohdr = encode_object_header(&messages, opts)?;
            buf.extend_from_slice(&ohdr);
        }
        StorageLayout::Chunked {
            ref chunk_dims,
            ref filters,
        } if ds.layout_version == 3 => {
            let element_size = ds.datatype.element_size();
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;
            let has_filters = !filters.is_empty();
            let filter_body = if has_filters {
                Some(encode_filter_pipeline(filters, element_size))
            } else {
                None
            };

            let chunk_coords_list = enumerate_chunks(&ds.shape, chunk_dims);
            let mut chunk_data_parts: Vec<Vec<u8>> = Vec::new();
            for coords in &chunk_coords_list {
                let mut chunk = extract_chunk_data(
                    &ds.data, &ds.shape, chunk_dims, coords, element_size as usize,
                );
                if has_filters {
                    chunk = apply_filters_forward(filters, chunk, element_size)?;
                }
                chunk_data_parts.push(chunk);
            }
            let chunk_sizes: Vec<u64> = chunk_data_parts.iter().map(|p| p.len() as u64).collect();

            let ndims = ds.shape.len();
            let dimensionality = ndims + 1;
            let layout_body_size = 3 + SIZE_OF_OFFSETS as usize + dimensionality * 4;

            let fixed_msg_sizes: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + filter_body.as_ref().map_or(0, |fb| 4 + fb.len())
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(fixed_msg_sizes, opts);

            let chunk_data_start = ohdr_addr + ohdr_size as u64;
            let mut chunk_addrs: Vec<u64> = Vec::new();
            let mut pos = chunk_data_start;
            for part in &chunk_data_parts {
                chunk_addrs.push(pos);
                pos += part.len() as u64;
            }

            let btree_addr = pos;

            let mut layout_body = Vec::with_capacity(layout_body_size);
            layout_body.push(3);
            layout_body.push(2);
            layout_body.push(dimensionality as u8);
            layout_body.extend_from_slice(&btree_addr.to_le_bytes());
            for d in 0..ndims {
                layout_body.extend_from_slice(&(chunk_dims[d] as u32).to_le_bytes());
            }
            layout_body.extend_from_slice(&element_size.to_le_bytes());
            debug_assert_eq!(layout_body.len(), layout_body_size);

            let mut messages: Vec<(u8, Vec<u8>)> = vec![
                (MessageType::Datatype.as_u8(), dt_body),
                (MessageType::Dataspace.as_u8(), ds_body),
                (MessageType::FillValue.as_u8(), fv_body),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb));
            }
            messages.push((MessageType::DataLayout.as_u8(), layout_body));
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body));
            }

            let ohdr = encode_object_header(&messages, opts)?;
            debug_assert_eq!(ohdr.len(), ohdr_size);
            buf.extend_from_slice(&ohdr);

            for part in &chunk_data_parts {
                buf.extend_from_slice(part);
            }

            write_btree_v1_chunk_index(
                buf,
                &chunk_addrs,
                &chunk_sizes,
                &chunk_coords_list,
                chunk_dims,
                element_size,
                ndims,
            )?;
        }
        StorageLayout::Chunked {
            ref chunk_dims,
            ref filters,
        } => {
            let element_size = ds.datatype.element_size();
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;

            let chunk_elements: u64 = chunk_dims.iter().product();
            let raw_chunk_bytes = chunk_elements * element_size as u64;

            let total_chunks = compute_chunk_count(&ds.shape, chunk_dims);
            let has_filters = !filters.is_empty();

            let filter_body = if has_filters {
                Some(encode_filter_pipeline(filters, element_size))
            } else {
                None
            };

            let n_unlimited = ds
                .max_dims
                .as_ref()
                .map_or(0, |md| md.iter().filter(|&&d| d == u64::MAX).count());
            let index_type_id = if n_unlimited >= 2 {
                5u8
            } else if total_chunks == 1 && n_unlimited == 0 {
                1u8
            } else if n_unlimited == 1 {
                4u8
            } else if !has_filters {
                2u8
            } else {
                3u8
            };

            let mut chunk_data_parts: Vec<Vec<u8>> = Vec::with_capacity(total_chunks);
            let chunk_coords_list = enumerate_chunks(&ds.shape, chunk_dims);

            for coords in &chunk_coords_list {
                let mut chunk = extract_chunk_data(
                    &ds.data,
                    &ds.shape,
                    chunk_dims,
                    coords,
                    element_size as usize,
                );
                if has_filters {
                    chunk = apply_filters_forward(filters, chunk, element_size)?;
                }
                chunk_data_parts.push(chunk);
            }

            let chunk_sizes: Vec<u64> = chunk_data_parts.iter().map(|p| p.len() as u64).collect();

            let dummy_layout = encode_chunked_layout(
                chunk_dims,
                index_type_id,
                0,
                if index_type_id == 1 && has_filters {
                    Some(chunk_sizes[0])
                } else {
                    None
                },
            );
            let layout_body_size = dummy_layout.len();

            let fixed_msg_sizes: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + filter_body.as_ref().map_or(0, |fb| 4 + fb.len())
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(fixed_msg_sizes, opts);

            let chunk_data_start = ohdr_addr + ohdr_size as u64;
            let mut chunk_addrs: Vec<u64> = Vec::new();
            let mut pos = chunk_data_start;
            for part in &chunk_data_parts {
                chunk_addrs.push(pos);
                pos += part.len() as u64;
            }

            let (chunk_index_addr, single_filtered_size) = match index_type_id {
                1 => (
                    chunk_addrs[0],
                    if has_filters {
                        Some(chunk_sizes[0])
                    } else {
                        None
                    },
                ),
                2 => (chunk_data_start, None),
                3 | 4 | 5 => (pos, None),
                _ => (pos, None),
            };

            let layout_body = encode_chunked_layout(
                chunk_dims,
                index_type_id,
                chunk_index_addr,
                single_filtered_size,
            );
            debug_assert_eq!(layout_body.len(), layout_body_size);

            let mut messages: Vec<(u8, Vec<u8>)> = vec![
                (MessageType::Datatype.as_u8(), dt_body),
                (MessageType::Dataspace.as_u8(), ds_body),
                (MessageType::FillValue.as_u8(), fv_body),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb));
            }
            messages.push((MessageType::DataLayout.as_u8(), layout_body));
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body));
            }

            let ohdr = encode_object_header(&messages, opts)?;
            debug_assert_eq!(ohdr.len(), ohdr_size);
            buf.extend_from_slice(&ohdr);

            for part in &chunk_data_parts {
                buf.extend_from_slice(part);
            }

            match index_type_id {
                3 => {
                    write_fixed_array_index(
                        buf,
                        &chunk_addrs,
                        &chunk_sizes,
                        has_filters,
                        raw_chunk_bytes,
                    )?;
                }
                4 => {
                    write_extensible_array_index(
                        buf,
                        &chunk_addrs,
                        &chunk_sizes,
                        has_filters,
                        total_chunks,
                    )?;
                }
                5 => {
                    write_btree_v2_chunk_index(
                        buf,
                        &chunk_addrs,
                        &chunk_sizes,
                        has_filters,
                        &ds.shape,
                        chunk_dims,
                        ds.datatype.element_size(),
                    )?;
                }
                _ => {}
            }
        }
    }

    Ok(ohdr_addr)
}

fn validate_dataset(ds: &DatasetNode) -> Result<()> {
    if ds.vlen_elements.is_some() {
        return Ok(());
    }
    let num_elements: u64 = if ds.shape.is_empty() {
        1
    } else {
        ds.shape.iter().product()
    };
    let expected = num_elements * ds.datatype.element_size() as u64;
    if ds.data.len() as u64 != expected {
        return Err(Error::Other {
            msg: format!(
                "dataset data size mismatch: expected {} bytes (shape {:?}, element_size {}), got {}",
                expected,
                ds.shape,
                ds.datatype.element_size(),
                ds.data.len()
            ),
        });
    }
    Ok(())
}
