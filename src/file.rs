use crate::btree2::{self, BTree2Header};
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, Result};
use crate::filters::FilterPipeline;
use crate::fractal_heap::{self, FractalHeapHeader};
use crate::io::ReadAt;
use crate::layout::DataLayout;
use crate::link::{Link, LinkTarget};
use crate::object_header::messages::MessageType;
use crate::object_header::ObjectHeader;
use crate::superblock::Superblock;

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 files. It holds a reference
/// to the underlying reader and the parsed superblock.
pub struct File<R: ReadAt + ?Sized> {
    reader: Box<R>,
    superblock: Superblock,
}

impl File<[u8]> {
    /// Open an HDF5 file from an in-memory byte buffer.
    pub fn from_bytes(data: Box<[u8]>) -> Result<File<[u8]>> {
        let superblock = Superblock::parse(&*data, 0)?;
        Ok(File {
            reader: data,
            superblock,
        })
    }
}

impl File<std::fs::File> {
    /// Open an HDF5 file from a filesystem path.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<File<std::fs::File>> {
        let f = std::fs::File::open(path).map_err(Error::Io)?;
        let superblock = Superblock::parse(&f, 0)?;
        Ok(File {
            reader: Box::new(f),
            superblock,
        })
    }
}

impl<R: ReadAt + ?Sized> File<R> {
    /// The parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the root group.
    pub fn root_group(&self) -> Result<Group<'_, R>> {
        let addr = self.superblock.root_group_object_header_address;
        let header = ObjectHeader::parse(
            &*self.reader,
            addr,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )?;
        Ok(Group {
            file: self,
            address: addr,
            header,
        })
    }

    /// Open a path like `"/group1/subgroup/dataset"`.
    ///
    /// Returns a `Node` which can be either a `Group` or a `Dataset`.
    pub fn open_path(&self, path: &str) -> Result<Node<'_, R>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        let root = self.root_group()?;

        if parts.is_empty() {
            return Ok(Node::Group(root));
        }

        let mut current_group = root;

        for (i, part) in parts.iter().enumerate() {
            let is_last = i == parts.len() - 1;
            let link = current_group.find_link(part)?;

            match link.target {
                LinkTarget::Hard { address } => {
                    let header = ObjectHeader::parse(
                        &*self.reader,
                        address,
                        self.superblock.size_of_offsets,
                        self.superblock.size_of_lengths,
                    )?;

                    if is_last {
                        // Determine if this is a group or dataset
                        let has_layout = header
                            .messages
                            .iter()
                            .any(|m| m.msg_type == MessageType::DataLayout);
                        if has_layout {
                            return Ok(Node::Dataset(Dataset {
                                file: self,
                                address,
                                header,
                            }));
                        } else {
                            return Ok(Node::Group(Group {
                                file: self,
                                address,
                                header,
                            }));
                        }
                    } else {
                        current_group = Group {
                            file: self,
                            address,
                            header,
                        };
                    }
                }
                LinkTarget::Soft { ref path } => {
                    // Resolve soft link by re-opening from root
                    let resolved = self.open_path(path)?;
                    if is_last {
                        return Ok(resolved);
                    }
                    match resolved {
                        Node::Group(g) => current_group = g,
                        Node::Dataset(_) => {
                            return Err(Error::NotAGroup {
                                path: part.to_string(),
                            })
                        }
                    }
                }
                LinkTarget::External { .. } => {
                    return Err(Error::Other {
                        msg: "external links not supported".into(),
                    });
                }
            }
        }

        Ok(Node::Group(current_group))
    }

    fn size_of_offsets(&self) -> u8 {
        self.superblock.size_of_offsets
    }

    fn size_of_lengths(&self) -> u8 {
        self.superblock.size_of_lengths
    }
}

/// A node in the HDF5 hierarchy — either a group or a dataset.
pub enum Node<'a, R: ReadAt + ?Sized> {
    Group(Group<'a, R>),
    Dataset(Dataset<'a, R>),
}

/// A group (directory-like container) in the HDF5 file.
pub struct Group<'a, R: ReadAt + ?Sized> {
    file: &'a File<R>,
    address: u64,
    header: ObjectHeader,
}

impl<'a, R: ReadAt + ?Sized> Group<'a, R> {
    /// List all child link names.
    pub fn members(&self) -> Result<Vec<String>> {
        let links = self.read_links()?;
        Ok(links.into_iter().map(|l| l.name).collect())
    }

    /// Get a specific child by name, returning the link.
    pub fn find_link(&self, name: &str) -> Result<Link> {
        let links = self.read_links()?;
        links
            .into_iter()
            .find(|l| l.name == name)
            .ok_or_else(|| Error::PathNotFound {
                path: name.to_string(),
            })
    }

    /// Open a child group by name.
    pub fn group(&self, name: &str) -> Result<Group<'a, R>> {
        let link = self.find_link(name)?;
        match link.target {
            LinkTarget::Hard { address } => {
                let header = ObjectHeader::parse(
                    &*self.file.reader,
                    address,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                )?;
                Ok(Group {
                    file: self.file,
                    address,
                    header,
                })
            }
            _ => Err(Error::NotAGroup {
                path: name.to_string(),
            }),
        }
    }

    /// Open a child dataset by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset<'a, R>> {
        let link = self.find_link(name)?;
        match link.target {
            LinkTarget::Hard { address } => {
                let header = ObjectHeader::parse(
                    &*self.file.reader,
                    address,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                )?;
                Ok(Dataset {
                    file: self.file,
                    address,
                    header,
                })
            }
            _ => Err(Error::NotADataset {
                path: name.to_string(),
            }),
        }
    }

    /// Read all attributes on this group.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        parse_attributes(&self.header, self.file)
    }

    /// Read all links from this group's object header.
    ///
    /// Links can be stored in two ways:
    /// 1. Directly as Link messages (0x0006) in the object header (compact storage).
    /// 2. In a fractal heap + B-tree v2, referenced by a Link Info message (0x0002).
    fn read_links(&self) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();

        // First check for direct Link messages
        let mut links: Vec<Link> = Vec::new();
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Link {
                links.push(Link::parse(&msg.data, so)?);
            }
        }
        if !links.is_empty() {
            return Ok(links);
        }

        // Otherwise, look for Link Info message → fractal heap + B-tree v2
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::LinkInfo {
                return self.read_links_from_link_info(&msg.data);
            }
        }

        Ok(links) // empty — no links
    }

    /// Parse a Link Info message and read links from fractal heap.
    fn read_links_from_link_info(&self, data: &[u8]) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();
        let sl = self.file.size_of_lengths();
        let o = so as usize;

        if data.len() < 2 {
            return Err(Error::InvalidObjectHeader {
                msg: "link info message too short".into(),
            });
        }

        let _version = data[0];
        let flags = data[1];
        let mut pos = 2;

        // Optional: max creation order (8 bytes, if flags bit 0 set)
        if (flags & 0x01) != 0 {
            pos += 8;
        }

        // Fractal heap address
        if pos + o > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link info: truncated at fractal heap address".into(),
            });
        }
        let fheap_addr = read_offset_from_slice(data, pos, so);
        pos += o;

        // B-tree v2 address (name index)
        if pos + o > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link info: truncated at B-tree address".into(),
            });
        }
        let bt2_addr = read_offset_from_slice(data, pos, so);

        if fheap_addr == u64::MAX || bt2_addr == u64::MAX {
            return Ok(Vec::new()); // no links stored yet
        }

        // Parse the fractal heap header
        let fheap = FractalHeapHeader::parse(&*self.file.reader, fheap_addr, so, sl)?;

        // Parse the B-tree v2 header
        let bt2 = BTree2Header::parse(&*self.file.reader, bt2_addr, so, sl)?;

        // Iterate B-tree records → each contains a heap ID → look up in fractal heap
        let mut links = Vec::new();
        let heap_id_len = fheap.heap_id_length as usize;

        btree2::iterate_records(&*self.file.reader, &bt2, so, |record| {
            // Type 5 record: hash (4 bytes) + heap_id (heap_id_len bytes)
            if let Some((_hash, heap_id)) =
                btree2::parse_link_name_record(&record.data, heap_id_len)
            {
                // Read the link message from the fractal heap
                let link_data = fractal_heap::read_managed_object(
                    &*self.file.reader,
                    &fheap,
                    &heap_id,
                    so,
                    sl,
                )?;
                let link = Link::parse(&link_data, so)?;
                links.push(link);
            }
            Ok(())
        })?;

        Ok(links)
    }
}

/// A dataset in the HDF5 file.
pub struct Dataset<'a, R: ReadAt + ?Sized> {
    file: &'a File<R>,
    address: u64,
    header: ObjectHeader,
}

impl<'a, R: ReadAt + ?Sized> Dataset<'a, R> {
    /// The dataset's datatype.
    pub fn datatype(&self) -> Result<Datatype> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Datatype {
                return Datatype::parse(&msg.data);
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no datatype message".into(),
        })
    }

    /// The dataset's dataspace (shape information).
    pub fn dataspace(&self) -> Result<Dataspace> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Dataspace {
                return Dataspace::parse(&msg.data);
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no dataspace message".into(),
        })
    }

    /// The dataset's shape (convenience wrapper).
    pub fn shape(&self) -> Result<Vec<u64>> {
        Ok(self.dataspace()?.shape().to_vec())
    }

    /// The data layout (contiguous, chunked, compact).
    pub fn layout(&self) -> Result<DataLayout> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::DataLayout {
                return DataLayout::parse(
                    &msg.data,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                );
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no layout message".into(),
        })
    }

    /// The filter pipeline, if any.
    pub fn filters(&self) -> Result<Option<FilterPipeline>> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::FilterPipeline {
                return Ok(Some(FilterPipeline::parse(&msg.data)?));
            }
        }
        Ok(None)
    }

    /// The fill value for this dataset, if defined.
    pub fn fill_value(&self) -> Result<FillValue> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::FillValue {
                return FillValue::parse(&msg.data);
            }
        }
        Ok(FillValue {
            defined: false,
            value: None,
        })
    }

    /// Read all attributes on this dataset.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        parse_attributes(&self.header, self.file)
    }

    /// Read the entire dataset as raw bytes.
    ///
    /// Returns the uncompressed, un-filtered data. The caller is responsible
    /// for interpreting the bytes according to `datatype()`.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        let layout = self.layout()?;
        let filters = self.filters()?;

        match layout {
            DataLayout::Compact { data } => {
                if let Some(pipeline) = filters {
                    pipeline.decompress(data)
                } else {
                    Ok(data)
                }
            }
            DataLayout::Contiguous { address, size } => {
                if address == u64::MAX {
                    // No data allocated — return fill value or zeros
                    let dtype = self.datatype()?;
                    let dspace = self.dataspace()?;
                    let total_size = dspace.num_elements() * dtype.element_size() as u64;
                    return Ok(vec![0u8; total_size as usize]);
                }
                let mut data = vec![0u8; size as usize];
                self.file
                    .reader
                    .read_exact_at(address, &mut data)
                    .map_err(Error::Io)?;
                if let Some(pipeline) = filters {
                    pipeline.decompress(data)
                } else {
                    Ok(data)
                }
            }
            DataLayout::Chunked { .. } => {
                let dtype = self.datatype()?;
                let dspace = self.dataspace()?;
                let dataset_dims = dspace.shape();
                let element_size = dtype.element_size();
                let max_dims = dspace.max_dimensions();
                crate::chunk::read_chunked(
                    &*self.file.reader,
                    &layout,
                    dataset_dims,
                    element_size,
                    &filters,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                    max_dims,
                )
            }
            DataLayout::Virtual { .. } => Err(Error::Other {
                msg: "virtual dataset reading not supported".into(),
            }),
        }
    }

    /// Read a variable-length dataset, resolving global heap references.
    ///
    /// Returns one `Vec<u8>` per element, containing the resolved vlen data.
    /// For vlen strings, each entry is the raw string bytes (use
    /// `read_vlen_strings()` for convenience).
    pub fn read_vlen(&self) -> Result<Vec<Vec<u8>>> {
        let raw = self.read_raw()?;
        let dspace = self.dataspace()?;
        let num_elements = dspace.num_elements() as usize;

        crate::global_heap::resolve_vlen_elements(
            &*self.file.reader,
            &raw,
            num_elements,
            self.file.size_of_offsets(),
            self.file.size_of_lengths(),
        )
    }

    /// Read a variable-length string dataset, returning Rust strings.
    ///
    /// Each element is converted from raw bytes to a `String`, stripping
    /// any null padding.
    pub fn read_vlen_strings(&self) -> Result<Vec<String>> {
        let vlen_data = self.read_vlen()?;
        Ok(vlen_data
            .into_iter()
            .map(|bytes| {
                String::from_utf8_lossy(&bytes)
                    .trim_end_matches('\0')
                    .to_string()
            })
            .collect())
    }
}

/// A parsed fill value from a fill value message (type 0x0005).
#[derive(Debug, Clone)]
pub struct FillValue {
    pub defined: bool,
    pub value: Option<Vec<u8>>,
}

impl FillValue {
    /// Parse a fill value message body.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Ok(FillValue {
                defined: false,
                value: None,
            });
        }
        let version = data[0];
        match version {
            1 | 2 => Self::parse_v1v2(data, version),
            3 => Self::parse_v3(data),
            _ => Err(Error::InvalidObjectHeader {
                msg: format!("unsupported fill value version {}", version),
            }),
        }
    }

    fn parse_v1v2(data: &[u8], version: u8) -> Result<Self> {
        // v1/v2: version(1) + space_alloc_time(1) + fill_write_time(1) + fill_defined(1)
        if data.len() < 4 {
            return Ok(FillValue {
                defined: false,
                value: None,
            });
        }
        let fill_defined = data[3];
        // v2: fill_defined == 2 means user-defined value follows
        // v1: value always follows (fill_defined field doesn't exist, size follows at byte 4)
        if version == 2 && fill_defined != 2 {
            return Ok(FillValue {
                defined: fill_defined == 1,
                value: None,
            });
        }
        if data.len() < 8 {
            return Ok(FillValue {
                defined: fill_defined != 0,
                value: None,
            });
        }
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if size == 0 || data.len() < 8 + size {
            return Ok(FillValue {
                defined: true,
                value: None,
            });
        }
        Ok(FillValue {
            defined: true,
            value: Some(data[8..8 + size].to_vec()),
        })
    }

    fn parse_v3(data: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1)
        // flags bits 0-1: space alloc time
        // flags bits 2-3: fill write time
        // flags bit 4: fill value undefined
        // flags bit 5: fill value defined
        if data.len() < 2 {
            return Ok(FillValue {
                defined: false,
                value: None,
            });
        }
        let flags = data[1];
        let undefined = (flags & 0x10) != 0;
        let defined = (flags & 0x20) != 0;
        if undefined || !defined {
            return Ok(FillValue {
                defined: false,
                value: None,
            });
        }
        if data.len() < 6 {
            return Ok(FillValue {
                defined: true,
                value: None,
            });
        }
        let size = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
        if size == 0 || data.len() < 6 + size {
            return Ok(FillValue {
                defined: true,
                value: None,
            });
        }
        Ok(FillValue {
            defined: true,
            value: Some(data[6..6 + size].to_vec()),
        })
    }
}

/// An attribute (name + value) on a group or dataset.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub datatype: Datatype,
    pub dataspace: Dataspace,
    pub raw_value: Vec<u8>,
}

/// Parse attribute messages from an object header.
fn parse_attributes<R: ReadAt + ?Sized>(
    header: &ObjectHeader,
    file: &File<R>,
) -> Result<Vec<Attribute>> {
    let mut attrs = Vec::new();

    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            if let Ok(attr) = parse_attribute_message(&msg.data) {
                attrs.push(attr);
            }
        }
    }

    if !attrs.is_empty() {
        return Ok(attrs);
    }

    // Look for AttributeInfo message → dense attribute storage via fractal heap + B-tree v2
    for msg in &header.messages {
        if msg.msg_type == MessageType::AttributeInfo {
            return parse_dense_attributes(&msg.data, file);
        }
    }

    Ok(attrs)
}

/// Parse an Attribute Info message and read attributes from fractal heap + B-tree v2.
///
/// Attribute Info message layout:
/// ```text
/// Byte 0:    Version (0)
/// Byte 1:    Flags (bit 0: max creation order present, bit 1: creation order index present)
/// [if bit 0]: Max creation order (2 bytes)
/// Fractal heap address (size_of_offsets)
/// Attribute Name B-tree v2 address (size_of_offsets)
/// [if bit 1]: Creation Order B-tree v2 address (size_of_offsets)
/// ```
fn parse_dense_attributes<R: ReadAt + ?Sized>(
    data: &[u8],
    file: &File<R>,
) -> Result<Vec<Attribute>> {
    let so = file.size_of_offsets();
    let sl = file.size_of_lengths();
    let o = so as usize;

    if data.len() < 2 {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute info message too short".into(),
        });
    }

    let _version = data[0];
    let flags = data[1];
    let mut pos = 2;

    // Optional max creation order
    if (flags & 0x01) != 0 {
        pos += 2;
    }

    if pos + o > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute info: truncated at fractal heap address".into(),
        });
    }
    let fheap_addr = read_offset_from_slice(data, pos, so);
    pos += o;

    if pos + o > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute info: truncated at B-tree address".into(),
        });
    }
    let bt2_name_addr = read_offset_from_slice(data, pos, so);

    if fheap_addr == u64::MAX || bt2_name_addr == u64::MAX {
        return Ok(Vec::new());
    }

    // Parse the fractal heap
    let fheap = FractalHeapHeader::parse(&*file.reader, fheap_addr, so, sl)?;

    // Parse the B-tree v2 (type 8: attribute name index)
    let bt2 = BTree2Header::parse(&*file.reader, bt2_name_addr, so, sl)?;

    let mut attrs = Vec::new();
    let heap_id_len = fheap.heap_id_length as usize;

    btree2::iterate_records(&*file.reader, &bt2, so, |record| {
        if let Some(heap_id) = btree2::parse_attribute_name_record(&record.data, heap_id_len) {
            let attr_data = fractal_heap::read_managed_object(
                &*file.reader,
                &fheap,
                &heap_id,
                so,
                sl,
            )?;
            if let Ok(attr) = parse_attribute_message(&attr_data) {
                attrs.push(attr);
            }
        }
        Ok(())
    })?;

    Ok(attrs)
}

/// Parse an attribute message body.
///
/// Attribute message layout (version 3):
/// ```text
/// Byte 0:    Version (1, 2, or 3)
/// Byte 1:    Flags (bit 0: datatype shared, bit 1: dataspace shared)
/// Byte 2-3:  Name size (u16)
/// Byte 4-5:  Datatype size (u16)
/// Byte 6-7:  Dataspace size (u16)
/// Byte 8:    Character set (version 3 only: 0=ASCII, 1=UTF-8)
/// Name (null-terminated, NOT padded in version 3)
/// Datatype message
/// Dataspace message
/// Value
/// ```
fn parse_attribute_message(data: &[u8]) -> Result<Attribute> {
    if data.len() < 6 {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute message too short".into(),
        });
    }

    let version = data[0];
    let _flags = data[1];
    let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
    let dt_size = u16::from_le_bytes([data[4], data[5]]) as usize;
    let ds_size = u16::from_le_bytes([data[6], data[7]]) as usize;

    let mut pos = match version {
        1 | 2 => 8,
        3 => 9, // extra charset byte
        _ => {
            return Err(Error::InvalidObjectHeader {
                msg: format!("unsupported attribute message version {}", version),
            })
        }
    };

    // Name
    if pos + name_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute name truncated".into(),
        });
    }
    let name = String::from_utf8_lossy(&data[pos..pos + name_size])
        .trim_end_matches('\0')
        .to_string();
    pos += name_size;

    // Version 1 pads name, datatype, dataspace to 8-byte boundaries
    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Datatype
    if pos + dt_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute datatype truncated".into(),
        });
    }
    let datatype = Datatype::parse(&data[pos..pos + dt_size])?;
    pos += dt_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Dataspace
    if pos + ds_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute dataspace truncated".into(),
        });
    }
    let dataspace = Dataspace::parse(&data[pos..pos + ds_size])?;
    pos += ds_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Value (remaining bytes)
    let raw_value = data[pos..].to_vec();

    Ok(Attribute {
        name,
        datatype,
        dataspace,
        raw_value,
    })
}

fn read_offset_from_slice(data: &[u8], offset: usize, size: u8) -> u64 {
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
