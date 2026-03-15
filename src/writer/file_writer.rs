use crate::error::{Error, Result};

use crate::writer::encode::{encode_superblock, SUPERBLOCK_SIZE};
use crate::writer::serialize::write_group;
use crate::writer::GroupNode;

/// Options controlling how the HDF5 file is written.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// If set, store these timestamps on every object header.
    /// Tuple: (access_time, modification_time, change_time, birth_time) as Unix seconds.
    pub timestamps: Option<(u32, u32, u32, u32)>,
}

/// Builds an HDF5 file in memory and writes it out.
///
/// # Example
/// ```
/// use hdf5_io::writer::FileWriter;
/// use hdf5_io::Datatype;
///
/// let mut w = FileWriter::new();
/// let data: Vec<u8> = (0..4i32).flat_map(|x| x.to_le_bytes()).collect();
/// w.root_mut().add_dataset("numbers", Datatype::native_i32(), &[4], data);
/// let bytes = w.to_bytes().unwrap();
/// ```
pub struct FileWriter {
    root: GroupNode,
    options: WriteOptions,
}

impl FileWriter {
    pub fn new() -> Self {
        FileWriter {
            root: GroupNode::new(),
            options: WriteOptions::default(),
        }
    }

    /// Create a writer with custom options.
    pub fn with_options(options: WriteOptions) -> Self {
        FileWriter {
            root: GroupNode::new(),
            options,
        }
    }

    pub fn root_mut(&mut self) -> &mut GroupNode {
        &mut self.root
    }

    /// Serialize the entire file to a byte vector.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; SUPERBLOCK_SIZE];
        let root_addr = write_group(&self.root, &mut buf, &self.options)?;
        let eof = buf.len() as u64;
        let sb = encode_superblock(root_addr, eof);
        buf[..SUPERBLOCK_SIZE].copy_from_slice(&sb);
        Ok(buf)
    }

    /// Serialize and write to a file on disk.
    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(Error::Io)
    }
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}
