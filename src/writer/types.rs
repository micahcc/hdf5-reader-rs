use crate::datatype::Datatype;
use crate::writer::DatasetNode;
use crate::writer::GroupNode;

/// A filter to apply in a chunked dataset's filter pipeline.
#[derive(Debug, Clone)]
pub enum ChunkFilter {
    /// Deflate (zlib) compression with a given level (0-9).
    Deflate(u32),
    /// Shuffle filter — reorders bytes for better compression.
    Shuffle,
    /// Fletcher32 checksum appended to each chunk.
    Fletcher32,
    /// Scale-offset filter for integer/float compression.
    ///
    /// Stores the 20 cd_values used by HDF5's scaleoffset filter.
    /// Use [`ScaleOffsetParams::from_int`] to construct.
    ScaleOffset(ScaleOffsetParams),
    /// N-bit filter — packs only the significant bits of each element.
    ///
    /// Use [`NbitParams::from_atomic`] to construct for simple integer/float types.
    Nbit(NbitParams),
    /// LZF compression (third-party filter, registered by h5py, filter ID 32000).
    Lzf,
}

/// Parameters for the scale-offset filter, computed from the datatype.
#[derive(Debug, Clone)]
pub struct ScaleOffsetParams {
    /// The 20 cd_values for the filter pipeline.
    pub(crate) cd_values: [u32; 20],
}

impl ScaleOffsetParams {
    /// Build scale-offset parameters for an integer datatype.
    ///
    /// `num_elements` is the total number of elements per chunk.
    /// `dtype_size` is the element size in bytes (e.g. 4 for i32).
    /// `signed` indicates whether the integer type is signed.
    /// `little_endian` indicates whether the byte order is LE.
    pub fn from_int(num_elements: u32, dtype_size: u32, signed: bool, little_endian: bool) -> Self {
        let mut cd = [0u32; 20];
        cd[0] = 2; // H5Z_SO_INT
        cd[1] = 0; // scale_factor (minbits_default)
        cd[2] = num_elements;
        cd[3] = 0; // class = integer
        cd[4] = dtype_size;
        cd[5] = if signed { 1 } else { 0 }; // sign: 0=unsigned, 1=signed (2's complement)
        cd[6] = if little_endian { 0 } else { 1 }; // order: 0=LE, 1=BE
        cd[7] = 1; // filavail = fill_defined (default fill = 0)
        // cd[8..19] = fill value bytes (0 for default)
        ScaleOffsetParams { cd_values: cd }
    }
}

/// Parameters for the N-bit filter, computed from the datatype.
#[derive(Debug, Clone)]
pub struct NbitParams {
    /// The cd_values for the filter pipeline message.
    pub(crate) cd_values: Vec<u32>,
}

impl NbitParams {
    /// Build N-bit parameters for a simple atomic (integer/float) type.
    ///
    /// `num_elements` is the total number of elements per chunk.
    /// `dtype_size` is the element size in bytes (e.g. 2 for uint16).
    /// `precision` is the number of significant bits.
    /// `bit_offset` is the bit offset within the element.
    /// `little_endian` indicates the byte order.
    pub fn from_atomic(
        num_elements: u32,
        dtype_size: u32,
        precision: u32,
        bit_offset: u32,
        little_endian: bool,
    ) -> Self {
        let order = if little_endian { 0u32 } else { 1 };
        // cd_values layout:
        // [0] = total parms size (parms count from cd[3] onward = 5)
        // [1] = need_not_compress (0)
        // [2] = d_nelmts
        // [3] = NBIT_ATOMIC (1)
        // [4] = size
        // [5] = order
        // [6] = precision
        // [7] = offset
        let cd = vec![
            8,
            0,
            num_elements,
            1,
            dtype_size,
            order,
            precision,
            bit_offset,
        ];
        NbitParams { cd_values: cd }
    }
}

/// Storage layout for a dataset.
#[derive(Debug, Clone, Default)]
pub enum StorageLayout {
    /// Data stored in a contiguous block after the object header.
    #[default]
    Contiguous,
    /// Data stored inline in the object header (small datasets only).
    Compact,
    /// Data stored in fixed-size chunks.
    Chunked {
        chunk_dims: Vec<u64>,
        filters: Vec<ChunkFilter>,
    },
}

/// Space allocation time for dataset storage.
///
/// Controls when raw data storage is allocated in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceAllocTime {
    /// Allocate immediately when the dataset is created (compact datasets).
    Early = 1,
    /// Allocate when data is first written (contiguous datasets).
    Late = 2,
    /// Allocate incrementally as chunks are written (chunked datasets).
    Incremental = 3,
}

pub(crate) enum ChildNode {
    Group(GroupNode),
    Dataset(DatasetNode),
    CommittedDatatype(Datatype),
}

pub(crate) struct AttrData {
    pub name: String,
    pub datatype: Datatype,
    pub shape: Vec<u64>,
    pub value: Vec<u8>,
    /// If set, this attribute references a committed (named) datatype.
    pub committed_type_name: Option<String>,
}
