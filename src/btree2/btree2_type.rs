use crate::error::Error;
use crate::error::Result;

/// B-tree v2 record type IDs.
///
/// Reference: H5B2pkg.h, the type field in the B-tree header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTree2Type {
    /// Type 1: Testing (internal use only)
    Testing = 1,
    /// Type 2: Indexing indirectly accessed, non-filtered 'huge' fractal heap objects
    HugeObjects = 2,
    /// Type 3: Indexing indirectly accessed, filtered 'huge' fractal heap objects
    HugeObjectsFiltered = 3,
    /// Type 4: Indexing directly accessed, non-filtered 'huge' fractal heap objects
    HugeObjectsDirect = 4,
    /// Type 5: Indexing group links (name hash → fractal heap ID) for new-style groups
    GroupLinks = 5,
    /// Type 6: Indexing group links by creation order
    GroupLinksCreationOrder = 6,
    /// Type 7: Indexing shared messages by hash
    SharedMessages = 7,
    /// Type 8: Indexing attribute names (name hash → fractal heap ID)
    AttributeNames = 8,
    /// Type 9: Indexing attributes by creation order
    AttributeCreationOrder = 9,
    /// Type 10: Chunked dataset (non-filtered, single dim unlimited)
    ChunkedData = 10,
    /// Type 11: Chunked dataset (filtered, single dim unlimited)
    ChunkedDataFiltered = 11,
}

impl BTree2Type {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Testing),
            2 => Ok(Self::HugeObjects),
            3 => Ok(Self::HugeObjectsFiltered),
            4 => Ok(Self::HugeObjectsDirect),
            5 => Ok(Self::GroupLinks),
            6 => Ok(Self::GroupLinksCreationOrder),
            7 => Ok(Self::SharedMessages),
            8 => Ok(Self::AttributeNames),
            9 => Ok(Self::AttributeCreationOrder),
            10 => Ok(Self::ChunkedData),
            11 => Ok(Self::ChunkedDataFiltered),
            _ => Err(Error::InvalidBTreeV2 {
                msg: format!("unknown B-tree v2 type {}", v),
            }),
        }
    }
}
