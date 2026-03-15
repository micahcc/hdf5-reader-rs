use crate::datatype::types::Datatype;

/// A member of a compound datatype.
#[derive(Debug, Clone)]
pub struct CompoundMember {
    pub name: String,
    pub byte_offset: u32,
    pub datatype: Datatype,
}

/// A member of an enumeration datatype.
#[derive(Debug, Clone)]
pub struct EnumMember {
    pub name: String,
    pub value: Vec<u8>,
}
