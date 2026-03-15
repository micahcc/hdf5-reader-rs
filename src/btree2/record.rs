/// A record from a B-tree v2 node.
///
/// The actual content depends on `BTree2Type`. We store raw bytes and provide
/// typed accessors.
#[derive(Debug, Clone)]
pub struct Record {
    pub data: Vec<u8>,
}
