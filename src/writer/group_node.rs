use crate::datatype::Datatype;
use crate::writer::DatasetNode;
use crate::writer::types::AttrData;
use crate::writer::types::ChildNode;

/// A group node in the file builder tree.
pub struct GroupNode {
    pub(crate) children: Vec<(String, ChildNode)>,
    pub(crate) attributes: Vec<AttrData>,
}

impl GroupNode {
    pub(crate) fn new() -> Self {
        GroupNode {
            children: vec![],
            attributes: vec![],
        }
    }

    /// Add a child group, returning a mutable reference to it.
    pub fn add_group(&mut self, name: &str) -> &mut GroupNode {
        self.children
            .push((name.to_string(), ChildNode::Group(GroupNode::new())));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Group(g) => g,
            _ => unreachable!(),
        }
    }

    /// Add a child dataset with raw data bytes.
    pub fn add_dataset(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        data: Vec<u8>,
    ) -> &mut DatasetNode {
        self.children.push((
            name.to_string(),
            ChildNode::Dataset(DatasetNode::new(datatype, shape, data)),
        ));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Dataset(d) => d,
            _ => unreachable!(),
        }
    }

    /// Add a variable-length dataset.
    ///
    /// Each element in `elements` is the raw bytes for one vlen entry.
    /// For vlen strings, each element is the UTF-8 bytes (no NUL terminator needed).
    /// For vlen sequences of T, each element is `count * sizeof(T)` bytes.
    pub fn add_vlen_dataset(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        elements: Vec<Vec<u8>>,
    ) -> &mut DatasetNode {
        self.children.push((
            name.to_string(),
            ChildNode::Dataset(DatasetNode::new_vlen(datatype, shape, elements)),
        ));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Dataset(d) => d,
            _ => unreachable!(),
        }
    }

    /// Add a committed (named) datatype to this group.
    pub fn commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.children.push((
            name.to_string(),
            ChildNode::CommittedDatatype(datatype),
        ));
    }

    /// Add a dataset that references a committed (named) datatype.
    ///
    /// The `committed_type_name` must match the name of a previously committed
    /// datatype in the same group.
    pub fn add_dataset_committed(
        &mut self,
        name: &str,
        committed_type_name: &str,
        datatype: Datatype,
        shape: &[u64],
        data: Vec<u8>,
    ) -> &mut DatasetNode {
        let mut ds = DatasetNode::new(datatype, shape, data);
        ds.committed_type_name = Some(committed_type_name.to_string());
        self.children.push((name.to_string(), ChildNode::Dataset(ds)));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Dataset(d) => d,
            _ => unreachable!(),
        }
    }

    /// Add an attribute to this group.
    pub fn add_attribute(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        value: Vec<u8>,
    ) -> &mut Self {
        self.attributes.push(AttrData {
            name: name.to_string(),
            datatype,
            shape: shape.to_vec(),
            value,
            committed_type_name: None,
        });
        self
    }
}
