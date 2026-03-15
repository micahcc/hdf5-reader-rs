use crate::io::ReadAt;

use crate::file::dataset::Dataset;
use crate::file::group::Group;

/// A node in the HDF5 hierarchy — either a group or a dataset.
pub enum Node<'a, R: ReadAt + ?Sized> {
    Group(Group<'a, R>),
    Dataset(Dataset<'a, R>),
}
