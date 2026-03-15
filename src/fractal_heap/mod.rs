pub(crate) mod header;
pub(crate) mod read_managed_object;

pub use header::FHDB_MAGIC;
pub use header::FHIB_MAGIC;
pub use header::FRHP_MAGIC;
pub use header::FractalHeapHeader;
pub use read_managed_object::read_managed_object;

#[cfg(test)]
mod tests;
