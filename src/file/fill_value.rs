use crate::error::Error;
use crate::error::Result;

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

    /// Parse an old-style fill value message (type 0x0004).
    ///
    /// Format: size(u32) + fill_value_bytes. No version or flags.
    pub fn parse_old(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Ok(FillValue {
                defined: false,
                value: None,
            });
        }
        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if size == 0 || data.len() < 4 + size {
            return Ok(FillValue {
                defined: true,
                value: None,
            });
        }
        Ok(FillValue {
            defined: true,
            value: Some(data[4..4 + size].to_vec()),
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
