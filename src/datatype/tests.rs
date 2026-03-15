use super::*;

/// Build a raw datatype message from class, version, class_bits, size, and properties.
fn build_dt_msg(class: u8, version: u8, class_bits: u32, size: u32, props: &[u8]) -> Vec<u8> {
    let cv = (class as u32) | ((version as u32) << 4) | (class_bits << 8);
    let mut buf = Vec::new();
    buf.extend_from_slice(&cv.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(props);
    buf
}

#[test]
fn parse_fixed_point_i32_le() {
    // class=0 (FixedPoint), version=1, signed (bit 3 of class_bits), LE (bit 0 = 0)
    let props = [0, 0, 32, 0]; // bit_offset=0, bit_precision=32
    let msg = build_dt_msg(0, 1, 0x08, 4, &props); // 0x08 = signed
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            assert_eq!(size, 4);
            assert_eq!(byte_order, ByteOrder::LittleEndian);
            assert!(signed);
            assert_eq!(bit_offset, 0);
            assert_eq!(bit_precision, 32);
        }
        other => panic!("expected FixedPoint, got {:?}", other),
    }
}

#[test]
fn parse_fixed_point_u8_be() {
    // class=0, version=1, unsigned (no bit 3), BE (bit 0 = 1)
    let props = [0, 0, 8, 0]; // bit_offset=0, bit_precision=8
    let msg = build_dt_msg(0, 1, 0x01, 1, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            ..
        } => {
            assert_eq!(size, 1);
            assert_eq!(byte_order, ByteOrder::BigEndian);
            assert!(!signed);
        }
        other => panic!("expected FixedPoint, got {:?}", other),
    }
}

#[test]
fn parse_float_f64_le() {
    // class=1, version=1, LE (byte_order_lo=0, byte_order_hi=0)
    let mut props = vec![0u8; 12];
    props[2..4].copy_from_slice(&64u16.to_le_bytes()); // bit_precision=64
    props[4] = 52; // exponent_location
    props[5] = 11; // exponent_size
    props[6] = 0; // mantissa_location
    props[7] = 52; // mantissa_size
    props[8..12].copy_from_slice(&1023u32.to_le_bytes()); // exponent_bias
    let msg = build_dt_msg(1, 1, 0, 8, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::FloatingPoint {
            size,
            byte_order,
            exponent_bias,
            mantissa_size,
            ..
        } => {
            assert_eq!(size, 8);
            assert_eq!(byte_order, ByteOrder::LittleEndian);
            assert_eq!(exponent_bias, 1023);
            assert_eq!(mantissa_size, 52);
        }
        other => panic!("expected FloatingPoint, got {:?}", other),
    }
}

#[test]
fn parse_string_ascii_nullterm() {
    // class=3, version=1, padding=NullTerminate(0), charset=Ascii(0)
    let msg = build_dt_msg(3, 1, 0x00, 10, &[]);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::String {
            size,
            padding,
            char_set,
        } => {
            assert_eq!(size, 10);
            assert_eq!(padding, StringPadding::NullTerminate);
            assert_eq!(char_set, CharacterSet::Ascii);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn parse_string_utf8_nullpad() {
    // padding=NullPad(1), charset=Utf8(1)
    // class_bits = (charset << 4) | padding = 0x11
    let msg = build_dt_msg(3, 1, 0x11, 32, &[]);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::String {
            padding, char_set, ..
        } => {
            assert_eq!(padding, StringPadding::NullPad);
            assert_eq!(char_set, CharacterSet::Utf8);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn parse_compound_v3_two_members() {
    // Compound v3: unpadded names, variable-width offsets
    // Compound size=8 → offset width=1 byte
    // Member 1: "a" at offset 0, type = u32 LE (FixedPoint, 4 bytes, class_bits=0x08)
    // Member 2: "b" at offset 4, type = u32 LE
    let mut props = Vec::new();
    // Member "a": name "a\0", offset=0, then u32 LE datatype
    props.extend_from_slice(b"a\0");
    props.push(0); // offset = 0
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
    // Member "b": name "b\0", offset=4, then u32 LE datatype
    props.extend_from_slice(b"b\0");
    props.push(4); // offset = 4
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

    // class=6 (Compound), version=3, class_bits lower 16 = nmembers=2
    let msg = build_dt_msg(6, 3, 2, 8, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 8);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "a");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "b");
            assert_eq!(members[1].byte_offset, 4);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn parse_enum_v3_i8() {
    // Enum v3: base type = i8 LE, 2 members
    let mut props = Vec::new();
    // Base type: i8 LE signed (class=0, version=3, class_bits=0x08, size=1)
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 1, &[0, 0, 8, 0]));
    // V3 names: unpadded, null-terminated
    props.extend_from_slice(b"OFF\0");
    props.extend_from_slice(b"ON\0");
    // Values: 0, 1
    props.push(0);
    props.push(1);

    // class=8 (Enum), version=3, nmembers=2
    let msg = build_dt_msg(8, 3, 2, 1, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Enum { base, members } => {
            assert_eq!(base.element_size(), 1);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "OFF");
            assert_eq!(members[0].value, vec![0]);
            assert_eq!(members[1].name, "ON");
            assert_eq!(members[1].value, vec![1]);
        }
        other => panic!("expected Enum, got {:?}", other),
    }
}

#[test]
fn parse_array_v2_1d() {
    // Array v2: ndims=1, dim[0]=5, element type = f32 LE
    let mut props = Vec::new();
    props.push(1); // ndims
    // v2: no padding after ndims
    props.extend_from_slice(&5u32.to_le_bytes()); // dimension[0] = 5
    // Element type: f32 LE
    let mut f32_props = vec![0u8; 12];
    f32_props[2..4].copy_from_slice(&32u16.to_le_bytes()); // bit_precision=32
    f32_props[4] = 23; // exponent_location
    f32_props[5] = 8; // exponent_size
    f32_props[6] = 0; // mantissa_location
    f32_props[7] = 23; // mantissa_size
    f32_props[8..12].copy_from_slice(&127u32.to_le_bytes()); // exponent_bias
    props.extend_from_slice(&build_dt_msg(1, 1, 0, 4, &f32_props));

    // class=10 (Array), version=2, size doesn't matter (computed from dims * element)
    let msg = build_dt_msg(10, 2, 0, 20, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Array {
            ref element_type,
            ref dimensions,
        } => {
            assert_eq!(dimensions, &vec![5]);
            assert_eq!(element_type.element_size(), 4);
            assert_eq!(dt.element_size(), 20); // 5 * 4
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn parse_array_v1_with_padding() {
    // Array v1: ndims=1, then 3 bytes padding, dim[0]=3, then 4 bytes perm, element type
    let mut props = Vec::new();
    props.push(1); // ndims
    props.extend_from_slice(&[0, 0, 0]); // v1 padding
    props.extend_from_slice(&3u32.to_le_bytes()); // dimension[0] = 3
    props.extend_from_slice(&0u32.to_le_bytes()); // v1 permutation[0] = 0
    // Element type: i16 LE signed
    props.extend_from_slice(&build_dt_msg(0, 1, 0x08, 2, &[0, 0, 16, 0]));

    let msg = build_dt_msg(10, 1, 0, 6, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Array {
            ref element_type,
            ref dimensions,
        } => {
            assert_eq!(dimensions, &vec![3]);
            assert_eq!(element_type.element_size(), 2);
            assert_eq!(dt.element_size(), 6); // 3 * 2
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn parse_opaque() {
    // class=5 (Opaque), tag_len in class_bits lower 8 = 4
    let props = b"test\0\0\0\0"; // tag "test", padded to 8
    let msg = build_dt_msg(5, 1, 4, 16, props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Opaque { size, tag } => {
            assert_eq!(size, 16);
            assert_eq!(tag, "test");
        }
        other => panic!("expected Opaque, got {:?}", other),
    }
}

#[test]
fn element_size_compound() {
    let dt = Datatype::Compound {
        size: 24,
        members: vec![],
    };
    assert_eq!(dt.element_size(), 24);
}

#[test]
fn element_size_enum() {
    let dt = Datatype::Enum {
        base: Box::new(Datatype::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }),
        members: vec![],
    };
    assert_eq!(dt.element_size(), 4);
}

#[test]
fn element_size_array_2d() {
    let dt = Datatype::Array {
        element_type: Box::new(Datatype::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 64,
        }),
        dimensions: vec![3, 4],
    };
    assert_eq!(dt.element_size(), 96); // 3 * 4 * 8
}

#[test]
fn reject_too_short_message() {
    let data = [0u8; 4];
    assert!(Datatype::parse(&data).is_err());
}

#[test]
fn parse_complex_f64() {
    // Complex of f64 LE: class=11, size=16, props = base f64 datatype
    let mut base_props = vec![0u8; 12];
    base_props[2..4].copy_from_slice(&64u16.to_le_bytes()); // bit_precision=64
    base_props[4] = 52; // exponent_location
    base_props[5] = 11; // exponent_size
    base_props[6] = 0; // mantissa_location
    base_props[7] = 52; // mantissa_size
    base_props[8..12].copy_from_slice(&1023u32.to_le_bytes()); // exponent_bias
    let base_msg = build_dt_msg(1, 1, 0, 8, &base_props);

    let msg = build_dt_msg(11, 1, 0, 16, &base_msg);
    let dt = Datatype::parse(&msg).unwrap();
    match &dt {
        Datatype::Complex { size, base } => {
            assert_eq!(*size, 16);
            assert_eq!(base.element_size(), 8);
            match base.as_ref() {
                Datatype::FloatingPoint {
                    byte_order,
                    mantissa_size,
                    ..
                } => {
                    assert_eq!(*byte_order, ByteOrder::LittleEndian);
                    assert_eq!(*mantissa_size, 52);
                }
                other => panic!("expected FloatingPoint base, got {:?}", other),
            }
        }
        other => panic!("expected Complex, got {:?}", other),
    }
    assert_eq!(dt.element_size(), 16);
}

#[test]
fn reject_unknown_class() {
    let msg = build_dt_msg(15, 1, 0, 4, &[]);
    assert!(Datatype::parse(&msg).is_err());
}

#[test]
fn compound_v2_name_padding_from_name_start() {
    // Compound v2: member 1 has String type (encoded_size=8), so member 2's
    // name starts at props offset 20 (8+4+8), which is NOT 8-aligned globally.
    // The v1/v2 padding must be relative to name_start, not absolute.
    let mut props = Vec::new();

    // Member "s": name padded to 8, offset(4), String type(8) = 20 bytes
    let mut name1 = vec![0u8; 8]; // ((1+8)/8)*8 = 8
    name1[0] = b's';
    props.extend_from_slice(&name1);
    props.extend_from_slice(&0u32.to_le_bytes()); // offset = 0
    props.extend_from_slice(&build_dt_msg(3, 2, 0x00, 4, &[])); // String ASCII

    // Member "n" starts at props offset 20. Name "n" padded to 8 from name_start.
    let mut name2 = vec![0u8; 8]; // ((1+8)/8)*8 = 8
    name2[0] = b'n';
    props.extend_from_slice(&name2);
    props.extend_from_slice(&4u32.to_le_bytes()); // offset = 4
    props.extend_from_slice(&build_dt_msg(0, 2, 0x08, 8, &[0, 0, 64, 0])); // i64 LE

    let msg = build_dt_msg(6, 2, 2, 12, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 12);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "s");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "n");
            assert_eq!(members[1].byte_offset, 4);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn compound_v3_three_byte_offset() {
    // Compound v3 with size > 65535 needs 3-byte member offset encoding.
    let compound_size: u32 = 70000;
    let mut props = Vec::new();

    // Member "a": name "a\0" (unpadded v3), 3-byte offset=0, i32 type
    props.extend_from_slice(b"a\0");
    props.extend_from_slice(&[0, 0, 0]); // 3 bytes LE = 0
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

    // Member "b": name "b\0", 3-byte offset=69996, i32 type
    props.extend_from_slice(b"b\0");
    let off = 69996u32.to_le_bytes();
    props.extend_from_slice(&off[..3]); // lower 3 bytes
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

    let msg = build_dt_msg(6, 3, 2, compound_size, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, compound_size);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "a");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "b");
            assert_eq!(members[1].byte_offset, 69996);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn compound_with_enum_member_encoded_size() {
    // Compound v3 with an enum member followed by i32. Tests that
    // encoded_size properly computes enum size (not data.len()).
    let mut props = Vec::new();

    // Member "e" at offset 0: Enum(i32, 2 members)
    props.extend_from_slice(b"e\0");
    props.push(0); // offset = 0 (1 byte, size=8)
    let mut enum_props = Vec::new();
    enum_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
    enum_props.extend_from_slice(b"A\0");
    enum_props.extend_from_slice(b"B\0");
    enum_props.extend_from_slice(&0i32.to_le_bytes());
    enum_props.extend_from_slice(&1i32.to_le_bytes());
    props.extend_from_slice(&build_dt_msg(8, 3, 2, 4, &enum_props));

    // Member "n" at offset 4: i32
    props.extend_from_slice(b"n\0");
    props.push(4); // offset = 4
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

    let msg = build_dt_msg(6, 3, 2, 8, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 8);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "e");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "n");
            assert_eq!(members[1].byte_offset, 4);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn compound_with_array_member_encoded_size() {
    // Compound v3 with an array member followed by i32.
    let mut props = Vec::new();

    // Member "arr" at offset 0: Array(i32, dim=[3])
    props.extend_from_slice(b"arr\0");
    props.push(0); // offset = 0
    let mut arr_props = Vec::new();
    arr_props.push(1); // ndims = 1
    arr_props.extend_from_slice(&3u32.to_le_bytes()); // dim[0] = 3
    arr_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
    props.extend_from_slice(&build_dt_msg(10, 3, 0, 12, &arr_props));

    // Member "n" at offset 12: i32
    props.extend_from_slice(b"n\0");
    props.push(12); // offset = 12
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

    let msg = build_dt_msg(6, 3, 2, 16, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 16);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "arr");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "n");
            assert_eq!(members[1].byte_offset, 12);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn compound_with_vlen_member_encoded_size() {
    // Compound v3 with a vlen member followed by i32.
    let mut props = Vec::new();

    // Member "v" at offset 0: VarLen(i32), sequence type
    props.extend_from_slice(b"v\0");
    props.push(0); // offset = 0
    let base = build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]);
    props.extend_from_slice(&build_dt_msg(9, 3, 0, 16, &base));

    // Member "n" at offset 16: i32
    props.extend_from_slice(b"n\0");
    props.push(16); // offset = 16
    props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

    let msg = build_dt_msg(6, 3, 2, 20, &props);
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 20);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "v");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "n");
            assert_eq!(members[1].byte_offset, 16);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn encoded_size_vlen() {
    // VarLen(i32): 8 header + 12 base type = 20
    let base = build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]);
    let msg = build_dt_msg(9, 3, 0, 16, &base);
    assert_eq!(Datatype::encoded_size(&msg).unwrap(), 20);
}

#[test]
fn encoded_size_enum() {
    // Enum v3(i32, 2 members "A","B"): 8 + 12(base) + 2("A\0") + 2("B\0") + 8(2*4 vals) = 32
    let mut enum_props = Vec::new();
    enum_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
    enum_props.extend_from_slice(b"A\0");
    enum_props.extend_from_slice(b"B\0");
    enum_props.extend_from_slice(&0i32.to_le_bytes());
    enum_props.extend_from_slice(&1i32.to_le_bytes());
    let msg = build_dt_msg(8, 3, 2, 4, &enum_props);
    assert_eq!(Datatype::encoded_size(&msg).unwrap(), 32);
}

#[test]
fn parse_float_vax_byte_order() {
    // VAX byte order requires BOTH bit 0 and bit 6 set: class_bits = 0x41
    // (1,1) = VAX — not (1,0) which was the old (wrong) mapping.
    let mut props = vec![0u8; 12];
    props[2..4].copy_from_slice(&32u16.to_le_bytes());
    props[4] = 23;
    props[5] = 8;
    props[6] = 0;
    props[7] = 23;
    props[8..12].copy_from_slice(&127u32.to_le_bytes());
    let msg = build_dt_msg(1, 3, 0x41, 4, &props); // 0x41 = bit0 + bit6
    let dt = Datatype::parse(&msg).unwrap();
    match dt {
        Datatype::FloatingPoint { byte_order, .. } => {
            assert_eq!(byte_order, ByteOrder::Vax);
        }
        other => panic!("expected FloatingPoint, got {:?}", other),
    }
}

#[test]
fn parse_float_invalid_byte_order_hi1_lo0() {
    // (byte_order_hi=1, byte_order_lo=0) = 0x40 → must be an error
    let mut props = vec![0u8; 12];
    props[2..4].copy_from_slice(&32u16.to_le_bytes());
    props[4] = 23;
    props[5] = 8;
    props[6] = 0;
    props[7] = 23;
    props[8..12].copy_from_slice(&127u32.to_le_bytes());
    let msg = build_dt_msg(1, 3, 0x40, 4, &props); // bit6 only, no bit0
    assert!(Datatype::parse(&msg).is_err());
}

#[test]
fn encoded_size_array() {
    // Array v3: 8 + 1(ndims) + 4(dim) + 12(i32 type) = 25
    let mut arr_props = Vec::new();
    arr_props.push(1);
    arr_props.extend_from_slice(&3u32.to_le_bytes());
    arr_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
    let msg = build_dt_msg(10, 3, 0, 12, &arr_props);
    assert_eq!(Datatype::encoded_size(&msg).unwrap(), 25);
}
