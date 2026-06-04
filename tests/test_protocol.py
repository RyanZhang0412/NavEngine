import struct

import pytest

from ui_ctrl.constants import CmdAck, CmdCtrl, CmdType, HEAD, PROTOCOL_CHAR_NUM, TAIL
from ui_ctrl.protocol import (
    build_ack,
    build_ctrl,
    build_data,
    build_frame,
    calc_checksum_bytes,
    parse_frame,
    verify_frame,
)


def test_build_ack_round_trip(ack_frame):
    assert verify_frame(ack_frame)
    msg = parse_frame(ack_frame)
    assert msg.msg_id == 0
    assert msg.cmd_type == CmdType.ACK
    assert msg.cmd == CmdAck.NORMAL
    assert msg.payload == bytes([0x05])
    assert len(msg.raw) == PROTOCOL_CHAR_NUM + 1


def test_build_ctrl_with_payload(ctrl_frame):
    assert verify_frame(ctrl_frame)
    msg = parse_frame(ctrl_frame)
    assert msg.cmd_type == CmdType.CTRL
    assert msg.cmd == CmdCtrl.POWER_ON
    assert msg.payload == b"\x01\x02"


def test_build_data_empty():
    frame = build_data(0x02, b"")
    assert verify_frame(frame)
    msg = parse_frame(frame)
    assert msg.cmd_type == CmdType.DATA
    assert msg.cmd == 0x02
    assert msg.payload == b""


def test_build_frame_custom():
    frame = build_frame(0x07, CmdType.INQUIRY, 0x01, b"")
    assert verify_frame(frame)
    msg = parse_frame(frame)
    assert msg.msg_id == 0x07
    assert msg.cmd_type == CmdType.INQUIRY


def test_crc_mismatch(ack_frame):
    bad = bytearray(ack_frame)
    bad[-3] ^= 0xFF
    assert not verify_frame(bytes(bad))


def test_bad_head(ack_frame):
    bad = bytearray(ack_frame)
    bad[0] = 0x00
    assert not verify_frame(bytes(bad))


def test_bad_tail(ack_frame):
    bad = bytearray(ack_frame)
    bad[-1] = 0x00
    assert not verify_frame(bytes(bad))


def test_length_mismatch(ack_frame):
    assert not verify_frame(ack_frame[:-1])


def test_checksum_region_length(ack_frame):
    region = ack_frame[2 : len(ack_frame) - 3]
    assert len(region) == len(ack_frame) - 5
    crc = calc_checksum_bytes(region)
    assert struct.unpack_from("<H", ack_frame, len(ack_frame) - 3)[0] == crc


def test_parse_invalid_raises(ack_frame):
    bad = bytearray(ack_frame)
    bad[-3] ^= 0xFF
    with pytest.raises(ValueError):
        parse_frame(bytes(bad))


def test_frame_structure(ack_frame):
    assert ack_frame[0:2] == HEAD
    assert ack_frame[-1] == TAIL
