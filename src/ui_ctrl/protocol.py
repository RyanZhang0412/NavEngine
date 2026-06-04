"""Frame build, parse, and verify (ui_ctrl_protocol.c)."""

from __future__ import annotations

import struct

from ui_ctrl.constants import (
    CHECKSUM_CHAR_NUM,
    CmdType,
    DATA_LEN_OFFSET,
    DATA_OFFSET,
    HEAD,
    HEAD_CHAR_NUM,
    PROTOCOL_CHAR_NUM,
    TAIL,
    TAIL_CHAR_NUM,
)
from ui_ctrl.crc16 import calc_crc16
from ui_ctrl.message import ProtocolMessage


def calc_checksum_bytes(header_through_payload: bytes) -> int:
    """CRC16 over ID through end of DATA (excludes HEAD, CRC, TAIL)."""
    return calc_crc16(header_through_payload)


def _checksum_region(frame: bytes) -> bytes:
    return frame[HEAD_CHAR_NUM : len(frame) - CHECKSUM_CHAR_NUM - TAIL_CHAR_NUM]


def build_frame(
    msg_id: int,
    cmd_type: int,
    cmd: int,
    payload: bytes = b"",
) -> bytes:
    """Build a complete protocol frame with CRC and tail."""
    payload = bytes(payload)
    data_len = len(payload)
    frame_len = PROTOCOL_CHAR_NUM + data_len

    header = bytearray(frame_len)
    header[0:2] = HEAD
    header[2] = msg_id & 0xFF
    header[3] = int(cmd_type) & 0xFF
    header[4] = int(cmd) & 0xFF
    struct.pack_into("<I", header, DATA_LEN_OFFSET, data_len)

    if data_len:
        header[DATA_OFFSET : DATA_OFFSET + data_len] = payload

    crc = calc_checksum_bytes(_checksum_region(bytes(header)))
    struct.pack_into("<H", header, DATA_OFFSET + data_len, crc)
    header[frame_len - 1] = TAIL

    return bytes(header)


def build_ack(ack_id: int, ack_type: int, msg_id: int = 0) -> bytes:
    """Build ACK frame (init_ack_msg)."""
    return build_frame(
        msg_id=msg_id,
        cmd_type=CmdType.ACK,
        cmd=ack_type,
        payload=bytes([ack_id & 0xFF]),
    )


def build_ctrl(ctrl_cmd: int, payload: bytes = b"", msg_id: int = 0) -> bytes:
    """Build CTRL frame (init_ctrl_msg)."""
    return build_frame(
        msg_id=msg_id,
        cmd_type=CmdType.CTRL,
        cmd=ctrl_cmd,
        payload=payload,
    )


def build_data(data_cmd: int, payload: bytes = b"", msg_id: int = 0) -> bytes:
    """Build DATA frame (init_data_msg)."""
    return build_frame(
        msg_id=msg_id,
        cmd_type=CmdType.DATA,
        cmd=data_cmd,
        payload=payload,
    )


def find_tail_pos(content: bytes) -> int:
    """Return tail byte index (find_tail_pos in ui_ctrl_protocol.c)."""
    if len(content) < PROTOCOL_CHAR_NUM:
        return 0

    data_len = struct.unpack_from("<I", content, DATA_LEN_OFFSET)[0]
    return (PROTOCOL_CHAR_NUM - TAIL_CHAR_NUM) + data_len


def verify_frame(raw: bytes) -> bool:
    """Verify frame head, length, CRC, and tail (verify_msg)."""
    if raw is None or len(raw) < PROTOCOL_CHAR_NUM:
        return False

    if raw[0:2] != HEAD:
        return False

    data_len = struct.unpack_from("<I", raw, DATA_LEN_OFFSET)[0]
    expected_len = PROTOCOL_CHAR_NUM + data_len

    if len(raw) != expected_len:
        return False

    tail_pos = find_tail_pos(raw)
    if tail_pos + 1 != len(raw):
        return False

    if raw[tail_pos] != TAIL:
        return False

    crc_expected = calc_checksum_bytes(_checksum_region(raw))
    crc_actual = struct.unpack_from("<H", raw, DATA_OFFSET + data_len)[0]

    return crc_actual == crc_expected


def parse_frame(raw: bytes) -> ProtocolMessage:
    """Parse frame bytes into ProtocolMessage (assumes valid frame)."""
    if not verify_frame(raw):
        raise ValueError("invalid frame")
    return ProtocolMessage.from_bytes(raw)
