"""Protocol message representation."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ui_ctrl.constants import (
    DATA_LEN_OFFSET,
    DATA_OFFSET,
    HEAD,
    PROTOCOL_CHAR_NUM,
    TAIL,
)


@dataclass
class ProtocolMessage:
    msg_id: int
    cmd_type: int
    cmd: int
    payload: bytes
    raw: bytes | None = None

    @classmethod
    def from_bytes(cls, raw: bytes) -> ProtocolMessage:
        if len(raw) < PROTOCOL_CHAR_NUM:
            raise ValueError(f"frame too short: {len(raw)} < {PROTOCOL_CHAR_NUM}")

        if raw[0:2] != HEAD:
            raise ValueError("invalid frame head")

        if raw[-1] != TAIL:
            raise ValueError("invalid frame tail")

        msg_id = raw[2]
        cmd_type = raw[3]
        cmd = raw[4]
        data_len = struct.unpack_from("<I", raw, DATA_LEN_OFFSET)[0]
        expected_len = PROTOCOL_CHAR_NUM + data_len

        if len(raw) != expected_len:
            raise ValueError(f"frame length mismatch: {len(raw)} != {expected_len}")

        payload = raw[DATA_OFFSET : DATA_OFFSET + data_len]
        return cls(
            msg_id=msg_id,
            cmd_type=cmd_type,
            cmd=cmd,
            payload=payload,
            raw=raw,
        )

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "cmd_type": self.cmd_type,
            "cmd": self.cmd,
            "payload": self.payload.hex() if self.payload else "",
            "len": len(self.raw) if self.raw else None,
        }
