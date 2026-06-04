"""Stream frame parser (ui_ctrl_port.c data_handler)."""

from __future__ import annotations

from collections.abc import Callable

from ui_ctrl.constants import COMPLETE_HEAD_U16, PROTOCOL_CHAR_NUM
from ui_ctrl.message import ProtocolMessage
from ui_ctrl.protocol import find_tail_pos, parse_frame, verify_frame


class StreamParser:
    """Incremental parser for UI control serial frames."""

    def __init__(
        self,
        on_frame: Callable[[ProtocolMessage], None] | None = None,
        on_bad_frame: Callable[[bytes], None] | None = None,
    ) -> None:
        self._on_frame = on_frame
        self._on_bad_frame = on_bad_frame
        self._buffer = bytearray()

    @property
    def buffer(self) -> bytes:
        return bytes(self._buffer)

    def feed(self, data: bytes) -> int:
        """Feed incoming bytes; return number of consumed bytes from input."""
        if not data:
            return 0

        self._buffer.extend(data)
        parsed_size = 0

        while True:
            buf = self._buffer
            size = len(buf)

            head_pos = self._find_head(buf)
            if head_pos < 0:
                # 保留末字节：帧头 0x3A3A 可能跨两次 read 边界
                if size > 1:
                    parsed_size += size - 1
                    self._buffer = bytearray(buf[-1:])
                break

            if size < PROTOCOL_CHAR_NUM:
                break

            if head_pos > 0:
                parsed_size += head_pos
                del buf[:head_pos]
                self._buffer = buf
                size = len(buf)

            if size < PROTOCOL_CHAR_NUM:
                break

            tail_pos = find_tail_pos(buf)
            if tail_pos == 0:
                parsed_size += 1
                del buf[0:1]
                self._buffer = buf
                continue

            frame_len = tail_pos + 1
            if frame_len > size:
                break

            frame = bytes(buf[:frame_len])
            if verify_frame(frame):
                msg = parse_frame(frame)
                if self._on_frame is not None:
                    self._on_frame(msg)
            elif self._on_bad_frame is not None:
                self._on_bad_frame(frame)

            # C data_handler always advances past a length-complete frame, even if verify fails
            parsed_size += frame_len
            del buf[:frame_len]
            self._buffer = buf

            if not buf:
                break

        return parsed_size

    @staticmethod
    def _find_head(data: bytes) -> int:
        """Find first 0x3A3A as little-endian uint16."""
        for pos in range(len(data) - 1):
            word = data[pos] | (data[pos + 1] << 8)
            if word == COMPLETE_HEAD_U16:
                return pos
        return -1

    def reset(self) -> None:
        self._buffer.clear()
