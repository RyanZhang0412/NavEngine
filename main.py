"""NavEngine entry: send and receive UI control frames over serial."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import serial
from serial import SerialException

from ui_ctrl.constants import CmdAck, CmdCtrl, CmdData, CmdInquiry, CmdType
from ui_ctrl.message import ProtocolMessage
from ui_ctrl.stream_parser import StreamParser
from ui_ctrl.training_ctrl import SEND_ACTIONS, build_send_frame

DEFAULT_PORT = "/dev/ttyCH341USB0"
DEFAULT_BAUD = 921600
DEFAULT_RECV_SEC = 2.0

_CMD_TYPE_NAMES = {e.value: e.name for e in CmdType}
_CMD_NAMES: dict[int, dict[int, str]] = {
    CmdType.CTRL: {e.value: e.name for e in CmdCtrl},
    CmdType.DATA: {e.value: e.name for e in CmdData},
    CmdType.INQUIRY: {e.value: e.name for e in CmdInquiry},
    CmdType.ACK: {e.value: e.name for e in CmdAck},
}


def resolve_port(port: str) -> str:
    """Map Windows-style COMn to the local device path."""
    if sys.platform == "win32":
        return port
    upper = port.upper()
    if upper.startswith("COM") and upper[3:].isdigit():
        return f"/dev/ttyS{int(upper[3:]) - 1}"
    return port


def cmd_name(cmd_type: int, cmd: int) -> str:
    names = _CMD_NAMES.get(cmd_type, {})
    return names.get(cmd, f"0x{cmd:02X}")


def decode_payload(payload: bytes) -> object:
    if not payload:
        return None
    text = payload.rstrip(b"\x00").decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def format_message(msg: ProtocolMessage) -> str:
    type_name = _CMD_TYPE_NAMES.get(msg.cmd_type, f"0x{msg.cmd_type:02X}")
    cmd_label = cmd_name(msg.cmd_type, msg.cmd)
    payload_view = decode_payload(msg.payload)
    lines = [
        f"[RX] id=0x{msg.msg_id:02X} type={type_name} cmd={cmd_label}",
        f"     payload_len={len(msg.payload)}",
    ]
    if payload_view is not None:
        if isinstance(payload_view, (dict, list)):
            lines.append(f"     json={json.dumps(payload_view, ensure_ascii=False)}")
        else:
            lines.append(f"     data={payload_view!r}")
    return "\n".join(lines)


class UiSerialSession:
    """Serial port with background receive and StreamParser."""

    def __init__(
        self,
        port: str,
        baud: int,
        *,
        read_chunk_timeout: float = 0.05,
    ) -> None:
        self._device = resolve_port(port)
        self._baud = baud
        self._read_chunk_timeout = read_chunk_timeout
        self._ser: serial.Serial | None = None
        self._parser = StreamParser(
            on_frame=self._on_frame,
            on_bad_frame=self._on_bad_frame,
        )
        self._messages: list[ProtocolMessage] = []
        self._lock = threading.Lock()
        self._rx_thread: threading.Thread | None = None
        self._running = False

    @property
    def messages(self) -> list[ProtocolMessage]:
        with self._lock:
            return list(self._messages)

    def _on_frame(self, msg: ProtocolMessage) -> None:
        with self._lock:
            self._messages.append(msg)
        print(format_message(msg))

    def _on_bad_frame(self, raw: bytes) -> None:
        print(f"[RX] 帧长完整但校验失败，长度={len(raw)}")

    def open(self) -> None:
        self._ser = serial.Serial(
            self._device,
            baudrate=self._baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._read_chunk_timeout,
        )
        self._ser.reset_input_buffer()
        print(f"已打开串口 {self._device} @ {self._baud}（已清空输入缓冲）")

    def start_receive(self) -> None:
        if self._ser is None:
            raise RuntimeError("serial port not open")
        if self._rx_thread is not None and self._rx_thread.is_alive():
            return
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name="ui-rx", daemon=True)
        self._rx_thread.start()
        print("接收线程已启动，持续解析入站帧…")

    def _rx_loop(self) -> None:
        assert self._ser is not None
        while self._running:
            waiting = self._ser.in_waiting
            if waiting:
                chunk = self._ser.read(waiting)
            else:
                chunk = self._ser.read(1)
            if chunk:
                self._parser.feed(chunk)
            elif not waiting:
                time.sleep(0.01)

    def send(self, frame: bytes) -> int:
        if self._ser is None:
            raise RuntimeError("serial port not open")
        written = self._ser.write(frame)
        self._ser.flush()
        print(f"[TX] 已发送 {written} 字节")
        return written

    def wait_receive(self, seconds: float) -> None:
        if seconds <= 0:
            return
        print(f"等待接收 {seconds:.1f}s …")
        time.sleep(seconds)

    def close(self) -> None:
        self._running = False
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        leftover = self._parser.buffer
        if leftover:
            print(
                f"警告: 残留 {len(leftover)} 字节未组成完整帧"
                "（可能从帧中间开始监听，需等待下一帧以 3a 3a 开头）"
            )
        if self._ser is not None:
            ser = self._ser
            self._ser = None
            try:
                if ser.is_open and hasattr(ser, "cancel_read"):
                    ser.cancel_read()
            except (SerialException, OSError):
                pass
            for op in (ser.reset_input_buffer, ser.reset_output_buffer):
                try:
                    if ser.is_open:
                        op()
                except (SerialException, OSError):
                    pass
            try:
                if ser.is_open:
                    ser.close()
            except (SerialException, OSError):
                pass
            time.sleep(0.15)
            print(f"已关闭串口 {self._device}")


def run_listen(
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    *,
    listen_sec: float | None = None,
) -> list[ProtocolMessage]:
    """Receive and parse until Ctrl+C or listen_sec elapses."""
    session = UiSerialSession(port, baud)
    stop = threading.Event()

    def _request_stop(signum: int, _frame) -> None:
        if signum == signal.SIGINT:
            print()
        stop.set()

    previous_int = signal.signal(signal.SIGINT, _request_stop)
    previous_term = signal.signal(signal.SIGTERM, _request_stop)
    session.open()
    session.start_receive()
    try:
        if listen_sec is not None and listen_sec > 0:
            print(f"监听 {listen_sec:.1f}s 后自动退出…")
            stop.wait(listen_sec)
        else:
            print("持续监听，按 Ctrl+C 退出…")
            while not stop.is_set():
                time.sleep(0.2)
        if stop.is_set() and listen_sec is None:
            print("停止监听")
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        session.close()
    return session.messages


def run_send(
    frame: bytes,
    action: str,
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    recv_sec: float = DEFAULT_RECV_SEC,
) -> list[ProtocolMessage]:
    """Send one control frame, then receive and parse responses."""
    session = UiSerialSession(port, baud)
    session.open()
    session.start_receive()
    try:
        print(f"[TX] 训练控制: {action}")
        session.send(frame)
        session.wait_receive(recv_sec)
    finally:
        session.close()
    return session.messages


def _print_summary(messages: list[ProtocolMessage], *, expect_rx: bool) -> None:
    if messages:
        print(f"\n共解析 {len(messages)} 帧")
    elif expect_rx:
        print("未收到有效帧（检查接线/波特率，或从完整帧头开始监听）")


def main(argv: list[str] | None = None) -> int:
    actions = ", ".join(sorted(SEND_ACTIONS))
    parser = argparse.ArgumentParser(
        description=f"UI 串口协议工具（默认 {DEFAULT_PORT} @ {DEFAULT_BAUD}）",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"串口设备路径（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"波特率（默认 {DEFAULT_BAUD}）",
    )
    parser.add_argument(
        "--recv-sec",
        type=float,
        default=DEFAULT_RECV_SEC,
        help=f"send 后等待接收时长（秒，默认 {DEFAULT_RECV_SEC}）",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listen_p = sub.add_parser("listen", help="持续接收并解析协议帧")
    listen_p.add_argument(
        "--listen-sec",
        type=float,
        default=None,
        metavar="SEC",
        help="监听指定秒数后退出（勿用 shell 的 timeout 强杀进程）",
    )
    send_p = sub.add_parser("send", help="发送训练控制帧并等待应答")
    send_p.add_argument(
        "action",
        choices=sorted(SEND_ACTIONS),
        help=f"训练控制动作: {actions}",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "listen":
            messages = run_listen(
                port=args.port,
                baud=args.baud,
                listen_sec=args.listen_sec,
            )
            _print_summary(messages, expect_rx=True)
        else:
            frame = build_send_frame(args.action)
            messages = run_send(
                frame,
                args.action,
                port=args.port,
                baud=args.baud,
                recv_sec=args.recv_sec,
            )
            _print_summary(messages, expect_rx=True)
    except (SerialException, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
