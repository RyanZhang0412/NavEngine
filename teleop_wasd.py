#!/usr/bin/env python3
"""WASD 键盘遥控：经 TRAINING_CHANGE 发送 Ranger.velo_fwd / velo_yaw。"""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import serial
from serial import SerialException

from ui_ctrl.constants import CmdCtrl
from ui_ctrl.protocol import build_ctrl, verify_frame
from ui_ctrl.training_ctrl import (
    ROBOT_KEY,
    TrainingProgram,
    build_training_init,
    build_training_start,
    build_training_stop,
)

DEFAULT_PORT = "/dev/ttyCH341USB0"
DEFAULT_BAUD = 921600
DEFAULT_RATE = 20.0
MAX_VELO = 0.1
# hold 模式：首键等待终端连发；连发后保持窗口须大于常见 repeat 间隔(~30–500ms)，否则速度会 0.1/0.0 抖动
DEFAULT_HOLD_INITIAL_SEC = 0.6
DEFAULT_HOLD_RELEASE_SEC = 0.15
# 连发中判定“仍按住”的最小保持时间（独立于松键停延迟，解决长按 fwd 一会 0.1 一会 0.0）
MIN_HOLD_REPEAT_SEC = 0.45


def _clamp_vel(value: float) -> float:
    return max(-MAX_VELO, min(MAX_VELO, value))


def build_velocity_change(fwd: float, yaw: float) -> bytes:
    """TRAINING_CHANGE，不修改 training_ctrl.py。"""
    body = {
        "id": 0,
        "autorun": 1,
        "training_set": 0,
        "training_program": int(TrainingProgram.ISOKINETIC),
        ROBOT_KEY: {
            "velo_fwd": _clamp_vel(fwd),
            "velo_yaw": _clamp_vel(yaw),
        },
    }
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8") + b"\x00"
    frame = build_ctrl(CmdCtrl.TRAINING_CHANGE, payload)
    if not verify_frame(frame):
        raise RuntimeError("built velocity frame failed verification")
    return frame


def resolve_port(port: str) -> str:
    if sys.platform == "win32":
        return port
    upper = port.upper()
    if upper.startswith("COM") and upper[3:].isdigit():
        return f"/dev/ttyS{int(upper[3:]) - 1}"
    return port


def velocities_from_latched(latched: set[str], *, speed: float) -> tuple[float, float]:
    """按下保持，直到反向键或空格清除（不依赖终端连发）。"""
    fwd = 0.0
    yaw = 0.0
    if "w" in latched:
        fwd += speed
    if "s" in latched:
        fwd -= speed
    if "a" in latched:
        yaw += speed
    if "d" in latched:
        yaw -= speed
    return _clamp_vel(fwd), _clamp_vel(yaw)


@dataclass
class _KeyHoldState:
    last_seen: float
    repeat_count: int = 1


def _key_hold_timeout(st: _KeyHoldState, *, hold_initial: float, hold_release: float) -> float:
    """首键用长窗口等连发；连发后保持窗口不能短于终端 repeat 间隔，否则长按会速度抖动。"""
    if st.repeat_count < 2:
        return hold_initial
    return max(hold_release, MIN_HOLD_REPEAT_SEC)


def velocities_from_hold(
    keys: dict[str, _KeyHoldState],
    *,
    speed: float,
    now: float,
    hold_initial: float,
    hold_release: float,
) -> tuple[float, float]:
    active = {
        k
        for k, st in keys.items()
        if now - st.last_seen <= _key_hold_timeout(
            st, hold_initial=hold_initial, hold_release=hold_release
        )
    }
    fwd = 0.0
    yaw = 0.0
    if "w" in active:
        fwd += speed
    if "s" in active:
        fwd -= speed
    if "a" in active:
        yaw += speed
    if "d" in active:
        yaw -= speed
    return _clamp_vel(fwd), _clamp_vel(yaw)


class TeleopSerial:
    def __init__(self, port: str, baud: int, *, verbose: bool) -> None:
        self._port = resolve_port(port)
        self._baud = baud
        self._verbose = verbose
        self._ser: serial.Serial | None = None
        self._last_print: tuple[float, float] | None = None
        self._last_sent: tuple[float, float] | None = None

    def open(self) -> None:
        self._ser = serial.Serial(
            self._port,
            baudrate=self._baud,
            timeout=0.05,
        )
        self._ser.reset_input_buffer()
        print(f"串口 {self._port} @ {self._baud}")

    def send_frame(self, frame: bytes) -> None:
        if self._ser is None:
            raise RuntimeError("serial not open")
        self._ser.write(frame)
        self._ser.flush()

    def send_velocity(self, fwd: float, yaw: float, *, force: bool = False) -> None:
        pair = (fwd, yaw)
        if not force and pair == self._last_sent:
            return
        frame = build_velocity_change(fwd, yaw)
        self.send_frame(frame)
        self._last_sent = pair
        if self._verbose and pair != self._last_print:
            print(f"  fwd={fwd:+.2f}  yaw={yaw:+.2f}  ({len(frame)} B)")
            self._last_print = pair

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


def register_key_latched(latched: set[str], char: str) -> None:
    c = char.lower()
    if c == "w":
        latched.discard("s")
        latched.add("w")
    elif c == "s":
        latched.discard("w")
        latched.add("s")
    elif c == "a":
        latched.discard("d")
        latched.add("a")
    elif c == "d":
        latched.discard("a")
        latched.add("d")
    elif c == " ":
        latched.clear()


def register_key_hold(keys: dict[str, _KeyHoldState], char: str, now: float) -> None:
    c = char.lower()
    if c == "w":
        keys.pop("s", None)
        _touch_hold_key(keys, "w", now)
    elif c == "s":
        keys.pop("w", None)
        _touch_hold_key(keys, "s", now)
    elif c == "a":
        keys.pop("d", None)
        _touch_hold_key(keys, "a", now)
    elif c == "d":
        keys.pop("a", None)
        _touch_hold_key(keys, "d", now)
    elif c == " ":
        keys.clear()


def _touch_hold_key(keys: dict[str, _KeyHoldState], key: str, now: float) -> None:
    st = keys.get(key)
    if st is not None and now - st.last_seen < 0.45:
        st.repeat_count += 1
        st.last_seen = now
    else:
        keys[key] = _KeyHoldState(last_seen=now, repeat_count=1)


def run_teleop(
    port: str,
    baud: int,
    *,
    speed: float,
    rate: float,
    setup: bool,
    verbose: bool,
    release_mode: str,
    hold_initial: float,
    hold_release: float,
) -> int:
    speed = min(speed, MAX_VELO)
    interval = 1.0 / rate
    keys: dict[str, _KeyHoldState] = {}
    latched: set[str] = set()

    link = TeleopSerial(port, baud, verbose=verbose)
    try:
        link.open()
        if setup:
            print("下发 init + start …")
            link.send_frame(build_training_init())
            time.sleep(0.2)
            link.send_frame(build_training_start())
            time.sleep(0.2)

        if release_mode == "latch":
            stop_help = "空格 急停 | 反向键取消该方向"
        else:
            stop_help = (
                f"松键约 {max(hold_release, MIN_HOLD_REPEAT_SEC):.2f}s 内停 "
                f"(连发前缓冲 {hold_initial:.2f}s) | 空格急停"
            )
        print(
            f"\nWASD 遥控 | 速度上限 {speed:.2f} | {rate:.0f} Hz | 模式 {release_mode}\n"
            "  W/S 前进/后退 (fwd)\n"
            "  A/D 左转/右转 (yaw 左正右负)\n"
            f"  {stop_help} | Q 退出\n"
        )

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            fwd, yaw = 0.0, 0.0
            while True:
                now = time.time()
                while select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ("\x03", "q", "Q"):
                        raise KeyboardInterrupt
                    if release_mode == "latch":
                        register_key_latched(latched, ch)
                    else:
                        register_key_hold(keys, ch, now)

                if release_mode == "latch":
                    fwd, yaw = velocities_from_latched(latched, speed=speed)
                else:
                    fwd, yaw = velocities_from_hold(
                        keys,
                        speed=speed,
                        now=now,
                        hold_initial=hold_initial,
                        hold_release=hold_release,
                    )
                link.send_velocity(fwd, yaw)
                time.sleep(interval)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    except KeyboardInterrupt:
        print("\n退出 …")
    finally:
        try:
            link.send_velocity(0.0, 0.0, force=True)
            if setup:
                link.send_frame(build_training_stop())
        except Exception:
            pass
        link.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WASD 串口速度遥控 (TRAINING_CHANGE)")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--speed",
        type=float,
        default=MAX_VELO,
        help=f"单轴按键速度 (默认 {MAX_VELO})",
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="发送频率 Hz")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="启动时先发 init + start",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="打印每次速度变化",
    )
    parser.add_argument(
        "--release-mode",
        choices=("hold", "latch"),
        default="hold",
        help="hold=松键停(默认,自适应连发); latch=按一下保持到反向/空格",
    )
    parser.add_argument(
        "--hold-initial",
        type=float,
        default=DEFAULT_HOLD_INITIAL_SEC,
        help=f"hold: 首键到终端连发前的保持时间 (默认 {DEFAULT_HOLD_INITIAL_SEC})",
    )
    parser.add_argument(
        "--hold-release",
        type=float,
        default=DEFAULT_HOLD_RELEASE_SEC,
        help=f"hold: 已连发后松键停延迟 (默认 {DEFAULT_HOLD_RELEASE_SEC})",
    )
    args = parser.parse_args(argv)

    try:
        return run_teleop(
            args.port,
            args.baud,
            speed=args.speed,
            rate=args.rate,
            setup=args.setup,
            verbose=args.verbose,
            release_mode=args.release_mode,
            hold_initial=args.hold_initial,
            hold_release=args.hold_release,
        )
    except (SerialException, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
