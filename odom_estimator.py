#!/usr/bin/env python3
"""里程估计 + 障碍判断（独立模块）。

思路（尽量简单，只为做对比）:
  1. 后台线程读串口，解析主控返回的 DATA 帧 JSON，取 velo_fwd / velo_yaw。
  2. 用差速模型把速度量化成左右轮「转动距离」（单位任意，能对比即可）:
        v_left  = velo_fwd - 0.5 * WHEEL_BASE * velo_yaw
        v_right = velo_fwd + 0.5 * WHEEL_BASE * velo_yaw
        d_left  = Σ v_left  * dt
        d_right = Σ v_right * dt
        total   = Σ velo_fwd * dt
  3. 只在巡线(FOLLOW)段累积里程；TURN/PAUSE/BLOCKED 不累积。
  4. 完整一段 = 上一趟 TURN 结束(捕获→FOLLOW) 到 本趟 TURN 开始(线尾掉头)。
     冷启动首段：从首次开跑到第一次 TURN 开始（尚无上一趟 TURN）。
  5. 丢线时判断: 当前 segment 里程 vs 基线
        - 远小于基线 → 障碍 → BLOCKED
        - 够长 → 线尾 → PAUSE→TURN，并把该 segment 记入历史

只在 FOLLOW 且 segment 已 begin 时累积；TURN/PAUSE/BLOCKED 不累积。

与 NavLink 共用一个串口（默认 /dev/ttyCH341USB0）：NavLink 写 init/运动，本模块读反馈 JSON。
用法见文件末尾 demo，或被 follow_line_yolo_2.py 导入。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ui_ctrl.constants import CmdCtrl, CmdType
from ui_ctrl.message import ProtocolMessage
from ui_ctrl.protocol import build_ctrl, verify_frame
from ui_ctrl.stream_parser import StreamParser
from ui_ctrl.training_ctrl import ROBOT_KEY, TrainingProgram

# ── 可调参数 ────────────────────────────────────────────────
WHEEL_BASE = 1.0               # 归一化轮距；单位任意，只做对比
HISTORY_SIZE = 6               # 保留最近 N 个正常 segment 作基线
BLOCKED_RATIO = 0.5            # 已废弃语义，保留兼容；请用 TURN_MIN_RATIO
TURN_MIN_RATIO = 0.65          # 当前段 >= 基线最小值×此比例 → 视为线尾，允许掉头
MIN_TURN_NO_BASELINE = 0.15    # 尚无历史基线时，至少累积这么多才允许丢线掉头
MIN_BASELINE_SEGMENTS = 2      # 基线至少要几个 segment 才用比例判断
ABS_MIN_SEGMENT = 0.05         # 绝对下限：低于此必判障碍（起步抖动）
FEEDBACK_TIMEOUT = 0.5         # 超过这么久没收到反馈，改用「指令速度」估算
VELO_KEYS = ("velo_fwd", "velo_yaw")  # 反馈 JSON 里的速度字段名

DEFAULT_SEGMENT_FILE = Path(__file__).resolve().parent / "data" / "odom_segments.jsonl"


def load_segment_records(path: Path) -> list[dict[str, Any]]:
    """读取历史段记录（JSONL，每行一段）。"""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_segment_record(
    path: Path,
    seg: Segment,
    *,
    segment_index: int,
    reason: str,
) -> Path:
    """段生命周期结束时追加一行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "segment": segment_index,
        "total": round(seg.total, 4),
        "left": round(seg.left, 4),
        "right": round(seg.right, 4),
        "dur_sec": round(max(0.0, seg.t1 - seg.t0), 2),
        "reason": reason,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def build_zero_velocity_change() -> bytes:
    """TRAINING_CHANGE 零速；勿用 build_training_change()，其默认 velo_yaw=0.5。"""
    body = {
        "id": 0,
        "autorun": 1,
        "training_set": 0,
        "training_program": int(TrainingProgram.ISOKINETIC),
        ROBOT_KEY: {"velo_fwd": 0.0, "velo_yaw": 0.0},
    }
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8") + b"\x00"
    frame = build_ctrl(CmdCtrl.TRAINING_CHANGE, payload)
    if not verify_frame(frame):
        raise RuntimeError("zero velocity frame failed verification")
    return frame


@dataclass
class Segment:
    total: float = 0.0
    left: float = 0.0
    right: float = 0.0
    t0: float = 0.0
    t1: float = 0.0


def _as_float(v: Any) -> float | None:
    """主控 RUNNING_INFO 里 velo 常为字符串，如 \"0.0675\"。"""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _find_velo(obj: Any) -> tuple[float | None, float | None]:
    """在嵌套 JSON 里递归找 velo_fwd / velo_yaw。"""
    if isinstance(obj, dict):
        fwd = _as_float(obj.get(VELO_KEYS[0]))
        yaw = _as_float(obj.get(VELO_KEYS[1]))
        if fwd is not None and yaw is not None:
            return fwd, yaw
        for v in obj.values():
            r = _find_velo(v)
            if r[0] is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_velo(v)
            if r[0] is not None:
                return r
    return None, None


class OdomEstimator:
    """从主控反馈 JSON 估算里程，并判断丢线时是否为障碍。"""

    def __init__(
        self,
        ser,
        *,
        wheel_base: float = WHEEL_BASE,
        segment_file: Path | str | None = DEFAULT_SEGMENT_FILE,
    ) -> None:
        self._ser = ser
        self._wheel_base = wheel_base
        self._segment_file = (
            Path(segment_file) if segment_file is not None else None
        )
        self._parser = StreamParser(on_frame=self._on_frame)
        self._lock = threading.Lock()
        self._running = False
        self._rx_thread: threading.Thread | None = None

        # 反馈速度（零阶保持）
        self._fb_fwd = 0.0
        self._fb_yaw = 0.0
        self._fb_time: float | None = None
        self._last_fb_time: float | None = None
        self._rx_frames = 0
        self._rx_velo_frames = 0

        # 指令速度（反馈超时时兜底）
        self._cmd_fwd = 0.0
        self._cmd_yaw = 0.0

        # 累积状态
        self._accumulating = False
        self._segment_open = False
        self._cur = Segment()
        self._history: list[float] = []
        self._last_segment_total = 0.0
        self._segment_count = 0
        self._bootstrap_from_file()

    def _bootstrap_from_file(self) -> None:
        if self._segment_file is None:
            return
        records = load_segment_records(self._segment_file)
        if not records:
            return
        totals = [float(r["total"]) for r in records if "total" in r]
        if not totals:
            return
        self._history = totals[-HISTORY_SIZE:]
        self._segment_count = len(records)
        self._last_segment_total = totals[-1]

    # ---------- 串口读取 ----------
    def start(self) -> None:
        if self._rx_thread and self._rx_thread.is_alive():
            return
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="odom-rx", daemon=True
        )
        self._rx_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None

    def _rx_loop(self) -> None:
        assert self._ser is not None
        while self._running:
            self.poll_serial()
            time.sleep(0.01)

    def poll_serial(self) -> None:
        """非阻塞读取串口可用字节并喂给解析器（供主循环每帧调用）。

        实现要点（Jetson + CH341 踩坑记录）：
        - 绝不调用 in_waiting：它是持有 GIL 的 ioctl，CH341 驱动异常时会在
          内核阻塞，冻结整个解释器（包括其他串口的读线程）。
        - 改用 timeout=0 的非阻塞 read：pyserial 内部走 select()，会正常
          释放 GIL，驱动异常时只会返回空数据，不会卡死。
        - 因此 odom 不再开后台读线程（start() 保留但主程序不再调用），
          由 follow 主循环在 on_timer 里调用本方法。
        """
        if self._ser is None:
            return
        try:
            if self._ser.timeout != 0:
                self._ser.timeout = 0
            chunk = self._ser.read(4096)
        except Exception:
            return
        if chunk:
            self._parser.feed(chunk)

    def _on_frame(self, msg: ProtocolMessage) -> None:
        if msg.cmd_type != CmdType.DATA:
            return
        with self._lock:
            self._rx_frames += 1
        try:
            text = msg.payload.rstrip(b"\x00").decode("utf-8", errors="replace")
            obj = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        fwd, yaw = _find_velo(obj)
        if fwd is None:
            return
        now = time.perf_counter()
        with self._lock:
            self._fb_fwd = fwd
            self._fb_yaw = yaw
            self._last_fb_time = now
            self._rx_velo_frames += 1

    # ---------- 指令兜底 ----------
    def feed_command(self, fwd: float, yaw: float) -> None:
        """主控循环每帧调用，记录当前下发速度（反馈超时时用它估算）。"""
        with self._lock:
            self._cmd_fwd = fwd
            self._cmd_yaw = yaw

    # ---------- 积分 ----------
    def _step(self) -> None:
        """用当前速度推进一个微元 dt，累加到当前 segment。"""
        now = time.perf_counter()
        with self._lock:
            last = self._fb_time if self._fb_time is not None else now
            dt = now - last
            self._fb_time = now
            if not self._accumulating or dt <= 0 or dt > 1.0:
                return

            # 选速度源：近期有反馈用反馈，否则用指令兜底
            use_fb = (
                self._last_fb_time is not None
                and (now - self._last_fb_time) <= FEEDBACK_TIMEOUT
            )
            fwd = self._fb_fwd if use_fb else self._cmd_fwd
            yaw = self._fb_yaw if use_fb else self._cmd_yaw

            half = 0.5 * self._wheel_base
            v_left = fwd - half * yaw
            v_right = fwd + half * yaw
            self._cur.total += fwd * dt
            self._cur.left += v_left * dt
            self._cur.right += v_right * dt
            self._cur.t1 = now

    # ---------- segment 控制 ----------
    def begin_segment(self) -> None:
        """TURN 结束进入 FOLLOW 时（或冷启动首段）：开新段，开始累积。"""
        with self._lock:
            self._cur = Segment(t0=time.perf_counter(), t1=time.perf_counter())
            self._fb_time = None
            self._accumulating = True
            self._segment_open = True

    def pause_accum(self) -> None:
        """PAUSE/BLOCKED/空格暂停：暂停累积，段不结束。"""
        with self._lock:
            self._accumulating = False

    def resume_accum(self) -> None:
        """同一段内恢复 FOLLOW：继续累积。"""
        with self._lock:
            if not self._segment_open:
                return
            self._fb_time = None
            self._accumulating = True

    def end_segment(self, *, reason: str = "") -> Segment:
        """线尾进入 TURN 时：结束当前段，记入历史，并写入段文件。"""
        with self._lock:
            self._accumulating = False
            seg = Segment(
                total=self._cur.total,
                left=self._cur.left,
                right=self._cur.right,
                t0=self._cur.t0,
                t1=self._cur.t1,
            )
            if self._segment_open:
                self._history.append(seg.total)
                if len(self._history) > HISTORY_SIZE:
                    self._history = self._history[-HISTORY_SIZE:]
                self._last_segment_total = seg.total
                self._segment_count += 1
                if self._segment_file is not None:
                    append_segment_record(
                        self._segment_file,
                        seg,
                        segment_index=self._segment_count,
                        reason=reason,
                    )
            self._segment_open = False
            self._cur = Segment()
            return seg

    # ---------- 查询 ----------
    @property
    def total(self) -> float:
        with self._lock:
            return self._cur.total

    @property
    def left(self) -> float:
        with self._lock:
            return self._cur.left

    @property
    def right(self) -> float:
        with self._lock:
            return self._cur.right

    @property
    def baseline_min(self) -> float:
        """历史正常段的最小里程（保守基线）。"""
        with self._lock:
            if len(self._history) < MIN_BASELINE_SEGMENTS:
                return 0.0
            return min(self._history)

    @property
    def history(self) -> list[float]:
        with self._lock:
            return list(self._history)

    @property
    def segment_open(self) -> bool:
        with self._lock:
            return self._segment_open

    @property
    def last_segment_total(self) -> float:
        with self._lock:
            return self._last_segment_total

    @property
    def segment_file(self) -> Path | None:
        return self._segment_file

    @property
    def expected_length(self) -> float | None:
        """预估本段全长：历史段均值；无历史时用上一段。"""
        with self._lock:
            if self._history:
                return sum(self._history) / len(self._history)
            if self._last_segment_total > 0:
                return self._last_segment_total
        return None

    def progress_pct(self) -> float | None:
        """当前段完成百分比（相对预估全长），无基线时返回 None。"""
        with self._lock:
            exp = None
            if self._history:
                exp = sum(self._history) / len(self._history)
            elif self._last_segment_total > 0:
                exp = self._last_segment_total
            if exp is None or exp <= 0:
                return None
            return 100.0 * self._cur.total / exp

    def progress_label(self) -> str:
        """UI 用：'50%' 或 '--%'（尚无预估全长）。"""
        pct = self.progress_pct()
        if pct is None:
            return "--%"
        return f"{min(999.0, pct):.0f}%"

    @property
    def segment_count(self) -> int:
        with self._lock:
            return self._segment_count

    def session_summary(self) -> str:
        """会话结束或写日志用：上一完整段 + 当前未结束段。"""
        with self._lock:
            open_s = "Y" if self._segment_open else "N"
            exp = None
            if self._history:
                exp = sum(self._history) / len(self._history)
            elif self._last_segment_total > 0:
                exp = self._last_segment_total
            if exp is not None and exp > 0:
                prog = f"{min(999.0, 100.0 * self._cur.total / exp):.0f}%"
            else:
                prog = "--%"
            exp_s = f"{exp:.3f}" if exp is not None else "n/a"
            return (
                f"seg#{self._segment_count} open={open_s} "
                f"cur_tot={self._cur.total:.3f} last_tot={self._last_segment_total:.3f} "
                f"exp={exp_s} prog={prog} hist={list(self._history)}"
            )

    def should_turn(self) -> bool:
        """里程够长：丢线应按线尾掉头。仅时间到（PAUSE_HOLD）不够。"""
        cur = self.total
        if cur < ABS_MIN_SEGMENT:
            return False
        with self._lock:
            n = len(self._history)
        if n < MIN_BASELINE_SEGMENTS:
            return cur >= MIN_TURN_NO_BASELINE
        base = self.baseline_min
        if base <= 0:
            return cur >= MIN_TURN_NO_BASELINE
        return cur >= base * TURN_MIN_RATIO

    def should_block(self) -> bool:
        """里程太短：丢线应停等障碍，永不因定时器掉头。"""
        return not self.should_turn()

    @property
    def rx_frames(self) -> int:
        with self._lock:
            return self._rx_frames

    @property
    def rx_velo_frames(self) -> int:
        with self._lock:
            return self._rx_velo_frames

    @property
    def last_velo(self) -> tuple[float, float]:
        with self._lock:
            return self._fb_fwd, self._fb_yaw

    def status(self) -> str:
        fwd, yaw = self.last_velo
        return (
            f"odom tot={self.total:.3f} L={self.left:.3f} R={self.right:.3f} "
            f"vel=({fwd:+.4f},{yaw:+.4f}) rx={self.rx_velo_frames}/{self.rx_frames} "
            f"base_min={self.baseline_min:.3f} hist={self.history} "
            f"turn={self.should_turn()}"
        )


def _send_stop(ser) -> None:
    """退出时零速 + stop，避免主控保持训练态。"""
    from ui_ctrl.training_ctrl import build_training_stop

    try:
        ser.write(build_zero_velocity_change())
        ser.flush()
        time.sleep(0.05)
        ser.write(build_training_stop())
        ser.flush()
    except Exception:
        pass


# ── demo: 单独跑，只听主控反馈并打印里程 ─────────────────────
def _demo() -> int:
    import argparse
    import os

    import serial

    from ui_ctrl.training_ctrl import build_training_init, build_training_start

    default_port = os.environ.get("NAV_SERIAL_PORT", "/dev/ttyCH341USB0")
    parser = argparse.ArgumentParser(
        description=(
            "odom demo：监听 RUNNING_INFO 里的 velo_fwd/velo_yaw。"
            " init 只回 TRAINING_STATE；要速度反馈需 start（--setup）。"
        ),
    )
    parser.add_argument("--port", default=default_port)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--setup",
        action="store_true",
        help="先发 init+start（零速），让主控开始推送 RUNNING_INFO",
    )
    args = parser.parse_args()

    ser = serial.Serial(args.port, baudrate=args.baud, timeout=0.05)
    ser.reset_input_buffer()
    print(f"串口 {args.port} @ {args.baud}")

    started = False
    if args.setup:
        for label, frame in (
            ("init", build_training_init()),
            ("start", build_training_start()),
            ("change", build_zero_velocity_change()),
        ):
            ser.write(frame)
            ser.flush()
            print(f"[TX] {label} ({len(frame)}B)")
        started = True
        time.sleep(0.3)
    else:
        print(
            "仅监听；init 无 RUNNING_INFO。"
            "加 --setup（init+start+零速）或先跑巡线。",
        )

    odom = OdomEstimator(ser)
    odom.start()
    odom.begin_segment()
    try:
        while True:
            odom.feed_command(0.0, 0.0)
            odom._step()
            print(odom.status(), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n退出 …")
    finally:
        odom.stop()
        if started:
            print("[TX] stop（关闭训练）")
            _send_stop(ser)
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
