#!/usr/bin/env python3
"""超声波模块实时监控（命令行调试工具，非巡线运行时组件）。

用途
    独立运行的串口监视器，用于排查超声波模块是否正常工作、协议是否正确、
    距离数据是否合理。巡线主程序不调用本文件，巡线时用的是 us_sensor.py。

硬件/协议（与 us_sensor.py 一致）
    - 默认 /dev/ttyCH341USB1 @ 9600。
    - 每帧一行 ASCII：T:<温度>C D:<距离>cm，以 \r\n 结尾。
    - D=-1 表示前方无回波 = 前方开阔无障碍（合法值，非错误）。
    - D>=0 表示检测到障碍，数值为距离 cm。

用法
    .venv/bin/python us_monitor.py                      # 实时显示 T/D
    .venv/bin/python us_monitor.py --raw                # 同时打印原始字节
    .venv/bin/python us_monitor.py --log logs/us.log    # 追加写日志文件
    .venv/bin/python us_monitor.py --stats              # 每 3 秒打印帧率/距离分布
    .venv/bin/python us_monitor.py --port /dev/ttyCH341USB1 --baud 9600

输出格式（每帧一行）
    HH:MM:SS.mmm  T=27.62C  D= -1.00cm  [OPEN]    # D<0：前方开阔
    HH:MM:SS.mmm  T=27.62C  D= 30.50cm  [    ]    # D>=0：障碍距离

按 Ctrl+C 退出（自动关闭串口）。
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import serial

DEFAULT_PORT = "/dev/ttyCH341USB1"
DEFAULT_BAUD = 9600

# 单帧正则：T:<温度>C D:<距离>cm，温度距离均允许负值。
_FRAME_RE = re.compile(
    r"T\s*:\s*(-?\d+(?:\.\d+)?)\s*C\s+D\s*:\s*(-?\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)

# D <= 该值表示前方无回波 = 前方开阔无障碍（模块合法返回值，非错误）。
# 统计时这类帧计入 invalid_d（不算作有效障碍距离样本），但仍是合法帧。
NO_OBSTACLE_CM = 0.0
RECONNECT_SEC = 1.0      # 串口打开失败后重试间隔（秒）
READ_TIMEOUT = 0.5       # 单次 read 超时（秒）


def parse_frame(line: str) -> tuple[float, float] | None:
    """从一行文本解析 (温度, 距离)；不匹配返回 None。"""
    m = _FRAME_RE.search(line)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def open_port(port: str, baud: int) -> serial.Serial | None:
    """打开串口并清空输入缓冲；失败时打印错误并返回 None（不抛异常）。"""
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=READ_TIMEOUT)
        ser.reset_input_buffer()
        return ser
    except Exception as e:
        print(f"[!] 打开 {port} 失败: {e}", file=sys.stderr, flush=True)
        return None


class Stats:
    """滑动窗口统计：帧率、有效距离分布、无效（无回波）帧计数。

    用于 --stats 模式下周期性输出模块健康度。窗口默认 100 帧。
    """

    def __init__(self, window: int = 100) -> None:
        """初始化统计器；window 为滑动窗口帧数（用于 fps 计算）。"""
        self.window = window
        self._stamps: deque[float] = deque(maxlen=window)   # 帧时间戳，用于算 fps
        self._valid_d: deque[float] = deque(maxlen=window)  # 有效距离样本（D>0）
        self.frames = 0          # 累计成功解析帧数
        self.bad = 0             # 累计无法解析行数
        self.invalid_d = 0       # 累计 D<=0（无回波）帧数

    def on_frame(self, t: float, d: float) -> None:
        """记录一帧：累加计数、推入时间戳；D>0 时还推入有效距离样本。"""
        self.frames += 1
        now = time.perf_counter()
        self._stamps.append(now)
        if d > NO_OBSTACLE_CM:
            self._valid_d.append(d)
        else:
            self.invalid_d += 1

    def on_bad(self) -> None:
        """记录一行无法解析的数据（格式不符）。"""
        self.bad += 1

    def fps(self) -> float:
        """按滑动窗口内首尾帧时间差估算帧率；样本不足返回 0。"""
        if len(self._stamps) < 2:
            return 0.0
        dt = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / dt if dt > 0 else 0.0

    def summary(self) -> str:
        """汇总成单行字符串：frames/bad/fps + 有效距离 min/max/avg + 无回波帧数。"""
        parts = [f"frames={self.frames}", f"bad={self.bad}", f"fps={self.fps():.1f}"]
        if self._valid_d:
            d = self._valid_d
            parts.append(
                f"valid={len(d)} d_min={min(d):.2f} d_max={max(d):.2f} "
                f"d_avg={statistics.mean(d):.2f}cm"
            )
        parts.append(f"invalid(sentinel)={self.invalid_d}")
        return " | ".join(parts)


def loop(
    port: str,
    baud: int,
    *,
    raw: bool,
    log_path: Path | None,
    stats_on: bool,
) -> int:
    """主循环：保持串口打开、持续读、按行解析打印。

    参数：
        port/ baud: 串口配置。
        raw: True 时同时打印每次 read 的原始字节（调试协议用）。
        log_path: 非 None 时把每行解析结果追加写入该文件。
        stats_on: True 时每 3 秒打印一次 Stats.summary()。

    串口掉线时自动重连（每 RECONNECT_SEC 秒）。Ctrl+C 退出时清理串口与日志。
    """
    ser: serial.Serial | None = None
    log_fp = open(log_path, "a", encoding="utf-8") if log_path else None
    st = Stats()
    buf = bytearray()                       # 跨 read 的不完整行缓冲
    last_stats_t = time.perf_counter()

    print(f"监控 {port} @ {baud}  (Ctrl+C 退出)", flush=True)
    if log_fp:
        print(f"日志: {log_path.resolve()}", flush=True)

    try:
        while True:
            # 串口未就绪：尝试打开，失败则等待后重试
            if ser is None:
                ser = open_port(port, baud)
                if ser is None:
                    time.sleep(RECONNECT_SEC)
                    continue
                print(f"[+] 已连接 {port}", flush=True)
                buf.clear()

            chunk = ser.read(128)
            if not chunk:
                # 无数据时也周期性打印统计
                if stats_on and time.perf_counter() - last_stats_t > 3.0:
                    print(f"[stats] {st.summary()}", flush=True)
                    last_stats_t = time.perf_counter()
                continue

            if raw:
                print(f"[raw {len(chunk)}B] {chunk!r}", flush=True)

            buf.extend(chunk)
            # 按换行符切分完整行（兼容 \n、\r\n、\r）
            while True:
                idx_lf = buf.find(0x0A)
                if idx_lf < 0:
                    break
                raw_line = bytes(buf[: idx_lf + 1])
                del buf[: idx_lf + 1]
                line = raw_line.decode("ascii", errors="replace").strip().rstrip("\r")
                if not line:
                    continue

                parsed = parse_frame(line)
                if parsed is None:
                    st.on_bad()
                    print(f"[?] 无法解析: {line!r}", flush=True)
                    if log_fp:
                        log_fp.write(f"{time_str()} PARSE? {line}\n")
                    continue

                t_c, d_cm = parsed
                st.on_frame(t_c, d_cm)
                # D<0 显示 OPEN（前方开阔），否则显示空 flag
                if d_cm < 0:
                    flag = "OPEN"
                else:
                    flag = "    "
                out = f"{time_str()}  T={t_c:>6.2f}C  D={d_cm:>8.2f}cm  [{flag}]"
                print(out, flush=True)
                if log_fp:
                    log_fp.write(out + "\n")

            # 有数据流时也按周期打印统计
            if stats_on and time.perf_counter() - last_stats_t > 3.0:
                print(f"[stats] {st.summary()}", flush=True)
                last_stats_t = time.perf_counter()

    except KeyboardInterrupt:
        print("\n退出 …", flush=True)
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        if log_fp:
            log_fp.close()
        if stats_on:
            print(f"[stats final] {st.summary()}", flush=True)
    return 0


def time_str() -> str:
    """当前本地时间字符串，精确到毫秒：HH:MM:SS.mmm。"""
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


def main(argv=None) -> int:
    """解析命令行参数并进入监控主循环。"""
    p = argparse.ArgumentParser(description="超声波模块串口监控")
    p.add_argument("--port", default=DEFAULT_PORT, help=f"串口设备 (默认 {DEFAULT_PORT})")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"波特率 (默认 {DEFAULT_BAUD})")
    p.add_argument("--raw", action="store_true", help="同时打印原始字节")
    p.add_argument("--log", default=None, help="追加写入的日志文件路径")
    p.add_argument("--stats", action="store_true", help="周期性打印帧率/距离统计")
    args = p.parse_args(argv)

    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    return loop(args.port, args.baud, raw=args.raw, log_path=log_path, stats_on=args.stats)


if __name__ == "__main__":
    raise SystemExit(main())
