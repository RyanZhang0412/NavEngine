#!/usr/bin/env python3
"""
从摄像头或 YOLO 流 / 其他网页快照按固定帧率保存图像，用于巡线模型训练数据。

直接读摄像头（与 follow_line 相同曝光/白平衡，便于和巡线一致）:
  python3 capture_dataset.py
  python3 capture_dataset.py --fps 1 --output datasets/line

YOLO 流已在跑时，从快照接口采集（不占 /dev/video0）:
  source .venv-yolo/bin/activate && python -m yolo.stream
  python3 capture_dataset.py --url http://127.0.0.1:8081/snapshot.jpg

按 Ctrl+C 停止；图像保存在 output/<会话时间>/ 下。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from follow_line import CameraSettings, open_camera

DEFAULT_CAMERA = 0
DEFAULT_FPS = 1.0
DEFAULT_OUTPUT = "datasets/line"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save camera frames for line-detection dataset")
    p.add_argument("--camera", default=str(DEFAULT_CAMERA), help="camera index or /dev/video0")
    p.add_argument(
        "--url",
        default="",
        help="HTTP snapshot URL (e.g. http://127.0.0.1:8080/snapshot.jpg); skips local camera",
    )
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="save rate in images per second")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="dataset root directory")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--max", type=int, default=0, help="stop after N images (0 = unlimited)")
    p.add_argument(
        "--no-lock",
        action="store_true",
        help="do not lock exposure/WB (default matches follow_line)",
    )
    return p.parse_args(argv)


def _parse_camera(camera_arg: str) -> int | str:
    try:
        return int(camera_arg)
    except ValueError:
        return camera_arg


def _device_path(camera: int | str) -> str | None:
    if isinstance(camera, int):
        return f"/dev/video{camera}"
    if str(camera).startswith("/dev/video"):
        return str(camera)
    return None


def _warn_if_busy(device: str) -> None:
    try:
        out = subprocess.run(
            ["fuser", "-v", device],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    if out.returncode == 0 and out.stderr.strip():
        print(f"警告: {device} 可能被占用:\n{out.stderr}", file=sys.stderr)
        print("可先结束占用摄像头的程序，或用 --url 从 YOLO 流快照采集。", file=sys.stderr)


def _fetch_url(url: str) -> np.ndarray | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"拉取快照失败: {e}", file=sys.stderr)
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def _make_session_dir(root: Path) -> Path:
    session = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    return session


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0:
        print("--fps 必须大于 0", file=sys.stderr)
        return 1

    interval = 1.0 / args.fps
    session_dir = _make_session_dir(Path(args.output))
    settings = CameraSettings(lock=not args.no_lock)

    use_url = bool(args.url.strip())
    cap: cv.VideoCapture | None = None

    if use_url:
        url = args.url.strip()
        print(f"数据源: {url}")
    else:
        camera = _parse_camera(args.camera)
        dev = _device_path(camera)
        if dev:
            _warn_if_busy(dev)
        cap = open_camera(camera, settings)
        if not cap.isOpened():
            print(f"无法打开摄像头: {args.camera}", file=sys.stderr)
            return 1
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"摄像头: {args.camera}  分辨率: {args.width}x{args.height}")

    print(f"保存目录: {session_dir.resolve()}")
    print(f"帧率: {args.fps} 张/秒  (间隔 {interval:.3f}s)")
    print("按 Ctrl+C 停止\n")

    saved = 0
    next_save = time.monotonic()

    try:
        while True:
            if args.max > 0 and saved >= args.max:
                break

            if use_url:
                now = time.monotonic()
                if now < next_save:
                    time.sleep(min(0.05, next_save - now))
                    continue
                frame = _fetch_url(args.url.strip())
                if frame is None:
                    time.sleep(0.2)
                    continue
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                now = time.monotonic()
                if now < next_save:
                    continue

            name = f"img_{saved + 1:06d}.jpg"
            path = session_dir / name
            if not cv.imwrite(str(path), frame):
                print(f"写入失败: {path}", file=sys.stderr)
                return 1

            saved += 1
            next_save += interval
            if next_save < now - interval:
                next_save = now + interval

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] 已保存 {saved}: {name}")

    except KeyboardInterrupt:
        print("\n停止采集")
    finally:
        if cap is not None:
            cap.release()

    print(f"共保存 {saved} 张 -> {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
