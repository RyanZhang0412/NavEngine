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

SSH 远程采集（浏览器预览，端口转发到你电脑）:
  python3 capture_dataset.py
  # 在你电脑上执行（重连 SSH 时加 -L）:
  ssh -L 8082:127.0.0.1:8082 orin@<orin的IP>
  # 浏览器打开 http://127.0.0.1:8082/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2 as cv
import numpy as np


# ── 摄像头默认参数（原 follow_line.py）─────────────────────────────
DEFAULT_EXPOSURE = 500
DEFAULT_GAIN = 60
DEFAULT_WB_TEMP = 4800


@dataclass(frozen=True)
class CameraSettings:
    """摄像头成像参数。默认 auto=True（与 follow_line_yolo_2 --auto-exposure 一致）。"""

    auto: bool = True
    exposure: float = DEFAULT_EXPOSURE
    gain: float = DEFAULT_GAIN
    wb_temperature: float = DEFAULT_WB_TEMP
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None


def configure_camera(cap: cv.VideoCapture, settings: CameraSettings) -> None:
    """配置曝光与白平衡。本机 USB Camera 经 V4L2/OpenCV 实测可用。"""
    if settings.auto:
        # 自动曝光：V4L2 常见值为 3；部分 UVC 驱动用 0.75
        if not cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3):
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv.CAP_PROP_AUTO_WB, 1)
    else:
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv.CAP_PROP_EXPOSURE, settings.exposure)
        cap.set(cv.CAP_PROP_GAIN, settings.gain)
        cap.set(cv.CAP_PROP_AUTO_WB, 0)
        cap.set(cv.CAP_PROP_WB_TEMPERATURE, settings.wb_temperature)

    for prop, value in (
        (cv.CAP_PROP_BRIGHTNESS, settings.brightness),
        (cv.CAP_PROP_CONTRAST, settings.contrast),
        (cv.CAP_PROP_SATURATION, settings.saturation),
    ):
        if value is not None:
            cap.set(prop, value)

    for _ in range(3):
        cap.read()

    ae = cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv.CAP_PROP_EXPOSURE)
    gain = cap.get(cv.CAP_PROP_GAIN)
    awb = cap.get(cv.CAP_PROP_AUTO_WB)
    wb = cap.get(cv.CAP_PROP_WB_TEMPERATURE)
    mode = "自动曝光" if settings.auto else "手动锁定"
    print(
        f"摄像头参数({mode}): ae={ae:.0f} exp={exp:.0f} gain={gain:.0f} | "
        f"awb={awb:.0f} temp={wb:.0f}K"
    )


def open_camera(camera, settings: CameraSettings | None = None) -> cv.VideoCapture:
    """优先 V4L2 打开摄像头，避免 GStreamer 管道卡住。"""
    settings = settings or CameraSettings()
    if isinstance(camera, int):
        cap = cv.VideoCapture(camera, cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(camera)
    else:
        device = str(camera)
        cap = cv.VideoCapture(device, cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(device)
    if cap.isOpened():
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        configure_camera(cap, settings)
    return cap


def ensure_display() -> None:
    """SSH 会话未继承 DISPLAY 时，默认连本机 :0 以便弹窗。"""
    if sys.platform == "win32":
        return
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print("DISPLAY 未设置，已自动设为 :0")


DEFAULT_CAMERA = 0
DEFAULT_FPS = 1.0
DEFAULT_OUTPUT = "datasets/line"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_HTTP_PORT = 8082


def _is_ssh_session() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


class PreviewFeed:
    """供 HTTP MJPEG 使用的最新帧缓存。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._saved = 0

    def update(self, frame: np.ndarray, saved: int) -> None:
        ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._saved = saved

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def get_saved(self) -> int:
        with self._lock:
            return self._saved


def _make_http_handler(feed: PreviewFeed) -> type[BaseHTTPRequestHandler]:
    boundary = b"frame"

    class Handler(BaseHTTPRequestHandler):
        server_version = "CaptureDataset/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[http] {fmt % args}")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send_index()
            elif self.path == "/stream.mjpg":
                self._send_mjpeg(boundary)
            elif self.path == "/snapshot.jpg":
                self._send_snapshot()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_index(self) -> None:
            saved = feed.get_saved()
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>capture_dataset</title>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; text-align:center; }}
  img {{ max-width:100%; border:1px solid #444; }}
  p {{ color:#aaa; font-size:14px; }}
</style></head>
<body>
<h1>采集预览</h1>
<img src="/stream.mjpg" alt="live">
<p>saved: {saved}</p>
<p><a href="/snapshot.jpg">snapshot.jpg</a></p>
</body></html>""".encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def _send_snapshot(self) -> None:
            data = feed.get_jpeg()
            if not data:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame yet")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_mjpeg(self, boundary: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}"
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            last: bytes | None = None
            try:
                while True:
                    data = feed.get_jpeg()
                    if data and data is not last:
                        last = data
                        self.wfile.write(b"--" + boundary + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def _start_http_preview(feed: PreviewFeed, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_http_handler(feed))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _print_ssh_hint(port: int) -> None:
    print(f"HTTP 预览: http://127.0.0.1:{port}/  （Orin 本机）")
    print("在你电脑上转发端口（新开终端，或重连 SSH 时加上 -L）:")
    print(f"  ssh -L {port}:127.0.0.1:{port} orin@<Orin的IP>")
    print(f"然后浏览器打开: http://127.0.0.1:{port}/")


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
        "--lock",
        action="store_true",
        help="固定曝光/白平衡（默认自动曝光，与 follow_line_2 --auto-exposure 一致）",
    )
    p.add_argument(
        "--exposure",
        type=float,
        default=None,
        help=f"手动曝光值（默认 {DEFAULT_EXPOSURE}；仅 --lock 时生效）",
    )
    p.add_argument(
        "--flip",
        action="store_true",
        default=True,
        help="水平镜像预览/保存（与 follow_line 一致，默认开启）",
    )
    p.add_argument("--no-flip", action="store_false", dest="flip")
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="不弹本地 OpenCV 窗口",
    )
    p.add_argument(
        "--http-port",
        type=int,
        default=None,
        help=f"HTTP 网页预览端口（SSH 默认 {_is_ssh_session() and DEFAULT_HTTP_PORT or '关闭'}）",
    )
    p.add_argument("--no-http", action="store_true", help="关闭 HTTP 网页预览")
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


def _annotate_frame(frame: np.ndarray, saved: int) -> np.ndarray:
    vis = frame.copy()
    cv.putText(
        vis, f"saved: {saved}", (10, 28),
        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2,
    )
    return vis


def _show_preview(frame: np.ndarray, saved: int, preview: bool) -> bool:
    """显示预览；返回 True 表示用户按 q 请求退出。"""
    if not preview:
        return False
    vis = _annotate_frame(frame, saved)
    cv.putText(vis, "q=quit", (10, 56), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv.imshow("capture_dataset", vis)
    # waitKey(30) 比 1ms 更利于 GTK 窗口在 Jetson/SSH 上刷新
    return (cv.waitKey(30) & 0xFF) == ord("q")


def _open_preview_window() -> None:
    cv.namedWindow("capture_dataset", cv.WINDOW_AUTOSIZE)
    cv.moveWindow("capture_dataset", 80, 80)
    if sys.platform != "win32":
        try:
            cv.startWindowThread()
        except cv.error:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0:
        print("--fps 必须大于 0", file=sys.stderr)
        return 1

    ensure_display()
    ssh = _is_ssh_session()
    if args.http_port is None:
        http_port = DEFAULT_HTTP_PORT if ssh and not args.no_http else 0
    else:
        http_port = max(0, int(args.http_port))
    if args.no_http:
        http_port = 0

    # SSH 下用浏览器预览；本地有显示器时用 OpenCV 窗口
    preview = (
        not args.no_preview
        and bool(os.environ.get("DISPLAY"))
        and http_port == 0
    )
    if ssh and http_port and not preview:
        print("检测到 SSH 会话，已启用 HTTP 网页预览（无需 Orin 本地显示器）")
    elif not preview and not args.no_preview and http_port == 0:
        print("未检测到 DISPLAY，仅保存不预览（可加 --http-port 8082）")

    interval = 1.0 / args.fps
    session_dir = _make_session_dir(Path(args.output))
    settings = CameraSettings(
        auto=not args.lock,
        exposure=args.exposure if args.exposure is not None else DEFAULT_EXPOSURE,
    )

    use_url = bool(args.url.strip())
    cap: cv.VideoCapture | None = None
    httpd: ThreadingHTTPServer | None = None
    http_feed: PreviewFeed | None = PreviewFeed() if http_port else None

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
    if http_port and http_feed is not None:
        httpd = _start_http_preview(http_feed, http_port)
        _print_ssh_hint(http_port)
    if preview:
        _open_preview_window()
        print("本地预览窗口已开启（q 或 Ctrl+C 停止）")
    elif not http_port:
        print("按 Ctrl+C 停止")
    print()

    saved = 0
    next_save = time.monotonic() + interval  # 先预览，间隔后再存第一张
    quit_requested = False

    # 预热摄像头并立刻弹出第一帧，避免“已开启但黑屏”
    if not use_url and cap is not None:
        for _ in range(5):
            ok, warm = cap.read()
            if ok and warm is not None:
                if args.flip:
                    warm = cv.flip(warm, 1)
                if http_feed is not None:
                    http_feed.update(_annotate_frame(warm, saved), saved)
                if preview:
                    _show_preview(warm, saved, preview=True)
                break
            time.sleep(0.05)

    try:
        while not quit_requested:
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
                if args.flip:
                    frame = cv.flip(frame, 1)
                if http_feed is not None:
                    http_feed.update(_annotate_frame(frame, saved), saved)
                if preview and _show_preview(frame, saved, preview=True):
                    quit_requested = True
                    continue
                now = time.monotonic()
                if now < next_save:
                    continue

            if use_url and args.flip:
                frame = cv.flip(frame, 1)
            if use_url and http_feed is not None:
                http_feed.update(_annotate_frame(frame, saved), saved)

            name = f"img_{saved + 1:06d}.jpg"
            path = session_dir / name
            if not cv.imwrite(str(path), frame):
                print(f"写入失败: {path}", file=sys.stderr)
                return 1

            saved += 1
            next_save = now + interval
            if http_feed is not None:
                http_feed.update(_annotate_frame(frame, saved), saved)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] 已保存 {saved}: {name}")

    except KeyboardInterrupt:
        print("\n停止采集")
    finally:
        if cap is not None:
            cap.release()
        if httpd is not None:
            httpd.shutdown()
        if preview:
            cv.destroyAllWindows()

    print(f"共保存 {saved} 张 -> {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
