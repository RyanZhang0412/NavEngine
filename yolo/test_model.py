#!/usr/bin/env python3
"""最小视频流验证：摄像头 + best.pt 分割推理 + 场景标签。

用法（必须在 NavEngine 根目录，或 python -m yolo.test_model）:
  source .venv-yolo/bin/activate
  python -m yolo.test_model

SSH 在你电脑转发后浏览器看:
  ssh -L 8081:127.0.0.1:8081 orin@<Orin的IP>
  http://127.0.0.1:8081/
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2 as cv
import numpy as np

from yolo import DEFAULT_MODEL
from yolo.scene import select_scene_label
from yolo.viz import render_frame

DEFAULT_PORT = 8081


def _is_ssh() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


def _open_camera(camera: int | str) -> cv.VideoCapture:
    if isinstance(camera, int):
        cap = cv.VideoCapture(camera, cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(camera)
    else:
        cap = cv.VideoCapture(str(camera), cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(str(camera))
    if cap.isOpened():
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    return cap


class _Feed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._scene = "无检测"
        self._fps = 0.0

    def update(self, vis: np.ndarray, scene: str, fps: float) -> None:
        ok, buf = cv.imencode(".jpg", vis, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._scene = scene
            self._fps = fps

    def snapshot(self) -> tuple[bytes | None, str, float]:
        with self._lock:
            return self._jpeg, self._scene, self._fps


def _http_handler(feed: _Feed) -> type[BaseHTTPRequestHandler]:
    boundary = b"frame"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                _, scene, fps = feed.snapshot()
                html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>yolo test</title></head><body style="margin:0;background:#111;color:#eee;text-align:center">
<h2>YOLO 验证流</h2><img src="/stream.mjpg" style="max-width:100%">
<p>{scene} | {fps:.1f} fps</p></body></html>""".encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        jpeg, _, _ = feed.snapshot()
                        if jpeg and jpeg is not last:
                            last = jpeg
                            self.wfile.write(b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg + b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path == "/snapshot.jpg":
                jpeg, _, _ = feed.snapshot()
                if not jpeg:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO 最小视频流验证")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--camera", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0", help="0=GPU, cpu=CPU")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--no-half", action="store_false", dest="half")
    p.add_argument("--port", type=int, default=None, help=f"HTTP 预览端口（SSH 默认 {DEFAULT_PORT}）")
    p.add_argument("--no-http", action="store_true")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print("请先: cd NavEngine && source .venv-yolo/bin/activate", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    use_half = args.half and use_cuda
    if use_cuda:
        torch.cuda.empty_cache()

    model = YOLO(args.model)
    print(f"model={args.model}")
    print(f"classes={model.names}  cuda={use_cuda} half={use_half} imgsz={args.imgsz}")

    try:
        cam: int | str = int(args.camera)
    except ValueError:
        cam = args.camera
    cap = _open_camera(cam)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}", file=sys.stderr)
        return 1

    http_port = 0 if args.no_http else (args.port if args.port is not None else (DEFAULT_PORT if _is_ssh() else 0))
    gui = bool(os.environ.get("DISPLAY")) and http_port == 0
    feed: _Feed | None = None
    httpd: ThreadingHTTPServer | None = None
    if http_port:
        feed = _Feed()
        httpd = ThreadingHTTPServer(("127.0.0.1", http_port), _http_handler(feed))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"HTTP 预览 http://127.0.0.1:{http_port}/")
        if _is_ssh():
            print(f"SSH 转发: ssh -L {http_port}:127.0.0.1:{http_port} orin@<Orin的IP>")
    if gui:
        print("本地窗口预览，按 q 退出")
    elif not http_port:
        print("无 DISPLAY 且未开 HTTP，仅终端打印结果")
    print("Ctrl+C 停止\n")

    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            t_inf = time.perf_counter()
            results = model.predict(
                frame, conf=args.conf, device=args.device,
                imgsz=args.imgsz, half=use_half, verbose=False,
            )
            infer_ms = (time.perf_counter() - t_inf) * 1000.0
            vis = render_frame(frame, results[0], min_conf=args.conf)
            scene = select_scene_label(results[0], min_conf=args.conf)
            scene_txt = f"{scene.class_name} {scene.confidence:.2f}" if scene else "无检测"

            n += 1
            elapsed = time.perf_counter() - t0
            fps = n / elapsed if elapsed > 0 else 0.0
            cv.putText(vis, f"fps:{fps:.1f} infer:{infer_ms:.0f}ms", (10, 58),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            if feed is not None:
                feed.update(vis, scene_txt, fps)
            if gui:
                cv.imshow("yolo-test", vis)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            if n == 1 or n % 30 == 0:
                print(f"frame {n} fps={fps:.1f} infer={infer_ms:.0f}ms scene={scene_txt}")
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        cap.release()
        if httpd is not None:
            httpd.shutdown()
        if gui:
            cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
