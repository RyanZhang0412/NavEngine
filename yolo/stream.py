#!/usr/bin/env python3
"""
YOLO 分割 MJPEG 网页流（SSH 端口转发后在 PC 浏览器观看）。

  cd ~/Desktop/NavEngine
  source .venv-yolo/bin/activate
  python -m yolo.stream
  python -m yolo.stream --imgsz 640 --port 8081
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv

from yolo import DEFAULT_MODEL
from yolo.scene import select_scene_label
from yolo.viz import render_frame

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_PORT = 8081


def open_camera(camera: int | str, width: int, height: int) -> cv.VideoCapture:
    if isinstance(camera, int):
        cap = cv.VideoCapture(camera, cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(camera)
    else:
        cap = cv.VideoCapture(str(camera), cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(str(camera))
    if cap.isOpened():
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    return cap


class YoloStreamSource:
    def __init__(
        self,
        cap: cv.VideoCapture,
        model,
        *,
        conf: float,
        device: str,
        imgsz: int,
        half: bool,
        quality: int,
    ) -> None:
        self._cap = cap
        self._model = model
        self._conf = conf
        self._device = device
        self._imgsz = imgsz
        self._half = half
        self._quality = quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._scene: str = ""
        self._fps: float = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        n = 0
        t0 = time.perf_counter()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            t_inf = time.perf_counter()
            results = self._model.predict(
                frame,
                conf=self._conf,
                device=self._device,
                imgsz=self._imgsz,
                half=self._half,
                verbose=False,
            )
            vis = render_frame(frame, results[0], min_conf=self._conf)
            infer_ms = (time.perf_counter() - t_inf) * 1000.0

            ok, buf = cv.imencode(".jpg", vis, [int(cv.IMWRITE_JPEG_QUALITY), self._quality])
            if not ok:
                continue

            n += 1
            elapsed = time.perf_counter() - t0
            fps = n / elapsed if elapsed > 0 else 0.0
            scene = "无检测"
            s = select_scene_label(results[0], min_conf=self._conf)
            if s:
                scene = f"{s.class_name} {s.confidence:.2f}"

            with self._lock:
                self._jpeg = buf.tobytes()
                self._scene = scene
                self._fps = fps

            if n == 1 or n % 30 == 0:
                print(f"frame {n} fps={fps:.1f} infer={infer_ms:.0f}ms scene={scene}")

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def get_status(self) -> tuple[str, float]:
        with self._lock:
            return self._scene, self._fps

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3.0)
        self._cap.release()


def make_handler(source: YoloStreamSource) -> type[BaseHTTPRequestHandler]:
    boundary = b"frame"

    class Handler(BaseHTTPRequestHandler):
        server_version = "NavEngineYolo/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[{self.address_string()}] {fmt % args}")

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
            scene, fps = source.get_status()
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLO Stream</title>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; }}
  main {{ padding:12px; text-align:center; }}
  img {{ max-width:100%; border:1px solid #444; }}
  p {{ font-size:14px; color:#aaa; }}
</style></head>
<body><main>
<h1>YOLO 分割流</h1>
<img src="/stream.mjpg" alt="live">
<p>scene: {scene} | fps: {fps:.1f}</p>
<p><a href="/stream.mjpg">/stream.mjpg</a> |
<a href="/snapshot.jpg">/snapshot.jpg</a></p>
</main></body></html>""".encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def _send_snapshot(self) -> None:
            data = source.get_jpeg()
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
            last_sent: bytes | None = None
            try:
                while True:
                    data = source.get_jpeg()
                    if data and data is not last_sent:
                        last_sent = data
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


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO seg MJPEG stream")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--camera", default="0")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--no-half", action="store_false", dest="half")
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--lan", action="store_true", help="listen 0.0.0.0")
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
    print(f"model={args.model} classes={model.names}")
    print(f"cuda={use_cuda} half={use_half} imgsz={args.imgsz}")

    try:
        camera: int | str = int(args.camera)
    except ValueError:
        camera = args.camera

    cap = open_camera(camera, args.width, args.height)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}", file=sys.stderr)
        return 1

    source = YoloStreamSource(
        cap, model,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        half=use_half,
        quality=max(1, min(100, args.quality)),
    )
    host = "0.0.0.0" if args.lan else args.host
    httpd = ThreadingHTTPServer((host, args.port), make_handler(source))

    print("YOLO 视频流已启动")
    print(f"  http://127.0.0.1:{args.port}/")
    print(f"  SSH: ssh -L {args.port}:127.0.0.1:{args.port} orin@<ip>")
    print("Ctrl+C 停止")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止…")
    finally:
        httpd.shutdown()
        source.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
