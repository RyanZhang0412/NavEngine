#!/usr/bin/env python3
"""验证 yolo/models/best.pt 推理与分割高亮。"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2 as cv

from yolo import DEFAULT_MODEL, YOLO_ROOT
from yolo.scene import select_scene_label
from yolo.viz import render_frame


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--camera", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0", help="0=GPU, cpu=CPU")
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸，显存不足时用 320/256")
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--no-half", action="store_false", dest="half")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", default=str(YOLO_ROOT / "preview.jpg"))
    p.add_argument("--frames", type=int, default=0)
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
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} half={use_half} imgsz={args.imgsz}")

    if use_cuda:
        torch.cuda.empty_cache()

    model = YOLO(args.model)
    print("classes:", model.names)

    try:
        cam_id: int | str = int(args.camera)
    except ValueError:
        cam_id = args.camera
    cap = cv.VideoCapture(cam_id, cv.CAP_V4L2)
    if not cap.isOpened():
        cap = cv.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}", file=sys.stderr)
        return 1
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    gui = not args.headless and has_display()
    if not gui:
        if not args.headless:
            print("未检测到 DISPLAY，自动 headless（结果写入文件）")
        out_path = Path(args.output)
        print(f"headless 模式，预览: {out_path}")
        print("Ctrl+C 停止" if not args.frames else f"跑 {args.frames} 帧后退出")
    else:
        print("按 q 退出")

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            results = model.predict(
                frame, conf=args.conf, device=args.device,
                imgsz=args.imgsz, half=use_half, verbose=False,
            )
            vis = render_frame(frame, results[0], min_conf=args.conf)
            scene = select_scene_label(results[0], min_conf=args.conf)
            n += 1
            det_msg = f"{scene.class_name} {scene.confidence:.2f}" if scene else "none"
            if gui:
                cv.imshow("yolo-seg test", vis)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv.imwrite(str(out_path), vis)
                if n == 1 or n % 30 == 0:
                    print(f"frame {n} scene={det_msg} -> {out_path}")
                if args.frames and n >= args.frames:
                    break
                time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        cap.release()
        if gui:
            cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
