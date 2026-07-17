#!/usr/bin/env python3
"""摄像头参数实时调节预览（不跑巡线，纯看画面效果）。

用键盘调曝光/gain/gamma/brightness，实时看画面变化。
q 退出，s 截图保存到当前目录。

用法：
  python3 cam_tune.py                    # 用默认摄像头
  python3 cam_tune.py --camera 0
  python3 cam_tune.py --auto-exposure    # 从自动曝光开始
"""
import argparse
import subprocess
import sys
import time

import cv2 as cv
import numpy as np

DEV = "/dev/video0"


def get_ctrl(name):
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", DEV, f"--get-ctrl={name}"],
            capture_output=True, text=True, timeout=1.0,
        )
        for line in r.stdout.splitlines():
            if ":" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return None


def set_ctrl(name, val):
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", DEV, f"--set-ctrl={name}={int(val)}"],
            capture_output=True, timeout=1.0,
        )
    except Exception:
        pass


# 各控件的调节步长
CONTROLS = [
    # (显示名, v4l2控件名, 步长, 最小, 最大)
    ("exposure", "exposure_time_absolute", 50, 1, 10000),
    ("gain", "gain", 5, 0, 128),
    ("gamma", "gamma", 25, 100, 500),
    ("brightness", "brightness", 5, -64, 64),
    ("contrast", "contrast", 5, 0, 100),
    ("saturation", "saturation", 5, 0, 100),
]


def main():
    p = argparse.ArgumentParser(description="摄像头参数实时调节预览")
    p.add_argument("--camera", default=0)
    p.add_argument("--auto-exposure", action="store_true", help="从自动曝光开始")
    args = p.parse_args()

    global DEV
    DEV = f"/dev/video{args.camera}" if isinstance(args.camera, int) else str(args.camera)

    cap = cv.VideoCapture(args.camera, cv.CAP_V4L2)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        return 1
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始模式
    if args.auto_exposure:
        set_ctrl("auto_exposure", 3)  # 自动
        mode = "AUTO"
    else:
        set_ctrl("auto_exposure", 1)  # 手动
        mode = "MANUAL"

    # 读当前值
    vals = {}
    for disp, ctrl, _, _, _ in CONTROLS:
        vals[disp] = get_ctrl(ctrl) or 0
    auto_exp = get_ctrl("auto_exposure")

    selected = 0  # 当前选中的控件索引
    print("摄像头参数调节预览")
    print("=" * 50)
    print("操作（纯字母键，不依赖方向键）:")
    print("  k / w    选中上一个控件")
    print("  j / x    选中下一个控件")
    print("  h / -    减小当前控件值")
    print("  l / =    增大当前控件值")
    print("  a        切换自动/手动曝光")
    print("  空格      截图保存")
    print("  r        重置到默认")
    print("  q / ESC  退出")
    print("=" * 50)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv.resize(frame, (640, 480))

        # 计算亮度直方图
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mean_y = float(gray.mean())
        # 过曝/欠曝像素比例
        over = float(np.sum(gray > 240)) / gray.size * 100
        under = float(np.sum(gray < 16)) / gray.size * 100

        # OSD
        osd = [
            f"mode={mode}  meanY={mean_y:.0f}  over>240={over:.0f}%  under<16={under:.0f}%",
            "",
        ]
        for i, (disp, ctrl, step, vmin, vmax) in enumerate(CONTROLS):
            marker = "▶" if i == selected else " "
            osd.append(f"  {marker} {disp:12s}: {vals[disp]:>6.0f}  (step {step}, {vmin}~{vmax})")

        y0 = 20
        for i, line in enumerate(osd):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv.putText(frame, line, (8, y0 + i * 22),
                        cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)

        # 亮度直方图（右下角小图）
        hist = cv.calcHist([gray], [0], None, [64], [0, 256])
        hist_h = 60
        hist_w = 160
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv.normalize(hist, hist, 0, hist_h, cv.NORM_MINMAX)
        for j in range(64):
            h = int(hist[j])
            x = j * hist_w // 64
            w = hist_w // 64
            color = (0, 0, 200) if j > 60 else (0, 200, 0) if j < 4 else (200, 200, 200)
            cv.rectangle(hist_img, (x, hist_h - h), (x + w, hist_h), color, -1)
        frame[480 - hist_h - 8: 480 - 8, 480 - hist_w - 8: 480 - 8] = hist_img

        cv.imshow("cam-tune", frame)
        key = cv.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:  # q 或 ESC
            break
        elif key == ord("a"):
            # 切换自动/手动
            if mode == "MANUAL":
                set_ctrl("auto_exposure", 3)
                mode = "AUTO"
                print("→ 自动曝光")
            else:
                set_ctrl("auto_exposure", 1)
                mode = "MANUAL"
                print("→ 手动曝光")
        elif key == ord(" "):  # 空格截图（不用 s，避免和别的冲突）
            fname = f"cam_snap_{int(time.time())}.jpg"
            cv.imwrite(fname, frame)
            print(f"截图: {fname}")
        elif key == ord("r"):
            # 重置到常用默认
            set_ctrl("auto_exposure", 1)
            mode = "MANUAL"
            set_ctrl("exposure_time_absolute", 500)
            set_ctrl("gain", 60)
            set_ctrl("gamma", 250)
            set_ctrl("brightness", 5)
            set_ctrl("contrast", 54)
            set_ctrl("saturation", 64)
            for disp, ctrl, _, _, _ in CONTROLS:
                vals[disp] = get_ctrl(ctrl) or 0
            print("→ 重置默认")
        elif key == ord("k") or key == ord("w"):
            # 上（选中上一个控件）
            selected = (selected - 1) % len(CONTROLS)
        elif key == ord("j") or key == ord("x"):
            # 下（选中下一个控件）
            selected = (selected + 1) % len(CONTROLS)
        elif key == ord("h") or key == ord("-") or key == ord("_"):
            # 减
            disp, ctrl, step, vmin, vmax = CONTROLS[selected]
            new_v = max(vmin, vals[disp] - step)
            set_ctrl(ctrl, new_v)
            vals[disp] = get_ctrl(ctrl) or new_v
            print(f"  {disp}: {vals[disp]:.0f}")
        elif key == ord("l") or key == ord("=") or key == ord("+"):
            # 增
            disp, ctrl, step, vmin, vmax = CONTROLS[selected]
            new_v = min(vmax, vals[disp] + step)
            set_ctrl(ctrl, new_v)
            vals[disp] = get_ctrl(ctrl) or new_v
            print(f"  {disp}: {vals[disp]:.0f}")

    cap.release()
    cv.destroyAllWindows()
    # 打印最终参数
    print("\n最终参数（可填入巡线程序）:")
    for disp, ctrl, _, _, _ in CONTROLS:
        v = get_ctrl(ctrl)
        if v is not None:
            print(f"  {disp:20s}: {v:.0f}")
    ae = get_ctrl("auto_exposure")
    print(f"  {'auto_exposure':20s}: {ae:.0f} ({'AUTO' if ae and ae>=3 else 'MANUAL'})")


if __name__ == "__main__":
    sys.exit(main())
