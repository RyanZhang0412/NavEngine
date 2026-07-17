#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

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

from follow_common import ManyImgs, color_follow, read_HSV, write_HSV

DEFAULT_PORT = "/dev/ttyCH341USB0"
DEFAULT_BAUD = 921600
DEFAULT_HSV_FILE = str(Path(__file__).resolve().parent / "colorHSV.text")
DEFAULT_CAMERA = 0
# 摄像头参数：直接改下面三个数（无命令行参数）
# AUTO_EXPOSURE=1 为手动曝光，AUTO_WB=0 后 WB_TEMPERATURE 生效
DEFAULT_EXPOSURE = 500
DEFAULT_GAIN = 60
DEFAULT_WB_TEMP = 4800


@dataclass(frozen=True)
class CameraSettings:
    """固定摄像头成像参数，避免巡线时 HSV 因自动曝光/白平衡漂移。"""

    lock: bool = True
    exposure: float = DEFAULT_EXPOSURE
    gain: float = DEFAULT_GAIN
    wb_temperature: float = DEFAULT_WB_TEMP
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None


def configure_camera(cap: cv.VideoCapture, settings: CameraSettings) -> None:
    """锁定曝光与白平衡。本机 USB Camera 经 V4L2/OpenCV 实测可用。"""
    if not settings.lock:
        return

    # 手动曝光：必须先设 AUTO_EXPOSURE=1，再设 EXPOSURE/GAIN。
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv.CAP_PROP_EXPOSURE, settings.exposure)
    cap.set(cv.CAP_PROP_GAIN, settings.gain)

    # 固定白平衡：先关 AUTO_WB，再设色温 (约 2800–6500)。
    cap.set(cv.CAP_PROP_AUTO_WB, 0)
    cap.set(cv.CAP_PROP_WB_TEMPERATURE, settings.wb_temperature)

    for prop, value in (
        (cv.CAP_PROP_BRIGHTNESS, settings.brightness),
        (cv.CAP_PROP_CONTRAST, settings.contrast),
        (cv.CAP_PROP_SATURATION, settings.saturation),
    ):
        if value is not None:
            cap.set(prop, value)

    # 丢弃前几帧，等驱动应用参数。
    for _ in range(3):
        cap.read()

    ae = cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv.CAP_PROP_EXPOSURE)
    gain = cap.get(cv.CAP_PROP_GAIN)
    awb = cap.get(cv.CAP_PROP_AUTO_WB)
    wb = cap.get(cv.CAP_PROP_WB_TEMPERATURE)
    print(
        f"摄像头固定参数: 手动曝光 ae={ae:.0f} exp={exp:.0f} gain={gain:.0f} | "
        f"固定白平衡 awb={awb:.0f} temp={wb:.0f}K"
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


def resolve_port(port: str) -> str:
    if sys.platform == "win32":
        return port
    upper = port.upper()
    if upper.startswith("COM") and upper[3:].isdigit():
        return f"/dev/ttyS{int(upper[3:]) - 1}"
    return port


def build_velocity_change(fwd: float, yaw: float) -> bytes:
    body = {
        "id": 0,
        "autorun": 1,
        "training_set": 0,
        "training_program": int(TrainingProgram.ISOKINETIC),
        ROBOT_KEY: {
            "velo_fwd": fwd,
            "velo_yaw": yaw,
        },
    }
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8") + b"\x00"
    frame = build_ctrl(CmdCtrl.TRAINING_CHANGE, payload)
    if not verify_frame(frame):
        raise RuntimeError("built velocity frame failed verification")
    return frame


class NavLink:
    def __init__(self, port: str, baud: int) -> None:
        self._port = resolve_port(port)
        self._baud = baud
        self._ser = None

    def open(self) -> None:
        self._ser = serial.Serial(self._port, baudrate=self._baud, timeout=0.05)
        self._ser.reset_input_buffer()
        print(f"串口 {self._port} @ {self._baud}")

    def _send(self, frame: bytes) -> None:
        if self._ser is None:
            raise RuntimeError("serial not open")
        self._ser.write(frame)
        self._ser.flush()

    def setup(self) -> None:
        self._send(build_training_init())
        time.sleep(0.2)
        self._send(build_training_start())
        time.sleep(0.2)

    def teardown(self) -> None:
        try:
            self._send(build_training_stop())
        except Exception:
            pass

    def stop(self) -> None:
        self._send(build_velocity_change(0.0, 0.0))

    def move(self, fwd: float, yaw: float) -> None:
        self._send(build_velocity_change(fwd, yaw))

    @property
    def serial(self):
        """与 OdomEstimator 共用同一串口（只写 move / 只读反馈）。"""
        return self._ser

    def close(self) -> None:
        if self._ser is not None:
            try:
                self.stop()
            except Exception:
                pass
            self._ser.close()
            self._ser = None


class LineDetectPoly:
    def __init__(
        self,
        link: NavLink,
        hsv_file: str = DEFAULT_HSV_FILE,
        camera=0,
        *,
        cam_settings: CameraSettings | None = None,
    ):
        self.link = link
        self._cam_settings = cam_settings or CameraSettings()
        self.end = 0.0
        self.dyn_update = False
        self.select_flags = False
        self.Track_state = 'identify'
        self.windows_name = 'frame'
        self.cols, self.rows = 0, 0
        self.Mouse_XY = (0, 0)
        self._quit = False

        self.hsv_text = hsv_file
        self.color = color_follow()
        
        # Speed
        self.linear = 0.1
        self.max_angular = 0.03

        # Centerline tracking params
        self.near_y_ratio = 0.82
        self.far_y_ratio = 0.28
        self.lookahead_y_ratio = 0.65  # 局部拟合参考高度
        self.kp_track = 0.03
        self.ka_track = 0.02
        self.kd_track = 0.03
        self.curvature_gain = 0.5
        self.min_pixels = 90
        self.center_deadband_px = 12
        self.row_step = 4
        self.min_run_width = 8
        self.endpoint_trim = 2
        self.local_fit_halfwin = 70
        self.local_fit_min_points = 10
        self.horizontal_mode_ratio = 2.2

        self.prev_error = 0.0
        self.last_ctrl_time = None
        self.last_angular = 0.0
        self.recovering_line = False
        self.lost_turn_angular = 0.28

        self.hsv_range = ()
        self.Roi_init = ()

        self.img_flip = True
        self._camera_id = camera
        self._cam_fail_count = 0
        self._last_cam_warn = 0.0
        self.capture = open_camera(camera)
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera}")

    def _reopen_camera(self) -> bool:
        if self.capture is not None:
            self.capture.release()
        self.capture = open_camera(self._camera_id)
        return self.capture.isOpened()

    def onMouse(self, event, x, y, flags, param):
        if event == 1:
            self.Track_state = 'init'
            self.select_flags = True
            self.Mouse_XY = (x, y)
        if event == 4:
            self.select_flags = False
            self.Track_state = 'mouse'
        if self.select_flags:
            self.cols = min(self.Mouse_XY[0], x), min(self.Mouse_XY[1], y)
            self.rows = max(self.Mouse_XY[0], x), max(self.Mouse_XY[1], y)
            self.Roi_init = (self.cols[0], self.cols[1], self.rows[0], self.rows[1])

    def _extract_binary_lane(self, rgb_img):
        """Build lane binary mask directly from HSV threshold."""
        h, w = rgb_img.shape[:2]
        work = rgb_img.copy()
        work[0:int(h * (1.0 / 3.0)), 0:w] = 0
        hsv_img = cv.cvtColor(work, cv.COLOR_BGR2HSV)
        hsv_img = cv.GaussianBlur(hsv_img, (5, 5), 0)
        lower = np.array(self.hsv_range[0], dtype=np.uint8)
        upper = np.array(self.hsv_range[1], dtype=np.uint8)
        mask = cv.inRange(hsv_img, lower, upper)

        kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        binary = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_close)
        _, binary = cv.threshold(binary, 10, 255, cv.THRESH_BINARY)
        return binary

    def _largest_lane_component(self, binary_img):
        num, labels, stats, _ = cv.connectedComponentsWithStats(binary_img, connectivity=8)
        if num <= 1:
            return None
        areas = stats[1:, cv.CC_STAT_AREA]
        best_id = 1 + int(np.argmax(areas))
        if stats[best_id, cv.CC_STAT_AREA] < self.min_pixels:
            return None
        mask = np.zeros_like(binary_img, dtype=np.uint8)
        mask[labels == best_id] = 255
        return mask

    def _extract_centerline_points(self, lane_mask):
        h, w = lane_mask.shape[:2]
        # Match centerline extraction ROI with binary ROI (keep bottom two-thirds).
        y0 = int(h * (1.0 / 3.0))
        step = max(1, int(self.row_step))
        min_w = max(2, int(self.min_run_width))

        left_points = []
        right_points = []
        for y in range(h - 1, y0 - 1, -step):
            xs = np.where(lane_mask[y] > 0)[0]
            if xs.size < 2:
                continue

            # Use row-wise outer boundaries to avoid inner-hole noise.
            left = float(np.min(xs))
            right = float(np.max(xs))
            if (right - left) < min_w:
                continue
            left_points.append((left, float(y)))
            right_points.append((right, float(y)))

        if len(left_points) < 8:
            return None

        left_pts = np.array(left_points, dtype=np.float64)
        right_pts = np.array(right_points, dtype=np.float64)
        left_pts = left_pts[np.argsort(left_pts[:, 1])]
        right_pts = right_pts[np.argsort(right_pts[:, 1])]

        ys = left_pts[:, 1]
        x_left = left_pts[:, 0]
        x_right = right_pts[:, 0]

        y_min_all = float(np.min(ys))
        y_max_all = float(np.max(ys))
        y_ref = y_min_all + float(np.clip(self.lookahead_y_ratio, 0.45, 0.9)) * (y_max_all - y_min_all)
        half_win = max(20, int(self.local_fit_halfwin))
        mask_local = np.abs(ys - y_ref) <= half_win
        min_pts = max(6, int(self.local_fit_min_points))

        # Local fit around lookahead region; fallback to global when points are insufficient.
        if np.count_nonzero(mask_local) >= min_pts:
            ys_fit = ys[mask_local]
            x_left_fit = x_left[mask_local]
            x_right_fit = x_right[mask_local]
            y_fit_min = float(np.min(ys_fit))
            y_fit_max = float(np.max(ys_fit))
        else:
            ys_fit = ys
            x_left_fit = x_left
            x_right_fit = x_right
            y_fit_min = y_min_all
            y_fit_max = y_max_all

        # When lane is near-horizontal in image, x=f(y) polynomial becomes ill-conditioned.
        # In that case, use raw boundary samples (smoothed) instead of quadratic fitting.
        x_span = float(np.max(x_right_fit) - np.min(x_left_fit))
        y_span = float(np.max(ys_fit) - np.min(ys_fit))
        ratio = x_span / max(1.0, y_span)
        use_horizontal_mode = ratio > float(self.horizontal_mode_ratio)

        if use_horizontal_mode:
            y_samples = ys_fit.copy()
            left_fit = x_left_fit.copy()
            right_fit = x_right_fit.copy()
            if left_fit.size >= 5:
                k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
                k /= np.sum(k)
                pad = len(k) // 2
                left_fit = np.convolve(np.pad(left_fit, (pad, pad), mode='edge'), k, mode='valid')
                right_fit = np.convolve(np.pad(right_fit, (pad, pad), mode='edge'), k, mode='valid')
        else:
            # Fit lane boundaries first, then derive centerline.
            left_coeff = np.polyfit(ys_fit, x_left_fit, 2)
            right_coeff = np.polyfit(ys_fit, x_right_fit, 2)
            y_samples = np.linspace(y_fit_min, y_fit_max, len(ys_fit))
            left_fit = np.polyval(left_coeff, y_samples)
            right_fit = np.polyval(right_coeff, y_samples)
        valid = (left_fit >= 0) & (right_fit < w) & ((right_fit - left_fit) >= min_w)
        if np.count_nonzero(valid) < 8:
            return None
        y_samples = y_samples[valid]
        left_fit = left_fit[valid]
        right_fit = right_fit[valid]
        center_x = 0.5 * (left_fit + right_fit)
        pts = np.stack([center_x, y_samples], axis=1)

        # Smooth centerline lightly with edge padding to avoid endpoint distortion.
        if pts.shape[0] >= 5:
            xs = pts[:, 0]
            kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
            kernel /= np.sum(kernel)
            pad = len(kernel) // 2
            xs_pad = np.pad(xs, (pad, pad), mode='edge')
            xs_sm = np.convolve(xs_pad, kernel, mode='valid')
            pts[:, 0] = xs_sm

        # Trim unstable endpoints (top/bottom few points).
        trim_n = max(0, int(self.endpoint_trim))
        if trim_n > 0 and pts.shape[0] > (2 * trim_n + 6):
            pts = pts[trim_n:-trim_n]
            left_fit = left_fit[trim_n:-trim_n]
            right_fit = right_fit[trim_n:-trim_n]
            y_samples = y_samples[trim_n:-trim_n]
        left_line = np.stack([left_fit, y_samples], axis=1)
        right_line = np.stack([right_fit, y_samples], axis=1)
        return pts, left_line, right_line

    def _compute_control(self, centerline_pts, img_w):
        ys = centerline_pts[:, 1]
        xs = centerline_pts[:, 0]
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        if y_max - y_min < 30:
            return None

        near_ratio = float(np.clip(self.near_y_ratio, 0.45, 0.95))
        far_ratio = float(np.clip(self.far_y_ratio, 0.05, 0.55))
        y_near = y_min + near_ratio * (y_max - y_min)
        y_far = y_min + far_ratio * (y_max - y_min)
        x_near = float(np.interp(y_near, ys, xs))
        x_far = float(np.interp(y_far, ys, xs))
        center_x = img_w / 2.0
        lateral = x_near - center_x
        heading = x_near - x_far

        if abs(lateral) < self.center_deadband_px:
            lateral = 0.0

        now = time.perf_counter()
        first = self.last_ctrl_time is None
        if first:
            dt = 0.03
        else:
            dt = max(1e-3, now - self.last_ctrl_time)
        self.last_ctrl_time = now

        if first:
            self.prev_error = lateral
        d_lateral = (lateral - self.prev_error) / dt
        self.prev_error = lateral

        dx_dy = np.gradient(xs, ys)
        d2x_dy2 = np.gradient(dx_dy, ys)
        near_idx = int(np.argmin(np.abs(ys - y_near)))
        dy = float(dx_dy[near_idx])
        ddy = float(d2x_dy2[near_idx])
        curvature = abs(ddy) / ((1.0 + dy * dy) ** 1.5 + 1e-6)

        sign = -1.0 if lateral >= 0 else 1.0
        omega = -(
            self.kp_track * lateral + self.kd_track * d_lateral
        ) + self.ka_track * heading + sign * self.curvature_gain * curvature

        omega = float(np.clip(omega, -self.max_angular, self.max_angular))
        omega = 0.75 * self.last_angular + 0.25 * omega
        self.last_angular = omega
        return omega, x_near, y_near, x_far, y_far, curvature, y_min, y_max

    def _start_recovery(self):
        self.recovering_line = True
        self.prev_error = 0.0
        self.last_ctrl_time = None
        self.last_angular = 0.0

    def _stop_recovery(self):
        self.recovering_line = False
        self.prev_error = 0.0
        self.last_ctrl_time = None
        self.last_angular = 0.0

    def recover_line(self):
        yaw = self.lost_turn_angular if not self.img_flip else -self.lost_turn_angular
        self.link.move(0.0, yaw)

    def execute(self, omega):
        yaw = omega if not self.img_flip else -omega
        self.link.move(self.linear, yaw)

    def process(self, rgb_img, action):
        rgb_img = cv.resize(rgb_img, (640, 480))
        if self.img_flip:
            rgb_img = cv.flip(rgb_img, 1)

        if action == 32:
            self.Track_state = 'tracking'
        elif action == ord('i'):
            self.Track_state = 'identify'
        elif action == ord('r'):
            self.Reset()
        elif action == ord('q'):
            self._quit = True

        if self.Track_state == 'init':
            cv.namedWindow(self.windows_name, cv.WINDOW_AUTOSIZE)
            cv.setMouseCallback(self.windows_name, self.onMouse, 0)
            if self.select_flags:
                cv.line(rgb_img, self.cols, self.rows, (255, 0, 0), 2)
                cv.rectangle(rgb_img, self.cols, self.rows, (0, 255, 0), 2)
                if self.Roi_init[0] != self.Roi_init[2] and self.Roi_init[1] != self.Roi_init[3]:
                    rgb_img, self.hsv_range = self.color.Roi_hsv(rgb_img, self.Roi_init)
                    self.dyn_update = True
        elif self.Track_state == 'identify':
            if os.path.exists(self.hsv_text):
                self.hsv_range = read_HSV(self.hsv_text)
            else:
                self.Track_state = 'init'

        binary = []
        if self.Track_state != 'init' and len(self.hsv_range) != 0:
            binary = self._extract_binary_lane(rgb_img)

            if self.dyn_update:
                write_HSV(self.hsv_text, self.hsv_range)
                self.dyn_update = False

            if isinstance(binary, np.ndarray):
                lane_mask = self._largest_lane_component(binary)
                center_data = None if lane_mask is None else self._extract_centerline_points(lane_mask)
                if center_data is None:
                    cv.putText(rgb_img, "fit: lost", (30, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    if self.Track_state == 'tracking':
                        self._start_recovery()
                        self.recover_line()
                else:
                    centerline, left_line, right_line = center_data
                    pts = np.array([[int(x), int(y)] for x, y in centerline], dtype=np.int32)
                    left_pts = np.array([[int(x), int(y)] for x, y in left_line], dtype=np.int32)
                    right_pts = np.array([[int(x), int(y)] for x, y in right_line], dtype=np.int32)
                    cv.polylines(rgb_img, [left_pts], isClosed=False, color=(0, 255, 0), thickness=2)
                    cv.polylines(rgb_img, [right_pts], isClosed=False, color=(255, 0, 0), thickness=2)
                    cv.polylines(rgb_img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                    for p in pts[::3]:
                        cv.circle(rgb_img, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)
                    if isinstance(lane_mask, np.ndarray):
                        lane_vis = cv.cvtColor(lane_mask, cv.COLOR_GRAY2BGR)
                        cv.polylines(lane_vis, [left_pts], isClosed=False, color=(0, 255, 0), thickness=2)
                        cv.polylines(lane_vis, [right_pts], isClosed=False, color=(255, 0, 0), thickness=2)
                        cv.polylines(lane_vis, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                        binary = lane_vis

                    ctrl = self._compute_control(centerline, rgb_img.shape[1])
                    if ctrl is None:
                        cv.putText(rgb_img, "ctrl: lost", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                        if self.Track_state == 'tracking':
                            self._start_recovery()
                            self.recover_line()
                        return rgb_img, lane_mask

                    omega, x_near, y_near, x_far, y_far, curv, y_min, y_max = ctrl
                    if self.recovering_line:
                        self._stop_recovery()
                    cv.circle(rgb_img, (int(x_near), int(y_near)), 6, (0, 255, 0), -1)
                    cv.circle(rgb_img, (int(x_far), int(y_far)), 6, (255, 128, 0), -1)
                    cv.putText(rgb_img, f"curv:{curv:.4f}", (30, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                    cv.putText(rgb_img, f"omega:{omega:.2f}", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                    cv.putText(rgb_img, f"state:{self.Track_state}", (30, 105), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                    cv.putText(rgb_img, f"y:[{int(y_min)},{int(y_max)}]", (30, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                    if self.Track_state == 'tracking':
                        self.execute(omega)
                    binary = lane_mask

        return rgb_img, binary

    def on_timer(self):
        ret, frame = self.capture.read()
        if not ret:
            self._cam_fail_count += 1
            now = time.time()
            if now - self._last_cam_warn >= 2.0:
                print(
                    f"摄像头读帧失败 ({self._cam_fail_count} 次)，"
                    f"设备={self._camera_id!r}，尝试重连…",
                    flush=True,
                )
                self._last_cam_warn = now
            if self._cam_fail_count % 30 == 0:
                self._reopen_camera()
            action = cv.waitKey(1) & 0xFF
            if action == ord('q'):
                self._quit = True
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv.putText(
                blank,
                "Camera read failed",
                (120, 230),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv.putText(
                blank,
                f"device={self._camera_id!r}  q=quit",
                (120, 270),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )
            cv.imshow('frame', blank)
            if self._quit:
                self.capture.release()
                cv.destroyAllWindows()
            return

        self._cam_fail_count = 0
        action = cv.waitKey(1) & 0xFF
        frame, binary = self.process(frame, action)

        if self.Track_state != 'tracking':
            self.link.stop()

        now = time.time()
        fps = 1.0 / max(1e-6, (now - self.end)) if self.end > 0 else 0.0
        self.end = now
        cv.putText(frame, f"FPS:{int(fps)}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)

        if isinstance(binary, np.ndarray) and binary.size > 0:
            cv.imshow('frame', ManyImgs(1, ([frame, binary])))
        else:
            cv.imshow('frame', frame)

        if action == ord('q') or self._quit:
            self.capture.release()
            cv.destroyAllWindows()
            self._quit = True

    def Reset(self):
        self.Track_state = 'init'
        self.hsv_range = ()
        self.Mouse_XY = (0, 0)
        self.prev_error = 0.0
        self.last_ctrl_time = None
        self.last_angular = 0.0
        self.recovering_line = False
        self.link.stop()
        print("Reset success!!!")


def ensure_display() -> None:
    """SSH 会话未继承 DISPLAY 时，默认连本机 :0 以便弹窗。"""
    if sys.platform == "win32":
        return
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print("DISPLAY 未设置，已自动设为 :0")


def main(argv=None):
    parser = argparse.ArgumentParser(description="follow_line_poly (NavEngine 串口版)")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="摄像头索引或 /dev/videoX")
    parser.add_argument("--hsv-file", default=DEFAULT_HSV_FILE)
    parser.add_argument("--no-setup", action="store_true")
    args = parser.parse_args(argv)

    ensure_display()

    try:
        camera = int(args.camera)
    except (TypeError, ValueError):
        camera = args.camera

    link = NavLink(args.port, args.baud)
    try:
        link.open()
        if not args.no_setup:
            link.setup()
        linedetect = LineDetectPoly(link, hsv_file=args.hsv_file, camera=camera)
        print("start it")
        while not linedetect._quit:
            linedetect.on_timer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n退出 …")
    except SerialException as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.no_setup:
            link.teardown()
        link.close()
        cv.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
