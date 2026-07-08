#!/usr/bin/env python3
"""YOLO 巡线 v2：在 follow_line_yolo 基础上增加 PAUSE/BLOCKED + 轻量反光抑制。

状态机:
  FOLLOW:  PD 巡线。
  PAUSE:   仅当里程够长（线尾）→ 缓停 → 停稳 PAUSE_HOLD_SEC → TURN。
  BLOCKED: 里程不足丢线 → 一直停等，线回来再 FOLLOW；不因时间掉头。
  TURN:    原地掉头，捕获线后回 FOLLOW。

反光: 预热后掩膜+位置同时跳变才 HOLD（上限见 _Internals）；帧间隔过久或检测连续稳定则释放。
巡线参数: 只改 FollowCfg；弯道降速/斜坡由 curve_slow、yaw_ramp_sec 派生。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import serial

from ui_ctrl.constants import CmdCtrl
from ui_ctrl.protocol import build_ctrl, verify_frame
from ui_ctrl.training_ctrl import (
    ROBOT_KEY,
    TrainingProgram,
    build_training_init,
    build_training_start,
    build_training_stop,
)

from odom_estimator import DEFAULT_SEGMENT_FILE, OdomEstimator
from yolo import DEFAULT_MODEL
from yolo.scene import ScenePrediction, select_scene_label
from yolo.viz import overlay_scene

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── 串口/摄像头默认参数（原 follow_line.py）─────────────────────────────
DEFAULT_PORT = "/dev/ttyCH341USB0"
DEFAULT_BAUD = 921600
DEFAULT_CAMERA = 0
# 手动曝光：AUTO_EXPOSURE=1 后 EXPOSURE/GAIN 生效；AUTO_WB=0 后 WB_TEMPERATURE 生效
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


def ensure_display() -> None:
    """SSH 会话未继承 DISPLAY 时，默认连本机 :0 以便弹窗。"""
    if sys.platform == "win32":
        return
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print("DISPLAY 未设置，已自动设为 :0")

# ── 巡线旋钮（只改 FollowCfg）──────────────────────────────────
@dataclass(frozen=True)
class FollowCfg:
    linear: float = 0.22         # 直道线速度
    max_yaw: float = 0.4         # 角速度上限
    kp: float = 0.0025
    ka: float = 0.001
    kd: float = 0.0055
    curve_slow: float = 0.15      # 弯里最低线速度 = linear × curve_slow
    yaw_ramp_sec: float = 0.2    # 角速度从 0 爬满 max_yaw 约需秒数


CFG = FollowCfg()

# ── 固定内部常数（一般不调）────────────────────────────────────
@dataclass(frozen=True)
class _Internals:
    lat_smooth: float = 0.5
    yaw_smooth: float = 0.65
    fwd_scale_smooth: float = 0.65
    cam_offset: float = -28
    lost_frames: int = 6
    decel_ramp_sec: float = 1.2
    pause_hold_sec: float = 2
    decel_stop_fwd: float = 0.015
    pause_recover_frames: int = 3
    block_recover_fwd_sec: float = 0.8
    steer_dt_max: float = 0.2
    d_lat_max: float = 150.0
    track_warmup: int = 12
    jump_lat_px: float = 42.0
    jump_mask_ratio: float = 0.35
    hold_max_frames: int = 4
    track_frame_gap_sec: float = 0.250
    near_y: float = 0.9          # 横向控制采样（0=远端顶部，1=近端脚下）
    far_y: float = 0.1           # 航向基准远端
    local_fit_halfwin: int = 70
    local_fit_min_points: int = 6
    local_fit_y_ratio: float = 0.3  # 局部拟合锚点（越小越往前看）
    ready_confirm: int = 3
    turn_capture_lat: float = 120
    turn_min_line_height: float = 80
    turn_min_aspect: float = 1.15
    line_side_hysteresis: float = 20.0
    lat_deadband_px: float = 10.0   # |横向偏差|小于此值当 0，减直道抖动
    follow_roi_top: float = 0.25     # FOLLOW 时去掉图像顶部比例（保留下半 1-top）


_I = _Internals()
DEBUG_PRINT_EVERY_FRAME = True


def _fwd_min() -> float:
    return max(0.04, CFG.linear * 0.18)


def _yaw_step_max() -> float:
    return CFG.max_yaw / max(CFG.yaw_ramp_sec * 30.0, 1.0)


def _jump_x_near_px() -> float:
    return _I.jump_lat_px * 0.76


def _hold_stable_lat_px() -> float:
    return _I.jump_lat_px * 0.48


def _curve_fwd_scale(yaw_demand: float, heading_px: float) -> float:
    """弯速仅看 heading（真弯道）；直线偏航修正不因 yaw_demand 降速。"""
    _ = yaw_demand
    floor = CFG.curve_slow
    ah = abs(heading_px)
    head_start = 38.0
    head_span = 42.0
    if ah <= head_start:
        return 1.0
    t = min(1.0, (ah - head_start) / head_span)
    return 1.0 - t * (1.0 - floor)


def _apply_follow_roi(mask: np.ndarray) -> np.ndarray:
    """FOLLOW 专用：去掉顶部区域，只在下半部拟合巡线。"""
    top = int(mask.shape[0] * _I.follow_roi_top)
    if top <= 0:
        return mask
    out = mask.copy()
    out[:top, :] = 0
    return out


_LOG = logging.getLogger("follow_line_yolo_2")
_LOG_DIR = Path(__file__).resolve().parent / "logs"


def setup_logging(log_path: str | Path | None = None, *, echo_console: bool) -> Path:
    _LOG.handlers.clear()
    _LOG.setLevel(logging.DEBUG)
    _LOG.propagate = False

    if log_path is None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"follow_line_yolo2_{datetime.now():%Y%m%d_%H%M%S}.log"
    else:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    _LOG.addHandler(fh)

    if echo_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        _LOG.addHandler(ch)

    return path


def _log_info(msg: str) -> None:
    _LOG.info(msg)


def _log_frame(osd_lines: list[str]) -> None:
    _LOG.info("--- frame ---\n%s", "\n".join(osd_lines))


@dataclass
class SteerBreakdown:
    dt: float = 0.0
    lateral: float = 0.0
    heading: float = 0.0
    d_lateral: float = 0.0
    p_term: float = 0.0
    d_term: float = 0.0
    a_term: float = 0.0
    yaw_demand: float = 0.0
    yaw_raw: float = 0.0
    yaw_out: float = 0.0
    fwd: float = 0.0
    fwd_scale: float = 1.0
    saturated: bool = False


def print_debug_config(*, conf: float, device: str, imgsz: int, half: bool,
                    camera, img_flip: bool, auto: bool, log_path: Path) -> None:
    c = CFG
    lines = [
        "=== follow_line_yolo_2 debug config ===",
        f"  log_file={log_path}",
        f"  CFG linear={c.linear} max_yaw={c.max_yaw} kp={c.kp} ka={c.ka} kd={c.kd}",
        f"  CFG curve_slow={c.curve_slow} yaw_ramp_sec={c.yaw_ramp_sec}",
        f"  derived fwd_min={_fwd_min():.3f} yaw_step={_yaw_step_max():.3f}",
        f"  yolo conf={conf} device={device} imgsz={imgsz} half={half}",
        f"  camera={camera} img_flip={img_flip} auto_start={auto}",
        "=====================================",
    ]
    for line in lines:
        _log_info(line)


def _put_text_block(vis: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 22),
                    line_h: int = 18, color=(0, 220, 255), scale: float = 0.42) -> None:
    x0, y0 = origin
    for i, line in enumerate(lines):
        cv.putText(vis, line, (x0, y0 + i * line_h),
                cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


class Mode(Enum):
    TURN = "turn"
    FOLLOW = "follow"
    PAUSE = "pause"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BoxGeom:
    x1: float
    y1: float
    x2: float
    y2: float
    bw: float
    bh: float


def analyze_box(box: tuple[float, float, float, float]) -> BoxGeom:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return BoxGeom(x1, y1, x2, y2, bw, bh)


def _sample_centerline_x(center: np.ndarray, y_ratio: float) -> tuple[float, float]:
    ys = center[:, 1]
    xs = center[:, 0]
    y_min, y_max = float(ys.min()), float(ys.max())
    y = y_min + float(np.clip(y_ratio, 0.0, 1.0)) * (y_max - y_min)
    x = float(np.interp(y, ys, xs))
    return x, y


@dataclass(frozen=True)
class TrackInfo:
    center: np.ndarray
    left_line: np.ndarray
    right_line: np.ndarray
    x_near: float
    y_near: float
    x_far: float
    y_far: float

    @property
    def heading_px(self) -> float:
        return self.x_near - self.x_far

    def lateral_px(self, center_x: float) -> float:
        return self.x_near - center_x

    def turn_capture_ready(self, center_x: float, geom: BoxGeom | None) -> tuple[bool, float, float, float]:
        lat = abs(self.lateral_px(center_x))
        aspect = (geom.bh / max(geom.bw, 1.0)) if geom is not None else 0.0
        bh = geom.bh if geom is not None else 0.0
        ok = (
            lat < _I.turn_capture_lat
            and bh >= _I.turn_min_line_height
            and aspect >= _I.turn_min_aspect
        )
        return ok, lat, aspect, bh


def _scan_boundaries(mask: np.ndarray):
    left_pts, right_pts = [], []
    for y in range(mask.shape[0] - 1, -1, -4):
        xs = np.where(mask[y] > 0)[0]
        if xs.size < 2:
            continue
        lx, rx = float(xs.min()), float(xs.max())
        if rx - lx < 8:
            continue
        left_pts.append((lx, float(y)))
        right_pts.append((rx, float(y)))
    return left_pts, right_pts


def _polyfit_segment(ys, x_left, x_right, y_ref: float):
    half_win = max(20, _I.local_fit_halfwin)
    local = np.abs(ys - y_ref) <= half_win
    if np.count_nonzero(local) >= _I.local_fit_min_points:
        ys_fit = ys[local]
        x_left_fit = x_left[local]
        x_right_fit = x_right[local]
    else:
        ys_fit, x_left_fit, x_right_fit = ys, x_left, x_right

    y_min = float(ys_fit.min())
    y_max = float(ys_fit.max())
    if y_max - y_min < 12:
        return None

    lc = np.polyfit(ys_fit, x_left_fit, 2)
    rc = np.polyfit(ys_fit, x_right_fit, 2)
    ys_s = np.linspace(y_min, y_max, max(len(ys_fit), 8))
    left_x = np.polyval(lc, ys_s)
    right_x = np.polyval(rc, ys_s)
    center_x = 0.5 * (left_x + right_x)

    center = np.stack([center_x, ys_s], axis=1)
    if center.shape[0] >= 5:
        xs = center[:, 0]
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        center[:, 0] = np.convolve(np.pad(xs, (pad, pad), mode="edge"), kernel, mode="valid")

    left_line = np.stack([left_x, ys_s], axis=1)
    right_line = np.stack([right_x, ys_s], axis=1)
    return center, left_line, right_line


def fit_centerline(mask: np.ndarray) -> TrackInfo | None:
    left_pts, right_pts = _scan_boundaries(mask)
    if len(left_pts) < 8:
        return None

    left = np.asarray(left_pts)
    right = np.asarray(right_pts)
    ys = left[:, 1]
    x_left = left[:, 0]
    x_right = right[:, 0]

    y_min_all = float(ys.min())
    y_max_all = float(ys.max())
    y_ref = y_min_all + float(np.clip(_I.local_fit_y_ratio, 0.28, 0.85)) * (y_max_all - y_min_all)

    seg = _polyfit_segment(ys, x_left, x_right, y_ref)
    if seg is None:
        return None
    center, left_line, right_line = seg

    x_near, y_near = _sample_centerline_x(center, _I.near_y)
    x_far, y_far = _sample_centerline_x(center, _I.far_y)
    return TrackInfo(center, left_line, right_line, x_near, y_near, x_far, y_far)


def draw_track(vis: np.ndarray, track: TrackInfo) -> None:
    for pts, color in (
        (track.left_line, (0, 255, 0)),
        (track.right_line, (0, 0, 255)),
        (track.center, (0, 255, 255)),
    ):
        poly = pts.astype(np.int32)
        cv.polylines(vis, [poly], False, color, 2)
    cv.circle(vis, (int(track.x_near), int(track.y_near)), 6, (0, 255, 0), -1)
    cv.circle(vis, (int(track.x_far), int(track.y_far)), 6, (255, 128, 0), -1)
    cx = int(vis.shape[1] / 2 + _I.cam_offset)
    cv.line(vis, (cx, 0), (cx, vis.shape[0]), (255, 0, 255), 1)


def draw_geom(vis: np.ndarray, geom: BoxGeom | None, mode: Mode) -> None:
    if geom is None:
        return
    p1 = (int(geom.x1), int(geom.y1))
    p2 = (int(geom.x2), int(geom.y2))
    if mode == Mode.FOLLOW:
        color = (0, 255, 0)
    elif mode == Mode.PAUSE:
        color = (0, 140, 255)
    elif mode == Mode.BLOCKED:
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)
    cv.rectangle(vis, p1, p2, color, 2)


@dataclass
class _GoodSample:
    lat: float = 0.0
    head: float = 0.0
    x_near: float = 0.0
    mask_px: int = 0


class YoloLineFollower:
    def __init__(self, link: NavLink, model, *, conf: float, device: str,
                imgsz: int, half: bool, camera=0, img_flip: bool = True,
                auto: bool = False, odom: OdomEstimator | None = None) -> None:
        self.link = link
        self.odom = odom
        self.model = model
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.img_flip = img_flip
        self.running = auto
        self._quit = False
        self._last_yaw = 0.0
        self._prev_error = 0.0
        self._lat_filt = 0.0
        self._last_ctrl_time: float | None = None
        self._mode = Mode.FOLLOW
        self._ready_streak = 0
        self._last_line_side = 1.0
        self._turn_yaw_dir = 1.0
        self._turn_enter_time = time.perf_counter()
        self._pause_enter_time = 0.0
        self._stop_time: float | None = None
        self._hold_accum = 0.0
        self._hold_tick_last: float | None = None
        self._pause_recover_streak = 0
        self._fwd_scale_filt = 1.0
        self._decel_t0: float | None = None
        self._decel_start_fwd = CFG.linear
        self._decel_start_yaw = 0.0
        self._last_cmd_fwd = 0.0
        self._last_cmd_yaw = 0.0
        self._lost_streak = 0
        self._good_streak = 0
        self._good = _GoodSample()
        self._hold_note = ""
        self._steer_dbg = SteerBreakdown()
        self._last_status = ""
        self._block_recover_t0: float | None = None
        self._inp_hold_streak = 0
        self._hold_stable_streak = 0
        self._hold_prev_lat: float | None = None
        self._hold_prev_head: float | None = None
        self._last_track_frame_time: float | None = None
        self._last_captured = False
        self._last_lat_abs = 0.0
        self._last_aspect = 0.0
        self._last_geom_bh = 0.0
        self._last_mask_px = 0
        self._last_mode_logged = self._mode

        self.capture = open_camera(camera)
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera}")

    def _drive(self, fwd: float, yaw: float) -> None:
        self.link.move(fwd, -yaw if self.img_flip else yaw)

    def _reset_steer(self) -> None:
        self._last_yaw = 0.0
        self._prev_error = 0.0
        self._lat_filt = 0.0
        self._last_ctrl_time = None

    def _reset_inp_hold(self) -> None:
        self._inp_hold_streak = 0
        self._hold_stable_streak = 0
        self._hold_prev_lat = None
        self._hold_prev_head = None

    def _reset_track_hold(self) -> None:
        """丢线/遮挡后旧横向参考不可信，勿继续 HOLD。"""
        self._good_streak = 0
        self._good = _GoodSample()
        self._hold_note = ""
        self._reset_inp_hold()
        self._last_track_frame_time = None

    def _steer_to(self, lateral_px: float, heading_px: float) -> float:
        now = time.perf_counter()
        first = self._last_ctrl_time is None
        dt = 0.03 if first else max(1e-3, now - self._last_ctrl_time)
        if not first and dt > _I.steer_dt_max:
            # BLOCKED/PAUSE 期间久未跑 PID，避免用旧状态 + 巨大 dt
            first = True
            dt = 0.03
        self._last_ctrl_time = now

        if abs(lateral_px) < _I.lat_deadband_px:
            lateral_px = 0.0

        if first:
            self._lat_filt = lateral_px
        else:
            self._lat_filt = (1.0 - _I.lat_smooth) * self._lat_filt + _I.lat_smooth * lateral_px
        lat_f = self._lat_filt

        if first:
            self._prev_error = lat_f
        d_lateral = (lat_f - self._prev_error) / dt
        d_lateral = float(np.clip(d_lateral, -_I.d_lat_max, _I.d_lat_max))
        self._prev_error = lat_f

        p_term = -(CFG.kp * lat_f)
        d_term = -(CFG.kd * d_lateral)
        a_term = CFG.ka * heading_px
        yaw_demand = p_term + d_term + a_term
        yaw_raw = float(np.clip(yaw_demand, -CFG.max_yaw, CFG.max_yaw))
        if not first:
            step = _yaw_step_max()
            yaw_raw = float(np.clip(
                yaw_raw,
                self._last_yaw - step,
                self._last_yaw + step,
            ))
        saturated = abs(yaw_demand) >= CFG.max_yaw - 1e-9
        alpha = float(np.clip(_I.yaw_smooth, 0.0, 1.0))
        yaw = (1.0 - alpha) * self._last_yaw + alpha * yaw_raw
        self._last_yaw = yaw

        fwd_scale = _curve_fwd_scale(yaw_demand, heading_px)
        alpha_slow = float(np.clip(_I.fwd_scale_smooth, 0.0, 0.99))
        alpha = 0.40 if fwd_scale > self._fwd_scale_filt else alpha_slow
        self._fwd_scale_filt = alpha * self._fwd_scale_filt + (1.0 - alpha) * fwd_scale
        fwd = float(np.clip(CFG.linear * self._fwd_scale_filt, _fwd_min(), CFG.linear))
        if self._block_recover_t0 is not None:
            t = (time.perf_counter() - self._block_recover_t0) / _I.block_recover_fwd_sec
            if t >= 1.0:
                self._block_recover_t0 = None
            else:
                fwd *= max(0.25, t)
        self._steer_dbg = SteerBreakdown(
            dt=dt, lateral=lat_f, heading=heading_px, d_lateral=d_lateral,
            p_term=p_term, d_term=d_term, a_term=a_term, yaw_demand=yaw_demand,
            yaw_raw=yaw_raw, yaw_out=yaw, fwd=fwd, fwd_scale=self._fwd_scale_filt,
            saturated=saturated,
        )
        return yaw

    def _scene_mask(self, result, scene: ScenePrediction, shape) -> np.ndarray | None:
        if scene.mask_index is None or result.masks is None:
            return None
        h, w = shape[:2]
        m = result.masks.data[scene.mask_index].cpu().numpy()
        if m.shape != (h, w):
            m = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8) * 255

    def _log_mode_change(self, reason: str) -> None:
        if self._mode != self._last_mode_logged:
            _log_info(
                f"MODE {self._last_mode_logged.value} -> {self._mode.value} | {reason} | "
                f"side={self._last_line_side:+.0f}"
            )
            self._last_mode_logged = self._mode
            self._hold_note = ""

    def _enter_follow(self, reason: str) -> None:
        self._mode = Mode.FOLLOW
        self._lost_streak = 0
        self._pause_recover_streak = 0
        self._clear_decel()
        if reason in ("线恢复", "误触恢复"):
            self._reset_steer()
            self._reset_track_hold()
            self._block_recover_t0 = time.perf_counter()
            _log_info(f"线恢复: 重置 PID/HOLD ({reason})")
        elif reason == "捕获→PID":
            self._block_recover_t0 = None
        if self.odom is not None:
            if reason == "捕获→PID":
                self.odom.begin_segment()
                _log_info("odom 段开始 (掉头结束→巡线) tot=0")
            else:
                self.odom.resume_accum()
        self._log_mode_change(reason)

    def _start_odom_segment_if_needed(self) -> None:
        """冷启动首段：尚无 TURN 完成过时，从开跑计到第一次 TURN。"""
        if self.odom is None or self.odom.segment_open:
            return
        self.odom.begin_segment()
        _log_info("odom 段开始 (冷启动→首段巡线) tot=0")

    def _log_odom_segment_end(self, seg, *, reason: str) -> None:
        dur = max(0.0, seg.t1 - seg.t0)
        file_s = ""
        if self.odom is not None and self.odom.segment_file is not None:
            file_s = f" file={self.odom.segment_file}"
        _log_info(
            f"odom 段结束 ({reason}) tot={seg.total:.3f} L={seg.left:.3f} "
            f"R={seg.right:.3f} dur={dur:.1f}s{file_s} | "
            f"{self.odom.session_summary() if self.odom else ''}"
        )

    def log_odom_session_end(self) -> None:
        if self.odom is None:
            return
        _log_info(f"odom 会话结束 | {self.odom.session_summary()}")

    def _update_line_side_follow(self, lat: float) -> None:
        if lat > _I.line_side_hysteresis:
            self._last_line_side = 1.0
        elif lat < -_I.line_side_hysteresis:
            self._last_line_side = -1.0

    def _line_valid(self, scene, geom, track_ok: bool) -> bool:
        return scene is not None and geom is not None and track_ok

    def _steer_inputs(
        self, track: TrackInfo, center_x: float, mask_px: int,
    ) -> tuple[float, float, str]:
        """返回 (lat, head) 供 PID；跳变帧用上一帧可信值（有上限）。"""
        lat = track.lateral_px(center_x)
        head = track.heading_px
        self._hold_note = ""
        now = time.perf_counter()
        gap = (
            None if self._last_track_frame_time is None
            else now - self._last_track_frame_time
        )
        self._last_track_frame_time = now

        if gap is not None and gap > _I.track_frame_gap_sec:
            self._good = _GoodSample(
                lat=lat, head=head, x_near=track.x_near, mask_px=mask_px,
            )
            self._reset_inp_hold()
            return lat, head, "gap"

        if self._good_streak < _I.track_warmup:
            self._good = _GoodSample(lat=lat, head=head, x_near=track.x_near, mask_px=mask_px)
            self._good_streak += 1
            self._reset_inp_hold()
            return lat, head, ""

        d_lat = abs(lat - self._good.lat)
        d_x = abs(track.x_near - self._good.x_near)
        ref_mask = max(self._good.mask_px, 1)
        d_mask = abs(mask_px - self._good.mask_px) / ref_mask
        jump_x = _jump_x_near_px()
        jumped = (
            (d_mask > _I.jump_mask_ratio and (d_lat > _I.jump_lat_px or d_x > jump_x))
            or (d_lat > _I.jump_lat_px and d_x > jump_x)
        )
        if jumped:
            self._inp_hold_streak += 1
            if self._hold_prev_lat is not None:
                stable = (
                    abs(lat - self._hold_prev_lat) < _hold_stable_lat_px()
                    and abs(head - self._hold_prev_head) < 30.0
                )
                self._hold_stable_streak = self._hold_stable_streak + 1 if stable else 0
            self._hold_prev_lat = lat
            self._hold_prev_head = head

            release = ""
            if self._hold_stable_streak >= 3:
                release = "stable"
            elif self._inp_hold_streak >= _I.hold_max_frames:
                release = "max"

            if release:
                self._good = _GoodSample(
                    lat=lat, head=head, x_near=track.x_near, mask_px=mask_px,
                )
                self._reset_inp_hold()
                self._hold_note = f"HOLD rel {release}"
                return lat, head, self._hold_note

            self._hold_note = f"HOLD inp {self._inp_hold_streak}/{_I.hold_max_frames}"
            return self._good.lat, self._good.head, self._hold_note

        self._good = _GoodSample(lat=lat, head=head, x_near=track.x_near, mask_px=mask_px)
        self._reset_inp_hold()
        return lat, head, ""

    def _start_decel(self) -> None:
        if self._decel_t0 is None:
            self._decel_t0 = time.perf_counter()
            self._decel_start_fwd = max(self._last_cmd_fwd, _fwd_min(), 0.05)
            self._decel_start_yaw = self._last_cmd_yaw

    def _reset_hold_timer(self) -> None:
        self._hold_accum = 0.0
        self._hold_tick_last = None

    def _clear_decel(self) -> None:
        self._decel_t0 = None
        self._stop_time = None
        self._reset_hold_timer()

    def _hold_elapsed(self) -> float:
        """停稳等待计时；run=False 时冻结（空格暂停不会攒出 11s）。"""
        if self._stop_time is None:
            return 0.0
        now = time.perf_counter()
        if self.running and self._mode in (Mode.PAUSE, Mode.BLOCKED):
            if self._hold_tick_last is not None:
                self._hold_accum += now - self._hold_tick_last
            self._hold_tick_last = now
        else:
            self._hold_tick_last = None
        return self._hold_accum

    def _decel_scale(self) -> float:
        if self._decel_t0 is None:
            return 1.0
        t = (time.perf_counter() - self._decel_t0) / _I.decel_ramp_sec
        return max(0.0, 1.0 - t)

    def _drive_decel_ramp(self) -> tuple[float, float]:
        """丢线/PAUSE 期间按时间线性降速。"""
        scale = self._decel_scale()
        fwd = self._decel_start_fwd * scale
        yaw = self._decel_start_yaw * scale
        if fwd <= _I.decel_stop_fwd:
            fwd = 0.0
            yaw = 0.0
            if self._stop_time is None:
                self._stop_time = time.perf_counter()
                self._reset_hold_timer()
                self._hold_tick_last = time.perf_counter()
        return fwd, yaw

    def _begin_pause(self, reason: str) -> str:
        if self._mode != Mode.PAUSE:
            self._pause_enter_time = time.perf_counter()
            self._pause_recover_streak = 0
            self._reset_track_hold()
            self._start_decel()
            if self.odom is not None:
                self.odom.pause_accum()
            self._log_mode_change(reason)
        self._mode = Mode.PAUSE
        self._lost_streak = 0
        return f"PAUSE: {reason}"

    def _begin_blocked(self, reason: str) -> str:
        if self._mode != Mode.BLOCKED:
            self._pause_enter_time = time.perf_counter()
            self._pause_recover_streak = 0
            self._reset_track_hold()
            self._start_decel()
            if self.odom is not None:
                self.odom.pause_accum()
            self._log_mode_change(reason)
        self._mode = Mode.BLOCKED
        self._lost_streak = 0
        return f"BLOCKED: {reason}"

    def _begin_turn(self, reason: str) -> str:
        if self._mode != Mode.TURN:
            self._reset_steer()
            self._turn_enter_time = time.perf_counter()
            self._turn_yaw_dir = self._last_line_side
            if self.odom is not None and self.odom.segment_open:
                seg = self.odom.end_segment(reason=reason)
                self._log_odom_segment_end(seg, reason=reason)
            elif self.odom is not None:
                _log_info("odom 段结束跳过 (无活跃段)")
        self._mode = Mode.TURN
        self._ready_streak = 0
        self._good_streak = 0
        self._pause_recover_streak = 0
        self._clear_decel()
        self._log_mode_change(reason)
        return f"TURN: {reason}"

    def _try_recover_from_pause(self, valid: bool) -> str | None:
        if not valid:
            self._pause_recover_streak = 0
            return None
        self._pause_recover_streak += 1
        if self._pause_recover_streak < _I.pause_recover_frames:
            return None
        self._enter_follow("线恢复")
        return "FOLLOW: 线恢复"

    def _step_mode(
        self,
        scene: ScenePrediction | None,
        geom: BoxGeom | None,
        track_ok: bool,
        track: TrackInfo | None,
        w: int,
    ) -> str:
        center_x = w / 2.0 + _I.cam_offset
        valid = self._line_valid(scene, geom, track_ok)

        if self._mode == Mode.FOLLOW:
            if valid and track is not None:
                self._lost_streak = 0
                self._clear_decel()
                self._update_line_side_follow(track.lateral_px(center_x))
                note = f" | {self._hold_note}" if self._hold_note else ""
                return f"FOLLOW{note}"
            self._lost_streak += 1
            self._start_decel()
            if self._lost_streak >= _I.lost_frames:
                if self.odom is not None and not self.odom.should_turn():
                    return self._begin_blocked(
                        f"里程不足 tot={self.odom.total:.3f} "
                        f"需>={self._turn_threshold_str()}",
                    )
                return self._begin_pause("线尾丢线")
            scale = self._decel_scale()
            return f"FOLLOW: 丢线({self._lost_streak}/{_I.lost_frames}) decel={scale:.2f}"

        if self._mode == Mode.BLOCKED:
            elapsed = time.perf_counter() - self._pause_enter_time
            scale = self._decel_scale()
            hold = self._hold_elapsed()
            if recovered := self._try_recover_from_pause(valid):
                return recovered
            if self._stop_time is not None:
                return (
                    f"BLOCKED: 等待障碍 hold={hold:.1f}s "
                    f"odom={self._odom_short()}"
                )
            return f"BLOCKED: 缓停 decel={scale:.2f} ({elapsed:.1f}s)"

        if self._mode == Mode.PAUSE:
            elapsed = time.perf_counter() - self._pause_enter_time
            scale = self._decel_scale()
            if valid and track is not None and elapsed < 0.4 and scale > 0.85:
                self._enter_follow("误触恢复")
                return "FOLLOW: 误触恢复"
            if recovered := self._try_recover_from_pause(valid):
                return recovered
            hold = self._hold_elapsed()
            if self._stop_time is not None:
                if hold >= _I.pause_hold_sec:
                    if self.odom is not None and not self.odom.should_turn():
                        return self._begin_blocked(
                            f"停稳{hold:.1f}s但里程不足 tot={self.odom.total:.3f}",
                        )
                    tot_s = (
                        f" tot={self.odom.total:.3f}" if self.odom is not None else ""
                    )
                    return self._begin_turn(f"线尾停稳{hold:.1f}s{tot_s}")
                return f"PAUSE: hold {hold:.1f}/{_I.pause_hold_sec:.1f}s decel={scale:.2f}"
            return f"PAUSE: 缓停 decel={scale:.2f} ({elapsed:.1f}s)"

        # TURN
        captured = False
        lat_px = aspect = bh = 0.0
        if track_ok and track is not None:
            captured, lat_px, aspect, bh = track.turn_capture_ready(center_x, geom)

        self._last_captured = captured
        self._last_lat_abs = lat_px
        self._last_aspect = aspect
        self._last_geom_bh = bh

        if not captured:
            self._ready_streak = 0
            if not track_ok:
                return "TURN: 无拟合线"
            return f"TURN: lat={lat_px:.0f} asp={aspect:.2f} bh={bh:.0f}"

        self._ready_streak += 1
        if self._ready_streak >= _I.ready_confirm:
            self._enter_follow("捕获→PID")
            return "FOLLOW: 捕获→PID"
        return f"TURN: 捕获 ({self._ready_streak}/{_I.ready_confirm})"

    def _turn_threshold_str(self) -> str:
        if self.odom is None:
            return "n/a"
        from odom_estimator import MIN_BASELINE_SEGMENTS, MIN_TURN_NO_BASELINE, TURN_MIN_RATIO

        if len(self.odom.history) < MIN_BASELINE_SEGMENTS:
            return f"{MIN_TURN_NO_BASELINE:.3f}(无基线)"
        base = self.odom.baseline_min
        return f"{base * TURN_MIN_RATIO:.3f}(基线×{TURN_MIN_RATIO})"

    def _odom_short(self) -> str:
        if self.odom is None:
            return "off"
        exp = self.odom.expected_length
        exp_s = f"{exp:.3f}" if exp is not None else "n/a"
        return (
            f"tot={self.odom.total:.3f} prog={self.odom.progress_label()} "
            f"exp={exp_s} base={self.odom.baseline_min:.3f} "
            f"turn={self.odom.should_turn()}"
        )

    def _draw_odom_progress(self, vis: np.ndarray) -> None:
        if self.odom is None:
            return
        label = self.odom.progress_label()
        x = max(8, vis.shape[1] - 96)
        cv.putText(
            vis, label, (x, 56),
            cv.FONT_HERSHEY_SIMPLEX, 1.35, (0, 40, 0), 4, cv.LINE_AA,
        )
        cv.putText(
            vis, label, (x, 56),
            cv.FONT_HERSHEY_SIMPLEX, 1.35, (0, 255, 140), 2, cv.LINE_AA,
        )

    def _feed_odom(self, fwd: float, yaw: float) -> None:
        if self.odom is None:
            return
        self.odom.feed_command(fwd, yaw)
        # 仅 FOLLOW 活跃段积分；TURN/PAUSE/BLOCKED 内 _accumulating=False
        if self._mode == Mode.FOLLOW:
            self.odom._step()

    def _apply_drive(self, mode: Mode, track: TrackInfo | None, track_ok: bool, w: int) -> None:
        if mode in (Mode.PAUSE, Mode.BLOCKED) or (
            mode == Mode.FOLLOW and self._decel_t0 is not None
        ):
            fwd, yaw = self._drive_decel_ramp()
            self._drive(fwd, yaw)
            self._steer_dbg = SteerBreakdown(fwd=fwd, yaw_out=yaw, yaw_raw=yaw)
            self._feed_odom(fwd, yaw)
            return

        if mode == Mode.FOLLOW and track_ok and track is not None:
            center_x = w / 2.0 + _I.cam_offset
            lat, head, hold_note = self._steer_inputs(track, center_x, self._last_mask_px)
            if hold_note.startswith("HOLD rel") or hold_note == "gap":
                self._lat_filt = lat
                self._prev_error = 0.0
            yaw = self._steer_to(lat, head)
            fwd = self._steer_dbg.fwd
            self._last_cmd_fwd = fwd
            self._last_cmd_yaw = yaw
            self._drive(fwd, yaw)
            self._feed_odom(fwd, yaw)
            return

        if mode == Mode.TURN:
            turn_yaw = -CFG.max_yaw * self._turn_yaw_dir
            self._drive(0.0, turn_yaw)
            self._steer_dbg = SteerBreakdown(
                fwd=0.0, yaw_out=turn_yaw, yaw_raw=turn_yaw,
            )
            self._feed_odom(0.0, turn_yaw)
            return

        self._drive(0.0, 0.0)
        self._steer_dbg = SteerBreakdown(fwd=0.0, yaw_out=0.0, yaw_raw=0.0)
        self._feed_odom(0.0, 0.0)

    def on_timer(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            return
        frame = cv.resize(frame, (640, 480))
        if self.img_flip:
            frame = cv.flip(frame, 1)

        results = self.model.predict(
            frame, conf=self.conf, device=self.device,
            imgsz=self.imgsz, half=self.half, verbose=False,
        )
        result = results[0]
        scene = select_scene_label(result, min_conf=self.conf)

        geom = None
        track = None
        mask = None
        if scene is not None and scene.box_xyxy is not None:
            geom = analyze_box(scene.box_xyxy)
            mask = self._scene_mask(result, scene, frame.shape)
            if mask is not None:
                if self._mode == Mode.FOLLOW:
                    mask = _apply_follow_roi(mask)
                track = fit_centerline(mask)

        track_ok = track is not None
        if mask is not None:
            self._last_mask_px = int(cv.countNonZero(mask))

        fwd_cmd = 0.0
        yaw_cmd = 0.0

        if self.running:
            self._start_odom_segment_if_needed()
            w = frame.shape[1]
            status = self._step_mode(scene, geom, track_ok, track, w)
            self._last_status = status
            self._apply_drive(self._mode, track, track_ok, w)
            fwd_cmd = self._steer_dbg.fwd
            yaw_cmd = self._steer_dbg.yaw_out
        else:
            self.link.stop()
            self._reset_steer()
            self._hold_tick_last = None
            status = "暂停(空格启动)"
            self._last_status = status
            self._steer_dbg = SteerBreakdown()
            self._feed_odom(0.0, 0.0)

        center_x = frame.shape[1] / 2.0 + _I.cam_offset
        geom_y1 = geom.y1 if geom is not None else float("nan")
        geom_bw = geom.bw if geom is not None else float("nan")
        geom_bh = geom.bh if geom is not None else float("nan")

        lat = track.lateral_px(center_x) if track is not None else float("nan")
        head = track.heading_px if track is not None else float("nan")
        x_near = track.x_near if track is not None else float("nan")
        y_near = track.y_near if track is not None else float("nan")
        x_far = track.x_far if track is not None else float("nan")
        y_far = track.y_far if track is not None else float("nan")

        dbg = self._steer_dbg
        side_s = "R" if self._last_line_side > 0 else "L"
        turn_side_s = "R" if self._turn_yaw_dir > 0 else "L"
        turn_elapsed = (
            time.perf_counter() - self._turn_enter_time
            if self._mode == Mode.TURN else 0.0
        )
        decel_s = self._decel_scale()
        hold_elapsed = self._hold_elapsed()
        osd_lines = [
            f"{self._mode.value} | {status}",
            f"run={self.running} side={side_s} turn_dir={turn_side_s} "
            f"streak={self._ready_streak}/{_I.ready_confirm} "
            f"t_turn={turn_elapsed:.1f}s decel={decel_s:.2f} hold={hold_elapsed:.1f}s",
            f"odom {self._odom_short()} last={self.odom.last_segment_total:.3f}"
            if self.odom is not None
            else "odom off",
            f"fwd={fwd_cmd:+.3f} scale={dbg.fwd_scale:.2f} yaw={yaw_cmd:+.3f} "
            f"(demand={dbg.yaw_demand:+.3f} raw={dbg.yaw_raw:+.3f} "
            f"sat={'Y' if dbg.saturated else 'N'})",
            f"lat={lat:+.1f} head={head:+.1f} |lat|={self._last_lat_abs:.1f} "
            f"warm={self._good_streak}/{_I.track_warmup} "
            f"lost={self._lost_streak}/{_I.lost_frames} {self._hold_note}",
            f"P={dbg.p_term:+.4f} D={dbg.d_term:+.4f} A={dbg.a_term:+.4f} "
            f"dt={dbg.dt:.3f}s lat_f={dbg.lateral:+.1f} d_lat={dbg.d_lateral:+.0f}",
            f"near=({x_near:.0f},{y_near:.0f}) far=({x_far:.0f},{y_far:.0f}) cx={center_x:.0f}",
            f"geom y1={geom_y1:.0f} bw={geom_bw:.0f} bh={geom_bh:.0f}",
            f"mask_px={self._last_mask_px} track_ok={track_ok} scene={scene is not None}",
        ]

        _log_frame(osd_lines)

        vis = overlay_scene(frame, result, scene)
        if self._mode == Mode.FOLLOW and _I.follow_roi_top > 0:
            y_roi = int(vis.shape[0] * _I.follow_roi_top)
            cv.line(vis, (0, y_roi), (vis.shape[1], y_roi), (80, 80, 80), 1, cv.LINE_AA)
        draw_geom(vis, geom, self._mode)
        if track is not None:
            draw_track(vis, track)
        if self._hold_note:
            cv.putText(
                vis, "HOLD", (vis.shape[1] - 72, 28),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2, cv.LINE_AA,
            )
        self._draw_odom_progress(vis)
        _put_text_block(vis, osd_lines)

        if os.environ.get("DISPLAY"):
            cv.imshow("yolo-follow-2", vis)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                self._quit = True
            elif key == 32:
                was_running = self.running
                self.running = not self.running
                if self.running and not was_running:
                    if self.odom is not None and self._mode == Mode.FOLLOW:
                        if self.odom.segment_open:
                            self.odom.resume_accum()
                        else:
                            self._start_odom_segment_if_needed()
                elif not self.running and was_running:
                    if self.odom is not None:
                        self.odom.pause_accum()

    def close(self) -> None:
        self.log_odom_session_end()
        self.capture.release()
        cv.destroyAllWindows()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO 巡线 v2 (停车等待/反光抑制)")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument("--log", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-odom", action="store_true", help="禁用里程计/障碍判断")
    parser.add_argument(
        "--odom-file",
        default=str(DEFAULT_SEGMENT_FILE),
        help="每段结束追加写入的 JSONL 文件（默认 data/odom_segments.jsonl）",
    )
    args = parser.parse_args(argv)

    echo_console = DEBUG_PRINT_EVERY_FRAME and not args.quiet
    log_path = setup_logging(args.log, echo_console=echo_console)
    _log_info(f"日志文件: {log_path}")
    if not echo_console:
        print(f"日志文件: {log_path}", flush=True)

    ensure_display()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        print("请先: cd NavEngine && source .venv-yolo/bin/activate", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
    model = YOLO(args.model)
    _log_info(f"model={args.model} classes={model.names} cuda={use_cuda}")

    try:
        camera = int(args.camera)
    except (TypeError, ValueError):
        camera = args.camera

    auto = not bool(os.environ.get("DISPLAY"))
    print_debug_config(
        conf=args.conf, device="0" if use_cuda else "cpu",
        imgsz=640, half=use_cuda, camera=camera, img_flip=True, auto=auto,
        log_path=log_path,
    )
    link = NavLink(args.port, args.baud)
    follower = None
    odom: OdomEstimator | None = None
    try:
        link.open()
        if not args.no_odom and link.serial is not None:
            odom_path = Path(args.odom_file)
            odom = OdomEstimator(link.serial, segment_file=odom_path)
            odom.start()
            _log_info(f"odom 段文件: {odom_path.resolve()}")
            if odom.history:
                _log_info(
                    f"odom 已加载历史 {len(odom.history)} 段 "
                    f"exp={odom.expected_length:.3f} hist={odom.history}"
                )
        if not args.no_setup:
            link.setup()
        follower = YoloLineFollower(
            link, model, conf=args.conf, device="0" if use_cuda else "cpu",
            imgsz=640, half=use_cuda, camera=camera, auto=auto, odom=odom,
        )
        _log_info("start it" + ("（自动运行）" if auto else "（空格启动，q 退出）"))
        if not echo_console:
            print("start it" + ("（自动运行）" if auto else "（空格启动，q 退出）"), flush=True)
        while not follower._quit:
            follower.on_timer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        _log_info("退出 …")
        if not echo_console:
            print("\n退出 …", flush=True)
    finally:
        if odom is not None:
            odom.stop()
        if not args.no_setup:
            link.teardown()
        link.close()
        if follower is not None:
            follower.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
