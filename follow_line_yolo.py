#!/usr/bin/env python3
"""YOLO 巡线：框几何 + 中心线双行采样（位置 + 航向）。

状态机（严格互斥）:
  FOLLOW: 线速度 + 双误差 PD；能拟合出中心线就一直跟（弯道不因框位置掉头）。
  TURN:   仅当无检测或 fit 中心线失败 → 开环原地转；线进入捕获区连续 N 帧 → FOLLOW。
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

# ── 巡线 PID（文件顶部，改这里）────────────────────────────
# 量级：|lat|=30px → |P|≈0.09；调参看 OSD sat=N（饱和说明 demand 仍超 MAX_YAW）。
LINEAR = 0.2           # 前进线速度（直道目标）
FWD_MIN = 0.04         # 弯道饱和降速下限（载人宜留余量，勿停死）
MAX_YAW = 0.6          # 角速度上限
KP = 0.003             # P：1px 横向偏差 → 0.001 角速度
KA = 0.0007             # A：航向 x_near−x_far，在 P/D 负号外
KD = 0.0065             # D：px/s；约为 KP 的 2 倍量级，勿与 KP 同数量级
LAT_SMOOTH = 0.35       # 横向 EMA（越小越平滑，压 YOLO 掩膜抖动）
YAW_SMOOTH = 0.65      # 输出 EMA（略减 D 依赖）
TURN_YAW = 0.4         # 原地转弯角速度
CAM_CENTER_OFFSET = -28   # 摄像头安装偏移（640 宽像素，正=光轴偏右）

# 双行采样：近处=位置，远处−近处 x 差=航向代理（ratio 越小=看得越远）
NEAR_Y_RATIO = 0.60    # 近处采样（略抬高，少盯脚下）
FAR_Y_RATIO = 0.01     # 远处采样（更靠图像上方）

# 轨道拟合
LOCAL_FIT_HALFWIN = 70
LOCAL_FIT_MIN_POINTS = 6
LOCAL_FIT_Y_RATIO = 0.42  # 局部拟合参考高度（偏远处）

# 状态切换（仅看能否拟合中心线，不用外接框 y1 判线尾）
READY_CONFIRM_FRAMES = 3   # 捕获区连续 N 帧才进 FOLLOW

# TURN 捕获区（比 ALIGN 宽：先进 FOLLOW，靠 PID 拉回）
TURN_CAPTURE_LAT_PX = 120    # |x_near−中心| < 此值视为线进入捕获区
TURN_MIN_LINE_HEIGHT = 80    # 框高 bh：拒绝贴地横条
TURN_MIN_ASPECT = 1.15       # bh/bw：高于宽才算「看到正道」
LINE_SIDE_HYSTERESIS_PX = 20   # FOLLOW 记线侧迟滞，防弯道左右抖
D_LAT_MAX = 150.0              # D 微分上限 (px/s)，配合低 KD 防单帧尖峰

DEBUG_PRINT_EVERY_FRAME = True  # True 时终端也回显每帧 debug（日志文件始终记录）

_LOG = logging.getLogger("follow_line_yolo")
_LOG_DIR = Path(__file__).resolve().parent / "logs"


def setup_logging(log_path: str | Path | None = None, *, echo_console: bool) -> Path:
    """初始化日志：始终写文件；终端是否回显由 echo_console 控制。"""
    _LOG.handlers.clear()
    _LOG.setLevel(logging.DEBUG)
    _LOG.propagate = False

    if log_path is None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"follow_line_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    """启动时记录全部可调参数。"""
    lines = [
        "=== follow_line_yolo debug config ===",
        f"  log_file={log_path}",
        f"  LINEAR={LINEAR}  FWD_MIN={FWD_MIN}  MAX_YAW={MAX_YAW}  TURN_YAW={TURN_YAW}  YAW_SMOOTH={YAW_SMOOTH}",
        f"  KP={KP}  KA={KA}  KD={KD}  LAT_SMOOTH={LAT_SMOOTH}  CAM_CENTER_OFFSET={CAM_CENTER_OFFSET}",
        f"  NEAR_Y_RATIO={NEAR_Y_RATIO}  FAR_Y_RATIO={FAR_Y_RATIO}",
        f"  LOCAL_FIT_HALFWIN={LOCAL_FIT_HALFWIN}  LOCAL_FIT_MIN_POINTS={LOCAL_FIT_MIN_POINTS}",
        f"  LOCAL_FIT_Y_RATIO={LOCAL_FIT_Y_RATIO}",
        f"  TURN_CAPTURE_LAT_PX={TURN_CAPTURE_LAT_PX}",
        f"  TURN_MIN_LINE_HEIGHT={TURN_MIN_LINE_HEIGHT}  TURN_MIN_ASPECT={TURN_MIN_ASPECT}",
        f"  LINE_SIDE_HYSTERESIS_PX={LINE_SIDE_HYSTERESIS_PX}  D_LAT_MAX={D_LAT_MAX}",
        f"  READY_CONFIRM_FRAMES={READY_CONFIRM_FRAMES}",
        f"  yolo conf={conf} device={device} imgsz={imgsz} half={half}",
        f"  camera={camera} img_flip={img_flip} auto_start={auto}",
        f"  DEBUG_PRINT_EVERY_FRAME={DEBUG_PRINT_EVERY_FRAME}",
        "=====================================",
    ]
    for line in lines:
        _log_info(line)


def _fwd_from_yaw_saturation(yaw_demand: float) -> tuple[float, float]:
    """角速度需求超限时按比例降线速度，直道 scale=1。"""
    if abs(yaw_demand) > MAX_YAW:
        scale = MAX_YAW / abs(yaw_demand)
    else:
        scale = 1.0
    fwd = float(np.clip(LINEAR * scale, FWD_MIN, LINEAR))
    return fwd, scale


def _put_text_block(vis: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 22),
                    line_h: int = 18, color=(0, 220, 255), scale: float = 0.42) -> None:
    x0, y0 = origin
    for i, line in enumerate(lines):
        cv.putText(vis, line, (x0, y0 + i * line_h),
                cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


class Mode(Enum):
    TURN = "turn"
    FOLLOW = "follow"


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
    """在中心线可见段上按高度比例取 (x, y)。y_ratio=0 为最远（图像上方）。"""
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
        """近处相对远处的横向偏移；竖直线时接近 0。"""
        return self.x_near - self.x_far

    def lateral_px(self, center_x: float) -> float:
        return self.x_near - center_x

    def turn_capture_ready(self, center_x: float, geom: BoxGeom | None) -> tuple[bool, float, float, float]:
        """TURN→FOLLOW：近处采样点进入捕获区 + 非横条（不要求已居中，交给 PID）。"""
        lat = abs(self.lateral_px(center_x))
        aspect = (geom.bh / max(geom.bw, 1.0)) if geom is not None else 0.0
        bh = geom.bh if geom is not None else 0.0
        ok = (
            lat < TURN_CAPTURE_LAT_PX
            and bh >= TURN_MIN_LINE_HEIGHT
            and aspect >= TURN_MIN_ASPECT
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
    half_win = max(20, LOCAL_FIT_HALFWIN)
    local = np.abs(ys - y_ref) <= half_win
    if np.count_nonzero(local) >= LOCAL_FIT_MIN_POINTS:
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
    y_ref = y_min_all + float(np.clip(LOCAL_FIT_Y_RATIO, 0.45, 0.9)) * (y_max_all - y_min_all)

    seg = _polyfit_segment(ys, x_left, x_right, y_ref)
    if seg is None:
        return None
    center, left_line, right_line = seg

    x_near, y_near = _sample_centerline_x(center, NEAR_Y_RATIO)
    x_far, y_far = _sample_centerline_x(center, FAR_Y_RATIO)
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
    cx = int(vis.shape[1] / 2 + CAM_CENTER_OFFSET)
    cv.line(vis, (cx, 0), (cx, vis.shape[0]), (255, 0, 255), 1)


def draw_geom(vis: np.ndarray, geom: BoxGeom | None, mode: Mode) -> None:
    if geom is None:
        return
    p1 = (int(geom.x1), int(geom.y1))
    p2 = (int(geom.x2), int(geom.y2))
    color = (0, 255, 0) if mode == Mode.FOLLOW else (0, 165, 255)
    cv.rectangle(vis, p1, p2, color, 2)


class YoloLineFollower:
    def __init__(self, link: NavLink, model, *, conf: float, device: str,
                imgsz: int, half: bool, camera=0, img_flip: bool = True,
                auto: bool = False) -> None:
        self.link = link
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
        self._last_line_side = 1.0  # +1 线偏右, -1 线偏左
        self._turn_yaw_dir = 1.0    # TURN 期间锁定，不随 lat 翻转
        self._turn_enter_time = time.perf_counter()
        self._steer_dbg = SteerBreakdown()
        self._last_status = ""
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

    def _steer_to(self, lateral_px: float, heading_px: float) -> float:
        """PD + 航向：yaw = −(KP·横向 + KD·d横向/dt) + KA·航向。"""
        now = time.perf_counter()
        first = self._last_ctrl_time is None
        dt = 0.03 if first else max(1e-3, now - self._last_ctrl_time)
        self._last_ctrl_time = now

        if first:
            self._lat_filt = lateral_px
        else:
            self._lat_filt = (1.0 - LAT_SMOOTH) * self._lat_filt + LAT_SMOOTH * lateral_px
        lat_f = self._lat_filt

        if first:
            self._prev_error = lat_f
        d_lateral = (lat_f - self._prev_error) / dt
        d_lateral = float(np.clip(d_lateral, -D_LAT_MAX, D_LAT_MAX))
        self._prev_error = lat_f

        p_term = -(KP * lat_f)
        d_term = -(KD * d_lateral)
        a_term = KA * heading_px
        yaw_demand = p_term + d_term + a_term
        yaw_raw = float(np.clip(yaw_demand, -MAX_YAW, MAX_YAW))
        saturated = abs(yaw_demand) >= MAX_YAW - 1e-9
        alpha = float(np.clip(YAW_SMOOTH, 0.0, 1.0))
        yaw = (1.0 - alpha) * self._last_yaw + alpha * yaw_raw
        self._last_yaw = yaw

        fwd, fwd_scale = _fwd_from_yaw_saturation(yaw_demand)
        self._steer_dbg = SteerBreakdown(
            dt=dt, lateral=lat_f, heading=heading_px, d_lateral=d_lateral,
            p_term=p_term, d_term=d_term, a_term=a_term, yaw_demand=yaw_demand,
            yaw_raw=yaw_raw, yaw_out=yaw, fwd=fwd, fwd_scale=fwd_scale,
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
                f"side={self._last_line_side:+.0f} streak={self._ready_streak}"
            )
            self._last_mode_logged = self._mode

    def _update_line_side_follow(self, lat: float) -> None:
        if lat > LINE_SIDE_HYSTERESIS_PX:
            self._last_line_side = 1.0
        elif lat < -LINE_SIDE_HYSTERESIS_PX:
            self._last_line_side = -1.0

    def _begin_turn(self, reason: str) -> str:
        """唯一进 TURN 入口：丢线 / 拟合失败。"""
        if self._mode != Mode.TURN:
            self._reset_steer()
            self._turn_enter_time = time.perf_counter()
            self._turn_yaw_dir = self._last_line_side
        self._mode = Mode.TURN
        self._ready_streak = 0
        self._log_mode_change(reason)
        return f"TURN: {reason}"

    def _step_mode(
        self,
        scene: ScenePrediction | None,
        geom: BoxGeom | None,
        track_ok: bool,
        track: TrackInfo | None,
        w: int,
    ) -> str:
        """状态机：只改 mode，不发指令。"""
        center_x = w / 2.0 + CAM_CENTER_OFFSET

        if scene is None or geom is None:
            return self._begin_turn("无检测")

        if self._mode == Mode.FOLLOW:
            if track_ok and track is not None:
                self._update_line_side_follow(track.lateral_px(center_x))
                return "FOLLOW"
            return self._begin_turn("无拟合线")

        # ---- TURN ----（仅 fit 失败时进入；找回线后捕获 → FOLLOW）
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
        if self._ready_streak >= READY_CONFIRM_FRAMES:
            self._mode = Mode.FOLLOW
            self._ready_streak = 0
            self._log_mode_change("捕获→PID")
            return "FOLLOW: 捕获→PID"
        return f"TURN: 捕获 ({self._ready_streak}/{READY_CONFIRM_FRAMES})"

    def _apply_drive(self, mode: Mode, track: TrackInfo | None, track_ok: bool, w: int) -> None:
        """FOLLOW 闭环巡线；TURN 开环原地转（方向进 TURN 时锁定）。"""
        if mode == Mode.FOLLOW and track_ok and track is not None:
            center_x = w / 2.0 + CAM_CENTER_OFFSET
            lat = track.lateral_px(center_x)
            yaw = self._steer_to(lat, track.heading_px)
            self._drive(self._steer_dbg.fwd, yaw)
        elif mode == Mode.TURN:
            turn_yaw = -TURN_YAW * self._turn_yaw_dir
            self._drive(0.0, turn_yaw)
            self._steer_dbg = SteerBreakdown(
                fwd=0.0, yaw_out=turn_yaw, yaw_raw=turn_yaw,
            )
        else:
            self._drive(0.0, 0.0)
            self._steer_dbg = SteerBreakdown(fwd=0.0, yaw_out=0.0, yaw_raw=0.0)

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
                track = fit_centerline(mask)

        track_ok = track is not None
        if mask is not None:
            self._last_mask_px = int(cv.countNonZero(mask))

        fwd_cmd = 0.0
        yaw_cmd = 0.0

        if self.running:
            w = frame.shape[1]
            status = self._step_mode(scene, geom, track_ok, track, w)
            self._last_status = status
            self._apply_drive(self._mode, track, track_ok, w)
            fwd_cmd = self._steer_dbg.fwd
            yaw_cmd = self._steer_dbg.yaw_out
        else:
            self.link.stop()
            self._reset_steer()
            status = "暂停(空格启动)"
            self._last_status = status
            self._steer_dbg = SteerBreakdown()

        center_x = frame.shape[1] / 2.0 + CAM_CENTER_OFFSET
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
        osd_lines = [
            f"{self._mode.value} | {status}",
            f"run={self.running} side={side_s} "
            f"turn_dir={turn_side_s} streak={self._ready_streak}/{READY_CONFIRM_FRAMES} "
            f"t={turn_elapsed:.1f}s",
            f"fwd={fwd_cmd:+.3f} scale={dbg.fwd_scale:.2f} yaw={yaw_cmd:+.3f} "
            f"(demand={dbg.yaw_demand:+.3f} raw={dbg.yaw_raw:+.3f} sat={'Y' if dbg.saturated else 'N'})",
            f"lat={lat:+.1f} head={head:+.1f} |lat|={self._last_lat_abs:.1f} "
            f"asp={self._last_aspect:.2f} bh={self._last_geom_bh:.0f} capture={self._last_captured}",
            f"P={dbg.p_term:+.4f} D={dbg.d_term:+.4f} A={dbg.a_term:+.4f} "
            f"dt={dbg.dt:.3f}s lat_f={dbg.lateral:+.1f} d_lat={dbg.d_lateral:+.0f}",
            f"near=({x_near:.0f},{y_near:.0f}) far=({x_far:.0f},{y_far:.0f}) cx={center_x:.0f}",
            f"geom y1={geom_y1:.0f} bw={geom_bw:.0f} bh={geom_bh:.0f}",
            f"mask_px={self._last_mask_px} track_ok={track_ok} scene={scene is not None}",
            f"KP={KP} KA={KA} KD={KD} LAT_SMOOTH={LAT_SMOOTH} "
            f"CAP<{TURN_CAPTURE_LAT_PX} ASP>{TURN_MIN_ASPECT} bh>{TURN_MIN_LINE_HEIGHT}",
        ]

        _log_frame(osd_lines)

        vis = overlay_scene(frame, result, scene)
        draw_geom(vis, geom, self._mode)
        if track is not None:
            draw_track(vis, track)
        _put_text_block(vis, osd_lines)

        if os.environ.get("DISPLAY"):
            cv.imshow("yolo-follow", vis)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                self._quit = True
            elif key == 32:
                self.running = not self.running

    def close(self) -> None:
        self.capture.release()
        cv.destroyAllWindows()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO 巡线 (框几何 + 中心线拟合)")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument(
        "--log",
        default=None,
        help="日志文件路径（默认 logs/follow_line_YYYYMMDD_HHMMSS.log）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="终端不回显 debug（仍写入日志文件）",
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
    try:
        link.open()
        if not args.no_setup:
            link.setup()
        follower = YoloLineFollower(
            link, model, conf=args.conf, device="0" if use_cuda else "cpu",
            imgsz=640, half=use_cuda, camera=camera, auto=auto,
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
        if not args.no_setup:
            link.teardown()
        link.close()
        if follower is not None:
            follower.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
