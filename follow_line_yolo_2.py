#!/usr/bin/env python3
"""YOLO 巡线 v2 主程序（NavEngine 巡线小车的核心控制器）。

本文件是整个巡线系统的"大脑"，独立运行（不依赖 follow_line.py / follow_common.py，
相关串口/摄像头代码已内联）。启动后从摄像头读帧 → YOLO 检测线 → 拟合中心线 →
PD 控制律算速度 → 通过串口发给主控驱动电机，循环往复。

═══════════════════════════════════════════════════════════════════════
硬件接线（Orin 上）
═══════════════════════════════════════════════════════════════════════
  /dev/ttyCH341USB0  主控串口（921600）：NavLink 发 init/运动，OdomEstimator 读反馈 JSON
  /dev/ttyCH341USB1  超声波模块（9600）：UltrasonicSensor 读前方障碍距离（可选，--no-us 跳过）
  /dev/video0        USB 摄像头：OpenCV V4L2 读取 640×480 帧
  （ttyCH341USB2    主控调试日志口，程序不使用；勿与 USB0 混淆）

═══════════════════════════════════════════════════════════════════════
状态机（互斥，_step_mode 每帧根据检测结果切换）
═══════════════════════════════════════════════════════════════════════
  FOLLOW   正常巡线：PD 控制律（横向误差 + 航向 + 曲率）→ 线速度/角速度。
           进入条件：TURN 捕获到线 / 障碍移开 / 线恢复。
           退出条件：① 前方 50cm 内有障碍（连续 3 帧）→ BLOCKED(障碍)
                     ② 丢线连续 6 帧：
                        - 里程足够（线尾）→ PAUSE
                        - 里程不足（中途障碍丢线）→ BLOCKED(里程)

  PAUSE    线尾停车：说明跑到线尽头。立即停车 → 等 PAUSE_HOLD_SEC → TURN。
           仅在里程达到预期（odom.should_turn）时进入，避免中途丢线误判为线尾。

  BLOCKED  停等避让，有两种触发来源（_us_blocked 标记区分）：
           ① 障碍触发（_us_blocked=True）：超声波检测到 <50cm 障碍。
              退出条件：障碍移开连续 3 帧 → FOLLOW。
           ② 里程不足触发（_us_blocked=False）：丢线但里程还没到线尾长度。
              退出条件：线恢复（valid 连续 pause_recover_frames 帧）→ FOLLOW。
              不会因超时掉头，避免在障碍前原地转。

  TURN     原地掉头：线尾停稳后执行。朝上次记住的线侧（_last_line_side）转，
           直到重新捕获线（turn_capture_ready：横向偏差/框高/宽高比达标连续
           ready_confirm 帧）→ FOLLOW。掉头期间 odom 结束当前段并开始新段。

═══════════════════════════════════════════════════════════════════════
反光/抖动抑制（_steer_inputs）
═══════════════════════════════════════════════════════════════════════
  YOLO 掩膜受反光影响会瞬间跳变。预热（track_warmup 帧）后，若掩膜面积和近点 x
  同时大幅跳变，怀疑是反光 → 用上一帧可信值（HOLD）保持控制量，直到连续稳定或
  达 hold_max_frames 上限后释放，避免被反光带偏。

═══════════════════════════════════════════════════════════════════════
参数调整指南
═══════════════════════════════════════════════════════════════════════
  - 巡线手感（速度/PID/弯道降速）：改 config/follow_params.json 的 cfg，或代码里 FollowCfg。
  - 障碍触发距离/确认帧数：改 internals 的 us_obstacle_cm / us_obstacle_confirm，
    或运行时用 --us-obstacle 命令行参数覆盖。
  - 其他内部常数（拟合窗口、迟滞、停稳等待等）：internals，一般不动。
  - 弯道最低速度 = linear × curve_slow；角速度爬升按 yaw_ramp_sec 与真实 dt。
  - 默认参数文件：config/follow_params.json（可用 --params / --no-params）。

═══════════════════════════════════════════════════════════════════════
运行
═══════════════════════════════════════════════════════════════════════
  source .venv-yolo/bin/activate   # 需要 ultralytics + torch + cv2
  python follow_line_yolo_2.py     # 有显示器时空格启动，SSH 下自动运行
  python follow_line_yolo_2.py --help   # 看所有参数
  关键参数：--port（主控，默认 USB0）--us-port（超声波，默认 USB1）
            --us-obstacle 50（障碍阈值）--no-us（禁用超声波）--no-odom（禁用里程）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, fields, replace
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
from us_sensor import UltrasonicSensor, DEFAULT_PORT as DEFAULT_US_PORT, DEFAULT_BAUD as DEFAULT_US_BAUD
from yolo import DEFAULT_MODEL
from yolo.scene import ScenePrediction, select_scene_label
from yolo.viz import overlay_scene

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── 串口/摄像头默认参数（原 follow_line.py）─────────────────────────────
DEFAULT_PORT = "/dev/ttyCH341USB0"   # 主控：NavLink 写 + OdomEstimator 读（同一口）
DEFAULT_BAUD = 921600
DEFAULT_CAMERA = 0
# V4L2/OpenCV：AUTO_EXPOSURE=1 手动，=3 自动（部分驱动用 0.75 表示自动）
DEFAULT_EXPOSURE = 400
DEFAULT_GAIN = 50
DEFAULT_GAMMA = 500
DEFAULT_BRIGHTNESS = 20
DEFAULT_CONTRAST = 50
DEFAULT_SATURATION = 65
DEFAULT_WB_TEMP = 4800


@dataclass(frozen=True)
class CameraSettings:
    """摄像头成像参数。

    lock=True（默认）：固定手动曝光/白平衡，巡线亮度稳定、少漂移。
    lock=False：打开自动曝光（过曝时可先这样试）；白平衡也放开。

    实测有效组合（cam_tune.py 调出）：exp=400 gain=50 gamma=500
    bright=20 contrast=50 sat=65。gamma 拉暗部提亮，不增加曝光时间（无运动模糊）。
    """

    lock: bool = True
    exposure: float = DEFAULT_EXPOSURE
    gain: float = DEFAULT_GAIN
    gamma: float = DEFAULT_GAMMA
    brightness: float = DEFAULT_BRIGHTNESS
    contrast: float = DEFAULT_CONTRAST
    saturation: float = DEFAULT_SATURATION
    wb_temperature: float = DEFAULT_WB_TEMP


def _v4l2_set(device: str, controls: dict[str, float]) -> None:
    """用 v4l2-ctl 设控件（OpenCV 对 gamma/brightness/contrast/saturation 支持不稳定）。"""
    if not controls:
        return
    args = ["v4l2-ctl", "-d", device]
    for name, val in controls.items():
        args.append(f"--set-ctrl={name}={int(val)}")
    try:
        subprocess.run(args, capture_output=True, timeout=2.0)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


def configure_camera(cap: cv.VideoCapture, settings: CameraSettings,
                     device: str = "/dev/video0") -> None:
    """配置曝光与白平衡。手动模式用 v4l2-ctl 设全部控件（OpenCV 对部分控件不可靠）。"""
    if settings.lock:
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv.CAP_PROP_EXPOSURE, settings.exposure)
        cap.set(cv.CAP_PROP_GAIN, settings.gain)
        cap.set(cv.CAP_PROP_AUTO_WB, 0)
        cap.set(cv.CAP_PROP_WB_TEMPERATURE, settings.wb_temperature)
        # v4l2-ctl 补设 OpenCV 设不稳的控件（gamma/brightness/contrast/saturation/gain）
        _v4l2_set(device, {
            "exposure_time_absolute": settings.exposure,
            "gain": settings.gain,
            "gamma": settings.gamma,
            "brightness": settings.brightness,
            "contrast": settings.contrast,
            "saturation": settings.saturation,
        })
    else:
        if not cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3):
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv.CAP_PROP_AUTO_WB, 1)
        # 自动曝光下仍设 gamma/brightness 等（不依赖曝光模式的画面调节）
        _v4l2_set(device, {
            "gamma": settings.gamma,
            "brightness": settings.brightness,
            "contrast": settings.contrast,
            "saturation": settings.saturation,
        })

    # 丢弃前几帧，等驱动应用参数。
    for _ in range(3):
        cap.read()

    ae = cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv.CAP_PROP_EXPOSURE)
    gain = cap.get(cv.CAP_PROP_GAIN)
    awb = cap.get(cv.CAP_PROP_AUTO_WB)
    wb = cap.get(cv.CAP_PROP_WB_TEMPERATURE)
    mode = "手动锁定" if settings.lock else "自动曝光"
    print(
        f"摄像头参数({mode}): ae={ae:.2f} exp={exp:.0f} gain={gain:.0f} "
        f"gamma={settings.gamma:.0f} bright={settings.brightness:.0f} | "
        f"awb={awb:.0f} temp={wb:.0f}K"
    )


def open_camera(camera, settings: CameraSettings | None = None) -> cv.VideoCapture:
    """优先 V4L2 打开摄像头，避免 GStreamer 管道卡住。"""
    settings = settings or CameraSettings()
    device = f"/dev/video{camera}" if isinstance(camera, int) else str(camera)
    if isinstance(camera, int):
        cap = cv.VideoCapture(camera, cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(camera)
    else:
        cap = cv.VideoCapture(device, cv.CAP_V4L2)
        if not cap.isOpened():
            cap = cv.VideoCapture(device)
    if cap.isOpened():
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        configure_camera(cap, settings, device)
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
        # start 后须先推一帧零速，主控才进入 ISOKINETIC 速度环（见 odom_estimator demo）
        self._send(build_velocity_change(0.0, 0.0))
        time.sleep(0.1)

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
        """与 OdomEstimator 共用同一串口（NavLink 写 init/运动，odom 读反馈 JSON）。"""
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
    linear: float = 0.4         # 直道线速度
    max_yaw: float = 0.6         # 巡线角速度上限
    turn_yaw: float = 0.4        # TURN 原地掉头角速度（勿绑 max_yaw，掉头太快易打滑）
    kp: float = 0.0015
    ka: float = 0.001           # A 项基础增益（向后兼容：未提供 ka_straight/ka_curve 时用此值）
    ka_straight: float = -1.0   # A 项直道/出弯增益（curve_w→0 生效，<0 时回退到 ka）
    ka_curve: float = -1.0      # A 项弯中增益（curve_w→1 生效，<0 时回退到 ka）
    kd: float = 0.0028           # D 项增益
    kff: float = 1.1              # 曲率前馈增益：ω_ff = kff × κ(1/m) × v(m/s)。从 0.5 起步调
    curve_slow: float = 0.25      # 弯里最低线速度 = linear × curve_slow
    yaw_ramp_sec: float = 0.2    # 角速度从 0 爬满目标的秒数（巡线 + TURN 共用）


CFG = FollowCfg()

# ── 固定内部常数（一般不调）────────────────────────────────────
#
# ⚠️ 本类字段是「设计常数」，不是调参项。
#   调巡线手感只允许动 FollowCfg（速度/转向 PD/前馈）。
#   本类的字段只在以下情况动：
#     - 换摄像头/机械结构 → 几何标定类（cam_offset/near_y/far_y/...）
#     - 换帧率量级       → 滤波类（*_smooth/steer_dt_max/d_lat_max）
#     - 反光场地启用     → 反光保持类（track_warmup/jump_*/hold_*）
#     - 改离散行为       → 状态机类（lost_frames/pause_*/turn_*/...）
#   新增字段前先回答两问：
#     1) 哪份日志的哪几帧证明需要它？
#     2) 什么证据出现时可以删掉它？答不上第二问的就是补丁，病根在上游。
#
@dataclass(frozen=True)
class _Internals:
    # ── 滤波（换帧率量级才动）──────────────────────────────────
    lat_smooth: float = 0.5
    yaw_smooth: float = 0.55     # 稍滤 D 尖峰，仍比原 0.65 快
    head_smooth: float = 0.5     # head 进 A 项前的 EMA（裸 head σ≈28px，放大 ka 后直道会抖）
    fwd_scale_smooth: float = 0.65
    steer_dt_max: float = 0.2
    d_lat_max: float = 25.0      # D 单项上限（原40：直道 D 饱和 0.16 远大于 P 0.033 导致左右飘；降到25 让 P 能主导）

    # ── 几何标定（换摄像头/机械结构才动）──────────────────────
    cam_offset: float = -28
    near_y: float = 0.8          # 横向控制采样（0=远端顶部，1=近端脚下）
    far_y: float = 0.2           # 航向基准远端
    local_fit_halfwin: int = 70
    local_fit_min_points: int = 5
    local_fit_y_ratio: float = 0.28  # 局部拟合锚点（越小越往前看）
    local_fit_extend_px: float = 40.0  # 拟合线段上下端外推长度（px），用于画更长/采样更稳

    # ── 曲率（κ 处理）──────────────────────────────────────────
    # kappa_abs_max：原始 κ EMA 前的硬限幅；kappa_cap_max：限速/FF 用的上限。
    # 真弯 κ 在 2.3~4.5/m，回摆污染会抬到 6~8；abs_max=8 形同虚设，cap_max=5 才是有效闸。
    kappa_w_lo: float = 1.0           # |κf|(1/m) 达此值弯道权重开始 >0（偏航斜视噪声 ≤1.0）
    kappa_w_hi: float = 2.5           # |κf| 达此值弯道权重=1（真弯 2.3~4.5/m）
    kappa_abs_max: float = 8.0        # κ 原始值硬限幅（1/m），先限幅再 EMA
    kappa_cap_max: float = 5.0        # 限速用 κ 上限：回摆污染可把 κf 抬到 6~8，别信
    kappa_vcap_margin: float = 0.85   # 曲率限速余量：v ≤ margin×max_yaw/|κ|，防转向饱和过冲
    vcap_floor_frac: float = 0.5      # 曲率限速下限 = frac×linear（原0.2太慢，弯道一顿一顿；0.3 让最低速提到0.18m/s，更丝滑）

    # ── 弯道转向衰减（补丁家族，待重构）────────────────────────
    # ⚠️ 以下三组（FF 退场 / P 衰减 / 大偏差限速）都是围着"κ 不可信"打补丁。
    #    病根在 κ 拟合质量，不在这里。验证 κ 限幅收紧后可逐步删减。
    ff_max_frac: float = 0.5          # FF 幅值上限 = frac×max_yaw，给 P 留回线余量
    ff_lat_fade_lo: float = 60.0      # |lat_f| 超过此值 FF 开始退场（κ 不可信）
    # 17:36 日志教训：原 80 太宽，lat<80 时 FF 全额，把车从线上推过去（84~86% lat 0→+30）。
    # 改 30：FF 在车快到线时就退场，给 P 留主导。ff_lat_fade_hi 也对应收紧。
    ff_lat_fade_hi: float = 160.0     # |lat_f| 达到此值 FF 完全归零
    ff_head_fade_lo: float = 10.0     # head 与 FF 反号时：|head| 超过此值 FF 开始退场
    ff_head_fade_hi: float = 40.0     # head 与 FF 反号时：|head| 达到此值 FF 归零（弯末反打抑制）
    # FF 收敛退场：lat 正在快速回归（车头已转够）时，FF 别再加码，防出弯过冲。
    # ff_converge_dlat: |d_lateral| 达到此值（px/s）FF 收敛退场满弓；10fps 下 lat 每帧变 5px → d_lat≈50
    # ff_converge_atten: 收敛时 FF 最多削到此比例（0.0=完全归零，0.3=保留30%）
    ff_converge_dlat: float = 60.0
    ff_converge_atten: float = 1.0
    # A 项（航向）lat 门限衰减：head 在 |lat| 大时失效（near 点看不到线，head 退化成残线角度），
    # 继续用会把 P 的修正抵消光，车冲出去就修不回。lat 超过 start 开始压，到 zero 压到 0。
    a_lat_fade_start: float = 60.0    # |lat_f| 超过此值 A 项开始线性衰减
    a_lat_fade_zero: float = 150.0    # |lat_f| 达到此值 A 项完全归零（让 P 单独修）
    p_curve_atten: float = 0.5   # 回退：0.85 掐 P 太狠导致直道抖。0.5 只做温和对抗抑制
    p_curve_atten_lat_only: float = 0.35  # 仅 lat 维度触发（弯道未确认）时更保守
    p_curve_start_px: float = 60.0   # 回退到 60：40 触发面太宽，直道瞬态误伤
    p_curve_confirm: int = 3        # head 维度连续确认帧数，确认后才允许 lat 维度满弓
    lat_slow_start: float = 60.0      # |lat_f| 超过此值开始大偏差限速
    lat_slow_hi: float = 120.0        # |lat_f| 达到此值限速到 lat_slow_frac
    lat_slow_frac: float = 0.55       # 大偏差限速下限 = frac×linear

    # ── 速度爬升 ──────────────────────────────────────────────
    fwd_up_sec: float = 1.2           # fwd 从 0 爬回 linear 所需秒数（降速无限制）

    # ── 反光保持（反光场地才触发，平时零触发）──────────────────
    track_warmup: int = 12
    jump_lat_px: float = 42.0
    jump_mask_ratio: float = 0.35
    hold_max_frames: int = 4
    track_frame_gap_sec: float = 0.50   # 放宽：9fps 下偶发 250ms 不再误判 gap（原 0.250）

    # ── 状态机行为（各管一个离散行为，互不耦合）────────────────
    lost_frames: int = 6
    pause_hold_sec: float = 2
    pause_recover_frames: int = 3
    block_recover_fwd_sec: float = 0.8
    ready_confirm: int = 3
    turn_capture_lat: float = 180
    turn_min_line_height: float = 50
    turn_min_aspect: float = 0.8
    turn_capture_box_cx_max: float = 90.0  # TURN 捕获额外条件：检测框水平中心离画面中线 < 此值(px)
    line_side_hysteresis: float = 20.0
    lat_deadband_px: float = 2.0    # 原 10.0 太大，中心死区导致直道不跟线来回漂
    follow_roi_top: float = 0.3     # FOLLOW 时去掉图像顶部比例（保留下半 1-top）
    min_mask_px: int = 3500          # mask 像素低于此值视为线尾残片，拒绝拟合（防 lat/κ 被残片带偏）
    # 段刚起步保护：里程 < 此值时丢线不进「里程不足 BLOCKED」。
    early_seg_protect_m: float = 1.0
    # 超声波避障（滑动窗口计数）：
    # 模块约 2Hz，腿/窄物体会产生"障碍-无回波-障碍"交替，连续帧计数攒不到。
    # 改成最近 us_window 帧里有 us_obstacle_confirm 次命中就触发，容忍偶尔无回波。
    us_obstacle_cm: float = 85.0
    us_window: int = 5              # 滑动窗口大小（帧）；5 帧 ≈ 0.5s @ 10fps
    us_obstacle_confirm: int = 2    # 窗口内命中此帧数即触发（旧连续 5 太严，撞人才触发）
    us_clear_confirm: int = 5


_I = _Internals()
DEBUG_PRINT_EVERY_FRAME = True


DEFAULT_PARAM_FILE = Path(__file__).resolve().parent / "config" / "follow_params.json"


def _override_dataclass(base_obj, overrides: dict, section_name: str):
    """按 dataclass 字段名做安全覆盖；出现未知键直接报错，防拼写错误静默生效。

    以 "_" 开头的键视为注释/分组标题，自动跳过（不写入 dataclass）。
    """
    valid = {f.name for f in fields(base_obj)}
    applied = {k: v for k, v in overrides.items() if not k.startswith("_")}
    bad = sorted(set(applied) - valid)
    if bad:
        raise ValueError(f"{section_name} 含未知键: {', '.join(bad)}")
    return replace(base_obj, **applied)


def load_param_file(path: Path) -> tuple[FollowCfg, _Internals]:
    """从 JSON 文件加载参数，未提供的字段沿用代码默认值。

    文件格式：
      {
        "cfg": {...FollowCfg 字段...},
        "internals": {..._Internals 字段...}
      }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("参数文件根节点必须是对象")
    cfg_data = data.get("cfg", {})
    internal_data = data.get("internals", {})
    if not isinstance(cfg_data, dict):
        raise ValueError("cfg 必须是对象")
    if not isinstance(internal_data, dict):
        raise ValueError("internals 必须是对象")
    return (
        _override_dataclass(CFG, cfg_data, "cfg"),
        _override_dataclass(_I, internal_data, "internals"),
    )


def _fwd_min() -> float:
    return max(0.04, CFG.linear * 0.18)


def _yaw_step_max() -> float:
    return CFG.max_yaw / max(CFG.yaw_ramp_sec * 30.0, 1.0)


def _jump_x_near_px() -> float:
    return _I.jump_lat_px * 0.76


def _hold_stable_lat_px() -> float:
    return _I.jump_lat_px * 0.48


def _curve_fwd_scale(yaw_demand: float, heading_px: float, lat_f: float = 0.0,
                     in_curve: bool = False) -> float:
    """弯道降速系数：看 heading 判断弯道，但偏航(lat 大)导致的 head 不算弯道。

    原版只看 heading_px，但车偏在中线一侧时线也呈斜向(head 大)，会被误判为深弯
    而压到地板速，导致偏航后速度太低无法修回。这里引入 lat 修正：lat 大时减小
    降速幅度，让车能维持速度把偏航修回来。

    in_curve=True（κ 确认真弯）时禁用 lat 放宽：真弯里过冲会让 lat 变大，若仍
    放宽降速会形成正反馈（越冲越快、越快越冲），2026-07-09 16:21 日志实测过冲
    168px 就是这条链路。
    """
    _ = yaw_demand
    floor = CFG.curve_slow
    ah = abs(heading_px)
    head_start = 35.0
    head_span = 40.0
    if ah <= head_start:
        return 1.0
    t = min(1.0, (ah - head_start) / head_span)
    # 偏航修正：|lat_f| > 30 时开始放宽降速，>80 时基本不降速（偏航≠弯道）
    alat = abs(lat_f)
    if alat > 30.0 and not in_curve:
        relief = min(1.0, (alat - 30.0) / 50.0)   # 0~1
        t = t * (1.0 - 0.7 * relief)               # 最多把 t 压回 30%
    return 1.0 - t * (1.0 - floor)


def _apply_follow_roi(mask: np.ndarray) -> np.ndarray:
    """FOLLOW 专用：去掉顶部区域，只在下半部拟合巡线。"""
    top = int(mask.shape[0] * _I.follow_roi_top)
    if top <= 0:
        return mask
    out = mask.copy()
    out[:top, :] = 0
    return out


# 形态学清理用的核（固定 3×3，不暴露参数）
_MORPH_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """清理 YOLO mask 的反光噪声：开运算去碎块 + 连通域保留主块。

    轮子架子/地面反光会在 mask 里产生两类噪声：
    1. 零散小碎块（几个像素的白点）→ 开运算（腐蚀+膨胀）直接消掉；
    2. 和线不相连的中等亮区（架子反光）→ 连通域分析，只保留面积最大的块。
    线本身是大连通块，不会被误删。
    """
    if mask is None:
        return mask
    # 开运算：去掉 <3px 的碎块噪声，线条（宽度 >8px）基本无损
    cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, _MORPH_KERNEL)
    # 连通域：找所有独立块，只保留面积最大的（线），去掉散块
    num, labels, stats, _ = cv.connectedComponentsWithStats(cleaned, connectivity=8)
    if num <= 1:
        return cleaned
    # stats[:, 4] 是面积；第 0 个是背景，从 1 开始找最大
    areas = stats[1:, cv.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == biggest] = 255
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
    ff_term: float = 0.0
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
        f"  derived fwd_min={_fwd_min():.3f} yaw_step@30fps={_yaw_step_max():.3f} "
        f"yaw_step@10fps≈{CFG.max_yaw/max(CFG.yaw_ramp_sec*10.0,1.0):.3f}",
        f"  yolo conf={conf} device={device} imgsz={imgsz} half={half}",
        f"  camera={camera} img_flip={img_flip} auto_start={auto}",
        f"  us obstacle<{_I.us_obstacle_cm:.0f}cm confirm={_I.us_obstacle_confirm} clear={_I.us_clear_confirm}",
        "=====================================",
    ]
    for line in lines:
        _log_info(line)


# OSD 文本中文→ASCII 翻译表（cv2 Hershey 字体不支持 Unicode，会画成 ?）。
# 日志文件用中文（可读性好），OSD 走 _osd_ascii() 翻译后再画。
# 排序规则：更长/更具体的词必须排在前（否则短词先匹配破坏长词，如"停稳"会吃掉"线尾停稳"）。
_OSD_ZH_TO_ASCII: list[tuple[str, str]] = [
    # 长串优先
    ("起步丢线保护",   "EARLY_LOST"),
    ("线尾停稳",       "LINE_END_STOP"),
    ("等障碍移开",     "WAIT_OBS_CLEAR"),
    ("等待障碍",       "WAIT_OBS"),
    ("前方障碍",       "OBS_AHEAD"),
    ("障碍预警",       "OBS_WARN"),
    ("障碍移开",       "OBS_CLEAR"),
    ("无拟合线",       "NO_LINE"),
    ("里程不足",       "ODOM_SHORT"),
    ("误触恢复",       "FAST_REC"),
    ("捕获→PID",       "CAPTURE"),
    ("线尾丢线",       "LINE_END_LOST"),
    ("线恢复",         "LINE_OK"),
    # 短词在后
    ("捕获",           "CAP"),
    ("丢线",           "LOST"),
    ("停稳",           "STOP"),
    ("线尾",           "LINE_END"),
    ("无基线",         "NOBASE"),
    ("基线",           "BASE"),
    ("需>=",           "NEED>="),
    ("×",              "x"),
    ("但",             "but"),
]


def _osd_ascii(s: str) -> str:
    """把 OSD 行里的中文片段替换成 ASCII 标签；剩余非 ASCII 字符替换成 '#'。"""
    out = s
    for zh, asc in _OSD_ZH_TO_ASCII:
        out = out.replace(zh, asc)
    # 兜底：还残留的非 ASCII（罕见希腊字母/未覆盖中文）改成 '#'，避免 cv2 画成 '?'。
    if not out.isascii():
        out = "".join(c if c.isascii() else "#" for c in out)
    return out


def _put_text_block(vis: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 22),
                    line_h: int = 18, color=(0, 220, 255), scale: float = 0.42) -> None:
    x0, y0 = origin
    for i, line in enumerate(lines):
        cv.putText(vis, _osd_ascii(line), (x0, y0 + i * line_h),
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
    curvature: float              # 中心线在 near 点的曲率 κ（1/px），右弯为正
    px_per_m: float               # 像素→米的纵向尺度（近处线宽标定）

    @property
    def heading_px(self) -> float:
        return self.x_near - self.x_far

    def lateral_px(self, center_x: float) -> float:
        return self.x_near - center_x

    def turn_capture_ready(self, center_x: float, geom: BoxGeom | None) -> tuple[bool, float, float, float]:
        lat = abs(self.lateral_px(center_x))
        aspect = (geom.bh / max(geom.bw, 1.0)) if geom is not None else 0.0
        bh = geom.bh if geom is not None else 0.0
        # 额外条件：检测框水平中心离画面中线要足够近。
        # 摄像头朝下后线总在顶部，near 点(脚下)可能根本没有线，lateral_px 不可信；
        # 改用 YOLO 检测框的横向中心判断"线是否已转到车正前方"。
        box_cx_offset = abs(((geom.x1 + geom.x2) * 0.5) - center_x) if geom is not None else 1e9
        ok = (
            lat < _I.turn_capture_lat
            and bh >= _I.turn_min_line_height
            and aspect >= _I.turn_min_aspect
            and box_cx_offset < _I.turn_capture_box_cx_max
        )
        return ok, lat, aspect, bh


def _estimate_curvature(center: np.ndarray, y_ref: float) -> float:
    """估计中心线在 y_ref 处的曲率 κ（1/px），右弯（x 随 y 增大而增大）为正。

    对中心线点做二次多项式 x = a·y² + b·y + c 拟合，取 y_ref 处的二阶导数：
      κ ≈ 2a / (1 + (2a·y_ref + b)²)^(3/2)
    图像里 y 向下为正，约定：x 随 y 增大而增大（线向右弯）→ κ > 0。
    """
    if center.shape[0] < 6:
        return 0.0
    ys = center[:, 1]
    xs = center[:, 0]
    span = float(ys.max() - ys.min())
    if span < 20.0:
        return 0.0
    try:
        coeff = np.polyfit(ys, xs, 2)   # [a, b, c]，x = a·y² + b·y + c
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    a = float(coeff[0]); b = float(coeff[1])
    dx_dy = 2.0 * a * y_ref + b
    kappa = (2.0 * a) / max(1e-6, (1.0 + dx_dy * dx_dy) ** 1.5)
    # 数值稳定性限幅，防止拟合炸裂
    return float(np.clip(kappa, -0.02, 0.02))


def _px_per_m_from_width(track_width_px: float) -> float:
    """用线的物理宽度（默认假设线宽 5cm）把横向像素换算成 1/px 的曲率单位→1/m。

    返回 px_per_m（像素每米）。若线宽检测不可用返回 0，调用方按 0 处理。
    """
    if track_width_px <= 1:
        return 0.0
    LINE_WIDTH_M = 0.05   # 线条实际宽度，可按需标定
    return track_width_px / LINE_WIDTH_M


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


def _polyfit_segment(ys, x_left, x_right, y_ref: float, *, y_min_all: float, y_max_all: float):
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
    # 线段两端适度外推：让可视化/采样更“长”，同时钳位在本帧实际扫描到的 y 范围内，
    # 避免外推到完全无数据的区域导致发散。
    ext = max(0.0, float(_I.local_fit_extend_px))
    y0 = max(float(y_min_all), y_min - ext)
    y1 = min(float(y_max_all), y_max + ext)
    ys_s = np.linspace(y0, y1, max(len(ys_fit), 8))
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

    seg = _polyfit_segment(ys, x_left, x_right, y_ref, y_min_all=y_min_all, y_max_all=y_max_all)
    if seg is None:
        return None
    center, left_line, right_line = seg

    x_near, y_near = _sample_centerline_x(center, _I.near_y)
    x_far, y_far = _sample_centerline_x(center, _I.far_y)

    # 曲率（1/px）：在局部中心线上、脚下 y_near 处求值（与改 κ 补丁前一致）。
    # 154204 全域 2a 补丁导致入弯 FF 过早贴顶、侧偏崩到 100~150，已回退。
    kappa = _estimate_curvature(center, y_near)
    # near_y 处的线宽（px）：在 y_near 附近取左右线 x 的差
    try:
        lw_near = float(np.interp(y_near, left_line[:, 1], left_line[:, 0]))
        rw_near = float(np.interp(y_near, right_line[:, 1], right_line[:, 0]))
        width_px = max(1.0, rw_near - lw_near)
    except (ValueError, IndexError):
        width_px = 0.0
    px_per_m = _px_per_m_from_width(width_px)

    return TrackInfo(center, left_line, right_line, x_near, y_near, x_far, y_far,
                     kappa, px_per_m)


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
                auto: bool = False, odom: OdomEstimator | None = None,
                us: UltrasonicSensor | None = None,
                cam_settings: CameraSettings | None = None,
                video_writer: cv.VideoWriter | None = None) -> None:
        self.link = link
        self.odom = odom
        self.us = us
        self._video_writer = video_writer
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
        self._head_filt = 0.0
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
        self._lost_streak = 0
        self._good_streak = 0
        self._good = _GoodSample()
        self._hold_note = ""
        self._steer_dbg = SteerBreakdown()
        self._last_status = ""
        self._block_recover_t0: float | None = None
        self._inp_hold_streak = 0
        # P 衰减的弯道确认计数：P/A 反向且 |head|>阈值连续 N 帧才允许 lat 维度参与衰减。
        # 防 head 噪声单帧越线（σ≈28px）在直道大偏移瞬态误掐 P。
        self._curve_confirm_streak = 0
        # κ EMA 滤波值（1/m）：前馈 FF 与曲率限速共用，防单帧 κ 尖峰
        self._kappa_filt = 0.0
        self._curve_w = 0.0           # 连续弯道权重 0~1（日志用）
        self._last_fwd = 0.0          # 上一帧 fwd，用于上升速率限制
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
        # FOLLOW "线已建立"标志：TURN 捕获后线在画面顶部、mask 小，此阶段不切 ROI
        # 不做 min_mask_px 门槛；等 mask 首次 ≥ min_mask_px 后武装，之后小 mask=线尾。
        self._follow_acquired = False
        # 超声波避障状态
        self._us_obstacle_cm: float = _I.us_obstacle_cm   # 障碍触发阈值（可被 CLI 覆盖）
        # 滑动窗口计数（替代旧的连续帧计数）：
        # 超声模块约 2Hz，偶尔丢读数/无回波，"连续 N 帧都障碍"根本攒不到。
        # 改成最近 N 帧里有 K 帧检测到障碍就触发，容忍偶尔的无回波。
        self._us_hit_window: list[bool] = []   # 最近各帧是否检测到障碍
        self._us_clear_streak = 0         # BLOCKED(障碍) 中连续障碍移开的帧数
        self._us_last_cm: float | None = None   # 最近一次距离（OSD 显示用）
        self._us_last_ts: float = 0.0     # 最近一次有效读数的时间戳（防过期读数）
        self._us_blocked = False          # BLOCKED 是否由障碍触发（决定退出条件）

        self._cam_settings = cam_settings or CameraSettings()
        self.capture = open_camera(camera, self._cam_settings)
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera}")

    def _drive(self, fwd: float, yaw: float) -> None:
        self.link.move(fwd, -yaw if self.img_flip else yaw)

    def _reset_steer(self) -> None:
        self._last_yaw = 0.0
        self._prev_error = 0.0
        self._lat_filt = 0.0
        self._head_filt = 0.0
        self._last_ctrl_time = None
        self._curve_confirm_streak = 0
        self._kappa_filt = 0.0
        self._last_fwd = 0.0

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

    def _steer_to(self, lateral_px: float, heading_px: float,
                  track: TrackInfo | None = None) -> float:
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
            self._head_filt = heading_px
        else:
            self._lat_filt = (1.0 - _I.lat_smooth) * self._lat_filt + _I.lat_smooth * lateral_px
            self._head_filt = (1.0 - _I.head_smooth) * self._head_filt + _I.head_smooth * heading_px
        lat_f = self._lat_filt
        head_f = self._head_filt

        if first:
            self._prev_error = lat_f
        d_lateral = (lat_f - self._prev_error) / dt
        d_lateral = float(np.clip(d_lateral, -_I.d_lat_max, _I.d_lat_max))
        self._prev_error = lat_f

        p_term = -(CFG.kp * lat_f)
        d_term = -(CFG.kd * d_lateral)
        # a_term 延后到 curve_w 计算之后（ka 按 curve_w 调度，见下方）

        # κ 滤波：单帧 κ 尖峰噪声大（16:50 日志实测单帧 0.0006→0.007 跳 10 倍），
        # 先硬限幅再重 EMA（新值权重 0.25），供前馈 FF 与曲率限速共用。
        kappa_raw = 0.0
        if track is not None and track.px_per_m > 1.0:
            kappa_raw = track.curvature * track.px_per_m
        kappa_raw = float(np.clip(kappa_raw, -_I.kappa_abs_max, _I.kappa_abs_max))
        if first:
            self._kappa_filt = kappa_raw
        else:
            # 非对称 EMA：κ 幅值增长（入弯）慢进 0.25 防噪声；收缩（出弯）快放 0.5。
            # 出弯 κf 拖 0.5~1s 不归零是弯末过头的另一半原因（FF 跟着多转）。
            a_k = 0.5 if abs(kappa_raw) < abs(self._kappa_filt) else 0.25
            self._kappa_filt = (1.0 - a_k) * self._kappa_filt + a_k * kappa_raw
        kappa_f = self._kappa_filt
        # 连续弯道权重（替代二值死区开关）：|κf| 在 lo→hi 之间线性 0→1。
        # 17:03 日志教训：硬开关在死区附近穿越一次，FF/限速就整体跳变，
        # fwd 0.07↔0.39 弹跳、FF -0.27↔0 跳变，弯里"一顿一顿"全是它。
        curve_w = float(np.clip(
            (abs(kappa_f) - _I.kappa_w_lo)
            / max(_I.kappa_w_hi - _I.kappa_w_lo, 1e-3), 0.0, 1.0))
        in_curve = curve_w >= 0.5
        self._curve_w = curve_w

        # A 项（航向）按 curve_w 调度：
        # 出弯瞬间 curve_w→0，κf≈0，此时 head 里没有曲率污染，就是纯航向误差，
        # 加大增益能主动修 head 残留（出弯 head=-33 拉不回 → lat 飙到 115 的根因）。
        # 弯中 curve_w→1，head 同时编码前方曲率，FF 已在处理，A 项压低给 FF 让路。
        # ka_straight/ka_curve <0 时回退到固定 ka（向后兼容旧 params）。
        ka_s = CFG.ka_straight if CFG.ka_straight >= 0 else CFG.ka
        ka_c = CFG.ka_curve if CFG.ka_curve >= 0 else CFG.ka
        ka_eff = ka_c + (ka_s - ka_c) * (1.0 - curve_w)
        a_term = ka_eff * head_f
        # lat 大时压 A 项：head = x_near - x_far 在 |lat| 大时失效（near 点已看不到线，
        # head 退化成残线角度，方向常与 P 相反，把 P 的回线修正抵消光 → 车冲出去修不回）。
        # 当 |lat| 超过 a_lat_fade_start 线性压 A，到 a_lat_fade_zero 压到 0，让 P 单独修。
        # 17:36 日志教训：lat=+165 head=+81 时 A=+0.24 把 P=-0.18 抵消，yaw_cmd≈0 车不转。
        alat_a = abs(lat_f)
        if alat_a > _I.a_lat_fade_start:
            a_span = max(_I.a_lat_fade_zero - _I.a_lat_fade_start, 1e-3)
            a_w = float(np.clip(
                1.0 - (alat_a - _I.a_lat_fade_start) / a_span, 0.0, 1.0))
            a_term *= a_w

        # 弯内 P/A 反向时衰减 P：P 的符号是 −sign(lat_f)，A 的符号是 sign(head)。
        # P/A 反向 ⟺ sign(lat_f)·sign(head) > 0（lat 与 head 同号）。
        # 衰减强度同时看 head（弯深）和 |lat_f|（偏航量）：lat 越大，P 越要给 A 让路。
        # 迟滞：head 维度需连续 p_curve_confirm 帧确认才允许 lat 维度满弓（防 head 噪声
        # σ≈28px 在直道大偏移瞬态单帧越线误掐 P）。
        p_sign = 1.0 if lat_f >= 0 else -1.0
        a_sign = 1.0 if heading_px >= 0 else -1.0
        if p_sign * a_sign > 0 and abs(heading_px) > _I.p_curve_start_px:
            curve_w_head = min(1.0, (abs(heading_px) - _I.p_curve_start_px) / 20.0)
            curve_w_lat = min(1.0, max(0.0, (abs(lat_f) - 30.0)) / 50.0)
            # head 维度连续确认计数
            self._curve_confirm_streak = min(self._curve_confirm_streak + 1, _I.p_curve_confirm)
            if self._curve_confirm_streak >= _I.p_curve_confirm:
                atten = _I.p_curve_atten * max(curve_w_head, curve_w_lat)
            else:
                # 弯道未确认：只允许较弱衰减（lat 维度上限封顶），避免直道瞬态误伤
                atten = _I.p_curve_atten_lat_only * curve_w_head
                atten = max(atten, _I.p_curve_atten_lat_only * min(curve_w_lat, 0.5))
            p_term = p_term * (1.0 - atten)
        else:
            self._curve_confirm_streak = 0

        fwd_scale = _curve_fwd_scale(p_term + d_term + a_term, heading_px, lat_f,
                                     in_curve=in_curve)
        # 曲率限速融入 scale（EMA 之前，与 head 降速共用平滑通道，不再瞬跳）：
        # ω = v×κ，要求 v ≤ margin × max_yaw / |κ|，否则转向饱和必过冲。
        # κ 限幅到 kappa_cap_max（回摆污染会把 κ 抬到 6~8，真弯只有 2.3~4.5），
        # 限速下限 vcap_floor_frac×linear（不再砸到 0.07 爬行），按 curve_w 连续混合。
        if curve_w > 0.0:
            kap = min(abs(kappa_f), _I.kappa_cap_max)
            v_cap = _I.kappa_vcap_margin * CFG.max_yaw / max(kap, 1e-6)
            v_cap = max(v_cap, _I.vcap_floor_frac * CFG.linear)
            cap_scale = min(1.0, v_cap / max(CFG.linear, 1e-6))
            cap_scale = 1.0 - curve_w * (1.0 - cap_scale)
            fwd_scale = min(fwd_scale, cap_scale)
        # 大偏差限速：弯道后半前方线段变直（κ→0、曲率限速释放），但车还挂在线外
        # 100px+，满速只会斜着滑走（17:15 日志：lat=-127 时 fwd 弹回 0.39，yaw 塌到
        # 0.02）。|lat| 大时压一个温和上限，让 P 先把车收敛回线。
        alat_now = abs(lat_f)
        if alat_now > _I.lat_slow_start:
            w_lat = min(1.0, (alat_now - _I.lat_slow_start)
                        / max(_I.lat_slow_hi - _I.lat_slow_start, 1e-3))
            lat_cap = 1.0 - w_lat * (1.0 - _I.lat_slow_frac)
            fwd_scale = min(fwd_scale, lat_cap)
        alpha_slow = float(np.clip(_I.fwd_scale_smooth, 0.0, 0.99))
        alpha = 0.40 if fwd_scale > self._fwd_scale_filt else alpha_slow
        self._fwd_scale_filt = alpha * self._fwd_scale_filt + (1.0 - alpha) * fwd_scale
        fwd = float(np.clip(CFG.linear * self._fwd_scale_filt, _fwd_min(), CFG.linear))
        # 上升速率限制：降速立即生效（安全），恢复限速按 fwd_up_sec 爬回，
        # 防 κ 噪声让限速松开/收紧交替造成 0.40↔0.07 弹跳（"一顿一顿"）。
        if not first:
            up_step = CFG.linear * dt / max(_I.fwd_up_sec, 1e-3)
            fwd = min(fwd, self._last_fwd + up_step)
        self._last_fwd = fwd
        if self._block_recover_t0 is not None:
            t = (time.perf_counter() - self._block_recover_t0) / _I.block_recover_fwd_sec
            if t >= 1.0:
                self._block_recover_t0 = None
            else:
                # 从 0 线性爬（原 max(0.25,t) 首帧就 25%×linear，TURN 捕获后易打滑）
                fwd *= float(np.clip(t, 0.0, 1.0))

        # 曲率前馈：ω_ff = kff × κ(1/m) × v(m/s)。κ 由中心线二次拟合得到（EMA 滤波），
        # 像素曲率经 px_per_m 换算到 1/m。直道 κ≈0，前馈自然失效；弯道按曲率给足转向。
        # 三道保险：
        # 1) 幅值上限 ff_max_frac×max_yaw，给 P 留回线余量，FF 永远不独占转向；
        # 2) |lat_f| 大时 κ 是斜视线条拟合出来的、不可信，FF 线性退场
        #    （|lat|≤ff_lat_fade_lo 全额 → ≥ff_lat_fade_hi 归零），把转向让给 P；
        # 3) 按 curve_w 连续加权，弯道边缘不再 0↔满幅跳变；
        # 4) 弯末反打抑制：head 与 FF 反号 = 车头已转过线的方向（FF 只看前方线的
        #    弯度，不知道车已经转够了）。17:33 日志：出弯 head=-33 时 FF 仍 +0.30
        #    顶着转，车以大角度穿线冲到对侧 +100px，直线上还要往回修。
        ff_term = 0.0
        if curve_w > 0.0 and CFG.kff > 0:
            ff_lim = _I.ff_max_frac * CFG.max_yaw
            ff_term = float(np.clip(-CFG.kff * kappa_f * fwd, -ff_lim, ff_lim))
            alat = abs(lat_f)
            span = max(_I.ff_lat_fade_hi - _I.ff_lat_fade_lo, 1e-3)
            ff_w = float(np.clip(1.0 - (alat - _I.ff_lat_fade_lo) / span, 0.0, 1.0))
            ff_term *= ff_w * curve_w
            # 收敛退场：lat 正在快速回归（lat×d_lateral<0）= 车头已转够，FF 别再加码。
            # 解决"出弯过冲"：FF 因 κf EMA 滞后在车追上弯道后还在满弓推，把车推过线。
            # 17:54 日志：lat -55→0 收敛 6 帧期间 FF 持续 +0.415，出弯冲到 +135。
            if lat_f * d_lateral < 0:
                converge_w = float(np.clip(
                    abs(d_lateral) / _I.ff_converge_dlat, 0.0, 1.0))
                ff_term *= 1.0 - converge_w * _I.ff_converge_atten
            if ff_term * heading_px < 0:
                # 弯末反打抑制：FF 与 head 反号 = 车头方向（head=near-far）已越过线走向，
                # FF 按 κf(EMA 滞后)还在同向加码 = 过推，该削。
                # 符号约定（已用 102745 日志 + 现实右弯校准，含摄像头水平翻转）：
                #   现实右弯 → flip后画面右弯 → head>0(near比far靠右), kf<0, FF=-kff×kf>0(向右转)
                #   入弯：FF>0 + head>0 → ff×head>0 → 不削（FF 推车追弯）✓
                #   弯末：FF>0(κf滞后) + head<0(线视觉反向) → ff×head<0 → 削 ✓
                # 曾试过 lat×head>0 和 ff×lat<0，都误削入弯/弯中的正常 FF，已回退。
                hspan = max(_I.ff_head_fade_hi - _I.ff_head_fade_lo, 1e-3)
                ff_term *= float(np.clip(
                    1.0 - (abs(heading_px) - _I.ff_head_fade_lo) / hspan, 0.0, 1.0))

        yaw_demand = p_term + d_term + a_term + ff_term
        yaw_raw = float(np.clip(yaw_demand, -CFG.max_yaw, CFG.max_yaw))
        if not first:
            # 按 dt 限制角速度斜率，不再假设 30fps（实际 ~10fps）。
            step = CFG.max_yaw * dt / max(CFG.yaw_ramp_sec, 1e-3)
            yaw_raw = float(np.clip(
                yaw_raw,
                self._last_yaw - step,
                self._last_yaw + step,
            ))
        saturated = abs(yaw_demand) >= CFG.max_yaw - 1e-9
        alpha = float(np.clip(_I.yaw_smooth, 0.0, 1.0))
        yaw = (1.0 - alpha) * self._last_yaw + alpha * yaw_raw
        self._last_yaw = yaw

        self._steer_dbg = SteerBreakdown(
            dt=dt, lateral=lat_f, heading=heading_px, d_lateral=d_lateral,
            p_term=p_term, d_term=d_term, a_term=a_term, ff_term=ff_term,
            yaw_demand=yaw_demand,
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
        mask = (m > 0.5).astype(np.uint8) * 255
        return _clean_mask(mask)

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
        # 重新走"线建立"流程：捕获后线在顶部 mask 小；正常线上 1 帧内即重新武装
        self._follow_acquired = False
        self._clear_stop()
        self._us_blocked = False
        self._us_hit_window.clear()
        self._us_clear_streak = 0
        if reason in ("线恢复", "误触恢复", "捕获→PID"):
            self._reset_steer()
            self._fwd_scale_filt = 0.0
            self._reset_track_hold()
            # 线速从 0 爬回，避免静止/掉头后瞬间给满 linear 打滑
            self._block_recover_t0 = time.perf_counter()
            if reason != "捕获→PID":
                _log_info(f"线恢复: 重置 PID/HOLD ({reason})")
            else:
                _log_info("TURN 捕获→FOLLOW: 线速软起步")
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

    def _reset_hold_timer(self) -> None:
        self._hold_accum = 0.0
        self._hold_tick_last = None

    def _mark_stopped(self) -> None:
        """PAUSE/BLOCKED 立即停车，并开始停稳计时。"""
        if self._stop_time is None:
            self._stop_time = time.perf_counter()
            self._reset_hold_timer()
            self._hold_tick_last = time.perf_counter()

    def _clear_stop(self) -> None:
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

    def _begin_pause(self, reason: str) -> str:
        if self._mode != Mode.PAUSE:
            self._pause_enter_time = time.perf_counter()
            self._pause_recover_streak = 0
            self._reset_track_hold()
            self._mark_stopped()
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
            self._mark_stopped()
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
        self._clear_stop()
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

    # ── 超声波避障 ────────────────────────────────────────────────
    def _us_osd(self) -> str:
        """OSD 单行：超声波状态 + 距离 + 触发计数。"""
        if self.us is None:
            return "us off"
        if not self.us.alive:
            return f"us DISCONNECTED (port={self.us.port})"
        # 直接读传感器当前值（_us_last_cm 只在 FOLLOW 中更新，暂停时会过期）
        d = self.us.distance_cm
        self._us_last_cm = d
        if d is None:
            d_s = "   n/a "
            flag = " "
        elif d < 0:
            d_s = "  clear"   # 无回波 = 前方开阔
            flag = " "
        else:
            flag = "!" if d < self._us_obstacle_cm else " "
            d_s = f"{d:6.1f}cm"
        hits = sum(self._us_hit_window)
        win = len(self._us_hit_window)
        return (
            f"us D={d_s}{flag} thr<{self._us_obstacle_cm:.0f} "
            f"obst={hits}/{win}(need {_I.us_obstacle_confirm}) "
            f"clear={self._us_clear_streak}/{_I.us_clear_confirm}"
        )

    def _us_obstacle_now(self) -> bool:
        """当前帧前方 threshold 内是否有障碍（含读数新鲜度检查）。

        D<0（无回波=前方开阔）或 D>=threshold 均视为无障碍。
        读数超过 STALE_SEC 视为过期（可能是模块卡住），也视为无障碍（保守）。
        """
        if self.us is None:
            return False
        d = self.us.distance_cm
        now = time.perf_counter()
        if d is not None and d >= 0:
            self._us_last_cm = d
            self._us_last_ts = now
            return d < self._us_obstacle_cm
        # d is None（从未收到）或 d<0（无回波）：不算障碍
        return False

    def _begin_us_blocked(self) -> str:
        """障碍触发的 BLOCKED（区别于里程不足的 BLOCKED，退出条件不同）。"""
        # 复用 _begin_blocked 的立即停车/odom pause 逻辑，再打障碍标记
        reason = f"前方障碍 D={self._us_last_cm:.1f}cm<{self._us_obstacle_cm:.0f}"
        msg = self._begin_blocked(reason)
        self._us_blocked = True
        self._us_clear_streak = 0
        self._us_hit_window.clear()
        return msg

    def _us_record_and_check(self) -> tuple[bool, int, int]:
        """记录本帧障碍状态到滑动窗口，返回（是否应触发 BLOCKED, 命中数, 窗口大小）。

        窗口逻辑（替代旧的"连续 N 帧都障碍"）：
        超声模块约 2Hz，腿/窄物体常产生"障碍-无回波-障碍"交替，
        连续计数永远攒不到。改成最近 window 帧里有 hits 次就触发。
        """
        hit = self._us_obstacle_now()
        self._us_hit_window.append(hit)
        # 限制窗口长度
        if len(self._us_hit_window) > _I.us_window:
            del self._us_hit_window[0:len(self._us_hit_window) - _I.us_window]
        hits = sum(self._us_hit_window)
        trigger = hits >= _I.us_obstacle_confirm
        return trigger, hits, len(self._us_hit_window)

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
                self._clear_stop()
                self._update_line_side_follow(track.lateral_px(center_x))
                # 超声波避障：滑动窗口计数（容忍偶尔无回波），命中数达阈值 → BLOCKED
                trigger, hits, win = self._us_record_and_check()
                if trigger:
                    return self._begin_us_blocked()
                note = f" | {self._hold_note}" if self._hold_note else ""
                if hits > 0:
                    return f"FOLLOW{note} | 障碍预警 {hits}/{win}"
                return f"FOLLOW{note}"
            self._lost_streak += 1
            if self._lost_streak >= _I.lost_frames:
                # 段刚起步：丢线多半是 ROI/mask 抖动，进 BLOCKED 会与「线恢复」死循环
                # （10:34 日志：tot=0 → BLOCKED → 恢复 → 再丢，车永远开不出去）
                if (self.odom is not None
                        and self.odom.total < _I.early_seg_protect_m):
                    self._lost_streak = _I.lost_frames  # 封顶，等线回来
                    return (
                        f"FOLLOW: 起步丢线保护 "
                        f"tot={self.odom.total:.3f}<{_I.early_seg_protect_m:.1f}m "
                        f"({self._lost_streak}/{_I.lost_frames})"
                    )
                if self.odom is not None and not self.odom.should_turn():
                    return self._begin_blocked(
                        f"里程不足 tot={self.odom.total:.3f} "
                        f"需>={self._turn_threshold_str()}",
                    )
                return self._begin_pause("线尾丢线")
            return f"FOLLOW: 丢线({self._lost_streak}/{_I.lost_frames})"

        if self._mode == Mode.BLOCKED:
            hold = self._hold_elapsed()
            if self._us_blocked:
                # 障碍 BLOCKED：靠障碍移开退出，不看线
                obstacle = self._us_obstacle_now()
                if not obstacle:
                    self._us_clear_streak += 1
                    if self._us_clear_streak >= _I.us_clear_confirm:
                        self._enter_follow("障碍移开")
                        return "FOLLOW: 障碍移开"
                else:
                    self._us_clear_streak = 0
                d = self._us_last_cm
                d_s = "clear" if (d is None or d < 0) else f"{d:.1f}cm"
                return (
                    f"BLOCKED: 等障碍移开 D={d_s} "
                    f"clear={self._us_clear_streak}/{_I.us_clear_confirm} hold={hold:.1f}s"
                )
            # 里程不足 BLOCKED：靠线恢复退出
            if recovered := self._try_recover_from_pause(valid):
                return recovered
            return (
                f"BLOCKED: 等待障碍 hold={hold:.1f}s "
                f"odom={self._odom_short()}"
            )

        if self._mode == Mode.PAUSE:
            elapsed = time.perf_counter() - self._pause_enter_time
            if valid and track is not None and elapsed < 0.4:
                self._enter_follow("误触恢复")
                return "FOLLOW: 误触恢复"
            if recovered := self._try_recover_from_pause(valid):
                return recovered
            hold = self._hold_elapsed()
            if hold >= _I.pause_hold_sec:
                if self.odom is not None and not self.odom.should_turn():
                    return self._begin_blocked(
                        f"停稳{hold:.1f}s但里程不足 tot={self.odom.total:.3f}",
                    )
                tot_s = (
                    f" tot={self.odom.total:.3f}" if self.odom is not None else ""
                )
                return self._begin_turn(f"线尾停稳{hold:.1f}s{tot_s}")
            return f"PAUSE: hold {hold:.1f}/{_I.pause_hold_sec:.1f}s"

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
            box_cx_off = abs(((geom.x1 + geom.x2) * 0.5) - center_x) if geom is not None else -1
            return (f"TURN: lat={lat_px:.0f} asp={aspect:.2f} bh={bh:.0f} "
                    f"box_cx={box_cx_off:.0f}/{_I.turn_capture_box_cx_max:.0f}")

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

    def _velo_fb_short(self, fwd_cmd: float, yaw_cmd: float) -> str:
        # 命令 vs 主控反馈的瞬时速度，用来判断电机是否达到命令值（max_yaw 上限探测）。
        # fb=(fwd_fb,yaw_fb)：odom 从串口 JSON 解析的真实 velo_fwd/velo_yaw。
        # rx：里程计累计收到的反馈帧数；rate：反馈帧率（与主控 ~?Hz 对照，判断串口是否丢包）。
        if self.odom is None:
            return f"velo cmd=({fwd_cmd:+.3f},{yaw_cmd:+.3f}) fb=off"
        fwd_fb, yaw_fb = self.odom.last_velo
        return (
            f"velo cmd=({fwd_cmd:+.3f},{yaw_cmd:+.3f}) "
            f"fb=({fwd_fb:+.3f},{yaw_fb:+.3f}) "
            f"rx={self.odom.rx_velo_frames}"
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
        # 同步读串口反馈（替代后台线程，避免 Jetson+CH341 多线程卡死）
        self.odom.poll_serial()
        self.odom.feed_command(fwd, yaw)
        # 仅 FOLLOW 活跃段积分；TURN/PAUSE/BLOCKED 内 _accumulating=False
        if self._mode == Mode.FOLLOW:
            self.odom._step()

    def _apply_drive(self, mode: Mode, track: TrackInfo | None, track_ok: bool, w: int) -> None:
        if mode in (Mode.PAUSE, Mode.BLOCKED):
            self._drive(0.0, 0.0)
            self._steer_dbg = SteerBreakdown(fwd=0.0, yaw_out=0.0, yaw_raw=0.0)
            self._feed_odom(0.0, 0.0)
            return

        if mode == Mode.FOLLOW and track_ok and track is not None:
            center_x = w / 2.0 + _I.cam_offset
            lat, head, hold_note = self._steer_inputs(track, center_x, self._last_mask_px)
            if hold_note.startswith("HOLD rel") or hold_note == "gap":
                self._lat_filt = lat
                self._head_filt = head
                self._prev_error = lat    # 原 0.0 会令下一帧 D 项喷出尖峰 → 直道抖动
            yaw = self._steer_to(lat, head, track)
            fwd = self._steer_dbg.fwd
            self._drive(fwd, yaw)
            self._feed_odom(fwd, yaw)
            return

        if mode == Mode.TURN:
            # 独立 turn_yaw（勿绑 max_yaw）；按 yaw_ramp_sec 爬到目标，避免原地猛甩
            target = -CFG.turn_yaw * self._turn_yaw_dir
            now = time.perf_counter()
            dt = 0.1 if self._last_ctrl_time is None else min(
                now - self._last_ctrl_time, _I.steer_dt_max)
            self._last_ctrl_time = now
            step = CFG.turn_yaw * dt / max(CFG.yaw_ramp_sec, 1e-3)
            delta = target - self._last_yaw
            turn_yaw = self._last_yaw + float(np.clip(delta, -step, step))
            self._last_yaw = turn_yaw
            self._drive(0.0, turn_yaw)
            self._steer_dbg = SteerBreakdown(
                fwd=0.0, yaw_out=turn_yaw, yaw_raw=target,
            )
            self._feed_odom(0.0, turn_yaw)
            return

        # FOLLOW 丢线但未满 lost_frames：直接停车
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
                    # ROI 武装条件：试切后 mask 仍 ≥ min_mask_px 才武装。
                    # 旧逻辑用全画面 raw_px 武装，下一帧一切 ROI 就掉到门槛下 →
                    # track_ok 翻转 → FOLLOW↔BLOCKED 振荡（10:34 日志 mask 3626→2590）。
                    if self._follow_acquired:
                        mask = _apply_follow_roi(mask)
                    else:
                        trial = _apply_follow_roi(mask)
                        trial_px = int(cv.countNonZero(trial))
                        if trial_px >= _I.min_mask_px:
                            self._follow_acquired = True
                            mask = trial
                        # else：保持全画面，继续未武装（线还在顶部/偏远）
                track = fit_centerline(mask)

        track_ok = track is not None
        if mask is not None:
            self._last_mask_px = int(cv.countNonZero(mask))
            # mask 太小（线尾残片）时拒绝拟合结果，进丢线流程。
            # 残片拟合出的 lat/κ 不可信，会让车在线尾被带偏（lat 单向漂移、κ 飙升）。
            # 注意：此门槛只在 FOLLOW 且线已建立后生效。TURN 捕获后线在顶部远处，
            # mask 天然小（1000~2000），若立即启用门槛会造成 FOLLOW↔BLOCKED 振荡。
            if (track_ok and self._mode == Mode.FOLLOW and self._follow_acquired
                    and self._last_mask_px < _I.min_mask_px):
                track_ok = False
                track = None

        fwd_cmd = 0.0
        yaw_cmd = 0.0

        if self.running:
            # 超声波必选：掉线（连接断或数据超时）→ 安全停车 PAUSE
            if self.us is not None and not self.us.alive:
                self.link.stop()
                if self._mode != Mode.PAUSE:
                    status = self._begin_pause("超声波掉线")
                    self._last_status = status
                else:
                    status = "PAUSE: 等超声波重连"
                    self._last_status = status
                self._apply_drive(self._mode, track, track_ok, frame.shape[1])
                fwd_cmd = self._steer_dbg.fwd
                yaw_cmd = self._steer_dbg.yaw_out
            else:
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
        hold_elapsed = self._hold_elapsed()
        osd_lines = [
            f"{self._mode.value} | {status}",
            f"run={self.running} side={side_s} turn_dir={turn_side_s} "
            f"streak={self._ready_streak}/{_I.ready_confirm} "
            f"t_turn={turn_elapsed:.1f}s hold={hold_elapsed:.1f}s",
            f"odom {self._odom_short()} last={self.odom.last_segment_total:.3f}"
            if self.odom is not None
            else "odom off",
            self._us_osd(),
            f"fwd={fwd_cmd:+.3f} scale={dbg.fwd_scale:.2f} yaw={yaw_cmd:+.3f} "
            f"(demand={dbg.yaw_demand:+.3f} raw={dbg.yaw_raw:+.3f} "
            f"sat={'Y' if dbg.saturated else 'N'})",
            self._velo_fb_short(fwd_cmd, yaw_cmd),
            f"lat={lat:+.1f} head={head:+.1f} |lat|={self._last_lat_abs:.1f} "
            f"warm={self._good_streak}/{_I.track_warmup} "
            f"lost={self._lost_streak}/{_I.lost_frames} {self._hold_note}",
            f"P={dbg.p_term:+.4f} D={dbg.d_term:+.4f} A={dbg.a_term:+.4f} FF={dbg.ff_term:+.4f} "
            f"dt={dbg.dt:.3f}s lat_f={dbg.lateral:+.1f} d_lat={dbg.d_lateral:+.0f}",
            f"near=({x_near:.0f},{y_near:.0f}) far=({x_far:.0f},{y_far:.0f}) cx={center_x:.0f}",
            f"kap={(track.curvature if track is not None else 0):+.5f}/px "
            f"px/m={(track.px_per_m if track is not None else 0):.0f} "
            f"kf={self._kappa_filt:+.2f}/m cw={self._curve_w:.2f}",
            f"geom y1={geom_y1:.0f} bw={geom_bw:.0f} bh={geom_bh:.0f}",
            f"mask_px={self._last_mask_px} track_ok={track_ok} scene={scene is not None} "
            f"acq={self._follow_acquired}",
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

        if self._video_writer is not None:
            self._video_writer.write(vis)

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
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self.capture.release()
        cv.destroyAllWindows()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO 巡线 v2 (停车等待/反光抑制)")
    parser.add_argument(
        "--params",
        default=str(DEFAULT_PARAM_FILE),
        help=f"参数文件(JSON)，默认 {DEFAULT_PARAM_FILE}",
    )
    parser.add_argument(
        "--no-params",
        action="store_true",
        help="不加载外部参数文件，使用代码内默认值",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"主控串口 init/运动 + odom 反馈（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="打开摄像头自动曝光/白平衡（过曝时可试；默认仍是手动锁定）",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help=f"手动曝光值（默认 {DEFAULT_EXPOSURE}）",
    )
    parser.add_argument("--gain", type=float, default=None, help=f"增益（默认 {DEFAULT_GAIN}）")
    parser.add_argument("--gamma", type=float, default=None, help=f"伽马（默认 {DEFAULT_GAMMA}，拉暗部提亮）")
    parser.add_argument("--brightness", type=float, default=None, help=f"亮度（默认 {DEFAULT_BRIGHTNESS}）")
    parser.add_argument("--contrast", type=float, default=None, help=f"对比度（默认 {DEFAULT_CONTRAST}）")
    parser.add_argument("--saturation", type=float, default=None, help=f"饱和度（默认 {DEFAULT_SATURATION}）")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.13)
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument("--log", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-odom", action="store_true", help="禁用里程计/障碍判断")
    parser.add_argument(
        "--odom-file",
        default=str(DEFAULT_SEGMENT_FILE),
        help="每段结束追加写入的 JSONL 文件（默认 data/odom_segments.jsonl）",
    )
    parser.add_argument(
        "--us-port",
        default=DEFAULT_US_PORT,
        help=f"超声波串口设备（默认 {DEFAULT_US_PORT}）",
    )
    parser.add_argument(
        "--us-baud",
        type=int,
        default=DEFAULT_US_BAUD,
        help=f"超声波波特率（默认 {DEFAULT_US_BAUD}）",
    )
    parser.add_argument(
        "--no-us",
        action="store_true",
        help="禁用超声波避障（默认必选：未连接会报错退出）",
    )
    parser.add_argument(
        "--us-obstacle",
        type=float,
        default=None,
        help="障碍触发距离 cm（优先覆盖参数文件）",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="录制巡线 OSD 画面到 videos/（与日志同名 .mp4）",
    )
    args = parser.parse_args(argv)

    # 参数加载优先级：
    # 1) 代码默认值（FollowCfg/_Internals）
    # 2) --params 文件覆盖（若存在）
    # 3) 命令行单项覆盖（如 --us-obstacle）
    global CFG, _I
    if not args.no_params:
        param_path = Path(args.params)
        if param_path.exists():
            try:
                cfg_new, internals_new = load_param_file(param_path)
            except Exception as exc:
                # 配置可读但内容非法时直接失败，避免带未知参数跑车。
                print(f"参数文件加载失败: {param_path} ({exc})", file=sys.stderr)
                return 1
            CFG = cfg_new
            _I = internals_new
        else:
            # 参数文件是“可选增强”：不存在时回落到代码默认值，避免首次运行失败。
            print(f"参数文件不存在，使用默认参数: {param_path}", flush=True)

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
    us: UltrasonicSensor | None = None
    try:
        link.open()
        if not args.no_odom and link.serial is not None:
            odom_path = Path(args.odom_file)
            odom = OdomEstimator(link.serial, segment_file=odom_path)
            # 不在后台线程读 odom：与超声波（USB1）并发访问 CH341 会卡死解释器。
            # 改为主循环 on_timer 里同步 poll_serial()。
            _log_info(f"odom 段文件: {odom_path.resolve()}")
            if odom.history:
                _log_info(
                    f"odom 已加载历史 {len(odom.history)} 段 "
                    f"exp={odom.expected_length:.3f} hist={odom.history}"
                )
        # 主控 init 尽早发送（与 GitHub 原版一致），不要等超声波就绪
        if not args.no_setup:
            link.setup()
            _log_info(f"主控已初始化: {args.port} @ {args.baud}")
        # 超声波：必选。--no-us 时跳过；否则必须连上并收到数据才放行
        if not args.no_us:
            us = UltrasonicSensor(args.us_port, baud=args.us_baud)
            us.start()
            _log_info(f"超声波: {args.us_port} @ {args.us_baud}，等待数据…")
            if not _wait_us_alive(us, timeout=5.0):
                raise RuntimeError(
                    f"超声波 {args.us_port} 5 秒内无数据：请检查接线/供电/端口。"
                    f" 用 --no-us 可跳过。"
                )
            _log_info(f"超声波就绪: {us.stats}")
        cam_settings = CameraSettings(
            lock=not args.auto_exposure,
            exposure=args.exposure if args.exposure is not None else DEFAULT_EXPOSURE,
            gain=args.gain if args.gain is not None else DEFAULT_GAIN,
            gamma=args.gamma if args.gamma is not None else DEFAULT_GAMMA,
            brightness=args.brightness if args.brightness is not None else DEFAULT_BRIGHTNESS,
            contrast=args.contrast if args.contrast is not None else DEFAULT_CONTRAST,
            saturation=args.saturation if args.saturation is not None else DEFAULT_SATURATION,
        )
        if args.auto_exposure:
            _log_info("摄像头: 自动曝光/白平衡（gamma/亮度等仍生效）")
        else:
            _log_info(f"摄像头: exp={cam_settings.exposure} gain={cam_settings.gain} "
                      f"gamma={cam_settings.gamma} bright={cam_settings.brightness}")

        video_writer = None
        if args.record:
            video_dir = Path("videos")
            video_dir.mkdir(exist_ok=True)
            video_path = video_dir / (log_path.stem + ".mp4")
            # OSD 是 640×480 BGR；用 mp4v 编码，帧率取实际控制周期约 15fps（回放近似）
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            video_writer = cv.VideoWriter(
                str(video_path), fourcc, 15.0, (640, 480)
            )
            if not video_writer.isOpened():
                _log_info(f"视频写入打开失败，跳过录制: {video_path}")
                video_writer = None
            else:
                _log_info(f"录像: {video_path}")

        follower = YoloLineFollower(
            link, model, conf=args.conf, device="0" if use_cuda else "cpu",
            imgsz=640, half=use_cuda, camera=camera, auto=auto, odom=odom, us=us,
            cam_settings=cam_settings,
            video_writer=video_writer,
        )
        if args.us_obstacle is not None:
            follower._us_obstacle_cm = args.us_obstacle
            _log_info(f"超声波障碍阈值已覆盖: {args.us_obstacle}cm")
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
    except RuntimeError as exc:
        _log_info(f"错误: {exc}")
        if not echo_console:
            print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if us is not None:
            us.stop()
        if odom is not None:
            odom.stop()
        if not args.no_setup:
            link.teardown()
        link.close()
        if follower is not None:
            follower.close()
    return 0


def _wait_us_alive(us: UltrasonicSensor, *, timeout: float) -> bool:
    """等待超声波收到第一帧有效数据；超时返回 False。"""
    import time as _time
    t0 = _time.perf_counter()
    while _time.perf_counter() - t0 < timeout:
        if us.alive:
            return True
        _time.sleep(0.1)
    return us.alive


if __name__ == "__main__":
    raise SystemExit(main())
 
