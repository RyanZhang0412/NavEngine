"""Training control frame builders (dx_train_tc.c)."""

from __future__ import annotations

import json
import math
from enum import IntEnum
from typing import Any, Callable

from ui_ctrl.constants import CmdCtrl
from ui_ctrl.protocol import build_ctrl, verify_frame

# dx_ctrl_model.h
class TrainingProgram(IntEnum):
    HOME = 0
    FREEDOM = 1
    RESISTANT = 2
    ELASCITY = 3
    PASSIVE_TRACK = 4
    LIMIT_TRACK = 5
    ASSISTANT_DELAY = 6
    ADAPATATION_TRACK = 7
    PASSIVE_STRETCH = 8
    ISOMETRIC = 9
    CYCLE_TRACK = 10
    ISOKINETIC = 11
    ISOKINETIC_TRACK = 12
    CUSTOM = 13
    CUSTOM_TRACK = 14


class JointType(IntEnum):
    WRIST_FLEXION_EXTENSION = 0
    WRIST_RADIAL_ULNAR_DEVIATION = 1
    FOREARM_PRONATION_SUPINATION = 2
    ENLBOW_FLEXION_EXTENSION = 3
    UPPER_EXTREMITY = 4
    SHOULDER_ROTATION = 5
    FINGER_GRASP = 6


class AffectSide(IntEnum):
    LEFT = 0
    RIGHT = 1


class TrainingCtrl(IntEnum):
    """em_training_ctrl_t — payload sent as ASCII digit (dx_train_tc.c)."""

    START = 1
    STOP = 2
    PAUSE = 3
    CONTINUE = 4
    RESTART = 5


ROBOT_KEY = "Ranger"


def _json_payload(obj: dict[str, Any], *, with_nul: bool = True) -> bytes:
    """C 端 cJSON_Print + strlen+1，保留末尾 NUL。"""
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if with_nul:
        data += b"\x00"
    return data


def _standard_data() -> list[float]:
    deg = math.pi / 180.0
    return [
        90.0 * deg,
        1.0,
        3.0,
        1.0,
        -90.0 * deg,
        1.0,
        3.0,
        1.0,
    ]


def _dynaxis_robot_init() -> dict[str, Any]:
    """test_training_init robot_Obj fields."""
    return {
        # "duration": 2,
        # "joint_type": int(JointType.FINGER_GRASP),
        # "affect_side": int(AffectSide.RIGHT),
        # "position_max": 80.0,
        # "position_min": 0.0,
        # "extra_max": 5.0,
        # "extra_min": -5.0,
        # "position_target": 40.0,
        # "velocity_target": 10.0,
        # "velocity_max": 40.0,
        # "torque": 1.0,
        # "torque_percent": 100.0,
        # "time_max": 1,
        # "time_min": 1,
        # "time_delay": 1,
        # "time_limit": 1,
        # "spasm_level": 1,
        # "action_max": 1,
        # "action_min": 1,
        # "contraction": 1,
        # "vibration": 0,
        # "standard_data": _standard_data(),
    }


def _dynaxis_robot_change(
    *,
    position_target: float = 20.0,
    joint_type: int = 0,
    vibration: int = 0,
    param: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """test_training_change robot_Obj fields (angles in radians)."""
    deg = math.pi / 180.0
    return {
        "velo_fwd": 0.0,
        "velo_yaw": 0.5,
        # "duration": 1,
        # "joint_type": joint_type,
        # "position_max": 85.0 * deg,
        # "position_min": -85.0 * deg,
        # "extra_max": 10.0 * deg,
        # "extra_min": -10.0 * deg,
        # "position_target": position_target * deg,
        # "velocity_target": 90.0 * deg,
        # "velocity_max": 90.0 * deg,
        # "torque": 10.0,
        # "torque_percent": 100.0,
        # "time_max": 1,
        # "time_min": 1,
        # "time_delay": 1,
        # "time_limit": 1,
        # "spasm_level": 1,
        # "action_max": 1,
        # "action_min": 1,
        # "contraction": 1,
        # "vibration": vibration,
        # "param_0": param[0],
        # "param_1": param[1],
        # "param_2": param[2],
        # "param_3": param[3],
        # "standard_data": _standard_data(),
    }


def _training_set_body(
    robot: dict[str, Any],
    *,
    training_program: int = 1,
    training_set: int = 0,
    autorun: int = 1,
    msg_id: int = 0,
) -> dict[str, Any]:
    return {
        "id": msg_id,
        "autorun": autorun,
        "training_set": training_set,
        "training_program": training_program,
        ROBOT_KEY: robot,
    }


def _build_ctrl_verified(cmd: CmdCtrl, payload: bytes) -> bytes:
    frame = build_ctrl(cmd, payload)
    if not verify_frame(frame):
        raise RuntimeError("built frame failed verification")
    return frame


def build_training_init(
    *,
    training_program: int = 1,
    robot: dict[str, Any] | None = None,
) -> bytes:
    """test_training_init → CTRL TRAINING_SET."""
    body = _training_set_body(
        robot or _dynaxis_robot_init(),
        training_program=training_program,
    )
    return _build_ctrl_verified(CmdCtrl.TRAINING_SET, _json_payload(body))


def build_training_change(
    *,
    position_target: float = 20.0,
    joint_type: int = 0,
    vibration: int = 0,
    param: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> bytes:
    """test_training_change → CTRL TRAINING_CHANGE."""
    body = _training_set_body(
        _dynaxis_robot_change(
            position_target=position_target,
            joint_type=joint_type,
            vibration=vibration,
            param=param,
        ),
        training_program=int(TrainingProgram.ISOKINETIC),
    )
    return _build_ctrl_verified(CmdCtrl.TRAINING_CHANGE, _json_payload(body))


def build_training_ctrl(ctrl: TrainingCtrl) -> bytes:
    """test_training_start/stop/pause → CTRL TRAINING_CTRL, ASCII digit payload."""
    return _build_ctrl_verified(CmdCtrl.TRAINING_CTRL, str(int(ctrl)).encode("ascii"))


def build_training_start() -> bytes:
    return build_training_ctrl(TrainingCtrl.START)


def build_training_stop() -> bytes:
    return build_training_ctrl(TrainingCtrl.STOP)


def build_training_pause() -> bytes:
    return build_training_ctrl(TrainingCtrl.PAUSE)


def build_training_clear() -> bytes:
    """test_training_clear → CTRL TRAINING_CLEAR, empty payload."""
    return _build_ctrl_verified(CmdCtrl.TRAINING_CLEAR, b"")


SEND_ACTIONS: dict[str, Callable[[], bytes]] = {
    "init": build_training_init,
    "start": build_training_start,
    "pause": build_training_pause,
    "stop": build_training_stop,
    "clear": build_training_clear,
    "change": build_training_change,
}


def build_send_frame(action: str) -> bytes:
    """Build frame for a named send action."""
    key = action.lower()
    if key not in SEND_ACTIONS:
        raise ValueError(f"unknown action: {action!r}")
    return SEND_ACTIONS[key]()
