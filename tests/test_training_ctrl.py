"""Training control frame tests (dx_train_tc.c)."""

import json
import struct

import pytest

from ui_ctrl.constants import CmdCtrl, CmdType, DATA_LEN_OFFSET, PROTOCOL_CHAR_NUM
from ui_ctrl.protocol import parse_frame, verify_frame
from ui_ctrl.training_ctrl import (
    TrainingCtrl,
    build_training_change,
    build_training_clear,
    build_training_init,
    build_training_pause,
    build_training_start,
    build_training_stop,
)


def test_training_init_frame():
    frame = build_training_init()
    assert verify_frame(frame)
    msg = parse_frame(frame)
    assert msg.cmd_type == CmdType.CTRL
    assert msg.cmd == CmdCtrl.TRAINING_SET
    assert msg.payload.endswith(b"\x00")
    body = json.loads(msg.payload.rstrip(b"\x00").decode())
    assert body["training_program"] == 1
    assert "Ranger" in body


def test_training_ctrl_frames():
    for builder, ctrl in (
        (build_training_start, TrainingCtrl.START),
        (build_training_stop, TrainingCtrl.STOP),
        (build_training_pause, TrainingCtrl.PAUSE),
    ):
        frame = builder()
        assert verify_frame(frame)
        msg = parse_frame(frame)
        assert msg.cmd == CmdCtrl.TRAINING_CTRL
        assert msg.payload == str(int(ctrl)).encode("ascii")
        assert struct.unpack_from("<I", frame, DATA_LEN_OFFSET)[0] == 1


def test_training_clear_frame():
    frame = build_training_clear()
    assert verify_frame(frame)
    msg = parse_frame(frame)
    assert msg.cmd == CmdCtrl.TRAINING_CLEAR
    assert msg.payload == b""
    assert len(frame) == PROTOCOL_CHAR_NUM


def test_training_change_frame():
    frame = build_training_change(position_target=30.0)
    assert verify_frame(frame)
    msg = parse_frame(frame)
    assert msg.cmd == CmdCtrl.TRAINING_CHANGE
    body = json.loads(msg.payload.rstrip(b"\x00").decode())
    assert body["training_program"] == 11
    assert body["Ranger"]["velo_fwd"] == 0.0
    assert body["Ranger"]["velo_yaw"] == 0.5
