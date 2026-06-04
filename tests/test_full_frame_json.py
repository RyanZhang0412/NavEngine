"""Full-frame tests with JSON payload (TX/RX round-trip)."""

import json
import struct

import pytest

from ui_ctrl.constants import (
    DATA_OFFSET,
    DATA_LEN_OFFSET,
    HEAD,
    PROTOCOL_CHAR_NUM,
    TAIL,
    CmdCtrl,
    CmdData,
    CmdInquiry,
    CmdType,
)
from ui_ctrl.protocol import build_ctrl, build_data, build_frame, parse_frame, verify_frame
from ui_ctrl.stream_parser import StreamParser

# POSE control JSON fields (ui_ctrl_port_helper.c parse_pose_ctrl_data)
POSE_JSON = {
    "jnt_qty": 1,
    "jnt_name": "height",
    "pos": 0.5,
    "speed": 0.1,
    "ctrl_cmd": 1,
}

# Firmware version DATA JSON (send_data_firmware_version)
FIRMWARE_JSON = {
    "name": "RobotMasterController",
    "currentVersion": "V1.4.0",
    "lastVersion": "V1.3.0",
    "factoryVersion": "V1.0.0",
}


@pytest.fixture
def pose_json_bytes() -> bytes:
    return json.dumps(POSE_JSON, separators=(",", ":")).encode("utf-8")


@pytest.fixture
def pose_ctrl_frame(pose_json_bytes: bytes) -> bytes:
    return build_ctrl(CmdCtrl.POSE, pose_json_bytes)


@pytest.fixture
def firmware_json_bytes() -> bytes:
    # C 端 send_data_packets 使用 strlen+1，保留末尾 NUL
    text = json.dumps(FIRMWARE_JSON, separators=(",", ":"))
    return text.encode("utf-8") + b"\x00"


@pytest.fixture
def firmware_data_frame(firmware_json_bytes: bytes) -> bytes:
    return build_data(CmdData.FIREWARE_VERSION, firmware_json_bytes)


def test_full_frame_ctrl_pose_json_tx_rx(pose_ctrl_frame, pose_json_bytes):
    """完整 CTRL+POSE 帧：组帧 -> 校验 -> 解析 -> JSON 字段一致。"""
    assert verify_frame(pose_ctrl_frame)

    expected_len = PROTOCOL_CHAR_NUM + len(pose_json_bytes)
    assert len(pose_ctrl_frame) == expected_len

    assert pose_ctrl_frame[0:2] == HEAD
    assert pose_ctrl_frame[2] == 0x00  # msg_id
    assert pose_ctrl_frame[3] == CmdType.CTRL
    assert pose_ctrl_frame[4] == CmdCtrl.POSE
    assert struct.unpack_from("<I", pose_ctrl_frame, DATA_LEN_OFFSET)[0] == len(
        pose_json_bytes
    )
    assert pose_ctrl_frame[-1] == TAIL

    msg = parse_frame(pose_ctrl_frame)
    assert msg.cmd_type == CmdType.CTRL
    assert msg.cmd == CmdCtrl.POSE
    assert msg.payload == pose_json_bytes

    decoded = json.loads(msg.payload.decode("utf-8"))
    assert decoded == POSE_JSON
    assert decoded["jnt_name"] == "height"
    assert decoded["pos"] == pytest.approx(0.5)
    assert decoded["ctrl_cmd"] == 1


def test_full_frame_data_firmware_json_tx_rx(firmware_data_frame, firmware_json_bytes):
    """完整 DATA+FIREWARE_VERSION 帧：JSON 收发往返。"""
    assert verify_frame(firmware_data_frame)

    msg = parse_frame(firmware_data_frame)
    assert msg.cmd_type == CmdType.DATA
    assert msg.cmd == CmdData.FIREWARE_VERSION
    assert msg.payload == firmware_json_bytes

    # 去掉 C 端附加的 NUL 再解析 JSON
    decoded = json.loads(msg.payload.rstrip(b"\x00").decode("utf-8"))
    assert decoded == FIRMWARE_JSON
    assert decoded["name"] == "RobotMasterController"
    assert decoded["currentVersion"] == "V1.4.0"


def test_full_frame_json_inquiry_response_round_trip():
    """模拟 INQUIRY 请求 + DATA JSON 应答的完整收发链。"""
    inquiry = build_frame(0x01, CmdType.INQUIRY, CmdInquiry.SYSTEM_INFO, b"")
    assert verify_frame(inquiry)

    response_body = json.dumps(
        {"systemState": "idle", "trainingId": 42},
        separators=(",", ":"),
    ).encode("utf-8")
    response = build_data(CmdData.SYSTEM_INFO, response_body, msg_id=0x01)

    frames = []
    parser = StreamParser(on_frame=frames.append)
    consumed = parser.feed(inquiry + response)
    assert consumed == len(inquiry) + len(response)
    assert len(frames) == 2

    req = frames[0]
    assert req.cmd_type == CmdType.INQUIRY
    assert req.payload == b""

    resp = frames[1]
    assert resp.cmd_type == CmdType.DATA
    assert resp.msg_id == 0x01
    assert json.loads(resp.payload.decode("utf-8"))["trainingId"] == 42


def test_full_frame_json_rebuild_from_parsed_fields(pose_ctrl_frame):
    """从解析字段重新组帧，字节级一致。"""
    msg = parse_frame(pose_ctrl_frame)
    rebuilt = build_ctrl(msg.cmd, msg.payload, msg_id=msg.msg_id)
    assert rebuilt == pose_ctrl_frame


def test_full_frame_json_payload_offset(pose_ctrl_frame, pose_json_bytes):
    """确认 JSON 载荷在帧中的偏移与长度。"""
    data_len = struct.unpack_from("<I", pose_ctrl_frame, DATA_LEN_OFFSET)[0]
    payload_in_frame = pose_ctrl_frame[DATA_OFFSET : DATA_OFFSET + data_len]
    assert payload_in_frame == pose_json_bytes
    assert json.loads(payload_in_frame.decode("utf-8"))["jnt_qty"] == 1
