"""OdomEstimator JSON parsing (no serial)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

from odom_estimator import (
    Segment,
    _as_float,
    _find_velo,
    append_segment_record,
    build_zero_velocity_change,
    load_segment_records,
    OdomEstimator,
)
from ui_ctrl.constants import CmdCtrl, CmdType
from ui_ctrl.protocol import parse_frame, verify_frame


def test_as_float_parses_strings():
    assert _as_float("0.0675") == 0.0675
    assert _as_float("-0.0308") == -0.0308
    assert _as_float(0.5) == 0.5
    assert _as_float("bad") is None


def test_find_velo_nested_ranger_strings():
    payload = {"Ranger": {"velo_fwd": "0.0897", "velo_yaw": "0.5131"}}
    fwd, yaw = _find_velo(payload)
    assert fwd == 0.0897
    assert yaw == 0.5131


def test_zero_velocity_change_frame():
    frame = build_zero_velocity_change()
    assert verify_frame(frame)
    msg = parse_frame(frame)
    assert msg.cmd_type == CmdType.CTRL
    assert msg.cmd == CmdCtrl.TRAINING_CHANGE
    body = json.loads(msg.payload.rstrip(b"\x00").decode())
    assert body["Ranger"]["velo_fwd"] == 0.0
    assert body["Ranger"]["velo_yaw"] == 0.0


def test_append_segment_record_jsonl(tmp_path):
    path = tmp_path / "seg.jsonl"
    seg = Segment(total=0.42, left=0.4, right=0.44, t0=1.0, t1=3.5)
    append_segment_record(path, seg, segment_index=1, reason="test")
    rows = load_segment_records(path)
    assert len(rows) == 1
    assert rows[0]["total"] == 0.42
    assert rows[0]["segment"] == 1


def test_progress_pct_from_history():
    odom = OdomEstimator(object(), segment_file=None)
    odom.begin_segment()
    with odom._lock:
        odom._history = [0.40, 0.40]
        odom._cur.total = 0.20
    assert odom.progress_pct() == 50.0
    assert odom.progress_label() == "50%"


def test_should_turn_requires_enough_mileage():
    odom = OdomEstimator(object(), segment_file=None)
    odom.begin_segment()
    assert not odom.should_turn()
    with odom._lock:
        odom._cur.total = 0.02
    assert not odom.should_turn()
    with odom._lock:
        odom._cur.total = 0.20
    assert odom.should_turn()


def test_end_segment_records_last_total(tmp_path):
    path = tmp_path / "seg.jsonl"
    odom = OdomEstimator(object(), segment_file=path)
    odom.begin_segment()
    with odom._lock:
        odom._cur.total = 0.42
    seg = odom.end_segment(reason="unit")
    assert seg.total == 0.42
    assert odom.last_segment_total == 0.42
    assert odom.segment_count == 1
    assert not odom.segment_open
    assert odom.total == 0.0
    rows = load_segment_records(path)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unit"


def test_should_turn_uses_baseline_ratio():
    odom = OdomEstimator(object(), segment_file=None)
    odom.begin_segment()
    with odom._lock:
        odom._history = [0.50, 0.60]
        odom._cur.total = 0.30
    assert not odom.should_turn()
    with odom._lock:
        odom._cur.total = 0.33
    assert odom.should_turn()
    assert not odom.should_block()
