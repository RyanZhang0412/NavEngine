import pytest

from ui_ctrl.constants import CmdAck, CmdCtrl, CmdType
from ui_ctrl.protocol import build_ack, build_ctrl


@pytest.fixture
def ack_frame() -> bytes:
    return build_ack(0x05, CmdAck.NORMAL)


@pytest.fixture
def ctrl_frame() -> bytes:
    return build_ctrl(CmdCtrl.POWER_ON, b"\x01\x02")


@pytest.fixture
def two_ack_frames(ack_frame: bytes) -> bytes:
    return ack_frame + ack_frame
