from ui_ctrl.protocol import build_ack, build_ctrl, verify_frame
from ui_ctrl.constants import CmdAck, CmdCtrl
from ui_ctrl.stream_parser import StreamParser


def test_single_frame(ack_frame):
    frames = []
    parser = StreamParser(on_frame=frames.append)
    consumed = parser.feed(ack_frame)
    assert consumed == len(ack_frame)
    assert len(frames) == 1
    assert frames[0].cmd == CmdAck.NORMAL
    assert parser.buffer == b""


def test_concatenated_frames(two_ack_frames, ack_frame):
    frames = []
    parser = StreamParser(on_frame=frames.append)
    consumed = parser.feed(two_ack_frames)
    assert consumed == len(two_ack_frames)
    assert len(frames) == 2
    assert all(f.cmd == CmdAck.NORMAL for f in frames)


def test_split_frame(ack_frame):
    frames = []
    parser = StreamParser(on_frame=frames.append)
    mid = len(ack_frame) // 2
    # incomplete frame is kept in buffer; nothing consumed yet
    assert parser.feed(ack_frame[:mid]) == 0
    assert len(frames) == 0
    assert len(parser.buffer) == mid
    assert parser.feed(ack_frame[mid:]) == len(ack_frame)
    assert len(frames) == 1


def test_noise_prefix(ack_frame):
    noise = b"\x00\x01\xff"
    frames = []
    parser = StreamParser(on_frame=frames.append)
    consumed = parser.feed(noise + ack_frame)
    assert consumed == len(noise) + len(ack_frame)
    assert len(frames) == 1


def test_bad_crc_then_good_frame(ack_frame):
    bad = bytearray(ack_frame)
    bad[-3] ^= 0xFF
    assert not verify_frame(bytes(bad))

    frames = []
    parser = StreamParser(on_frame=frames.append)
    consumed = parser.feed(bytes(bad) + ack_frame)
    assert consumed == len(bad) + len(ack_frame)
    assert len(frames) == 1


def test_no_head_keeps_last_byte():
    parser = StreamParser()
    consumed = parser.feed(b"\x00\x01\x02\x03")
    assert consumed == 3
    assert parser.buffer == b"\x03"


def test_head_split_across_chunks(ack_frame):
    frames = []
    parser = StreamParser(on_frame=frames.append)
    assert parser.feed(ack_frame[:1]) == 0
    assert parser.buffer == ack_frame[:1]
    assert parser.feed(ack_frame[1:]) == len(ack_frame)
    assert len(frames) == 1


def test_reset(ack_frame):
    parser = StreamParser()
    parser.feed(ack_frame[:5])
    parser.reset()
    assert parser.buffer == b""
