from ui_ctrl.crc16 import calc_crc16, calc_crc16_continue


def test_empty_crc():
    assert calc_crc16(b"") == 0x0000


def test_known_sequence():
    data = bytes([0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x05])
    assert calc_crc16(data) == 0xCD87


def test_continue_matches_single_pass():
    part1 = bytes([0x00, 0x03, 0x01])
    part2 = bytes([0x00, 0x00, 0x00, 0x00, 0x05])
    full = part1 + part2

    crc_full = calc_crc16(full)
    crc_split = calc_crc16_continue(part2, calc_crc16(part1))
    assert crc_split == crc_full
