"""Protocol constants and command enumerations (from ui_ctrl_protocol.h)."""

from enum import IntEnum

HEAD = b"\x3a\x3a"
TAIL = 0x2E

HEAD_CHAR_NUM = 2
ID_CHAR_NUM = 1
CHECKSUM_CHAR_NUM = 2
TAIL_CHAR_NUM = 1
CMD_TYPE_NUM = 1
CMD_NUM = 1
DATA_LEN_NUM = 4  # DEF_UI_CTRL_PROTOCOL_DATA_LEN_NUM

PROTOCOL_CHAR_NUM = (
    HEAD_CHAR_NUM
    + ID_CHAR_NUM
    + CHECKSUM_CHAR_NUM
    + TAIL_CHAR_NUM
    + CMD_TYPE_NUM
    + CMD_NUM
    + DATA_LEN_NUM
)

DATA_LEN_OFFSET = HEAD_CHAR_NUM + ID_CHAR_NUM + CMD_TYPE_NUM + CMD_NUM
DATA_OFFSET = DATA_LEN_OFFSET + DATA_LEN_NUM

COMPLETE_HEAD_U16 = (HEAD[1] << 8) | HEAD[0]  # 0x3A3A, little-endian uint16


class CmdType(IntEnum):
    """em_ui_ctrl_protocol_cmd_type_t"""

    CTRL = 0x00
    INQUIRY = 0x01
    DATA = 0x02
    ACK = 0x03


class CmdData(IntEnum):
    """em_ui_ctrl_protocol_cmd_data_t"""

    UNDEFINED = 0x00
    SYSTEM_INFO = 0x01
    SYSTEM_STATE = 0x02
    TRAINING_STATE = 0x03
    SYSTEM_ERROR = 0x04
    TRAINING_ERROR = 0x05
    EMERG_STATE = 0x06
    RUNNING_INFO = 0x07
    SYSTEM_DATE = 0x08
    BUTTON_INFO = 0x09
    FIRMWARE_UPDATE = 0x0A
    FIREWARE_VERSION = 0x0B  # C header: N_UI_CP_CMD_DATA_FIREWARE_VERSION
    FIRMWARE_INFO = 0x0C
    FIRMWARE_OVER = 0x0D
    FIRMWARE_PROGRESS = 0x0E
    JOINT_TYPE = 0x0F
    TRAINING_DATA = 0x10


class CmdCtrl(IntEnum):
    """em_ui_ctrl_protocol_cmd_ctrl_t"""

    UNDEFINED = 0x00
    POWER_ON = 0x01
    REQUEST_POWER_OFF = 0x02
    POWER_OFF = 0x03
    REBOOT = 0x04
    CALIBRATE = 0x05
    TRAINING_SET = 0x06
    TRAINING_CTRL = 0x07
    TRAINING_CLEAR = 0x08
    TRAINING_CHANGE = 0x09
    POSE = 0x0A
    SYSSET_BASIC = 0x0B
    SYSSET_ROOT = 0x0C
    SYSTEM_CONFIG = 0x0D
    AUDIO = 0x0E
    SYSTEM_TIME = 0x0F
    FIRMWARE_OPERATE = 0x10
    REQ_SENSOR_INIT = 0x11
    RUNNING_INFO_CTRL = 0x12


class CmdInquiry(IntEnum):
    """em_ui_ctrl_protocol_cmd_inquiry_t"""

    UNDEFINED = 0x00
    SYSTEM_INFO = 0x01
    SYSTEM_STATE = 0x02
    TRAINING_STATE = 0x03
    SYSTEM_ERROR = 0x04
    TRAINING_ERROR = 0x05
    EMERG_STATE = 0x06
    SYSTEM_DATE = 0x07
    SYSSET_BASIC = 0x08
    SYSSET_ROOT = 0x09
    FIRMWARE_INFO = 0x0A
    FIRMWARE_SEND = 0x0B
    FIRMWARE_RESEND = 0x0C
    FIRMWARE_VERSION = 0x0D
    TRAINING_DATA = 0x0E


class CmdAck(IntEnum):
    """em_ui_ctrl_protocol_cmd_ack_t"""

    UNDEFINED = 0x00
    NORMAL = 0x01
    ERROR = 0x02


class EmergType(IntEnum):
    """em_ui_ctrl_protocol_emerg_state_t"""

    CLEAR = 0x00
    SOFTWARE = 0x01
    BTN = 0x02
    SPASM = 0x04
    VOICE = 0x08
    MAGNET = 0x10
    ERR = 0xFF


# Backward-compatible alias
EmergState = EmergType
