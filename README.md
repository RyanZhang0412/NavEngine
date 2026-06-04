# NavEngine UI 串口协议

Python 实现 RobotMasterController 中 `ui_ctrl_protocol` / `ui_ctrl_port` 的串口通信协议层：帧组帧、CRC16-XMODEM 校验、帧解析与流式搜帧。

## 帧格式

| 偏移 | 字段 | 说明 |
|------|------|------|
| 0-1 | HEAD | `0x3A 0x3A` |
| 2 | ID | 消息 ID |
| 3 | CMD_TYPE | CTRL=0, INQUIRY=1, DATA=2, ACK=3 |
| 4 | CMD | 子命令 |
| 5-8 | DATA_LEN | `uint32` 小端 |
| 9.. | DATA | 载荷 |
| ... | CRC16 | `uint16` 小端，覆盖 ID 至 DATA 末尾 |
| 末 | TAIL | `0x2E` |

## 快速使用

```python
from ui_ctrl.protocol import build_ack, build_ctrl, parse_frame, verify_frame
from ui_ctrl.constants import CmdAck, CmdCtrl
from ui_ctrl.stream_parser import StreamParser

# 组帧
frame = build_ack(0x05, CmdAck.NORMAL)
assert verify_frame(frame)

msg = parse_frame(frame)
print(msg.to_dict())

# 流式解析
frames = []
parser = StreamParser(on_frame=lambda m: frames.append(m))
parser.feed(frame)
```

## 运行测试

```bash
cd NavEngine
pip install -r requirements.txt
PYTHONPATH=src pytest tests/ -v
```

## 参考

- `RobotMasterController/src/ctrlport/ui_ctrl_protocol.c`
- `RobotMasterController/src/ctrlport/ui_ctrl_port.c`
