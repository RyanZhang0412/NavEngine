#!/usr/bin/env python3
"""超声波测距传感器驱动（后台线程读串口，线程安全）。

所属系统
    NavEngine 巡线小车。超声波模块用于在巡线（FOLLOW）时检测前方近距离障碍，
    触发 follow_line_yolo_2.py 的 BLOCKED 状态停车避让。

硬件/接线
    - 模块：CH340 USB 转串口的超声波测距模块（实测品牌 QinHeng CH340，1a86:7523）。
    - 默认接在 /dev/ttyCH341USB1，波特率 9600（与主控 USB0 独立，互不干扰）。
      （主控调试日志口），不要混淆；超声波是主动上报数据的那一个。

数据协议（实测）
    模块每帧主动发送一行 ASCII，以 \r\n 结尾：
        T:<温度>C D:<距离>cm
    例如：
        T:27.62C D:-1.00cm
        T:25.00C D:30.50cm

    距离 D 的语义（重要）：
        D = -1（或 <= NO_OBSTACLE_CM）：前方无回波 = 前方开阔无障碍。
            这是模块的合法返回值，不是错误，不要当哨兵值丢弃。
        D >= 0：检测到障碍，数值为前方障碍距离（单位 cm）。
        协议没有"查询命令"，模块按自身节奏（约 2Hz）持续上报。

对外接口（巡线主程序用法）
    sensor = UltrasonicSensor("/dev/ttyCH341USB1")
    sensor.start()                       # 启动后台读线程
    if sensor.alive:                     # 连接且数据新鲜
        d = sensor.distance_cm           # None=从未收到的帧；-1=前方开阔；>=0=障碍距离
        if sensor.has_obstacle(50):      # 50cm 内是否有障碍
            ...
    sensor.stop()                        # 退出时调用，关闭串口+线程

设计要点
    - 后台线程持续解析，主循环只读缓存值，零阻塞（YOLO 推理 + 控制循环不受影响）。
    - 串口异常/掉线自动重连（每 RECONNECT_SEC 秒尝试一次），不抛异常到主循环。
    - alive 同时要求"串口已连接"和"近期有有效帧"，用于判断是否可信。
    - connected 仅表示串口是否打开，用于日志/重连状态展示。
"""
from __future__ import annotations

import logging
import re
import threading
import time

import serial

_LOG = logging.getLogger("us_sensor")

# 单帧正则：T:<温度>C D:<距离>cm，温度和距离都允许负值（温度可能为负，距离 -1 表示无障碍）。
# 大小写不敏感，允许 T/D 与数字间有空格。
_FRAME_RE = re.compile(
    r"T\s*:\s*(-?\d+(?:\.\d+)?)\s*C\s+D\s*:\s*(-?\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
# 无障碍阈值：模块返回 D<=该值表示前方开阔（无回波）。D=-1 是典型值。
NO_OBSTACLE_CM = 0.0

DEFAULT_PORT = "/dev/ttyCH341USB1"   # 默认串口（超声波专用口）
DEFAULT_BAUD = 9600                  # 默认波特率
RECONNECT_SEC = 1.0                  # 串口打开失败后重试间隔（秒）
READ_TIMEOUT = 0.5                   # 单次串口 read 超时（秒），防止永久阻塞
READ_CHUNK = 128                     # 单次 read 最大字节数
STALE_SEC = 2.0                      # 超过该秒数没有收到任何有效帧，视为掉线


class UltrasonicSensor:
    """后台线程读超声波串口，线程安全暴露最近距离。

    线程模型：start() 启动一个 daemon 线程，持续 read 串口并按行解析，
    把结果写入受 _lock 保护的字段。主循环通过只读 property 取值，不会阻塞。
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        *,
        stale_sec: float = STALE_SEC,
    ) -> None:
        """初始化传感器（不打开串口，真正连接发生在 start() 之后）。

        参数：
            port: 串口设备路径，默认 /dev/ttyCH341USB1。
            baud: 波特率，默认 9600。
            stale_sec: alive 判定的数据新鲜窗口（秒），超过该秒数无帧视为掉线。
        """
        self._port = port
        self._baud = baud
        self._stale_sec = stale_sec
        self._ser: serial.Serial | None = None
        self._stop = threading.Event()           # 用于通知后台线程退出
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()            # 保护下面所有共享字段
        self._distance_cm: float | None = None   # 最近一帧距离（含 -1）；None=从未收到
        self._temperature_c: float | None = None
        self._last_good_ts: float = 0.0          # 最近一次有效帧的时间戳（perf_counter）
        self._frames = 0                         # 累计成功解析的帧数
        self._bad = 0                            # 累计无法解析的行数
        self._connected = False                  # 串口是否当前已打开

    @property
    def port(self) -> str:
        """配置的串口设备路径。"""
        return self._port

    @property
    def baud(self) -> int:
        """配置的波特率。"""
        return self._baud

    @property
    def connected(self) -> bool:
        """串口是否已打开。掉线（I/O 错误等）后变 False，自动重连成功后回 True。

        仅反映"串口能否打开/读取"，不代表数据是否新鲜；要判断数据可信用 alive。
        """
        return self._connected

    @property
    def alive(self) -> bool:
        """数据流是否可信：既要求串口已连接，又要求近期收到过有效帧。

        判定：connected 且 (now - last_good_ts) < stale_sec。
        巡线主循环据此判断传感器是否健康，掉线时进入安全停车。
        """
        if not self._connected:
            return False
        with self._lock:
            ts = self._last_good_ts
        if ts <= 0:
            return False
        return (time.perf_counter() - ts) < self._stale_sec

    @property
    def distance_cm(self) -> float | None:
        """最近一次模块返回的距离（cm），线程安全。

        返回值语义：
          - None：从未收到任何数据帧（刚启动或长期掉线）。
          - 负值（典型 -1.0）：前方无回波 = 前方开阔无障碍。
          - >=0：检测到障碍，数值为前方障碍距离 cm。

        注意：-1 是有效帧的返回值，不是错误；不要用它判断"是否有障碍"，
        请用 has_obstacle()。
        """
        with self._lock:
            return self._distance_cm

    def has_obstacle(self, threshold_cm: float) -> bool:
        """当前是否检测到 threshold_cm 以内的障碍。

        判定：距离存在 且 D>NO_OBSTACLE_CM（排除 -1 无回波）且 D<threshold。
        即 D=-1（前方开阔）或 D>=threshold（太远）都返回 False。
        """
        with self._lock:
            d = self._distance_cm
        return d is not None and d > NO_OBSTACLE_CM and d < threshold_cm

    @property
    def temperature_c(self) -> float | None:
        """最近一次模块返回的温度（℃），无数据时为 None。"""
        with self._lock:
            return self._temperature_c

    @property
    def stats(self) -> str:
        """调试用：累计帧数/坏帧数/连接/存活状态汇总字符串。

        注意：先取快照再拼字符串，不能在持有 _lock 时调用 self.alive
        （alive 内部也要拿 _lock，Lock 不可重入会自死锁）。
        """
        with self._lock:
            frames = self._frames
            bad = self._bad
        return (
            f"frames={frames} bad={bad} "
            f"connected={self._connected} alive={self.alive}"
        )

    def start(self) -> None:
        """启动后台读线程（幂等：重复调用不会创建多个线程）。"""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="us-sensor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止后台线程并关闭串口（最多等待 2 秒让线程退出）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_serial()

    def _close_serial(self) -> None:
        """关闭串口并把连接状态置 False；忽略关闭时的异常。"""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._connected = False

    def _run(self) -> None:
        """后台线程主循环：保持串口打开 + 持续读取并按行解析。

        串口未打开时调用 _open_blocking() 重连；读异常时关闭后下轮重连。
        用 bytearray 缓冲跨 read 的不完整行，按 \\n 切分交给 _handle_line。
        """
        buf = bytearray()
        while not self._stop.is_set():
            if self._ser is None:
                self._open_blocking()
                if self._ser is None:
                    continue
                buf.clear()

            try:
                chunk = self._ser.read(READ_CHUNK)
            except Exception as e:
                _LOG.warning("US 读失败，重连: %s", e)
                self._close_serial()
                continue

            if not chunk:
                continue
            buf.extend(chunk)
            # 按换行符切分完整行（兼容 \n 与 \r\n）
            while True:
                idx = buf.find(0x0A)
                if idx < 0:
                    break
                line = bytes(buf[: idx + 1]).decode("ascii", errors="replace")
                del buf[: idx + 1]
                self._handle_line(line.strip().rstrip("\r"))

    def _open_blocking(self) -> None:
        """尝试打开串口；失败则等待 RECONNECT_SEC 后返回（由外层重试）。

        成功时把 self._ser/_connected 置位并打日志；失败时清空连接状态。
        不抛异常，确保后台线程不会因单次打开失败而退出。
        """
        try:
            ser = serial.Serial(self._port, baudrate=self._baud, timeout=READ_TIMEOUT)
            ser.reset_input_buffer()
        except Exception as e:
            if not self._connected:
                # 仅在状态翻转时打日志，避免刷屏
                pass
            self._connected = False
            self._stop.wait(RECONNECT_SEC)
            return
        self._ser = ser
        self._connected = True
        _LOG.info("US 已连接 %s @ %d", self._port, self._baud)

    def _handle_line(self, line: str) -> None:
        """解析一行原始文本，更新共享字段。

        匹配成功：累加 frames，更新 temperature/distance/last_good_ts。
        匹配失败或数值非法：累加 bad 计数，不影响现有距离值。
        全程在 _lock 内修改共享字段。
        """
        if not line:
            return
        m = _FRAME_RE.search(line)
        if not m:
            with self._lock:
                self._bad += 1
            return
        try:
            t_c = float(m.group(1))
            d_cm = float(m.group(2))
        except ValueError:
            with self._lock:
                self._bad += 1
            return

        now = time.perf_counter()
        with self._lock:
            self._frames += 1
            self._temperature_c = t_c
            # D=-1（无回波）也是合法帧，表示前方开阔；始终更新距离与时间戳。
            self._distance_cm = d_cm
            self._last_good_ts = now
