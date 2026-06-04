"""NavEngine YOLO 分割工作流。"""

from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = YOLO_ROOT / "models" / "best.pt"
