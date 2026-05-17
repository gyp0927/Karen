"""Pytest 共享配置。"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，使 import core/agents/graph 等能正确解析
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
