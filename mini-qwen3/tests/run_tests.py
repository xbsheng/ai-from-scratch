import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""全部测试统一入口：uv run python run_tests.py

按功能模块拆分后的聚合 runner，逐个模块执行所有 test_* 函数。
单个模块也可独立运行：uv run python test_<模块>.py
"""

import importlib
import sys

from test_common import run_module

MODULES = [
    "test_rms_norm",
    "test_ffn",
    "test_rope",
    "test_gqa",
    "test_tf_block",
    "test_model",
    "test_moe",
]


def main() -> int:
    total_failed = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        print(f"\n===== {mod_name} =====")
        total_failed += run_module(mod.__dict__)

    print(f"\n{'✅ 全部通过' if total_failed == 0 else f'❌ {total_failed} 个测试失败'}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
