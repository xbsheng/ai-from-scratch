"""模型运行时资源占用统计（零额外依赖：stdlib + torch）。

统计项：参数量 / 内存(RSS) / MPS 显存 / prefill 延迟 / 生成速度 / 峰值内存。

用法:
    uv run python profile.py              # 随机权重（测资源占用，无需下载）
    uv run python profile.py --official   # 加载官方权重后统计

注: 不加载官方权重时内存占用同样准确（权重内容不影响内存量）。
"""

import resource
import sys
import time

import torch
from config import QWEN_CONFIG_0_6_B
from model import Qwen3


def rss_mb() -> float:
    """进程峰值物理内存。macOS 的 ru_maxrss 单位是字节，Linux 是 KB。"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def main() -> None:
    print(f"device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"dtype:  {QWEN_CONFIG_0_6_B['dtype']}")

    model = Qwen3(QWEN_CONFIG_0_6_B)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e9:.3f}B | fp32 权重理论内存: {n_params * 4 / 1e9:.2f}GB")

    if "--official" in sys.argv:
        from load_qwen3 import load_official_weights

        load_official_weights(model)

    model = model.to(QWEN_CONFIG_0_6_B["dtype"]).eval()
    print(f"实例化+转dtype后 RSS: {rss_mb():.0f} MB")
    if torch.backends.mps.is_available():
        print(f"MPS 显存: {torch.mps.driver_allocated_memory() / 1e6:.0f} MB")

    # ---- prefill 计时 ----
    torch.manual_seed(0)
    in_idx = torch.randint(0, QWEN_CONFIG_0_6_B["vocab_size"], (1, 8))
    cache: dict = {}
    with torch.no_grad():
        t0 = time.perf_counter()
        model(in_idx, cache)
        prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"prefill 8 tokens: {prefill_ms:.0f} ms")

    # ---- 生成计时（10 tokens）----
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(10):
            model(in_idx[:, -1:], cache)
        gen_s = time.perf_counter() - t0
    print(f"生成 10 tokens: {gen_s:.1f}s | {10 / gen_s:.1f} tokens/s")

    print(f"峰值 RSS: {rss_mb():.0f} MB")
    if torch.backends.mps.is_available():
        print(f"MPS 显存(峰值后): {torch.mps.driver_allocated_memory() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
