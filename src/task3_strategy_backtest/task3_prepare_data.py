"""
任务三 · 数据准备阶段
=====================

目的：把"任务二训练好的三分类模型"的输出，与"真实日期 + 真实 OHLC 价格"对齐，
为 2024-2025 年真实口径回测准备三份缓存文件。

背景（关键的对齐事实）
----------------------
任务二的数据集 X_*.npy / y_*.npy 本身不含日期，且 y_test.npy 是把 test.csv 按
label 做**稳定排序**后得到的（已在本脚本中断言验证）。因此存在唯一确定的置换：

    perm = np.argsort(test.csv.label, kind="stable")
    y_test[i] == test.csv.label[perm[i]]
    => X_test.npy 的第 i 行  <->  test.csv 的第 perm[i] 行  <->  (code, date)

据此可以给每个测试样本恢复真实的股票代码与交易日，从而用真实价格回测。

产出（写入 task3_2024_2025_results/）
-------------------------------------
1. signals_full.csv      全部 21330 个测试样本：code,date,label,pred,p0,p1,p2
2. signals_2024_2025.csv 其中日期落在 2024-01-01 ~ 2025-12-31 的信号（用于回测）
3. panel_ohlc.csv        回测票池的真实日线 open/high/low/close（含 2023-12 缓冲）
4. market_daily.csv      全市场等权日收益（买入持有基准）
5. model_check.json      模型 checkpoint 复现校验信息

用法：
    .venv/bin/python task3_prepare_data.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "task3_2024_2025_results"
OUT.mkdir(exist_ok=True)

# ---- 输入文件 -------------------------------------------------------------
CHECKPOINT = ROOT / "deep_model_results1024" / "best_model.pt"
X_TRAIN = ROOT / "X_train.npy"
X_TEST = ROOT / "X_test.npy"
Y_TEST = ROOT / "y_test.npy"
TEST_CSV = ROOT / "test.csv"

# 源 CSV 可能位于不同目录名（历史版本为 预处理/，现为 original_data/）
SOURCE_CANDIDATES = [
    ROOT / "original_data" / "all_2016_2025_with_indicators.csv",
    ROOT / "预处理" / "all_2016_2025_with_indicators.csv",
    ROOT / "all_2016_2025_with_indicators.csv",
]

# ---- 回测窗口 -------------------------------------------------------------
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-12-31"
# 持有期最长 20 个交易日，需要信号日之后的真实价格路径
PANEL_START = "2023-12-01"   # 留出缓冲，覆盖 2024-01 首个交易日的"前一日"
PANEL_END = "2025-12-31"
INPUT_SIZE = 33


def find_source_csv() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    hits = list(ROOT.glob("**/all_2016_2025_with_indicators.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError("找不到 all_2016_2025_with_indicators.csv")


# ---------------------------------------------------------------------------
# 1. 模型 -> 概率
# ---------------------------------------------------------------------------
def train_channel_stats(chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """按任务二训练脚本的口径，用 X_train 计算每通道 mean/std。

    **必须与 train_deep_sequence_models.load_arrays() 逐位一致**：
    该函数是在整块 float32 数组上一次性算 `mean(axis=(0,1))` / `std(axis=(0,1))` 的。
    如果改用分块累加（float64），浮点求和顺序不同，会让约 1% 的样本在 argmax 上翻转
    （top1 与 top2 概率差通常 < 0.05），从而无法逐位复现任务二保存的预测。
    这里优先用整块 float32；内存不足时回退到分块并给出警告。
    """
    try:
        x = np.load(X_TRAIN).astype(np.float32)
        mean = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
        del x
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std
    except MemoryError:
        print("  [warn] 整块加载 X_train 内存不足，回退到分块累加（可能与任务二存在 ~1% 的 argmax 差异）",
              flush=True)
    x = np.load(X_TRAIN, mmap_mode="r")
    c = x.shape[2]
    s1 = np.zeros(c, np.float64)
    s2 = np.zeros(c, np.float64)
    seen = 0
    for st in range(0, len(x), chunk):
        b = x[st:st + chunk].astype(np.float64)
        s1 += b.sum(axis=(0, 1))
        s2 += (b ** 2).sum(axis=(0, 1))
        seen += b.shape[0] * b.shape[1]
    mean = s1 / seen
    var = np.maximum(s2 / seen - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


@torch.no_grad()
def model_probabilities(checkpoint: Path = CHECKPOINT) -> np.ndarray:
    """加载任务二最优模型（TCN），输出测试集三分类 softmax 概率 (N,3)。"""
    from train_deep_sequence_models import TCNClassifier

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TCNClassifier(INPUT_SIZE, cfg["hidden_size"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    mean, std = train_channel_stats()
    X = np.load(X_TEST).astype(np.float64)
    X = ((X - mean) / std).astype(np.float32)

    outs = []
    for st in range(0, len(X), 1024):
        logits = model(torch.from_numpy(X[st:st + 1024]))
        outs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outs), cfg


# ---------------------------------------------------------------------------
# 2. (code, date) 对齐
# ---------------------------------------------------------------------------
def build_signals() -> tuple[pd.DataFrame, dict]:
    probs, cfg = model_probabilities()
    pred = probs.argmax(axis=1)

    # 正确的 npy <-> (code,date) 对齐：价格指纹全局双射（task3_remap）
    # 旧版用 argsort(label, stable)，但 test.csv 本身已按类排序 => 恒等置换 => 300/300 错配
    from task3_remap import remap_test_to_dates, verify_mapping
    sig = remap_test_to_dates()
    mapping_stats = verify_mapping(sig)
    sig["pred"] = pred
    sig["p0"] = probs[:, 0]
    sig["p1"] = probs[:, 1]
    sig["p2"] = probs[:, 2]
    sig["p_true"] = probs[np.arange(len(sig)), sig["label"].values]
    sig["date"] = pd.to_datetime(sig["date"])

    acc = float((sig["pred"] == sig["label"]).mean())
    y = np.load(Y_TEST)
    acc_vs_sorted_y = float((pred == y).mean())
    saved_pred_path = ROOT / "deep_model_results1024" / "predictions_test_tcn_lr0.001_ep20.npy"
    agree = None
    if saved_pred_path.exists():
        agree = float((np.load(saved_pred_path) == pred).mean())

    check = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "model_config": cfg,
        "mapping_method": "price_fingerprint_bijection (task3_remap)",
        "mapping_verification": mapping_stats,
        "test_accuracy_vs_true_labels": acc,
        "test_accuracy_vs_sorted_y_test_artifact": acc_vs_sorted_y,
        "agreement_with_saved_predictions": agree,
        "n_test_samples": int(len(sig)),
        "date_range_test": [str(sig["date"].min().date()), str(sig["date"].max().date())],
    }
    return sig, check


# ---------------------------------------------------------------------------
# 3. 真实价格面板 + 全市场基准
# ---------------------------------------------------------------------------
def build_panel(universe_codes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = find_source_csv()
    print(f"[panel] 流式读取源数据: {src}", flush=True)
    panel_rows: list[tuple] = []
    mkt_sum: dict[str, float] = {}
    mkt_cnt: dict[str, int] = {}

    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ci = {c: i for i, c in enumerate(header)}
        i_code, i_date = ci["code"], ci["date"]
        i_o, i_h, i_l, i_c = ci["open"], ci["high"], ci["low"], ci["close"]
        i_pct = ci["pct_chg"]
        n = 0
        for row in reader:
            d = row[i_date]
            if d < PANEL_START:
                continue
            n += 1
            code = row[i_code]
            # 全市场等权日收益（仅回测年份）
            if d >= BACKTEST_START and d <= BACKTEST_END:
                try:
                    v = float(row[i_pct])
                    if np.isfinite(v):
                        mkt_sum[d] = mkt_sum.get(d, 0.0) + v
                        mkt_cnt[d] = mkt_cnt.get(d, 0) + 1
                except (ValueError, TypeError):
                    pass
            if code not in universe_codes:
                continue
            try:
                panel_rows.append((
                    code, d,
                    float(row[i_o]), float(row[i_h]),
                    float(row[i_l]), float(row[i_c]),
                ))
            except (ValueError, TypeError):
                continue
    print(f"[panel] 扫描 {n} 行（>={PANEL_START}），票池记录 {len(panel_rows)} 条", flush=True)

    panel = pd.DataFrame(panel_rows, columns=["code", "date", "open", "high", "low", "close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna().sort_values(["code", "date"]).reset_index(drop=True)

    mkt = pd.DataFrame({
        "date": pd.to_datetime(list(mkt_sum.keys())),
        "mkt_return": [mkt_sum[d] / mkt_cnt[d] / 100.0 for d in mkt_sum],
        "n_codes": [mkt_cnt[d] for d in mkt_sum],
    }).sort_values("date").reset_index(drop=True)
    return panel, mkt


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

    print("[1/3] 复现任务二模型并恢复真实日期...", flush=True)
    sig, check = build_signals()
    sig.to_csv(OUT / "signals_full.csv", index=False)
    (OUT / "model_check.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)

    mask = (sig["date"] >= BACKTEST_START) & (sig["date"] <= BACKTEST_END)
    sub = sig[mask].reset_index(drop=True)
    sub.to_csv(OUT / "signals_2024_2025.csv", index=False)
    print(f"[2/3] 2024-2025 信号 {len(sub)} 条，覆盖 {sub['code'].nunique()} 只股票 / "
          f"{sub['date'].nunique()} 个交易日", flush=True)

    print("[3/3] 提取真实 OHLC 面板与全市场基准...", flush=True)
    panel, mkt = build_panel(set(sub["code"].unique()))
    panel.to_csv(OUT / "panel_ohlc.csv", index=False)
    mkt.to_csv(OUT / "market_daily.csv", index=False)
    print(f"      panel={panel.shape}  market={mkt.shape}", flush=True)
    print("完成。输出目录:", OUT, flush=True)


if __name__ == "__main__":
    main()
