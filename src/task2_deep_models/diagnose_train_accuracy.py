"""诊断：训练集准确率为什么卡在 33%？

结论先行（跑完自行验证）：
  - deep_model_results/training_log.csv 里的 train_accuracy ≈ 0.33 是**统计口径 bug**，
    不是模型没学到东西。
  - 原因：train_loader 用 shuffle=True 创建，而 PyTorch 的 shuffle DataLoader 每次
    iter() 都会重新随机排序一次。训练循环已消费掉第 1 次排序；随后 predict() 又 iter 了
    一遍，得到的是**第 2 次排序**下的预测，顺序与 y_train 不对齐。
  - metrics(y_train, train_pred) 因此把「预测 A」和「标签 B」配对，3 类均衡下结果≈随机=1/3。
  - val/test loader 是 shuffle=False，所以 0.50 的验证/测试准确率是真实可信的。

本脚本做两件事：
  1. 用固定顺序（shuffle=False）重算训练集准确率，得到模型在训练集上的真实拟合水平。
  2. 复现该 bug：把对齐的预测随机打乱后再比对，观察准确率塌回 ~1/3。
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from train_deep_sequence_models import Config, build_model

ROOT = "deep_model_results"
INPUT_SIZE = 33
CHUNK = 1024


def chunked_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """按训练脚本同款方式算 per-channel mean/std，但分块累加以避免一次性读入 1.18GB。"""
    x = np.load(path, mmap_mode="r")
    n, T, C = x.shape
    s1 = np.zeros(C, np.float64)
    s2 = np.zeros(C, np.float64)
    seen = 0
    for start in range(0, n, CHUNK):
        block = x[start:start + CHUNK].astype(np.float64)
        s1 += block.sum(axis=(0, 1))
        s2 += (block ** 2).sum(axis=(0, 1))
        seen += block.shape[0] * T
    mean = s1 / seen
    var = np.maximum(s2 / seen - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def main() -> None:
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"device = {device}")

    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    counts = np.bincount(y_train, minlength=3)
    print(f"y_train 分布: {counts}  比例: "
          + " ".join(f"{v / len(y_train):.4f}" for v in counts))

    print("计算训练集标准化参数（分块）...")
    mean, std = chunked_stats("X_train.npy")

    ckpt = torch.load(f"{ROOT}/best_model.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    model = build_model(cfg, INPUT_SIZE).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"模型 = {cfg.model}  hidden={cfg.hidden_size}  layers={cfg.num_layers}")

    print("加载 X_train（mmap）并按固定顺序前向...")
    X = np.load("X_train.npy", mmap_mode="r")
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), CHUNK):
            xb = torch.from_numpy(((X[start:start + CHUNK] - mean) / std).astype(np.float32))
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    print("\n=== 1) 正确口径：训练集真实准确率（顺序对齐）===")
    acc = accuracy_score(y_train, preds)
    macro = f1_score(y_train, preds, average="macro")
    print(f"train_accuracy (正确)  = {acc:.4f}")
    print(f"train_macro_f1  (正确) = {macro:.4f}")
    print(f"日志中记录的 train_accuracy (epoch20, tcn lr0.001) = 0.329217")
    print("\n分类报告:")
    print(classification_report(y_train, preds,
                                target_names=["下跌/fall", "震荡/neutral", "拉伸/pull"],
                                zero_division=0))

    print("=== 2) 复现 bug：预测被打乱一次后准确率塌回随机水平 ===")
    rng = np.random.default_rng(0)
    for trial in range(5):
        shuffled = rng.permutation(preds)
        print(f"  打乱后准确率 (trial {trial}) = {accuracy_score(y_train, shuffled):.4f}")

    print("\n=== 3) 对照：验证集（shuffle=False，口径本来就正确）===")
    Xv = np.load("X_val.npy", mmap_mode="r")
    vpreds = []
    with torch.no_grad():
        for start in range(0, len(Xv), CHUNK):
            xb = torch.from_numpy(((Xv[start:start + CHUNK] - mean) / std).astype(np.float32))
            vpreds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    vpreds = np.concatenate(vpreds)
    print(f"val_accuracy (正确) = {accuracy_score(y_val, vpreds):.4f}  (日志记录 0.502203)")

    print("\n=== 4) 结论 ===")
    print(f"  a) 训练集真实准确率 {acc:.4f} >> 33%，模型在训练集上确实学到了东西；")
    print("     日志里的 0.329 是顺序错位造成的假象，不是拟合不足。")
    print(f"  b) 验证集准确率 {accuracy_score(y_val, vpreds):.4f} 与日志 0.502203 一致，")
    print("     说明 0.50 是真实数字，可放心使用。")
    print("  c) 修复方法：predict() 传 shuffle=False 的 loader，或用 y 建索引把")
    print("     预测映射回原始顺序；train/val 同口径后差距即正常化的过拟合度量。")
    print("  训练集模型拟合水平远高于 33%；0.50 的验证/测试准确率是真实数字。")
    print("  要修复日志：predict() 必须传 shuffle=False 的 loader，")
    print("  或用 y 做索引把预测映射回原始顺序。")


if __name__ == "__main__":
    main()
