"""
任务三 · 多空双向策略回测（2024-2025）
====================================
规则：
  预测拉伸(2) 且 p2>=0.5  -> 次日开盘 做多 10% 仓位
  预测下跌(0) 且 p0>=0.5  -> 次日开盘 做空 10% 仓位
  止损 5% / 止盈 15%（两个方向镜像）/ 最长持有 20 个交易日
  反向信号次日开盘平仓；最多同时 10 仓（多空共用）；手续费双边各 0.1%
引擎语义与 task3_backtest_2024_2025.py 完全一致（先验证：仅做多 应精确复现 +105.66%）
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "task3_2024_2025_results"
OUT = ROOT / "task3_long_short_results"
OUT.mkdir(exist_ok=True)

P_BUY, P_SHORT = 0.5, 0.5
STOP, TAKE, MAX_HOLD = 0.05, 0.15, 20
POS_FRAC, MAX_POS, COST = 0.10, 10, 0.001

# ---------------------------------------------------------------- 数据
sig = pd.read_csv(CACHE / "signals_2024_2025.csv", parse_dates=["date"])
panel = pd.read_csv(CACHE / "panel_ohlc.csv", parse_dates=["date"])
mkt = pd.read_csv(CACHE / "market_daily.csv", parse_dates=["date"])
calendar = [pd.Timestamp(x) for x in sorted(mkt["date"].unique())]
pos_in_cal = {d: i for i, d in enumerate(calendar)}

panel_idx = {}
for r in panel.itertuples():
    panel_idx[(r.code, r.date)] = (r.open, r.high, r.low, r.close)

def bar(code, date):
    return panel_idx.get((code, date))

# 信号 -> 执行日
buy_at_open = {}    # ed -> [(code, dir, prob)]
reverse_at_open = {}  # ed -> [(code, dir)]  多头遇下跌信号平多/空头遇拉伸信号平空
n_long_sig = n_short_sig = 0
for r in sig.itertuples():
    i = pos_in_cal.get(r.date)
    if i is None or i + 1 >= len(calendar):
        continue
    ed = calendar[i + 1]
    if r.pred == 2 and r.p2 >= P_BUY:
        buy_at_open.setdefault(ed, []).append((r.code, +1, r.p2))
        reverse_at_open.setdefault(ed, []).append((r.code, +1))
        n_long_sig += 1
    elif r.pred == 0 and r.p0 >= P_SHORT:
        buy_at_open.setdefault(ed, []).append((r.code, -1, r.p0))
        reverse_at_open.setdefault(ed, []).append((r.code, -1))
        n_short_sig += 1
# 同一天信号按置信度降序（与原版一致：仓位不足时高置信度优先）
for ed in buy_at_open:
    buy_at_open[ed].sort(key=lambda x: -x[2])
print(f"多头信号 {n_long_sig}，空头信号 {n_short_sig}", flush=True)

# ---------------------------------------------------------------- 引擎（镜像原版语义）
def run(use_long=True, use_short=True, stop=STOP, take=TAKE, name=""):
    cash, equity = 1.0, 1.0
    positions = {}  # code -> dict
    trades, daily_rows = [], []

    def close_position(code, date, price, reason):
        nonlocal cash
        p = positions.pop(code)
        if p["dir"] == +1:
            proceeds = p["shares"] * price * (1 - COST)
        else:  # 空头平仓: 返还保证金 + 价差收益 - 买回手续费
            proceeds = p["cost_basis"] + p["shares"] * (p["entry_price"] - price) - p["shares"] * price * COST
        cash += proceeds
        trades.append({"code": code, "dir": "long" if p["dir"] == 1 else "short",
                       "entry_date": p["entry_date"], "exit_date": date,
                       "entry_price": p["entry_price"], "exit_price": price,
                       "bars_held": p["bars"], "return": proceeds / p["cost_basis"] - 1.0,
                       "pnl": proceeds - p["cost_basis"], "reason": reason})

    for d in calendar:
        # 1) 开盘：反向信号平仓
        for code, sdir in reverse_at_open.get(d, []):
            if code in positions and positions[code]["dir"] != sdir:
                r = bar(code, d)
                if r and np.isfinite(r[0]) and r[0] > 0:
                    close_position(code, d, r[0], "signal_reverse")
        # 2) 开盘：新开仓
        for code, direction, prob in buy_at_open.get(d, []):
            if direction == +1 and not use_long:
                continue
            if direction == -1 and not use_short:
                continue
            if code in positions or len(positions) >= MAX_POS:
                continue
            r = bar(code, d)
            if not r or not np.isfinite(r[0]) or r[0] <= 0:
                continue
            notional = min(POS_FRAC * equity, cash)
            if notional < 1e-6:
                continue
            fee = notional * COST
            shares = (notional - fee) / r[0]
            if shares <= 0:
                continue
            cash -= notional
            positions[code] = {"dir": direction, "entry_date": d, "entry_price": r[0],
                               "shares": shares, "cost_basis": notional, "bars": 0}
        # 3) 盘中：止损/止盈（含入场当日，与原版一致）
        for code in list(positions.keys()):
            r = bar(code, d)
            if not r:
                continue
            p = positions[code]
            ep = p["entry_price"]
            if stop is not None:
                if p["dir"] == +1:
                    sp = ep * (1 - stop)
                    if np.isfinite(r[2]) and r[2] <= sp:
                        close_position(code, d, min(r[0], sp) if np.isfinite(r[0]) else sp, "stop_loss")
                        continue
                else:
                    sp = ep * (1 + stop)
                    if np.isfinite(r[1]) and r[1] >= sp:
                        close_position(code, d, max(r[0], sp) if np.isfinite(r[0]) else sp, "stop_loss")
                        continue
            if take is not None:
                if p["dir"] == +1:
                    tp = ep * (1 + take)
                    if np.isfinite(r[1]) and r[1] >= tp:
                        close_position(code, d, max(r[0], tp) if np.isfinite(r[0]) else tp, "take_profit")
                        continue
                else:
                    tp = ep * (1 - take)
                    if np.isfinite(r[2]) and r[2] <= tp:
                        close_position(code, d, min(r[0], tp) if np.isfinite(r[0]) else tp, "take_profit")
                        continue
        # 4) 收盘：到期平仓
        for code in list(positions.keys()):
            p = positions[code]
            p["bars"] += 1
            if p["bars"] >= MAX_HOLD:
                r = bar(code, d)
                if r:
                    close_position(code, d, r[3], "max_hold")
        # 5) 收盘估值
        mtm = 0.0
        for code, p in positions.items():
            r = bar(code, d)
            if not r:
                continue
            if p["dir"] == +1:
                mtm += p["shares"] * r[3]
            else:
                mtm += p["cost_basis"] + p["shares"] * (p["entry_price"] - r[3])
        equity = cash + mtm
        daily_rows.append({"date": d, "equity": equity, "n_positions": len(positions)})

    for code in list(positions.keys()):
        r = bar(code, calendar[-1])
        if r:
            close_position(code, calendar[-1], r[3], "end_of_backtest")
    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trades)
    print(f"  [{name}] 期末净值 {daily['equity'].iloc[-1]:.4f}  交易 {len(trades)} 笔", flush=True)
    return daily, trades

def perf(daily, trades, name):
    eq = daily["equity"].to_numpy()
    ret = np.diff(eq) / eq[:-1]
    cagr = eq[-1] ** (244 / len(eq)) - 1
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(244) if ret.std() > 0 else 0
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    wins = trades[trades["pnl"] > 0]
    pf = wins["pnl"].sum() / abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()) if (trades["pnl"] < 0).any() else np.inf
    return {"name": name, "total_return": float(eq[-1] - 1), "cagr": float(cagr),
            "sharpe": float(sharpe), "max_drawdown": float(mdd),
            "n_trades": int(len(trades)), "win_rate": float((trades["return"] > 0).mean()) if len(trades) else 0,
            "avg_trade_return": float(trades["return"].mean()) if len(trades) else 0,
            "avg_win": float(trades.loc[trades["return"] > 0, "return"].mean()) if (trades["return"] > 0).any() else 0,
            "avg_loss": float(trades.loc[trades["return"] <= 0, "return"].mean()) if (trades["return"] <= 0).any() else 0,
            "best_trade": float(trades["return"].max()) if len(trades) else 0,
            "worst_trade": float(trades["return"].min()) if len(trades) else 0,
            "profit_factor": float(pf)}

# ---------------------------------------------------------------- 先验证引擎：仅做多应复现 +105.66%
results, curves = {}, {}
daily, trades = run(use_short=False, name="仅做多(引擎验证)")
assert len(trades) == 291 and abs(daily["equity"].iloc[-1] - 2.0565515) < 1e-4, \
    f"引擎验证失败: 净值 {daily['equity'].iloc[-1]:.4f} 交易 {len(trades)}"
print("  ✓ 引擎验证通过：与原版 +105.66% 一致\n", flush=True)
results["仅做多"] = perf(daily, trades, "仅做多")
curves["仅做多"] = daily.set_index("date")["equity"]
daily.to_csv(OUT / "daily_equity_仅做多.csv", index=False)
trades.to_csv(OUT / "trades_仅做多.csv", index=False)

for name, kw in [("多空组合", {}),
                 ("仅做空", {"use_long": False}),
                 ("多空无风控", {"stop": None, "take": None})]:
    daily, trades = run(name=name, **kw)
    daily.to_csv(OUT / f"daily_equity_{name}.csv", index=False)
    trades.to_csv(OUT / f"trades_{name}.csv", index=False)
    results[name] = perf(daily, trades, name)
    curves[name] = daily.set_index("date")["equity"]

# 多空腿分解
tr_ls = pd.read_csv(OUT / "trades_多空组合.csv")
leg = {}
for side in ["long", "short"]:
    t = tr_ls[tr_ls["dir"] == side]
    if len(t):
        leg[side] = {"n": int(len(t)), "win_rate": float((t["return"] > 0).mean()),
                     "avg_return": float(t["return"].mean()), "pnl_sum": float(t["pnl"].sum()),
                     "best": float(t["return"].max()), "worst": float(t["return"].min())}
json.dump({"config": {"p_buy": P_BUY, "p_short": P_SHORT, "stop": STOP, "take": TAKE,
                      "max_hold": MAX_HOLD, "pos_frac": POS_FRAC, "max_pos": MAX_POS, "cost": COST},
           "results": results, "long_short_legs": leg},
          open(OUT / "metrics.json", "w"), ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 图
fig, ax = plt.subplots(figsize=(10, 5.2))
for name, s in curves.items():
    ax.plot(s.index, s.values, label=f"{name} ({s.iloc[-1]-1:+.1%})", lw=1.4)
mkt_curve = (1 + mkt.set_index("date")["mkt_return"]).cumprod()
ax.plot(mkt_curve.index, mkt_curve.values, label=f"全市场等权 ({mkt_curve.iloc[-1]-1:+.1%})",
        lw=1.1, color="gray", alpha=0.8)
ax.axhline(1, color="k", lw=0.6, alpha=0.4)
ax.set_title("多空双向策略 vs 对照 · 净值曲线（2024-2025）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig01_equity.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.6))
for name in ["多空组合", "仅做多", "仅做空"]:
    s = curves[name]
    dd = s / s.cummax() - 1
    ax.plot(dd.index, dd.values, label=f"{name} (最大 {dd.min():.1%})", lw=1.2)
ax.set_title("回撤对比")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig02_drawdown.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, side, title in [(axes[0], "long", "多头腿"), (axes[1], "short", "空头腿")]:
    t = tr_ls[tr_ls["dir"] == side]
    ax.hist(t["return"] * 100, bins=40, color="seagreen" if side == "long" else "indianred", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{title}: {len(t)}笔 胜率{(t['return']>0).mean():.1%} 均益{t['return'].mean():+.2%}")
    ax.grid(alpha=0.3)
fig.suptitle("多空组合 · 两腿交易收益分布")
fig.tight_layout()
fig.savefig(OUT / "fig03_legs.png")
plt.close(fig)

print(json.dumps({"results": results, "legs": leg}, ensure_ascii=False, indent=2))
