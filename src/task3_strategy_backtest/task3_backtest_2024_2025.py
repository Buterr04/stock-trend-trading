"""
任务三 · 基于三分类预测的中低频量化策略回测（2024-2025 真实日期口径）
=====================================================================

策略逻辑
--------
任务二最优模型（TCN）在样本日 t 输出三分类概率：
    0 = 下跌类(fall)   1 = 震荡类(neutral)   2 = 拉伸类(pull)

交易规则（全部按开盘价撮合，不使用未来信息）：
    * 开仓：pred == 2(拉伸) 且 p(拉伸) >= p_buy  ->  次日开盘买入
    * 平仓：pred == 0(下跌)                      ->  次日开盘卖出
    * 止损：持仓期间触及 entry*(1-stop_loss)      ->  盘中触发价成交（跳空则按开盘价）
    * 止盈：持仓期间触及 entry*(1+take_profit)    ->  盘中触发价成交（跳空则按开盘价）
    * 最长持有 max_hold_days 个交易日             ->  当日收盘卖出

时点约定（重要）
----------------
原始特征构造 preprocess.py 中 `window = code_data[start:target]`，即 60 日窗口为
[t-60, t-1]，不含样本日 t 本身；而标签基准是 close[t]。
因此"t 日开盘执行"在信息集上是合法的（t 开盘晚于 t-1 收盘）。
本脚本默认仍采用更保守的 **exec_delay=1（t+1 开盘成交）**，并在正文中给出
exec_delay=0（t 日开盘成交）的对照结果。

风险控制
--------
    固定仓位 position_fraction、最大持仓数 max_positions、
    止损 / 止盈 / 最长持有期、双边交易成本 cost_rate。

基准
----
    * market_bh     : 全市场等权日收益复利（买入持有）
    * always_long   : 票池内所有样本信号无条件持有 max_hold 日（等权，信号无关基准）
    * random_signal : 同样的信号日随机抽取同等数量标的，重复 N 次（安慰剂检验）
    * no_risk       : 关闭止损/止盈的同一策略

用法：
    .venv/bin/python task3_backtest_2024_2025.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "task3_2024_2025_results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

plt.rcParams["font.family"] = ["Heiti TC", "Heiti SC", "Arial Unicode MS", "PingFang HK", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

TRADING_DAYS = 244
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-12-31"


# ===========================================================================
# 配置
# ===========================================================================
@dataclass(frozen=True)
class StrategyConfig:
    buy_class: int = 2
    sell_class: int = 0
    p_buy: float = 0.50                # p(拉伸) 开仓阈值
    stop_loss: float | None = 0.05
    take_profit: float | None = 0.15
    max_hold_days: int = 20
    cost_rate: float = 0.001           # 单边 0.10%（佣金+印花税+滑点近似）
    position_fraction: float = 0.10    # 单仓位占当前权益比例
    max_positions: int = 10
    exec_delay: int = 1                # 0: 样本日 t 开盘成交；1: t+1 开盘成交

    def tag(self) -> str:
        sl = "off" if self.stop_loss is None else f"{self.stop_loss:.0%}"
        tp = "off" if self.take_profit is None else f"{self.take_profit:.0%}"
        return (f"p{self.p_buy:.2f}_sl{sl}_tp{tp}_hold{self.max_hold_days}"
                f"_pos{self.position_fraction:.0%}x{self.max_positions}_lag{self.exec_delay}")


DEFAULT = StrategyConfig()


# ===========================================================================
# 价格面板
# ===========================================================================
class Panel:
    """按股票组织的价格面板，提供 O(log n) 的 (code, date) -> 行号查找。"""

    def __init__(self, df: pd.DataFrame):
        self.data: dict[str, dict] = {}
        for code, g in df.groupby("code", sort=False):
            g = g.sort_values("date")
            self.data[code] = {
                "date": g["date"].to_numpy(),
                "open": g["open"].to_numpy(np.float64),
                "high": g["high"].to_numpy(np.float64),
                "low": g["low"].to_numpy(np.float64),
                "close": g["close"].to_numpy(np.float64),
            }

    def idx(self, code: str, date) -> int | None:
        d = self.data.get(code)
        if d is None:
            return None
        arr = d["date"]
        i = int(np.searchsorted(arr, np.datetime64(date)))
        if i < len(arr) and arr[i] == np.datetime64(date):
            return i
        return None

    def row(self, code: str, date) -> dict | None:
        i = self.idx(code, date)
        if i is None:
            return None
        d = self.data[code]
        return {"open": float(d["open"][i]), "high": float(d["high"][i]),
                "low": float(d["low"][i]), "close": float(d["close"][i])}

    def close_at(self, code: str, date) -> float | None:
        i = self.idx(code, date)
        return None if i is None else float(self.data[code]["close"][i])


# ===========================================================================
# 信号 -> 开盘执行撮合表
# ===========================================================================
def build_signal_maps(sig: pd.DataFrame, cfg: StrategyConfig, calendar: np.ndarray):
    """把"样本日信号"映射为"实际成交日开盘执行"的订单表。

    返回 (buy_at_open, sell_at_open, meta)：
        buy_at_open[date]  = [code, ...]  在该日开盘买入
        sell_at_open[date] = [code, ...]  在该日开盘卖出
        meta[(date, code)] = 建仓时的模型元信息
    """
    cal = [pd.Timestamp(x) for x in calendar]
    pos = {d: i for i, d in enumerate(cal)}
    buy_at_open: dict[pd.Timestamp, list[str]] = {}
    sell_at_open: dict[pd.Timestamp, list[str]] = {}
    meta: dict[tuple, dict] = {}

    buys = sig[(sig["pred"] == cfg.buy_class) & (sig[f"p{cfg.buy_class}"] >= cfg.p_buy)].copy()
    buys["_date_key"] = buys["date"].map(lambda x: pd.Timestamp(x))
    # 同一天信号按模型置信度（p 拉伸）降序排列：仓位额度不足时优先给高置信度信号，
    # 避免"先到先得"带来的人为路径依赖。
    for d, g in buys.sort_values(f"p{cfg.buy_class}", ascending=False).groupby("_date_key", sort=False):
        i = pos.get(d)
        if i is None:
            continue
        j = i + cfg.exec_delay
        if j >= len(cal):
            continue
        ed = cal[j]
        for r in g.itertuples():
            buy_at_open.setdefault(ed, []).append(r.code)
            meta[(ed, r.code)] = {"pred": int(r.pred),
                                  "p_buy": float(getattr(r, f"p{cfg.buy_class}")),
                                  "label": int(r.label)}

    sells = sig[sig["pred"] == cfg.sell_class]
    for r in sells.itertuples():
        i = pos.get(pd.Timestamp(r.date))
        if i is None:
            continue
        j = i + cfg.exec_delay
        if j >= len(cal):
            continue
        sell_at_open.setdefault(cal[j], []).append(r.code)
    return buy_at_open, sell_at_open, meta


# ===========================================================================
# 回测引擎
# ===========================================================================
def run_backtest(panel: Panel, calendar: np.ndarray,
                 buy_at_open: dict, sell_at_open: dict,
                 cfg: StrategyConfig, meta: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = meta or {}
    cal = [pd.Timestamp(x) for x in calendar]

    cash = 1.0
    equity = 1.0
    positions: dict[str, dict] = {}
    daily_rows: list[dict] = []
    trades: list[dict] = []

    def close_position(code: str, date, price: float, reason: str):
        nonlocal cash
        p = positions.pop(code)
        proceeds = p["shares"] * price * (1 - cfg.cost_rate)
        cash += proceeds
        trades.append({
            "code": code,
            "entry_date": p["entry_date"], "exit_date": date,
            "entry_price": p["entry_price"], "exit_price": price,
            "shares": p["shares"], "bars_held": p["bars"],
            "return": proceeds / p["cost_basis"] - 1.0,
            "pnl": proceeds - p["cost_basis"],
            "cost_basis": p["cost_basis"], "proceeds": proceeds,
            "reason": reason,
            "pred_at_entry": p["pred"], "p_buy_at_entry": p["p_buy"],
            "label_at_entry": p["label"],
        })

    for d in cal:
        # ---------- 1) 开盘：先执行卖出（释放现金与仓位额度） ----------
        for code in sell_at_open.get(d, []):
            if code not in positions:
                continue
            r = panel.row(code, d)
            if r and np.isfinite(r["open"]) and r["open"] > 0:
                close_position(code, d, r["open"], "signal_fall")
        # ---------- 2) 开盘：执行买入 ----------
        for code in buy_at_open.get(d, []):
            if code in positions or len(positions) >= cfg.max_positions:
                continue
            r = panel.row(code, d)
            if not r or not np.isfinite(r["open"]) or r["open"] <= 0:
                continue
            notional = min(cfg.position_fraction * equity, cash)
            if notional < 1e-6:
                continue
            fee = notional * cfg.cost_rate
            shares = (notional - fee) / r["open"]
            if shares <= 0:
                continue
            cash -= notional
            m = meta.get((d, code), {})
            positions[code] = {
                "entry_date": d, "entry_price": r["open"], "shares": shares,
                "cost_basis": notional, "bars": 0,
                "pred": m.get("pred", -1), "p_buy": m.get("p_buy", np.nan),
                "label": m.get("label", -1),
            }
        # ---------- 3) 盘中：止损 / 止盈 ----------
        for code in list(positions.keys()):
            r = panel.row(code, d)
            if not r:
                continue
            ep = positions[code]["entry_price"]
            if cfg.stop_loss is not None:
                stop = ep * (1 - cfg.stop_loss)
                if np.isfinite(r["low"]) and r["low"] <= stop:
                    fill = min(r["open"], stop) if np.isfinite(r["open"]) else stop
                    close_position(code, d, fill, "stop_loss")
                    continue
            if cfg.take_profit is not None:
                take = ep * (1 + cfg.take_profit)
                if np.isfinite(r["high"]) and r["high"] >= take:
                    fill = max(r["open"], take) if np.isfinite(r["open"]) else take
                    close_position(code, d, fill, "take_profit")
                    continue
        # ---------- 4) 收盘：最长持有期强制平仓 ----------
        for code in list(positions.keys()):
            p = positions[code]
            p["bars"] += 1
            if p["bars"] >= cfg.max_hold_days:
                c = panel.close_at(code, d)
                if c:
                    close_position(code, d, c, "max_hold")
        # ---------- 5) 收盘估值 ----------
        mtm = sum(p["shares"] * (panel.close_at(code, d) or 0.0)
                  for code, p in positions.items())
        equity = cash + mtm
        daily_rows.append({
            "date": d, "equity": equity, "cash": cash, "invested": mtm,
            "n_positions": len(positions),
            "position_ratio": mtm / equity if equity > 0 else 0.0,
        })
    # 收尾：最后一日仍持仓则按收盘平掉
    for code in list(positions.keys()):
        c = panel.close_at(code, cal[-1])
        if c:
            close_position(code, cal[-1], c, "end_of_backtest")

    daily = pd.DataFrame(daily_rows)
    daily["daily_return"] = daily["equity"].pct_change().fillna(daily["equity"].iloc[0] - 1.0)
    daily["cum_return"] = daily["equity"] - 1.0
    daily["peak"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["peak"] - 1.0
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["entry_date"] = pd.to_datetime(tr["entry_date"])
        tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    return daily, tr


# ===========================================================================
# 绩效指标
# ===========================================================================
def performance(daily: pd.DataFrame, trades: pd.DataFrame | None, label: str = "") -> dict:
    eq = daily["equity"].to_numpy()
    n = len(eq)
    ret = daily["daily_return"].to_numpy()
    years = n / TRADING_DAYS
    total = float(eq[-1] - 1)
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else np.nan
    vol = float(np.std(ret, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else np.nan
    mean_ann = float(np.mean(ret) * TRADING_DAYS)
    sharpe = mean_ann / vol if vol and vol > 0 else np.nan
    downside = ret[ret < 0]
    dvol = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan
    sortino = mean_ann / dvol if dvol and dvol > 0 else np.nan
    mdd = float(daily["drawdown"].min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    out = {
        "name": label,
        "start": str(daily["date"].iloc[0].date()),
        "end": str(daily["date"].iloc[-1].date()),
        "trading_days": n,
        "total_return": total,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "avg_position_ratio": float(daily["position_ratio"].mean()),
        "avg_n_positions": float(daily["n_positions"].mean()),
    }
    if trades is not None and len(trades):
        wins = trades[trades["return"] > 0]
        losses = trades[trades["return"] <= 0]
        gp = float(wins["return"].sum())
        gl = float(-losses["return"].sum())
        traded = float(trades["cost_basis"].sum() + trades["proceeds"].sum())
        out.update({
            "n_trades": int(len(trades)),
            "win_rate": float(len(wins) / len(trades)),
            "avg_trade_return": float(trades["return"].mean()),
            "avg_win": float(wins["return"].mean()) if len(wins) else np.nan,
            "avg_loss": float(losses["return"].mean()) if len(losses) else np.nan,
            "profit_factor": gp / gl if gl > 0 else np.nan,
            "avg_bars_held": float(trades["bars_held"].mean()),
            "best_trade": float(trades["return"].max()),
            "worst_trade": float(trades["return"].min()),
            "annual_turnover": traded / max(daily["equity"].mean(), 1e-9) / years,
        })
    return out


def segment_performance(sub: pd.DataFrame, trades: pd.DataFrame | None, label: str) -> dict:
    """区间绩效（权益重新归一到 1）。"""
    sub = sub.sort_values("date").reset_index(drop=True).copy()
    ret = sub["daily_return"].to_numpy()
    eq = np.cumprod(1 + ret)
    sub["equity"] = eq
    sub["drawdown"] = eq / np.maximum.accumulate(eq) - 1
    return performance(sub, trades, label)


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.set_index("date")["equity"].sort_index()
    me = d.resample("ME").last()
    base = me.shift(1)
    base.iloc[0] = d.iloc[0] / (1 + daily["daily_return"].iloc[0])
    mom = me / base - 1
    out = pd.DataFrame({"month_return": mom.values, "year": me.index.year,
                        "month": me.index.month}, index=me.index)
    return out.reset_index(drop=True)


# ===========================================================================
# 基准
# ===========================================================================
def always_long_returns(panel: Panel, sig: pd.DataFrame, calendar: np.ndarray,
                        cfg: StrategyConfig) -> dict:
    """票池内全部样本：按同样时点买入并持有 max_hold 个交易日，等权平均（信号无关）。"""
    cal = [pd.Timestamp(x) for x in calendar]
    pos = {d: i for i, d in enumerate(cal)}
    rets = []
    for r in sig.itertuples():
        i = pos.get(pd.Timestamp(r.date))
        if i is None:
            continue
        j = i + cfg.exec_delay
        exit_i = min(j + cfg.max_hold_days - 1, len(cal) - 1)
        if j >= len(cal):
            continue
        e = panel.row(r.code, cal[j])
        x = panel.close_at(r.code, cal[exit_i])
        if not e or x is None or e["open"] <= 0:
            continue
        rets.append(x / e["open"] * (1 - cfg.cost_rate) / (1 + cfg.cost_rate) - 1)
    rets = np.array(rets)
    return {"n": int(len(rets)),
            "mean": float(np.mean(rets)) if len(rets) else np.nan,
            "median": float(np.median(rets)) if len(rets) else np.nan,
            "win_rate": float((rets > 0).mean()) if len(rets) else np.nan}


def random_signal_mc(panel: Panel, sig: pd.DataFrame, calendar: np.ndarray, cfg: StrategyConfig,
                     n_iter: int = 200, seed: int = 20260922) -> pd.DataFrame:
    """安慰剂检验：在同样的信号日、按同样的买入数量随机选股，重复 n_iter 次。

    随机组合只保留"交易数量与择时分布"同构，剔除模型选股能力，
    用于回答：策略收益有多少来自"在 2024-2025 的均衡票池里做多头"而非"选对股票"。
    """
    rng = np.random.default_rng(seed)
    cal = [pd.Timestamp(x) for x in calendar]
    by_date = {pd.Timestamp(d): g["code"].tolist() for d, g in sig.groupby("date")}
    real_buy, _, _ = build_signal_maps(sig, cfg, calendar)
    n_per_day = {d: len(set(v)) for d, v in real_buy.items()}

    rows = []
    for it in range(n_iter):
        buy = {}
        for d, k in n_per_day.items():
            j = cal.index(d)
            si = j - cfg.exec_delay
            if si < 0:
                continue
            pool = by_date.get(cal[si], [])
            if not pool:
                continue
            pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
            buy[d] = list(pick)
        daily, trades = run_backtest(panel, calendar, buy, {}, cfg, {})
        m = performance(daily, trades, f"random_{it}")
        rows.append({"iter": it, "total_return": m["total_return"], "cagr": m["cagr"],
                     "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "n_trades": m.get("n_trades", 0), "win_rate": m.get("win_rate", np.nan)})
    return pd.DataFrame(rows)


# ===========================================================================
# 绘图
# ===========================================================================
def plot_equity_curve(curves: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for name, s in curves.items():
        axes[0].plot(s.index, s.values, lw=1.8, label=name)
    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("净值 (初始=1)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)
    main = curves["主策略"]
    dd = main / main.cummax() - 1
    axes[1].fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.4)
    axes[1].set_ylabel("回撤 (%)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drawdown(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(daily["date"], daily["drawdown"] * 100, 0, color="#c0392b", alpha=0.45)
    ax.plot(daily["date"], daily["drawdown"] * 100, color="#7b241c", lw=1)
    ax.set_title("策略回撤曲线（主策略）")
    ax.set_ylabel("回撤 (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_monthly_heatmap(mr: pd.DataFrame, path: Path) -> None:
    pivot = mr.pivot_table(index="year", columns="month", values="month_return") * 100
    fig, ax = plt.subplots(figsize=(12, 2.6))
    vmax = np.nanmax(np.abs(pivot.values)) or 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [f"{m}月" for m in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("月度收益率 (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trades(trades: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(trades["return"] * 100, bins=40, color="#2ecc71", edgecolor="white")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("单笔交易净收益率分布 (%)")
    axes[1].hist(trades["bars_held"], bins=range(1, int(trades["bars_held"].max()) + 2),
                 color="#3498db", edgecolor="white")
    axes[1].set_title("持仓天数分布")
    vr = trades["reason"].value_counts()
    colors = {"stop_loss": "#c0392b", "take_profit": "#27ae60", "max_hold": "#2980b9",
              "signal_fall": "#8e44ad", "end_of_backtest": "#7f8c8d"}
    axes[2].bar(vr.index, vr.values, color=[colors.get(i, "#95a5a6") for i in vr.index])
    axes[2].set_title("平仓原因统计")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_signal_quality(sig: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cm = pd.crosstab(sig["label"], sig["pred"]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    im = axes[0].imshow(cm.values, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, int(cm.values[i, j]), ha="center", va="center",
                         color="white" if cm.values[i, j] > cm.values.max() * 0.6 else "black")
    axes[0].set_xticks(range(3), ["预测跌", "预测震", "预测拉"])
    axes[0].set_yticks(range(3), ["实际跌", "实际震", "实际拉"])
    axes[0].set_title("2024-2025 混淆矩阵")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    bins = [0, .25, .35, .45, .55, .65, 1.01]
    s = sig.copy()
    s["bin"] = pd.cut(s["p2"], bins)
    labels_bin, hit_pull, hit_fall, ns = [], [], [], []
    for key, g in s.groupby("bin", observed=True):
        labels_bin.append(str(key))
        hit_pull.append(float((g["label"] == 2).mean()))
        hit_fall.append(float((g["label"] == 0).mean()))
        ns.append(len(g))
    x = np.arange(len(hit_pull))
    axes[1].bar(x - 0.2, hit_pull, 0.4, label="实际为拉伸类", color="#27ae60")
    axes[1].bar(x + 0.2, hit_fall, 0.4, label="实际为下跌类", color="#c0392b")
    axes[1].axhline(1 / 3, color="gray", ls="--", lw=1, label="随机基准 1/3")
    axes[1].set_xticks(x, [f"{l}\nn={n}" for l, n in zip(labels_bin, ns)], rotation=0, fontsize=7)
    axes[1].set_title("p(拉伸) 分箱校准")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for cls, name, color in [(0, "下跌类", "#c0392b"), (1, "震荡类", "#7f8c8d"), (2, "拉伸类", "#27ae60")]:
        sub = sig[sig["pred"] == cls]
        if len(sub):
            axes[2].bar(name, len(sub), color=color)
            axes[2].text(name, len(sub), f"{len(sub)}\n准确率{(sub['label'] == cls).mean():.1%}",
                         ha="center", va="bottom", fontsize=8)
    axes[2].set_title("各预测类别的样本量与准确率")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sensitivity(th: pd.DataFrame, sltp: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(th["p_buy"], th["total_return"] * 100, "o-", color="#2980b9", label="累计收益(%)")
    axes[0].set_xlabel("开仓阈值 p(拉伸)")
    axes[0].set_ylabel("累计收益 (%)")
    axes[0].set_title("阈值敏感性：收益与交易数")
    ax2 = axes[0].twinx()
    ax2.plot(th["p_buy"], th["n_trades"], "s--", color="#e67e22")
    ax2.set_ylabel("交易笔数")
    axes[0].grid(alpha=0.3)

    axes[1].plot(th["p_buy"], th["sharpe"], "o-", color="#8e44ad", label="夏普")
    axes[1].set_xlabel("开仓阈值 p(拉伸)")
    axes[1].set_ylabel("夏普比率")
    axes[1].set_title("阈值敏感性：夏普与最大回撤")
    ax3 = axes[1].twinx()
    ax3.plot(th["p_buy"], th["max_drawdown"] * 100, "^--", color="#c0392b")
    ax3.set_ylabel("最大回撤 (%)")
    axes[1].grid(alpha=0.3)

    g = sltp.copy()
    g["sl_label"] = g["stop_loss"].map(lambda v: "无" if v < 0 else f"{v:.0%}")
    g["tp_label"] = g["take_profit"].map(lambda v: "无" if v < 0 else f"{v:.0%}")
    piv = g.pivot_table(index="sl_label", columns="tp_label", values="sharpe")
    piv = piv.reindex(index=[l for l in ["无", "3%", "5%", "8%", "10%"] if l in piv.index],
                      columns=[l for l in ["无", "8%", "15%", "20%", "30%"] if l in piv.columns])
    im = axes[2].imshow(piv.values, cmap="viridis", aspect="auto")
    axes[2].set_xticks(range(len(piv.columns)), list(piv.columns))
    axes[2].set_yticks(range(len(piv.index)), list(piv.index))
    axes[2].set_xlabel("止盈")
    axes[2].set_ylabel("止损")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                axes[2].text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
    axes[2].set_title("止损/止盈网格：夏普比率")
    fig.colorbar(im, ax=axes[2], shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mc(mc: pd.DataFrame, real: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(mc["total_return"] * 100, bins=30, color="#95a5a6", edgecolor="white")
    axes[0].axvline(real["total_return"] * 100, color="#c0392b", lw=2.5, label="真实模型信号")
    axes[0].set_title("随机信号安慰剂检验：累计收益 (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].hist(mc["sharpe"], bins=30, color="#95a5a6", edgecolor="white")
    axes[1].axvline(real["sharpe"], color="#c0392b", lw=2.5, label="真实模型信号")
    axes[1].set_title("随机信号安慰剂检验：夏普比率")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_capacity(cap: pd.DataFrame, yearly: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    lbl = [f"{r.position_fraction:.1%}×{r.max_positions}" for r in cap.itertuples()]
    x = np.arange(len(cap))
    axes[0].bar(x, cap["total_return"] * 100, color="#2980b9")
    axes[0].set_xticks(x, lbl, rotation=30, fontsize=8)
    axes[0].set_title("仓位容量敏感性：累计收益 (%)")
    axes[0].grid(alpha=0.3)
    axes[1].bar(x, cap["sharpe"], color="#8e44ad")
    axes[1].set_xticks(x, lbl, rotation=30, fontsize=8)
    axes[1].set_title("仓位容量敏感性：夏普比率")
    axes[1].grid(alpha=0.3)
    y = np.arange(len(yearly))
    axes[2].bar(y - 0.2, yearly["total_return"] * 100, 0.4, label="年度收益(%)", color="#27ae60")
    axes[2].bar(y + 0.2, yearly["max_drawdown"] * 100, 0.4, label="最大回撤(%)", color="#c0392b")
    ax2 = axes[2].twinx()
    ax2.plot(y, yearly["sharpe"], "o--", color="#2c3e50", label="夏普")
    ax2.set_ylabel("夏普")
    axes[2].set_xticks(y, [f"{int(v)}年" for v in yearly["year"]])
    axes[2].set_title("分年度表现")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hold(hold: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(hold["max_hold_days"], hold["total_return"] * 100, "o-", color="#2980b9")
    axes[0].set_xlabel("最长持仓天数")
    axes[0].set_ylabel("累计收益 (%)")
    axes[0].set_title("持仓期敏感性：累计收益")
    axes[0].grid(alpha=0.3)
    ax = axes[1]
    ax.plot(hold["max_hold_days"], hold["sharpe"], "o-", color="#8e44ad", label="夏普")
    ax.plot(hold["max_hold_days"], hold["win_rate"], "s--", color="#27ae60", label="胜率")
    ax.plot(hold["max_hold_days"], hold["avg_trade_return"] * 10, "^--", color="#e67e22",
            label="单笔收益×10")
    ax.set_xlabel("最长持仓天数")
    ax.set_title("持仓期敏感性：质量指标")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> None:
    print("[load] 读取缓存...", flush=True)
    sig_all = pd.read_csv(OUT / "signals_2024_2025.csv", parse_dates=["date"])
    panel = Panel(pd.read_csv(OUT / "panel_ohlc.csv", parse_dates=["date"]))
    mkt = pd.read_csv(OUT / "market_daily.csv", parse_dates=["date"])

    calendar = np.array(sorted(mkt["date"].unique()), dtype="datetime64[ns]")
    calendar = calendar[(calendar >= np.datetime64(BACKTEST_START)) & (calendar <= np.datetime64(BACKTEST_END))]
    print(f"[load] 信号 {len(sig_all)} 条 / 交易日 {len(calendar)} 天", flush=True)

    cfg = DEFAULT
    buy, sell, meta = build_signal_maps(sig_all, cfg, calendar)

    print("[run] 主策略...", flush=True)
    daily, trades = run_backtest(panel, calendar, buy, sell, cfg, meta)
    main_metrics = performance(daily, trades, "主策略")

    print("[run] 无风控对照...", flush=True)
    cfg_nr = replace(cfg, stop_loss=None, take_profit=None)
    buy_nr, sell_nr, meta_nr = build_signal_maps(sig_all, cfg_nr, calendar)
    daily_nr, trades_nr = run_backtest(panel, calendar, buy_nr, sell_nr, cfg_nr, meta_nr)
    nr_metrics = performance(daily_nr, trades_nr, "无风控")

    print("[run] 即时执行对照 (exec_delay=0)...", flush=True)
    cfg_l0 = replace(cfg, exec_delay=0)
    buy_l0, sell_l0, meta_l0 = build_signal_maps(sig_all, cfg_l0, calendar)
    daily_l0, trades_l0 = run_backtest(panel, calendar, buy_l0, sell_l0, cfg_l0, meta_l0)
    l0_metrics = performance(daily_l0, trades_l0, "t日开盘执行")

    print("[run] 买入持有基准...", flush=True)
    mkt = mkt[(mkt["date"] >= BACKTEST_START) & (mkt["date"] <= BACKTEST_END)].sort_values("date")
    mkt_eq = (1 + mkt["mkt_return"]).cumprod()
    mkt_daily = pd.DataFrame({
        "date": mkt["date"].values, "equity": mkt_eq.values,
        "daily_return": mkt["mkt_return"].values,
    })
    mkt_daily["drawdown"] = mkt_daily["equity"] / mkt_daily["equity"].cummax() - 1
    mkt_daily["position_ratio"] = 1.0
    mkt_daily["n_positions"] = 0
    mkt_metrics = performance(mkt_daily, None, "全市场等权买入持有")

    print("[run] 全样本无条件持有基准...", flush=True)
    al = always_long_returns(panel, sig_all, calendar, cfg)

    print("[run] 随机信号安慰剂检验 (200 次)...", flush=True)
    mc = random_signal_mc(panel, sig_all, calendar, cfg, n_iter=200)
    mc_summary = {
        "total_return_median": float(mc["total_return"].median()),
        "total_return_p05": float(mc["total_return"].quantile(0.05)),
        "total_return_p95": float(mc["total_return"].quantile(0.95)),
        "sharpe_median": float(mc["sharpe"].median()),
        "sharpe_p95": float(mc["sharpe"].quantile(0.95)),
        "max_dd_median": float(mc["max_drawdown"].median()),
        "real_total_return_percentile": float((mc["total_return"] < main_metrics["total_return"]).mean()),
        "real_sharpe_percentile": float((mc["sharpe"] < main_metrics["sharpe"]).mean()),
    }

    print("[run] 阈值敏感性...", flush=True)
    th_rows = []
    for p in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        c = replace(cfg, p_buy=p)
        b, s, m = build_signal_maps(sig_all, c, calendar)
        d, t = run_backtest(panel, calendar, b, s, c, m)
        mm = performance(d, t, f"thr{p}")
        th_rows.append({"p_buy": p, "n_trades": mm.get("n_trades", 0),
                        "total_return": mm["total_return"], "cagr": mm["cagr"],
                        "sharpe": mm["sharpe"], "max_drawdown": mm["max_drawdown"],
                        "win_rate": mm.get("win_rate", np.nan),
                        "avg_position_ratio": mm["avg_position_ratio"]})
    th = pd.DataFrame(th_rows)

    print("[run] 止损/止盈网格...", flush=True)
    sltp_rows = []
    for sl in [None, 0.03, 0.05, 0.08, 0.10]:
        for tp in [None, 0.08, 0.15, 0.20, 0.30]:
            c = replace(cfg, stop_loss=sl, take_profit=tp)
            b, s, m = build_signal_maps(sig_all, c, calendar)
            d, t = run_backtest(panel, calendar, b, s, c, m)
            mm = performance(d, t, "sl/tp")
            sltp_rows.append({"stop_loss": -1 if sl is None else sl,
                              "take_profit": -1 if tp is None else tp,
                              "total_return": mm["total_return"], "sharpe": mm["sharpe"],
                              "max_drawdown": mm["max_drawdown"],
                              "win_rate": mm.get("win_rate", np.nan)})
    sltp = pd.DataFrame(sltp_rows)

    print("[run] 仓位容量敏感性...", flush=True)
    cap_rows = []
    for frac, mx in [(0.20, 5), (0.10, 10), (0.10, 20), (0.05, 20), (0.05, 30), (1 / 30, 30), (0.03, 40)]:
        c = replace(cfg, position_fraction=round(frac, 5), max_positions=mx)
        b, s, m = build_signal_maps(sig_all, c, calendar)
        d, t = run_backtest(panel, calendar, b, s, c, m)
        mm = performance(d, t, "capacity")
        cap_rows.append({"position_fraction": round(frac, 5), "max_positions": mx,
                         "max_gross": round(frac * mx, 3),
                         "n_trades": mm.get("n_trades", 0),
                         "total_return": mm["total_return"], "cagr": mm["cagr"],
                         "sharpe": mm["sharpe"], "max_drawdown": mm["max_drawdown"],
                         "avg_position_ratio": mm["avg_position_ratio"],
                         "win_rate": mm.get("win_rate", np.nan)})
    cap = pd.DataFrame(cap_rows)

    print("[run] 持仓期敏感性（标签视界为 50 日）...", flush=True)
    hold_rows = []
    for h in [5, 10, 20, 30, 40, 50]:
        c = replace(cfg, max_hold_days=h)
        b, s, m = build_signal_maps(sig_all, c, calendar)
        d, t = run_backtest(panel, calendar, b, s, c, m)
        mm = performance(d, t, f"hold{h}")
        hold_rows.append({"max_hold_days": h, "n_trades": mm.get("n_trades", 0),
                          "total_return": mm["total_return"], "cagr": mm["cagr"],
                          "sharpe": mm["sharpe"], "max_drawdown": mm["max_drawdown"],
                          "win_rate": mm.get("win_rate", np.nan),
                          "avg_bars_held": mm.get("avg_bars_held", np.nan),
                          "avg_trade_return": mm.get("avg_trade_return", np.nan)})
    hold = pd.DataFrame(hold_rows)

    print("[run] 分年度与分月统计...", flush=True)
    daily["year"] = daily["date"].dt.year
    yearly = []
    for y, g in daily.groupby("year"):
        ty = trades[trades["entry_date"].dt.year == y] if len(trades) else trades
        mm = segment_performance(g, ty if len(ty) else None, str(y))
        yearly.append({"year": int(y), "total_return": mm["total_return"], "sharpe": mm["sharpe"],
                       "max_drawdown": mm["max_drawdown"], "avg_position_ratio": mm["avg_position_ratio"],
                       "n_trades": mm.get("n_trades", 0), "win_rate": mm.get("win_rate", np.nan)})
    yearly = pd.DataFrame(yearly)
    mr = monthly_returns(daily)

    # ---------------- 保存 ----------------
    daily.to_csv(OUT / "daily_equity.csv", index=False)
    daily_nr.to_csv(OUT / "daily_equity_no_risk.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    th.to_csv(OUT / "sensitivity_threshold.csv", index=False)
    sltp.to_csv(OUT / "sensitivity_stop_takeprofit.csv", index=False)
    cap.to_csv(OUT / "sensitivity_capacity.csv", index=False)
    hold.to_csv(OUT / "sensitivity_hold_days.csv", index=False)
    mc.to_csv(OUT / "monte_carlo_random_signal.csv", index=False)
    mr.to_csv(OUT / "monthly_returns.csv", index=False)
    yearly.to_csv(OUT / "yearly_performance.csv", index=False)
    exit_reason = trades["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    exit_reason.to_csv(OUT / "exit_reasons.csv", index=False)
    hit_rows = []
    for cls in [0, 1, 2]:
        g = sig_all[sig_all["pred"] == cls]
        hit_rows.append({"pred": cls, "n": int(len(g)),
                         "hit_rate": float((g["label"] == cls).mean()) if len(g) else np.nan,
                         "hit_pull": float((g["label"] == 2).mean()) if len(g) else np.nan,
                         "hit_fall": float((g["label"] == 0).mean()) if len(g) else np.nan})
    hit = pd.DataFrame(hit_rows)
    hit.to_csv(OUT / "signal_hit_rate.csv", index=False)

    bundle = {
        "config": asdict(cfg),
        "backtest_window": [BACKTEST_START, BACKTEST_END],
        "trading_days": int(len(calendar)),
        "n_signals": int(len(sig_all)),
        "main": main_metrics,
        "no_risk": nr_metrics,
        "exec_delay_0": l0_metrics,
        "market_buy_hold": mkt_metrics,
        "always_long_20d": al,
        "random_signal": mc_summary,
        "capacity": cap.to_dict(orient="records"),
        "hold_days": hold.to_dict(orient="records"),
        "yearly": yearly.to_dict(orient="records"),
        "exit_reasons": exit_reason.to_dict(orient="records"),
        "signal_hit_rate": hit.to_dict(orient="records"),
    }
    (OUT / "metrics.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 绘图 ----------------
    print("[plot] 生成图表...", flush=True)
    plot_equity_curve({
        "主策略": daily.set_index("date")["equity"],
        "无风控对照": daily_nr.set_index("date")["equity"],
        "全市场等权买入持有": mkt_daily.set_index("date")["equity"],
    }, OUT / "fig01_equity_curve.png", "2024-2025 策略净值曲线（真实价格、真实交易日）")
    plot_drawdown(daily, OUT / "fig02_drawdown.png")
    plot_monthly_heatmap(mr, OUT / "fig03_monthly_heatmap.png")
    plot_trades(trades, OUT / "fig04_trades.png")
    plot_signal_quality(sig_all, OUT / "fig05_signal_quality.png")
    plot_sensitivity(th, sltp, OUT / "fig06_sensitivity.png")
    plot_mc(mc, main_metrics, OUT / "fig07_random_signal_mc.png")
    plot_capacity(cap, yearly, OUT / "fig08_capacity_and_yearly.png")
    plot_hold(hold, OUT / "fig09_hold_sensitivity.png")

    print(json.dumps(bundle, ensure_ascii=False, indent=2), flush=True)
    print("输出目录:", OUT, flush=True)


if __name__ == "__main__":
    main()
