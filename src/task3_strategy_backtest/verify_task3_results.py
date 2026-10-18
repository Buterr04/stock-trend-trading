"""独立校验任务三结果（不依赖任务三脚本自身的断言）"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "task3_2024_2025_results"
ok = lambda b: "PASS" if b else "FAIL"

print("=" * 70)
print("[A] 账目恒等式：sum(trades.pnl) == 期末净值-1（期末应全为现金）")
tr = pd.read_csv(OUT / "trades.csv", parse_dates=["entry_date", "exit_date"])
daily = pd.read_csv(OUT / "daily_equity.csv", parse_dates=["date"])
pnl_sum = tr["pnl"].sum()
eq_end = daily["equity"].iloc[-1] - 1.0
print(f"  sum(pnl)={pnl_sum:.6f}  equity-1={eq_end:.6f}  diff={abs(pnl_sum-eq_end):.2e}  {ok(abs(pnl_sum-eq_end)<1e-6)}")
print(f"  期末持仓数={int(daily['n_positions'].iloc[-1])}（应为0）  交易数={len(tr)}")

print("=" * 70)
print("[B] 绩效指标独立复算（244 交易日年化）")
ret = daily["daily_return"].to_numpy()
n = len(daily); years = n / 244
eq = daily["equity"].to_numpy()
cagr = eq[-1] ** (1 / years) - 1
sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(244)
mdd = (eq / np.maximum.accumulate(eq) - 1).min()
mj = json.loads((OUT / "metrics.json").read_text())["main"]
print(f"  total_return: 复算={eq[-1]-1:.4f} 记录={mj['total_return']:.4f} {ok(abs(eq[-1]-1-mj['total_return'])<1e-9)}")
print(f"  cagr:         复算={cagr:.4f} 记录={mj['cagr']:.4f} {ok(abs(cagr-mj['cagr'])<1e-9)}")
print(f"  sharpe:       复算={sharpe:.4f} 记录={mj['sharpe']:.4f} {ok(abs(sharpe-mj['sharpe'])<1e-9)}")
print(f"  max_dd:       复算={mdd:.4f} 记录={mj['max_drawdown']:.4f} {ok(abs(mdd-mj['max_drawdown'])<1e-9)}")

print("=" * 70)
print("[C] 抽样手工复算 3 笔交易（止损/止盈/到期各一）")
panel = pd.read_csv(OUT / "panel_ohlc.csv", parse_dates=["date"])
prow = {(r.code, r.date): (r.open, r.high, r.low, r.close) for r in panel.itertuples()}
for reason in ["stop_loss", "take_profit", "max_hold"]:
    t = tr[tr["reason"] == reason].iloc[0]
    ep, xp = t["entry_price"], t["exit_price"]
    o0 = prow[(t["code"], t["entry_date"])][0]
    line = f"  {t['code']} {reason}: 入场价={ep} vs 面板开盘={o0} {ok(abs(ep-o0)<1e-9)}"
    if reason == "stop_loss":
        stop = ep * 0.95
        od = prow[(t["code"], t["exit_date"])]
        line += f" | 止损价={stop:.4f} 当日low={od[2]} open={od[0]} 成交={xp} 理论={min(od[0],stop):.4f} {ok(abs(xp-min(od[0],stop))<1e-9 and od[2]<=stop)}"
    elif reason == "take_profit":
        take = ep * 1.15
        od = prow[(t["code"], t["exit_date"])]
        line += f" | 止盈价={take:.4f} 当日high={od[1]} open={od[0]} 成交={xp} 理论={max(od[0],take):.4f} {ok(abs(xp-max(od[0],take))<1e-9 and od[1]>=take)}"
    else:
        c = prow[(t["code"], t["exit_date"])][3]
        line += f" | 到期按收盘平仓 收盘={c} 成交={xp} {ok(abs(xp-c)<1e-9)} | bars={t['bars_held']}(应=20) {ok(t['bars_held']==20)}"
    print(line)

print("=" * 70)
print("[D] 信号平仓为何 0 次？统计'持仓期间出现同票 pred=0 信号'的交易数")
sig = pd.read_csv(OUT / "signals_2024_2025.csv", parse_dates=["date"])
mkt = pd.read_csv(OUT / "market_daily.csv", parse_dates=["date"])
cal = list(pd.to_datetime(sorted(mkt["date"].unique())))
pos_in_cal = {d: i for i, d in enumerate(cal)}
sells = sig[sig["pred"] == 0]
sell_exec = {}
for r in sells.itertuples():
    i = pos_in_cal.get(r.date)
    if i is not None and i + 1 < len(cal):
        sell_exec.setdefault(r.code, []).append(cal[i + 1])
n_overlap = 0
for t in tr.itertuples():
    for ed in sell_exec.get(t.code, []):
        if t.entry_date < ed <= t.exit_date:
            n_overlap += 1
            break
print(f"  持仓窗口内存在同票下跌信号(次日开盘可执行)的交易数 = {n_overlap}")
print(f"  记录中 signal_fall 平仓次数 = {(tr['reason']=='signal_fall').sum()}")
print(f"  => {'一致：确实没有可触发的信号平仓' if n_overlap==0 else '不一致！引擎可能漏卖了 '+str(n_overlap)+' 笔'}")

print("=" * 70)
print("[E] 信号→成交时点：入场日应 = 信号日+1 个交易日")
buy_sig = sig[(sig["pred"] == 2) & (sig["p2"] >= 0.5)]
sig_day = {}
for r in buy_sig.itertuples():
    i = pos_in_cal.get(r.date)
    if i is not None and i + 1 < len(cal):
        sig_day[(r.code, cal[i + 1])] = r.p2
hit = sum(1 for t in tr.itertuples() if (t.code, t.entry_date) in sig_day)
print(f"  {len(tr)} 笔交易中入场日=信号日次日开盘的笔数 = {hit}  {ok(hit==len(tr))}")

print("=" * 70)
print("[F] npy↔(code,date) 映射的独立特征级验证（最强检验）")
print("  对 300 个随机测试样本：取源数据该票 t-1 日前 60 日 pct_chg 序列，")
print("  与 X_test[i] 全部 33 通道做 Spearman 相关，正确映射应存在 |rho|≈1 的通道；")
print("  日期平移 +5 日作为对照组，相关应崩溃。")
from scipy.stats import spearmanr
df = pd.read_csv(ROOT / "test.csv", parse_dates=["date"])
perm = np.argsort(df["label"].values, kind="stable")
y_test = np.load(ROOT / "y_test.npy")
assert np.array_equal(y_test, df["label"].values[perm]), "标签对齐断言失败"
X = np.load(ROOT / "X_test.npy", mmap_mode="r")
use = ["code", "date", "pct_chg"]
src = pd.read_csv(ROOT / "original_data" / "all_2016_2025_with_indicators.csv",
                  usecols=use, dtype={"pct_chg": np.float64})
src["date"] = pd.to_datetime(src["date"])
series = {c: (g["date"].to_numpy(), g["pct_chg"].to_numpy())
          for c, g in src.groupby("code", sort=False)}
rng = np.random.default_rng(7)
idx = rng.choice(len(df), 300, replace=False)

def best_corr(i, shift):
    r = df.iloc[perm[i]]
    s = series.get(r["code"])
    if s is None:
        return np.nan, -1
    dates, pct = s
    j = int(np.searchsorted(dates, np.datetime64(r["date"])))
    if j >= len(dates) or dates[j] != np.datetime64(r["date"]):
        return np.nan, -1
    win = pct[j - 60 + shift: j + shift]
    if len(win) < 60:
        return np.nan, -1
    xi = np.asarray(X[i])
    best, bc = 0.0, -1
    for c in range(33):
        v = xi[:, c]
        if np.isnan(v).any() or np.isnan(win).any():
            m = ~(np.isnan(v) | np.isnan(win))
            if m.sum() < 30:
                continue
            rho = spearmanr(v[m], win[m]).statistic
        else:
            rho = spearmanr(v, win).statistic
        if np.isfinite(rho) and abs(rho) > abs(best):
            best, bc = rho, c
    return best, bc

good = 0; ctrl_good = 0; chans = {}
for k, i in enumerate(idx):
    rho, c = best_corr(int(i), 0)
    rho5, _ = best_corr(int(i), 5)
    if np.isfinite(rho) and abs(rho) > 0.99:
        good += 1
        chans[c] = chans.get(c, 0) + 1
    if np.isfinite(rho5) and abs(rho5) > 0.99:
        ctrl_good += 1
print(f"  正确日期：|rho|>0.99 的样本 {good}/300  命中通道分布={chans}")
print(f"  平移+5日：|rho|>0.99 的样本 {ctrl_good}/300（应接近 0）")
print(f"  => {ok(good>=295 and ctrl_good<=5)}  X_test 行与 (code,date) 的映射在特征层面成立")

print("=" * 70)
print("[G] 命中率先验复核 + 测试集时间分布")
for cls in [0, 1, 2]:
    g = sig[sig["pred"] == cls]
    print(f"  pred={cls}: n={len(g)} 真实下跌占比={ (g['label']==0).mean():.4f} 真实拉伸占比={(g['label']==2).mean():.4f}")
sig_full = pd.read_csv(OUT / "signals_full.csv", parse_dates=["date"])
by_year = sig_full["date"].dt.year.value_counts().sort_index()
print("  全部测试样本按年分布:", by_year.to_dict())
print("  （测试集横跨 2016-2025 => 任务二划分是随机划分而非时间外推，报告应披露）")
