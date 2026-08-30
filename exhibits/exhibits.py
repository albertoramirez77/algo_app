"""
exhibits.py — the complete exhibit page, in one file.

    python exhibits.py --prices px_clean.parquet

Produces EXHIBITS.pdf and EXHIBITS.png: six panels on one portrait page, sized to drop
straight into the pitch as its final page.

    1  hedge quality by economic proximity      the thesis, in one image
    2  equity curve with drawdown               what the strategy did, including now
    3  placebo distribution                     signal separated from machinery
    4  parameter surface                        a plateau, not a spike
    5  trade statistics                         the fund's own reporting vocabulary
    6  bootstrap drawdown distribution          what could have happened, not what did

Panels 1 to 4 describe the research. Panels 5 and 6 describe the same result in the
vocabulary the fund's own strategy documents use - trade counts, win rates, profit factors,
and simulated drawdown paths. Nothing in 5 or 6 is a new claim.

WHICH BOOK EACH PANEL USES

Panels 2, 5 and 6 use the TRANCHED book marked daily, which is the specification being
pitched. Panels 3 and 4 use the single-grid monthly book, because a placebo and a parameter
sweep are properties of the SIGNAL rather than of the rebalancing schedule, and running
either across twenty-one grids would multiply the compute for no additional information.
Panel 1 is computed from daily returns directly and involves no portfolio at all.

THE BOOTSTRAP IS BLOCKED, NOT RESHUFFLED

Reordering months independently would destroy the clustering of losses and make any
strategy look safer than it is. Six-month blocks are resampled with replacement so runs of
consecutive losses survive into the simulated paths.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL, VOL_TARGET, IDM, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6
N_GRIDS, BLOCK, N_PATHS = 21, 6, 5000
COST_MULTIPLE = 3.0
CHAINS = {"ZS": ["ZM", "ZL"], "MCL": ["HO", "RB"]}

INK, MUTE, ACC, NEG, GRN = "#1a1a1a", "#8f8f8f", "#c1440e", "#2b6a8f", "#5d8a3a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------

def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df = df[df["asset"] == "commodity"].copy()
    # Universe rule, applied before anything else and computed from contract
    # specifications alone: exclude any instrument whose ex-ante round-trip cost exceeds
    # three times the universe median. On this data that is E-mini natural gas, whose
    # $0.005 tick was set when gas traded above $10 and was never rescaled.
    med_px = df.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med_px.index:
        inst = BY_SYMBOL[s]
        notional = med_px[s] * inst.dollar_price_mult
        tick_bp = inst.tick_value / notional * 1e4
        cost[s] = 1.5 * tick_bp + inst.commission / notional * 1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    if drop:
        print(f"  universe rule excludes {sorted(drop)} "
              f"(cost > {COST_MULTIPLE:.0f}x median of {cs.median():.2f}bp)")
        df = df[~df["symbol"].isin(drop)].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
               - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


# ----------------------------------------------------------------------------------
# books
# ----------------------------------------------------------------------------------

def monthly_book(m: pd.DataFrame, bps=3.0, seed=None, J_=None, vw=None, min_n=6):
    if J_ or vw:
        m = m.copy(); g = m.groupby("symbol")
        if J_:
            m["bm"] = (g["r0"].transform(lambda s: s.rolling(J_, min_periods=J_).sum())
                       - g["r1"].transform(lambda s: s.rolling(J_, min_periods=J_).sum()))
        if vw:
            m["vol"] = (g["r0"].transform(
                lambda s: s.rolling(vw, min_periods=3).std()) * np.sqrt(12)
                ).groupby(m["symbol"]).shift(1)
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        sv = s["bm"]
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        r = sv.rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0; held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]; dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * IDM / den))
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index()


def grid_targets(df: pd.DataFrame, offset: int, min_n: int = 6) -> pd.DataFrame:
    d = df.sort_values(["symbol", "date"]).copy()
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"] == offset][["symbol", "ym", "date", "c0", "c1", "settle_0"]].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"] = g["c0"].diff(); snap["r1"] = g["c1"].diff()
    snap["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
                  - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    snap["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(snap["symbol"]).shift(1)
    snap["px_entry"] = g["settle_0"].shift(1)
    rows = []
    for dt, gg in snap.groupby("date"):
        s = gg[["symbol", "bm", "vol", "px_entry"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px in zip(s["symbol"], w, s["vol"], s["px_entry"]):
            inst = BY_SYMBOL[sym]
            den = inst.dollar_price_mult * px * vol
            if den > 0:
                rows.append(dict(date=dt, symbol=sym,
                                 target=wi * CAPITAL * VOL_TARGET * IDM / den))
    return pd.DataFrame(rows)


def tranched_book(df: pd.DataFrame, frames: list[pd.DataFrame], bps: float = 3.0):
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    syms = sorted(df["symbol"].unique())
    ret = df.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = df.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = []
    for tf in frames:
        if tf.empty:
            continue
        stacks.append((tf.pivot_table(index="date", columns="symbol", values="target")
                         .reindex(index=dates, columns=syms).ffill()).to_numpy())
    S = np.stack(stacks, axis=0)
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    N = np.round(T)
    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    P = np.nan_to_num(px.to_numpy(), nan=0.0)
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)
    held = N[:-1]
    pnl = np.nansum(held * dpm * P[:-1] * np.expm1(R[1:]), axis=1)
    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * P[:-1] * bps / 1e4), axis=1)
    return (pd.Series((pnl - cost) / CAPITAL, index=dates[1:]),
            pd.Series(pnl / CAPITAL, index=dates[1:]),
            dict(in_market=float((np.abs(held).sum(axis=1) > 0).mean()),
                 trades_per_month=float(trades.sum(axis=1).mean() * 21)))


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------

def sharpe(r):
    r = r.dropna()
    if len(r) < 36:
        return np.nan
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


def max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 250 or y.var() <= 0:
        return np.nan
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    return float(1.0 - (y - A @ b).var() / y.var())


def proximity(df: pd.DataFrame) -> dict:
    p0 = df.pivot_table(index="date", columns="symbol", values="r0").sort_index()
    p1 = df.pivot_table(index="date", columns="symbol", values="r1").sort_index()
    idx = p0.index.union(p1.index)
    p0, p1 = p0.reindex(idx), p1.reindex(idx)
    syms = [s for s in p0.columns if p0[s].notna().sum() > 500]
    curve, peer, chain, mkt, pcs, wins = [], [], [], [], [], 0
    for s in syms:
        y = p0[s].to_numpy()
        c = r2(y, p1[s].to_numpy().reshape(-1, 1)) if s in p1.columns else np.nan
        others = [o for o in syms if o != s]
        best = max((r2(y, p0[o].to_numpy().reshape(-1, 1)) for o in others), default=np.nan)
        if np.isfinite(c):
            curve.append(c)
        if np.isfinite(best):
            peer.append(best)
        if np.isfinite(c) and np.isfinite(best) and c > best:
            wins += 1
        mv = r2(y, p0[others].mean(axis=1).to_numpy().reshape(-1, 1))
        if np.isfinite(mv):
            mkt.append(mv)
        A = p0[others].fillna(0.0).to_numpy()
        Ac = A - A.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(Ac, full_matrices=False)
            v = r2(y, (U * S)[:, :min(8, U.shape[1])])
            if np.isfinite(v):
                pcs.append(v)
        except np.linalg.LinAlgError:
            pass
        if s in CHAINS:
            legs = [l for l in CHAINS[s] if l in p0.columns]
            if legs:
                v = r2(y, p0[legs].to_numpy())
                if np.isfinite(v):
                    chain.append((v, len(legs)))
    return dict(curve=np.mean(curve) if curve else np.nan,
                peer=np.mean(peer) if peer else np.nan,
                market=np.mean(mkt) if mkt else np.nan,
                pca8=np.mean(pcs) if pcs else np.nan,
                chain=np.mean([c for c, _ in chain]) if chain else np.nan,
                chain_n=np.mean([n for _, n in chain]) if chain else 2.0,
                wins=wins, n=len(syms))


# ----------------------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------------------

def panel_proximity(ax, px):
    tiers = [
        ("Deferred contract", px["curve"], 1.0, ACC),
        ("Crush / crack products", px["chain"], px["chain_n"], GRN),
        ("Best peer (hindsight)", px["peer"], 1.0, MUTE),
        ("8 principal components", px["pca8"], 8.0, MUTE),
        ("Equal-weighted market", px["market"], 1.0, MUTE),
    ]
    tiers = [t for t in tiers if np.isfinite(t[1])]
    tiers.sort(key=lambda t: t[1] / max(t[2], 1))
    y = np.arange(len(tiers))
    vals = [t[1] / max(t[2], 1) for t in tiers]
    ax.barh(y, vals, color=[t[3] for t in tiers], height=0.58, zorder=3,
            edgecolor=INK, linewidth=0.4)
    for i, t in enumerate(tiers):
        lab = f"R\u00b2 {t[1]:.2f}" + (f" / {t[2]:.0f}" if t[2] > 1 else "")
        ax.text(vals[i] + max(vals) * 0.025, i, lab, va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in tiers], fontsize=6.5)
    ax.set_xlabel("common variance removed, per regressor", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.34)
    ax.set_title("1 \u00b7 Hedge quality tracks economic proximity",
                 fontsize=8.2, loc="left", weight="bold", pad=9)


def panel_equity(ax, r):
    eq = (1 + r).cumprod()
    x = eq.index
    dd = (eq / eq.cummax() - 1) * 100
    ax2 = ax.twinx()
    ax2.fill_between(x, dd.to_numpy(), 0, color=NEG, alpha=0.17, lw=0, zorder=1)
    ax2.set_ylim(dd.min() * 2.7, 0)
    ax2.set_ylabel("drawdown (%)", fontsize=7, color=NEG)
    ax2.tick_params(axis="y", colors=NEG, labelsize=6.4)
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(NEG)
    ax2.spines["top"].set_visible(False)
    ax.plot(x, eq.to_numpy(), color=ACC, lw=1.25, zorder=3)
    ax.set_ylabel("growth of $1, net", fontsize=7)
    ax.axhline(1.0, color=MUTE, lw=0.5, ls=":", zorder=2)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title("2 \u00b7 Equity curve and drawdown", fontsize=8.2, loc="left",
                 weight="bold", pad=9)


def panel_placebo(ax, real_t, pt):
    lo, hi = min(pt.min(), real_t) - 0.5, max(pt.max(), real_t) + 0.5
    ax.hist(pt, bins=np.linspace(lo, hi, 20), color=MUTE, alpha=0.75,
            edgecolor="white", lw=0.4)
    ax.axvline(real_t, color=ACC, lw=1.7)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("t-statistic", fontsize=7)
    ax.set_ylabel("shuffles", fontsize=7)
    ax.set_title("3 \u00b7 Placebo: signal, not machinery", fontsize=8.2, loc="left",
                 weight="bold", pad=9)
    ax.annotate(f"real  {real_t:+.2f}", xy=(real_t, ax.get_ylim()[1] * 0.62),
                xytext=(-58, 0), textcoords="offset points", fontsize=6.6, color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=0.7))


def panel_grid(ax, grid):
    ks = sorted({k for k, _ in grid}); vws = sorted({v for _, v in grid})
    Z = np.array([[grid[(k, v)] for v in vws] for k in ks])
    ax.imshow(Z, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(vws))); ax.set_xticklabels(vws, fontsize=6.6)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=6.6)
    ax.set_xlabel("volatility lookback (months)", fontsize=7)
    ax.set_ylabel("formation window (months)", fontsize=7)
    for i in range(len(ks)):
        for j in range(len(vws)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.3, color=INK)
    ax.set_title("4 \u00b7 Parameter surface is a plateau", fontsize=8.2, loc="left",
                 weight="bold", pad=9)


def panel_stats(ax, stats):
    ax.axis("off")
    ax.set_title("5 \u00b7 Trade statistics, monthly", fontsize=8.2, loc="left",
                 weight="bold", pad=9)
    tbl = ax.table(cellText=[[k, v] for k, v in stats], colWidths=[0.63, 0.37],
                   loc="upper center", cellLoc="left", bbox=[0.0, 0.0, 1.0, 0.97])
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.6)
    bold = {"Sharpe ratio, net", "Profit factor", "Maximum drawdown"}
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#d4d4d4"); c.set_linewidth(0.4)
        c.PAD = 0.035
        if j == 1:
            c.get_text().set_ha("right")
        if i % 2 == 1:
            c.set_facecolor("#f5f5f5")
        if stats[i][0] in bold:
            c.get_text().set_weight("bold")


def panel_mc(ax, dds, realised, pct):
    ax.hist(dds * 100, bins=50, color=MUTE, alpha=0.75, edgecolor="white", lw=0.35)
    p5 = np.percentile(dds, 5) * 100
    ax.axvline(realised * 100, color=ACC, lw=1.7)
    ax.axvline(p5, color=NEG, lw=1.0, ls="--")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.52)
    ax.annotate(f"realised {realised*100:.1f}%", xy=(realised * 100, ymax * 0.42),
                xytext=(16, -4), textcoords="offset points", fontsize=6.5, color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=0.7))
    ax.annotate(f"5th pct {p5:.1f}%", xy=(p5, ymax * 0.30), xytext=(-56, 6),
                textcoords="offset points", fontsize=6.5, color=NEG,
                arrowprops=dict(arrowstyle="->", color=NEG, lw=0.7))
    ax.text(0.02, 0.985,
            f"{N_PATHS:,} paths, {BLOCK}-month blocks\n"
            f"median {np.median(dds)*100:.1f}%,  worst {dds.min()*100:.1f}%\n"
            f"realised worse than {pct:.0%} of paths",
            transform=ax.transAxes, fontsize=6.2, color=INK, va="top", linespacing=1.55,
            bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="none", alpha=0.88))
    ax.set_xlabel("maximum drawdown (%)", fontsize=7)
    ax.set_ylabel("simulated paths", fontsize=7)
    ax.set_title("6 \u00b7 Drawdowns that could have occurred", fontsize=8.2,
                 loc="left", weight="bold", pad=9)


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="EXHIBITS")
    a = ap.parse_args()

    df = load_daily(a.prices)
    m = monthly(df)
    print(f"  {m['symbol'].nunique()} commodities, {m['ym'].nunique()} months")

    print("  building tranched book...")
    frames = [f for f in (grid_targets(df, o) for o in range(N_GRIDS)) if not f.empty]
    net_d, gross_d, aux = tranched_book(df, frames)
    net = net_d.resample("ME").sum(); net = net[net != 0]
    gross = gross_d.resample("ME").sum().reindex(net.index)

    print("  placebo...")
    base_m = monthly_book(m)
    yrs_m = len(base_m) / 12
    real_t = sharpe(base_m) * np.sqrt(yrs_m)
    pt = np.array([v for v in (sharpe(monthly_book(m, seed=s)) * np.sqrt(yrs_m)
                               for s in range(a.seeds)) if np.isfinite(v)])

    print("  parameter surface...")
    grid = {(k, vw): sharpe(monthly_book(m, J_=k, vw=vw))
            for k in (6, 9, 12, 15) for vw in (3, 6, 12)}

    print("  hedge proximity...")
    px_prox = proximity(df)

    print("  bootstrapping drawdowns...")
    x = net.to_numpy(); n = len(x)
    realised = max_dd(x)
    rng = np.random.default_rng(0)
    nb = int(np.ceil(n / BLOCK))
    st = rng.integers(0, n - BLOCK + 1, size=(N_PATHS, nb))
    idx = (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_PATHS, -1)[:, :n]
    dds = np.array([max_dd(p) for p in x[idx]])
    pct = float((dds < realised).mean())

    wins, losses = net[net > 0], net[net < 0]
    ann, vol = net.mean() * 12, net.std(ddof=1) * np.sqrt(12)
    eq = (1 + net).cumprod(); under = (eq < eq.cummax() * (1 - 1e-12)).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    stats = [
        ("Rebalances (months)", f"{len(net)}"),
        ("Winning months", f"{(net > 0).mean():.0%}"),
        ("Average winning month", f"{wins.mean()*100:+.2f}%"),
        ("Average losing month", f"{losses.mean()*100:+.2f}%"),
        ("Win / loss ratio", f"{wins.mean()/abs(losses.mean()):.2f}"),
        ("Profit factor", f"{wins.sum()/abs(losses.sum()):.2f}"),
        ("Annual return, net", f"{ann*100:+.2f}%"),
        ("Cost / gross profit", f"{(gross.sum()-net.sum())/gross.sum():.1%}"),
        ("Annualised volatility", f"{vol*100:.1f}%"),
        ("Sharpe ratio, net", f"{ann/vol:.2f}"),
        ("Maximum drawdown", f"{realised*100:.1f}%"),
        ("Longest drawdown (mo)", f"{longest}"),
        ("Best / worst month", f"{net.max()*100:+.1f}% / {net.min()*100:+.1f}%"),
        ("Contracts / month", f"{aux['trades_per_month']:.0f}"),
        ("Days holding a position", f"{aux['in_market']:.0%}"),
    ]

    fig = plt.figure(figsize=(8.5, 11.0))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.22],
                  hspace=0.46, wspace=0.30,
                  left=0.165, right=0.945, top=0.900, bottom=0.115)
    panel_proximity(fig.add_subplot(gs[0, 0]), px_prox)
    panel_equity(fig.add_subplot(gs[0, 1]), net)
    panel_placebo(fig.add_subplot(gs[1, 0]), real_t, pt)
    panel_grid(fig.add_subplot(gs[1, 1]), grid)
    panel_stats(fig.add_subplot(gs[2, 0]), stats)
    panel_mc(fig.add_subplot(gs[2, 1]), dds, realised, pct)

    fig.suptitle("The Same Barrel \u2014 Supporting Exhibits", fontsize=11,
                 weight="bold", x=0.055, ha="left", y=0.962)
    fig.text(0.055, 0.940,
             f"{m['symbol'].nunique()} CME commodity futures, {m['ym'].min()} to "
             f"{m['ym'].max()}, {len(net)} monthly observations, net of 3bp per side",
             fontsize=7.2, color=MUTE, ha="left")
    fig.text(0.055, 0.038,
             "Panels 2, 5 and 6 use the tranched book marked daily. Panels 3 and 4 use the "
             "single-grid monthly book, since a placebo and a\nparameter sweep are properties "
             "of the signal rather than of the rebalancing schedule; panel 1 is computed from "
             "daily returns and\ninvolves no portfolio. Panel 6 resamples six-month blocks with "
             "replacement so runs of consecutive losses survive into the paths.",
             fontsize=6.1, color=MUTE, linespacing=1.55)

    fig.savefig(f"{a.out}.pdf"); fig.savefig(f"{a.out}.png", dpi=200)
    print(f"\n  -> {a.out}.pdf and {a.out}.png\n")
    for k, v in stats:
        print(f"    {k:26s} {v:>18s}")
    print(f"\n    {'placebo separation':26s} "
          f"{(real_t-pt.mean())/max(pt.std(ddof=1),1e-9):>+17.1f} sd")
    print(f"    {'grid cells above 0.35':26s} "
          f"{sum(1 for v in grid.values() if np.isfinite(v) and v>0.35):>14d} of {len(grid)}")
    print(f"    {'curve beats peer in':26s} {px_prox['wins']:>14d} of {px_prox['n']}")
    print(f"    {'bootstrap 5th percentile':26s} {np.percentile(dds,5)*100:>17.1f}%")


if __name__ == "__main__":
    main()