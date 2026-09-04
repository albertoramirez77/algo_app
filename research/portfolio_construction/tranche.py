"""
tranche.py — the rebalance date is arbitrary. Averaging over all of them removes noise.

    python tranche.py --prices data/px_clean.parquet

THE OBSERVATION

The strategy rebalances on one trading day of the month. There are roughly 21 such days, and
each produces a different return series from the same signal, the same universe and the same
sizing. One of them was chosen and its Sharpe reported.

That means two separate things, and both matter.

    NOBODY KNOWS HOW LUCKY THE CHOSEN GRID WAS. If the 21 grids run from 0.3 to 0.9 and the
    reported one is 0.586, the result is mid-pack and honest. If they run 0.1 to 0.6, then
    shifting the rebalance date by three days destroys the strategy — and that is a question
    a portfolio manager will ask.

    THE VARIATION ACROSS GRIDS IS PURE NOISE. It carries no information about the signal.
    Averaging it away raises the Sharpe ratio with no new data, no new parameter, and no
    search over specifications.

THE ARITHMETIC

Averaging K series with mean pairwise correlation rho leaves the mean return unchanged and
scales volatility by sqrt((1 + (K-1)rho) / K). At rho = 0.7 and K = 21 that is a 15%
reduction in volatility, which is a 15% increase in Sharpe for free.

THE IMPLEMENTATION DETAIL THAT MAKES IT WORK AT $450,000

Twenty-one separate books would each hold $21,000 and round to zero constantly. So the
tranches are combined BEFORE rounding: compute 21 target position vectors, average them,
and round once at full size. The averaged target moves more smoothly than any single grid,
so turnover FALLS and integer granularity IMPROVES. This is the opposite of what running
parallel books would do.

WHY THIS IS NOT DATA MINING

Every refinement tested in this project was an empirical claim about markets, and eleven of
twelve failed. This is arithmetic. The variance of an average of correlated series is a
theorem, not a hypothesis. The only empirical question is how correlated the grids are, and
that is measured below rather than assumed.

Real funds call this tranching, or overlapping portfolios. It is standard practice and
rarely done in student work.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL, VOL_TARGET, IDM, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6
N_GRIDS = 21          # trading days in a month; one grid per possible rebalance day


def load_daily(path: str) -> pd.DataFrame:
    """Daily panel, returns chained strictly within each contract's own life."""
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
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    return df


def grid_panel(df: pd.DataFrame, offset: int) -> pd.DataFrame:
    """
    Build the monthly panel for one rebalance grid.

    `offset` is the trading day within the month on which the book is struck. Offset 0 is
    the first trading day, which is the current specification. Every grid uses identical
    signal, sizing and cost rules; only the observation dates differ.
    """
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    d["dom"] = d.groupby(["symbol", "ym"]).cumcount()
    # the strike date for this grid, per instrument-month
    marks = d[d["dom"] == offset][["symbol", "ym", "date"]].rename(
        columns={"date": "mark"})
    if marks.empty:
        return pd.DataFrame()

    # cumulative log price from the start, so any two marks give the return between them
    d = d.sort_values(["symbol", "date"])
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d.merge(marks, left_on=["symbol", "ym", "date"],
                   right_on=["symbol", "ym", "mark"], how="inner")
    snap = snap[["symbol", "ym", "date", "c0", "c1", "settle_0", "settle_1"]]
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")

    # period returns between consecutive marks
    snap["r0"] = g["c0"].diff()
    snap["r1"] = g["c1"].diff()
    snap["mom0"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    snap["mom1"] = g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    snap["bm"] = snap["mom0"] - snap["mom1"]
    snap["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(snap["symbol"]).shift(1)
    snap["px_entry"] = g["settle_0"].shift(1)
    snap["fwd"] = g["r0"].shift(-1)
    return snap


def targets(panel: pd.DataFrame, min_n: int = 6) -> pd.DataFrame:
    """
    FRACTIONAL target contracts per instrument-month, before rounding. Tranches must be
    averaged before rounding, so rounding happens once at full size rather than 21 times
    at a twenty-first of the size.
    """
    rows = []
    for ym, g in panel.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            den = inst.dollar_price_mult * px * vol
            if den <= 0:
                continue
            rows.append(dict(ym=ym, symbol=sym,
                             target=wi * CAPITAL * VOL_TARGET * IDM / den,
                             px=px, fwd=fwd))
    return pd.DataFrame(rows)


def pnl_from_targets(tg: pd.DataFrame, bps: float = 3.0, integer: bool = True):
    """Round (or not), mark, and charge costs. One book, one rounding step."""
    prev, out, turn, zero, tot = {}, {}, [], 0, 0
    for ym, g in tg.groupby("ym"):
        pnl = cost = 0.0
        held = {}
        for _, row in g.iterrows():
            sym = row["symbol"]
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            n = float(np.round(row["target"])) if integer else float(row["target"])
            tot += 1
            if n == 0 and abs(row["target"]) > 1e-9:
                zero += 1
            held[sym] = n
            if np.isfinite(row["fwd"]):
                pnl += n * dpm * row["px"] * (np.exp(row["fwd"]) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * row["px"] * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        turn.append(sum(abs(held.get(k, 0) - prev.get(k, 0)) for k in set(held) | set(prev)))
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return (pd.Series(out).sort_index(),
            dict(zero_share=zero / max(tot, 1), turn=float(np.mean(turn[1:]))
                 if len(turn) > 1 else np.nan))


def st(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 48:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), yrs=yrs, sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12,
                vol=av, dd=float((eq / eq.cummax() - 1).min()))


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--grids", type=int, default=N_GRIDS)
    a = ap.parse_args()

    df = load_daily(a.prices)
    print("=" * 80)
    print("1. EVERY REBALANCE GRID, SAME STRATEGY")
    print("=" * 80)
    print(f"  {df['symbol'].nunique()} commodities, {df['date'].nunique():,} trading days")
    print(f"  building {a.grids} grids — identical signal and sizing, different strike day\n")

    tgts, series, stats_ = {}, {}, {}
    for off in range(a.grids):
        panel = grid_panel(df, off)
        if panel.empty:
            continue
        tg = targets(panel)
        if tg.empty:
            continue
        r, aux = pnl_from_targets(tg)
        s = st(r)
        if not np.isfinite(s["sharpe"]):
            continue
        tgts[off], series[off], stats_[off] = tg, r, s

    if len(series) < 5:
        raise SystemExit("  too few usable grids — check the price file")

    srs = pd.Series({k: v["sharpe"] for k, v in stats_.items()}).sort_index()
    print(f"  {'grid':>6s} {'Sharpe':>8s} {'return':>9s} {'vol':>7s} {'maxDD':>8s}")
    for off in srs.index:
        s = stats_[off]
        tag = "  <- current spec" if off == 0 else ""
        print(f"  day {off:>2d} {s['sharpe']:>8.3f} {s['ann']*100:>8.2f}% "
              f"{s['vol']*100:>6.1f}% {s['dd']*100:>7.1f}%{tag}")

    print(f"\n  across {len(srs)} grids:  mean {srs.mean():.3f}   median {srs.median():.3f}")
    print(f"                    min {srs.min():.3f}   max {srs.max():.3f}   "
          f"spread {srs.max()-srs.min():.3f}")
    if 0 in srs.index:
        pct = (srs < srs.loc[0]).mean()
        print(f"\n  THE REPORTED GRID (day 0) SITS AT THE {pct:.0%} PERCENTILE.")
        if pct > 0.75:
            print("  That is a lucky draw. The honest number to report is the mean across")
            print("  grids, not the one that happened to be chosen.")
        elif pct < 0.25:
            print("  That is an unlucky draw — the chosen grid understates the strategy.")
        else:
            print("  Mid-pack, so the reported figure was not a lucky calendar choice.")
            print("  Worth saying explicitly: a PM will wonder.")

    print("\n" + "=" * 80)
    print("2. HOW MUCH OF THIS IS NOISE?")
    print("=" * 80)
    M = pd.DataFrame(series).dropna(how="all")
    C = M.corr()
    rho = float(np.nanmean(C.to_numpy()[np.triu_indices_from(C.to_numpy(), k=1)]))
    K = M.shape[1]
    factor = np.sqrt((1 + (K - 1) * rho) / K)
    print(f"  mean pairwise correlation between grids   {rho:.3f}")
    print(f"  grids averaged                            {K}")
    print(f"  predicted volatility reduction            x{factor:.3f}")
    print(f"  predicted Sharpe uplift                   x{1/factor:.3f}   "
          f"({srs.mean():.3f} -> {srs.mean()/factor:.3f})")
    print("\n  Averaging K correlated series leaves the mean return unchanged and scales")
    print("  volatility by sqrt((1+(K-1)rho)/K). That is a theorem, not a hypothesis. The")
    print("  only empirical input is rho, measured above.")

    print("\n" + "=" * 80)
    print("3. THE TRANCHED BOOK — averaged BEFORE rounding")
    print("=" * 80)
    print("  Twenty-one separate books would each hold $21,000 and round to zero")
    print("  constantly. Instead the fractional targets are averaged into one vector and")
    print("  rounded once, at full size.\n")
    allt = pd.concat([t.assign(grid=k) for k, t in tgts.items()], ignore_index=True)
    avg = (allt.groupby(["ym", "symbol"])
                .agg(target=("target", "mean"), px=("px", "mean"),
                     fwd=("fwd", "mean")).reset_index())
    r_tr, aux_tr = pnl_from_targets(avg)
    s_tr = st(r_tr)

    r0, aux0 = pnl_from_targets(tgts[0]) if 0 in tgts else (None, None)
    s0 = st(r0) if r0 is not None else None

    print(f"  {'':22s} {'Sharpe':>8s} {'t':>7s} {'return':>9s} {'vol':>7s} "
          f"{'maxDD':>8s} {'zeroed':>8s}")
    if s0:
        print(f"  {'single grid (day 0)':22s} {s0['sharpe']:>8.3f} {s0['t']:>+7.2f} "
              f"{s0['ann']*100:>8.2f}% {s0['vol']*100:>6.1f}% {s0['dd']*100:>7.1f}% "
              f"{aux0['zero_share']*100:>7.1f}%")
    print(f"  {'mean of grid Sharpes':22s} {srs.mean():>8.3f}")
    print(f"  {'TRANCHED (averaged)':22s} {s_tr['sharpe']:>8.3f} {s_tr['t']:>+7.2f} "
          f"{s_tr['ann']*100:>8.2f}% {s_tr['vol']*100:>6.1f}% {s_tr['dd']*100:>7.1f}% "
          f"{aux_tr['zero_share']*100:>7.1f}%")
    if s0:
        print(f"\n  gain over the single reported grid   "
              f"{s_tr['sharpe']-s0['sharpe']:+.3f}")
        print(f"  gain over the mean grid              {s_tr['sharpe']-srs.mean():+.3f}")
        print(f"  turnover: single {aux0['turn']:.1f} contracts/month, "
              f"tranched {aux_tr['turn']:.1f}")
        if aux_tr["turn"] < aux0["turn"]:
            print("  Turnover FELL. An averaged target moves more smoothly than any single")
            print("  grid, so the tranched book trades less and pays less.")
        if aux_tr["zero_share"] < aux0["zero_share"]:
            print("  Fewer positions round away, because the averaged target is rounded")
            print("  once at full size rather than in twenty-one small pieces.")

    print("\n" + "=" * 80)
    print("4. COST SENSITIVITY OF THE TRANCHED BOOK")
    print("=" * 80)
    for bps in (3, 10, 20, 40):
        rb, _ = pnl_from_targets(avg, bps=bps)
        sb = st(rb)
        line = f"  {bps:>2d}bp per side   tranched {sb['sharpe']:>+6.3f}"
        if 0 in tgts:
            r0b, _ = pnl_from_targets(tgts[0], bps=bps)
            s0b = st(r0b)
            line += f"   single grid {s0b['sharpe']:>+6.3f}   diff {sb['sharpe']-s0b['sharpe']:>+6.3f}"
        print(line)

    print("\n" + "=" * 80)
    print("5. IS THE GAIN LEVERAGE OR NOISE REMOVAL?")
    print("=" * 80)
    print("  Rescale both to the same realised volatility. Anything that survives is the")
    print("  removal of timing noise. Anything that disappears was a size effect.\n")
    if s0 and np.isfinite(s0["vol"]) and s0["vol"] > 0:
        tgt_vol = s0["vol"]
        for lbl, ser, sx in (("single grid", r0, s0), ("tranched", r_tr, s_tr)):
            if np.isfinite(sx["vol"]) and sx["vol"] > 0:
                rescaled = ser * (tgt_vol / sx["vol"])
                sr2 = st(rescaled)
                print(f"  {lbl:16s} SR {sr2['sharpe']:>+6.3f}   dd {sr2['dd']*100:>+6.1f}%  "
                      f"(rescaled to {tgt_vol*100:.1f}% vol)")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    gain = s_tr["sharpe"] - (s0["sharpe"] if s0 else srs.mean())
    print(f"  grid dispersion   {srs.min():.3f} to {srs.max():.3f}   "
          f"(spread {srs.max()-srs.min():.3f})")
    print(f"  tranched Sharpe   {s_tr['sharpe']:.3f}")
    print(f"  gain              {gain:+.3f}")
    print()
    if gain > 0.05:
        print("  ADOPT IT. The gain comes from arithmetic rather than from an empirical")
        print("  claim about markets, which is why it succeeded where eleven refinements")
        print("  failed. Report the tranched figure as the headline and the grid dispersion")
        print("  in Analytical Evidence — the dispersion is the honest answer to 'what if")
        print("  you had rebalanced three days later', and almost no applicant can answer")
        print("  that question at all.")
    elif gain > 0:
        print("  Small but real, and free. Adopt it and report the grid dispersion, which")
        print("  is worth more than the Sharpe gain: it demonstrates the result does not")
        print("  depend on an arbitrary calendar choice.")
    else:
        print("  No gain. The grids are correlated enough that averaging adds nothing, so")
        print("  keep the single-grid implementation for simplicity. Still report the grid")
        print("  dispersion — showing the result is stable across rebalance dates is worth")
        print("  a paragraph whether or not it improves the number.")


if __name__ == "__main__":
    main()