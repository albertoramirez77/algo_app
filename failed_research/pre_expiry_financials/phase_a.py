"""
phase_a.py — is the financials pre-expiry effect real, or did the residualisation make it?

    python phase_a.py --prices data/px_clean.parquet

THE RESULT UNDER INVESTIGATION

Financial futures (4 equity index, 6 rates, 8 FX), basis-residualised long-second /
short-front spread returns, by days to front expiry:

    0-5    +0.0168%/d  t=+5.02        21-30  +0.0137%/d  t=+12.64
    6-10   +0.0129%/d  t=+9.58        31-45  +0.0023%/d  t=+2.95
    11-15  +0.0143%/d  t=+10.70       46-90  -0.0024%/d  t=-6.27
    16-20  +0.0174%/d  t=+11.86

near-minus-far +0.0191%/day, t=+10.29. Ten times more significant than the commodity
result the test was built to find, and unexplained.

WHY IT MIGHT BE NOTHING

The basis slope removed from FX was -0.03277 — fourteen times any other asset class. In FX
the "basis" is the interest-rate differential, not an inventory signal. Regressing spread
returns on it may have injected structure rather than removed it. The raw FX numbers are
small and noisy while the residuals are enormous. That gap is what an artifact looks like.

FIVE CHECKS, ONE PASS OVER THE SAME DATA

  1  per asset class          one effect, or one contaminated asset class?
  2  no residualisation       does it exist in raw returns at all?
  3  per-instrument residual  does a finer control kill it?
  4  roll days excluded       is it a continuous-contract stitching artifact?
  5  sign convention by hand  does spread_ret mean what the code claims?

Check 2 is decisive. An effect that appears only after a regression is a property of the
regression.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

BUCKETS = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 45), (46, 90)]
NEAR, FAR = 10, (31, 90)


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))

    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        df[f"blk_{leg}"] = blk
        df[f"ret_{leg}"] = (df[f"settle_{leg}"] /
                            df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1) - 1.0)
        # a roll day is the first observation of a new contract block
        df[f"roll_{leg}"] = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: s != s.shift(1)).fillna(True)

    df["spread_ret"] = df["ret_1"] - df["ret_0"]
    df["is_roll"] = df["roll_0"] | df["roll_1"]
    df["dte"] = (df["expiry_0"] - df["date"]).dt.days
    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    df["gap"] = gap
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (gap / 365.25)
    df.loc[(gap <= 0) | (gap > 400), "basis"] = np.nan
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["physical"] = df["symbol"].map(
        lambda s: BY_SYMBOL[s].physical if s in BY_SYMBOL else False)
    return df.dropna(subset=["spread_ret", "dte"])


def add_residuals(d: pd.DataFrame) -> pd.DataFrame:
    """Two residualisations: pooled within asset class (as before), and per instrument."""
    d = d.copy()
    d["res_asset"] = np.nan
    d["res_inst"] = np.nan
    ok = d["basis"].notna()
    for a, g in d[ok].groupby("asset"):
        X = np.column_stack([np.ones(len(g)), g["basis"].to_numpy()])
        y = g["spread_ret"].to_numpy()
        d.loc[g.index, "res_asset"] = y - X @ (np.linalg.pinv(X.T @ X) @ (X.T @ y))
    for s, g in d[ok].groupby("symbol"):
        if len(g) < 100:
            continue
        X = np.column_stack([np.ones(len(g)), g["basis"].to_numpy()])
        y = g["spread_ret"].to_numpy()
        d.loc[g.index, "res_inst"] = y - X @ (np.linalg.pinv(X.T @ X) @ (X.T @ y))
    return d


def cmean(s: pd.DataFrame, col: str) -> tuple[float, float, int]:
    g = s.groupby("date")[col].mean().dropna()
    if len(g) < 30:
        return np.nan, np.nan, len(g)
    return g.mean(), g.mean() / (g.std(ddof=1) / np.sqrt(len(g))), len(g)


def nmf(s: pd.DataFrame, col: str) -> tuple[float, float]:
    n = s[s["dte"] <= NEAR].groupby("date")[col].mean().dropna()
    f = s[(s["dte"] >= FAR[0]) & (s["dte"] <= FAR[1])].groupby("date")[col].mean().dropna()
    if len(n) < 30 or len(f) < 30:
        return np.nan, np.nan
    dd = n.mean() - f.mean()
    se = np.sqrt(n.var(ddof=1) / len(n) + f.var(ddof=1) / len(f))
    return dd, (dd / se if se > 0 else np.nan)


def show_profile(s: pd.DataFrame, col: str, label: str) -> None:
    print(f"\n  {label}")
    for lo, hi in BUCKETS:
        sub = s[(s["dte"] >= lo) & (s["dte"] <= hi)]
        if len(sub) < 60:
            continue
        m, t, nd = cmean(sub, col)
        flag = " *" if abs(t) > 2 else ""
        print(f"    {lo:>3d}-{hi:<3d}  {m*100:>+9.4f}%/d  t={t:>+7.2f}  n={nd:>5,}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    df = load(a.prices)
    d = add_residuals(df[(df["dte"] >= 0) & (df["dte"] <= 90)])
    fin = d[~d["physical"]]

    print("=" * 78)
    print("CHECK 5 (FIRST) — DOES spread_ret MEAN WHAT THE CODE CLAIMS?")
    print("=" * 78)
    ex = fin[(fin["ret_0"].notna()) & (fin["ret_1"].notna()) &
             (~fin["is_roll"])].iloc[len(fin) // 2]
    hand0 = ex["ret_0"]
    hand1 = ex["ret_1"]
    print(f"  {ex['symbol']} on {ex['date']:%Y-%m-%d}")
    print(f"    front  {ex['contract_0']}  settle {ex['settle_0']:.5f}  ret {hand0:+.6f}")
    print(f"    second {ex['contract_1']}  settle {ex['settle_1']:.5f}  ret {hand1:+.6f}")
    print(f"    spread_ret = ret_1 - ret_0 = {hand1 - hand0:+.6f}   "
          f"stored {ex['spread_ret']:+.6f}   match "
          f"{abs((hand1 - hand0) - ex['spread_ret']) < 1e-12}")
    print(f"    expiry_0 {ex['expiry_0']:%Y-%m-%d}  expiry_1 {ex['expiry_1']:%Y-%m-%d}  "
          f"gap {ex['gap']:.0f}d  dte {ex['dte']:.0f}d")
    print("  POSITIVE spread_ret means the DEFERRED contract outperformed.")

    print("\n" + "=" * 78)
    print("CHECK 2 — DOES THE EFFECT EXIST WITHOUT RESIDUALISATION?")
    print("=" * 78)
    print("  This is the decisive one. An effect that appears only after a regression is")
    print("  a property of the regression.")
    for col, lab in (("spread_ret", "RAW spread return"),
                     ("res_asset", "residualised per ASSET CLASS (the original)"),
                     ("res_inst", "residualised per INSTRUMENT")):
        m, t = nmf(fin, col)
        print(f"  near-minus-far, {lab:<42s} {m*100:>+8.4f}%/d  t={t:>+7.2f}")

    print("\n" + "=" * 78)
    print("CHECK 1 — IS IT ONE EFFECT, OR ONE CONTAMINATED ASSET CLASS?")
    print("=" * 78)
    rows = []
    for asset in ("equity", "rates", "fx", "commodity"):
        s = d[d["asset"] == asset]
        if s.empty:
            continue
        r = dict(asset=asset, n_inst=s["symbol"].nunique(), rows=len(s))
        for col, tag in (("spread_ret", "raw"), ("res_asset", "res_a"),
                         ("res_inst", "res_i")):
            m, t = nmf(s, col)
            r[f"{tag}_bp"] = m * 1e4 if np.isfinite(m) else np.nan
            r[f"{tag}_t"] = t
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n  raw = no residualisation, res_a = per asset class, res_i = per instrument")
    print("  Units are basis points per day of near-minus-far.")

    for asset in ("equity", "rates", "fx"):
        s = d[d["asset"] == asset]
        if len(s) > 500:
            show_profile(s, "spread_ret", f"{asset.upper()} — RAW")
            show_profile(s, "res_inst", f"{asset.upper()} — per-instrument residual")

    print("\n" + "=" * 78)
    print("CHECK 4 — IS IT A CONTINUOUS-CONTRACT STITCHING ARTIFACT?")
    print("=" * 78)
    print(f"  roll days: {fin['is_roll'].mean():.2%} of financial rows")
    for lab, s in (("all rows", fin), ("roll days EXCLUDED", fin[~fin["is_roll"]]),
                   ("roll days ONLY", fin[fin["is_roll"]])):
        m, t = nmf(s, "spread_ret")
        m2, t2 = nmf(s, "res_asset")
        print(f"  {lab:<22s} raw {m*100:>+8.4f}%/d t={t:>+7.2f}    "
              f"res {m2*100:>+8.4f}%/d t={t2:>+7.2f}")
    # where does the P&L actually sit
    rr = fin.groupby("is_roll")["spread_ret"].agg(["mean", "count"])
    print(f"\n  mean raw spread return, non-roll days {rr.loc[False,'mean']*1e4:+.3f} bp "
          f"(n={rr.loc[False,'count']:,})")
    if True in rr.index:
        print(f"  mean raw spread return, roll days     {rr.loc[True,'mean']*1e4:+.3f} bp "
              f"(n={rr.loc[True,'count']:,})")

    print("\n" + "=" * 78)
    print("CHECK 3 — HOW BIG IS THE RESIDUALISATION'S FOOTPRINT?")
    print("=" * 78)
    for asset, g in d.groupby("asset"):
        gg = g.dropna(subset=["basis", "spread_ret"])
        if len(gg) < 100:
            continue
        X = np.column_stack([np.ones(len(gg)), gg["basis"].to_numpy()])
        y = gg["spread_ret"].to_numpy()
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        fitted = X @ b
        print(f"  {asset:10s} slope {b[1]:>+10.5f}   sd(basis) {gg['basis'].std():>7.4f}   "
              f"sd(fitted) {fitted.std()*1e4:>7.3f}bp   sd(raw) {y.std()*1e4:>8.3f}bp   "
              f"R2 {1 - (y - fitted).var()/y.var():.4f}")
    print("\n  If sd(fitted) is a large fraction of sd(raw), the regression is moving the")
    print("  series a lot, and the residual is substantially the regression's own work.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    raw_m, raw_t = nmf(fin, "spread_ret")
    ra_m, ra_t = nmf(fin, "res_asset")
    ri_m, ri_t = nmf(fin, "res_inst")
    nr_m, nr_t = nmf(fin[~fin["is_roll"]], "spread_ret")
    fx_raw = nmf(d[d["asset"] == "fx"], "spread_ret")[1]
    eq_raw = nmf(d[d["asset"] == "equity"], "spread_ret")[1]
    rt_raw = nmf(d[d["asset"] == "rates"], "spread_ret")[1]

    print(f"  raw                     t={raw_t:+.2f}")
    print(f"  per asset class         t={ra_t:+.2f}")
    print(f"  per instrument          t={ri_t:+.2f}")
    print(f"  raw, no roll days       t={nr_t:+.2f}")
    print(f"  raw by class: equity t={eq_raw:+.2f}  rates t={rt_raw:+.2f}  "
          f"fx t={fx_raw:+.2f}")
    print()
    if not np.isfinite(raw_t) or abs(raw_t) < 2:
        print("  ARTIFACT. The effect does not exist in raw returns. It is created by")
        print("  regressing spread returns on the basis, which in FX is the interest-rate")
        print("  differential and is mechanically related to the spread. Discard it and")
        print("  move to Phase B.")
    elif abs(nr_t) < 2:
        print("  STITCHING ARTIFACT. The effect lives on roll days, where the continuous")
        print("  series switches contract. It is a property of the series construction,")
        print("  not of the market. Discard and move to Phase B.")
    elif np.sign(raw_t) != np.sign(ra_t):
        print("  ARTIFACT. Raw and residualised disagree in SIGN. The residualisation is")
        print("  not removing a nuisance, it is inverting the series.")
    elif sum(abs(x) > 2 for x in (eq_raw, rt_raw, fx_raw)) <= 1:
        print("  ONE ASSET CLASS ONLY. Not a general term-structure premium. It may still")
        print("  be real within that class, but the cross-asset framing is wrong and the")
        print("  breadth argument collapses to a handful of instruments.")
    else:
        print("  SURVIVES. Present in raw returns, off roll days, in more than one asset")
        print("  class, with a consistent sign. Pre-register before measuring anything")
        print("  further.")


if __name__ == "__main__":
    main()