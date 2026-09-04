"""
test_rollpressure.py — does the size of a forced roll, relative to the capacity available to
absorb it, predict the calendar spread?

    python test_rollpressure.py --prices data/px_clean.parquet

Read 11_preregistration_rollpressure.md first. Predictions were fixed before this ran.

THE IDENTITY

Open interest in an expiring contract must reach zero. Always. That is a contract
specification, not a behavioural claim. Three observable quantities follow:

    pressure = open_interest_front / (ADV_front x days_to_expiry)

Dimensionless: how many days of normal volume must trade, every remaining day, purely to
complete a roll that happens regardless of price.

THE CRITICAL DESIGN POINT

days_to_expiry sits in the denominator, so pressure rises mechanically as expiry nears. An
uncontrolled test would rediscover the calendar — which is published years ahead, known to
everyone, and already arbitraged (the index-roll literature is dead post-2012).

So pressure is normalised WITHIN instrument and WITHIN days-to-expiry bucket, on an
expanding window. The signal is "unusual pressure for this contract at this point in its own
cycle." The script also reports the uncontrolled version: if removing the control
STRENGTHENS the result, the result is the calendar and the hypothesis is dead.

NO LOOK-AHEAD

Signal observed at settlement on day t. Position taken at settlement on t+1. Forward return
measured t+1 to t+1+h. ADV is lagged two days on top of that, because cleared volume is an
exchange figure whose publication lag is not established.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

DTE_LO, DTE_HI = 1, 30
DTE_BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30)]
PRIMARY_H = 5
HORIZONS = [1, 3, 5, 10]
ADV_WINDOW = 20
ADV_LAG = 2
MIN_COVERAGE = 0.60


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------

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
        df[f"ret_{leg}"] = (df[f"settle_{leg}"] /
                            df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1) - 1.0)

    df["spread_ret"] = df["ret_1"] - df["ret_0"]
    df["dte"] = (df["expiry_0"] - df["date"]).dt.days
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    return df


def coverage_gate(df: pd.DataFrame) -> bool:
    """
    Open interest is missing on a large share of rows because Databento leaves ts_ref
    undefined on many statistics records. If that missingness concentrates inside roll
    windows, every number downstream is contaminated. Check before testing, not after.
    """
    print("=" * 78)
    print("0. COVERAGE GATE — is open interest present WHERE THE TEST LIVES?")
    print("=" * 78)
    inw = df[(df["dte"] >= DTE_LO) & (df["dte"] <= DTE_HI)]
    out = df[(df["dte"] > DTE_HI) & (df["dte"] <= 120)]
    c_in = inw["oi_0"].notna().mean()
    c_out = out["oi_0"].notna().mean()
    v_in = inw["vol_0"].notna().mean()
    print(f"  rows in roll window (dte {DTE_LO}-{DTE_HI}): {len(inw):,}")
    print(f"  open interest present, inside window  {c_in:.1%}")
    print(f"  open interest present, outside window {c_out:.1%}")
    print(f"  volume present, inside window         {v_in:.1%}")
    print(f"  difference in/out: {c_in - c_out:+.1%}   "
          f"(large gap = systematic, not random)")

    per = inw.groupby("symbol")["oi_0"].apply(lambda s: s.notna().mean())
    bad = per[per < MIN_COVERAGE]
    print(f"\n  instruments below {MIN_COVERAGE:.0%} coverage in-window: "
          f"{len(bad)} of {len(per)}")
    if len(bad):
        print("   ", ", ".join(f"{s}:{v:.0%}" for s, v in bad.items()))

    ok = c_in >= MIN_COVERAGE and abs(c_in - c_out) < 0.15
    print(f"\n  GATE {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  Open interest is either too sparse or systematically absent where the")
        print("  test lives. Refetch statistics with the sentinel handled, or drop the")
        print("  instruments below threshold and rerun. Do not interpret what follows.")
    return ok


def build_signal(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ADV: rolling median of cleared volume, lagged. Cleared volume is an exchange figure
    # published with settlement and its lag is not established; two days is conservative.
    d["adv"] = (d.groupby("symbol")["vol_0"]
                  .transform(lambda s: s.rolling(ADV_WINDOW, min_periods=5).median())
                  .groupby(d["symbol"]).shift(ADV_LAG))

    d["oi_lag"] = d.groupby("symbol")["oi_0"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["pressure"] = d["oi_lag"] / (d["adv"] * d["dte"].clip(lower=1))
    d.loc[~np.isfinite(d["pressure"]), "pressure"] = np.nan
    d["log_pressure"] = np.log(d["pressure"].where(d["pressure"] > 0))
    d["log_adv"] = np.log(d["adv"].where(d["adv"] > 0))

    # bucket by days to expiry
    d["dte_bucket"] = np.nan
    for i, (lo, hi) in enumerate(DTE_BUCKETS):
        d.loc[(d["dte"] >= lo) & (d["dte"] <= hi), "dte_bucket"] = i

    # THE CONTROL. Expanding z-score within (instrument, dte bucket). Expanding, not
    # full-sample: a full-sample mean is look-ahead.
    g = d.groupby(["symbol", "dte_bucket"])["log_pressure"]
    mu = g.transform(lambda s: s.expanding().mean().shift(1))
    sd = g.transform(lambda s: s.expanding().std().shift(1))
    d["z_ctrl"] = ((d["log_pressure"] - mu) / sd).clip(-3, 3)

    # uncontrolled comparison: cross-sectional rank of raw pressure, no dte adjustment
    d["z_raw"] = d.groupby("date")["log_pressure"].transform(
        lambda s: (s - s.mean()) / s.std()).clip(-3, 3)

    # forward spread returns, starting the day AFTER the signal
    for h in HORIZONS:
        fwd = (d.groupby("symbol")["spread_ret"]
                 .transform(lambda s: s.shift(-1).rolling(h, min_periods=h).sum()))
        d[f"fwd{h}"] = fwd
    return d


# ----------------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------------

def fm(panel: pd.DataFrame, x: str, y: str) -> dict:
    """Fama-MacBeth: one cross-sectional slope per date, t-test on the series of slopes."""
    slopes = []
    for _, g in panel.groupby("date"):
        s = g[[x, y]].dropna()
        if len(s) < 4 or s[x].std() == 0:
            continue
        b = np.polyfit(s[x].to_numpy(), s[y].to_numpy(), 1)[0]
        if np.isfinite(b):
            slopes.append(b)
    if len(slopes) < 50:
        return dict(n=len(slopes))
    a = np.array(slopes)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return dict(n=len(a), slope=a.mean(), se=se, t=a.mean() / se, mde=2 * se)


def show(label: str, r: dict, note: str = "") -> None:
    if "slope" in r:
        flag = " *" if abs(r["t"]) > 2 else ""
        print(f"  {label:38s} {r['slope']*1e4:>+8.3f} bp  t={r['t']:>+6.2f}  "
              f"n={r['n']:>5,}{flag}  {note}")
    else:
        print(f"  {label:38s} too few cross-sections ({r.get('n', 0)})")


# ----------------------------------------------------------------------------------
# portfolio
# ----------------------------------------------------------------------------------

def portfolio(panel: pd.DataFrame, xcol: str, h: int) -> dict:
    """
    Long-short the calendar spread on normalised pressure. Weights are the cross-sectionally
    demeaned signal scaled to unit gross, so the book is dollar-neutral by construction and
    the return series is a tradeable object rather than a regression coefficient.
    """
    rows = []
    for dt, g in panel.groupby("date"):
        s = g[[xcol, f"fwd{h}", "symbol"]].dropna()
        if len(s) < 6:
            continue
        w = s[xcol] - s[xcol].mean()
        gross = w.abs().sum()
        if gross <= 0:
            continue
        w = w / gross
        rows.append(dict(date=dt, ret=float((w * s[f"fwd{h}"]).sum()), n=len(s)))
    if len(rows) < 100:
        return dict(n=len(rows))
    r = pd.DataFrame(rows).set_index("date")["ret"]
    # overlapping h-day windows: scale to a per-day series before annualising
    daily = r / h
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    eq = (1 + daily.fillna(0)).cumprod()
    return dict(n=len(r), ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=ann_ret / ann_vol if ann_vol > 0 else np.nan,
                max_dd=float((eq / eq.cummax() - 1).min()),
                hit=float((daily > 0).mean()),
                top20_share=float(daily.nlargest(20).sum() / daily.sum())
                if daily.sum() != 0 else np.nan)


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    df = load(a.prices)
    gate = coverage_gate(df)
    d = build_signal(df)
    w = d[(d["dte"] >= DTE_LO) & (d["dte"] <= DTE_HI)].copy()
    panel = w.dropna(subset=["z_ctrl", f"fwd{PRIMARY_H}"])

    print("\n" + "=" * 78)
    print("1. SAMPLE AND POWER — stated before the result")
    print("=" * 78)
    print(f"  {len(panel):,} instrument-days, {panel['symbol'].nunique()} instruments, "
          f"{panel['date'].nunique():,} dates")
    print(f"  pressure: median {w['pressure'].median():.3f}  "
          f"p10 {w['pressure'].quantile(.10):.3f}  p90 {w['pressure'].quantile(.90):.3f}")
    print("    (days of normal volume that must clear per remaining day)")
    pre = fm(panel, "z_ctrl", f"fwd{PRIMARY_H}")
    if "mde" in pre:
        print(f"  minimum detectable slope at t=2: {pre['mde']*1e4:.3f} bp per 1sd "
              f"of pressure over {PRIMARY_H} days")
        print(f"  a 1bp effect would register at t = {1e-4/pre['se']:.1f}")

    print("\n" + "=" * 78)
    print(f"2. P1 — DOES PRESSURE PREDICT THE SPREAD?  (primary: {PRIMARY_H}-day forward)")
    print("=" * 78)
    print("  positive = deferred outperforms front = the front is being sold\n")
    for h in HORIZONS:
        sub = w.dropna(subset=["z_ctrl", f"fwd{h}"])
        show(f"controlled for dte, {h:>2d}-day forward", fm(sub, "z_ctrl", f"fwd{h}"),
             "<- PRIMARY" if h == PRIMARY_H else "")

    print("\n  uncontrolled comparison — if this is STRONGER, the signal is the calendar")
    print("  and the hypothesis is dead:")
    for h in (PRIMARY_H,):
        sub = w.dropna(subset=["z_raw", f"fwd{h}"])
        show(f"NOT controlled for dte, {h}-day forward", fm(sub, "z_raw", f"fwd{h}"))

    r_ctrl = fm(panel, "z_ctrl", f"fwd{PRIMARY_H}")
    r_raw = fm(w.dropna(subset=["z_raw", f"fwd{PRIMARY_H}"]), "z_raw", f"fwd{PRIMARY_H}")
    p1 = "slope" in r_ctrl and r_ctrl["t"] > 3
    calendar = ("t" in r_raw and "t" in r_ctrl and abs(r_raw["t"]) > abs(r_ctrl["t"]) * 1.2)

    print("\n" + "=" * 78)
    print("3. BY ASSET CLASS")
    print("=" * 78)
    for asset, g in panel.groupby("asset"):
        if g["symbol"].nunique() < 3:
            continue
        show(f"{asset} ({g['symbol'].nunique()} instruments)",
             fm(g, "z_ctrl", f"fwd{PRIMARY_H}"))

    print("\n" + "=" * 78)
    print("4. P2 — IS IT STRONGER WHERE CAPACITY IS SCARCER?")
    print("=" * 78)
    med = panel["log_adv"].median()
    for lab, sub in (("low ADV half (scarcer capacity)", panel[panel["log_adv"] <= med]),
                     ("high ADV half", panel[panel["log_adv"] > med])):
        show(lab, fm(sub, "z_ctrl", f"fwd{PRIMARY_H}"))
    lo = fm(panel[panel["log_adv"] <= med], "z_ctrl", f"fwd{PRIMARY_H}")
    hi = fm(panel[panel["log_adv"] > med], "z_ctrl", f"fwd{PRIMARY_H}")
    p2 = ("slope" in lo and "slope" in hi and lo["slope"] > hi["slope"])
    print(f"  P2 {'consistent' if p2 else 'NOT consistent'} — predicted low-ADV stronger")

    print("\n" + "=" * 78)
    print("5. P4 — IS IT A LAST-TWO-DAYS ILLIQUIDITY ARTIFACT?")
    print("=" * 78)
    show("all, dte 1-30", fm(panel, "z_ctrl", f"fwd{PRIMARY_H}"))
    show("excluding dte < 3", fm(panel[panel["dte"] >= 3], "z_ctrl", f"fwd{PRIMARY_H}"))
    show("dte 6-30 only", fm(panel[panel["dte"] >= 6], "z_ctrl", f"fwd{PRIMARY_H}"))
    r_ex = fm(panel[panel["dte"] >= 3], "z_ctrl", f"fwd{PRIMARY_H}")
    p4 = "slope" in r_ex and r_ex["t"] > 3

    print("\n" + "=" * 78)
    print("6. P3 — PLACEBO: shuffle pressure across instruments within each date")
    print("=" * 78)
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(a.seeds):
        p = panel.copy()
        p["z_ctrl"] = p.groupby("date")["z_ctrl"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        rr = fm(p, "z_ctrl", f"fwd{PRIMARY_H}")
        if "t" in rr:
            ts.append(rr["t"])
    p3 = False
    if ts:
        ts = np.array(ts)
        z = ((r_ctrl.get("t", np.nan) - ts.mean()) / max(ts.std(ddof=1), 1e-9)
             if "t" in r_ctrl else np.nan)
        print(f"  placebo t: {ts.mean():+.2f} ± {ts.std(ddof=1):.2f} over {len(ts)} seeds")
        print(f"  real sits {z:+.1f} placebo sd from the placebo mean")
        p3 = abs(z) > 2

    print("\n" + "=" * 78)
    print("7. THE PORTFOLIO — is there alpha, or only a coefficient?")
    print("=" * 78)
    pf = portfolio(panel, "z_ctrl", PRIMARY_H)
    if "sharpe" in pf:
        print(f"  gross Sharpe        {pf['sharpe']:>7.3f}")
        print(f"  annual return       {pf['ann_ret']*100:>7.2f}%   "
              f"vol {pf['ann_vol']*100:.2f}%")
        print(f"  max drawdown        {pf['max_dd']*100:>7.1f}%")
        print(f"  hit rate            {pf['hit']:>7.1%}")
        print(f"  best 20 days        {pf['top20_share']*100:>7.1f}% of total P&L")
        print(f"  rebalances          {pf['n']:,}")
        print("\n  GROSS. Costs at ~1.85% of capital per year, dominated by spread, come")
        print("  off this. A gross Sharpe under about 0.35 does not survive them.")
        print(f"  t on this Sharpe over 16 years = {pf['sharpe']*4:.2f}")
    else:
        print(f"  too few rebalances ({pf.get('n', 0)})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for k, v in (("coverage gate", gate), ("P1 slope t>3", p1),
                 ("P2 scarcer capacity stronger", p2),
                 ("P3 placebo distinguishes", p3),
                 ("P4 survives excluding dte<3", p4)):
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if calendar:
        print("\n  WARNING: the uncontrolled version is stronger than the controlled one.")
        print("  The signal is the calendar, not the pressure. That is hypothesis 2, which")
        print("  is already dead.")
    print()
    if gate and p1 and p3 and p4:
        print("  SURVIVES. Move to costs, integer sizing at $450k, and capacity. The")
        print("  economics can be right and the trade still too small to express.")
    elif not gate:
        print("  UNINTERPRETABLE. Fix open-interest coverage before reading anything above.")
    else:
        print("  DEAD as specified. Report it as dead, with power, and move on. Do not")
        print("  reparameterise to rescue it — that is the failure mode this project has")
        print("  spent three hypotheses learning to avoid.")


if __name__ == "__main__":
    main()