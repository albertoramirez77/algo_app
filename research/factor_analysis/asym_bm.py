"""
asym_bm.py — is basis-momentum asymmetric because storage bounds the basis on one side?

    python asym_bm.py --prices data/px_clean.parquet

THE POWER PROBLEM THAT GOVERNS THIS ROUND

With 182 months, the minimum Sharpe difference detectable at t=2 is 2/sqrt(16) = 0.50. Path
consistency added 0.229. The units correction added 0.184. BOTH WERE BELOW THE DETECTION
THRESHOLD BEFORE THEY WERE RUN. They did not fail because they were wrong; they failed
because this sample cannot resolve effects of that size.

So this test does not chase Sharpe. It tests a PATTERN across four cells of the same
portfolio, which uses the cross-section instead of collapsing everything into one time
series, and therefore has materially more power than any Sharpe comparison.

THE CLAIM

Storage theory bounds the basis on one side only.

    CONTANGO IS BOUNDED. If the futures price exceeds spot by more than full carry
    (financing plus physical storage), the arbitrage is riskless: buy spot, store it, sell
    forward, deliver. Capital enters and the spread closes. So basis >= -(r + u).

    BACKWARDATION IS NOT. To arbitrage it you would have to short physical inventory you do
    not own. You cannot borrow soybeans that are not in a warehouse. The basis can run as
    far as scarcity demands.

The basis therefore lives on a HALF-LINE, not a line.

Basis-momentum bets on continuation of basis changes. Its SHORT leg bets on further
contango - into a wall. Its LONG leg bets on further backwardation - into open space. The
factor should be asymmetric, and the asymmetry should concentrate where the wall is near.

THE DISCRIMINATING PREDICTION

Split the portfolio's P&L four ways: whether the position is long or short, and whether that
instrument is currently backwardated or contangoed.

    storage bound     SHORT-IN-CONTANGO is the weakest cell. That is the only cell pressed
                      against the arbitrage boundary.
    symmetric curve   no cell structure at all: the factor is the factor.
    market beta       both LONG cells beat both SHORT cells regardless of curve state.

Three explanations, three different observable patterns, one table. This is the property
that every successful test in this project has had and every failed one has lacked.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL = 450_000.0
VOL_TARGET = 0.20
IDM_CAP = 2.5
J = 12
VOL_WINDOW = 6


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
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    df["gap"] = gap.where((gap > 0) & (gap <= 400))
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (df["gap"] / 365.25)
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")

    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis=("basis", "last"), px=("settle_0", "last"),
                n_days=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m["sector"] = m["symbol"].map(lambda s: BY_SYMBOL[s].sector if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["spread"] = m["r0"] - m["r1"]
    m["bm"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


def idm_of(m: pd.DataFrame) -> float:
    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    if not np.isfinite(rho):
        rho = 0.2
    return min(1.0 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), IDM_CAP)


def run(m: pd.DataFrame, idm: float, bps: float = 3.0, min_n: int = 6,
        shuffle_tag_seed: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    The frozen strategy, but recording every position's P&L with its side and its curve
    state so the four-cell decomposition can be computed from the SAME book. Nothing is
    re-optimised; this is an accounting split of the strategy that already exists.
    """
    rng = np.random.default_rng(shuffle_tag_seed) if shuffle_tag_seed is not None else None
    prev, out, rows = {}, {}, []
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd", "basis"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        tags = (s["basis"] > 0).to_numpy()
        if rng is not None:
            tags = rng.permutation(tags)          # placebo: curve state, wrong instruments
        pnl_tot = cost_tot = 0.0
        held = {}
        for i, (sym, wi, vol, px, fwd) in enumerate(
                zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"])):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * idm / den))
            held[sym] = n
            pnl = n * dpm * px * (np.exp(fwd) - 1.0)
            pnl_tot += pnl
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost_tot += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
            if n != 0:
                rows.append(dict(ym=ym, symbol=sym, side="long" if n > 0 else "short",
                                 state="backwardated" if tags[i] else "contangoed",
                                 pnl=pnl / CAPITAL, notional=abs(n) * dpm * px / CAPITAL,
                                 basis=s["basis"].iloc[i]))
        for sym in set(prev) - set(held):
            cost_tot += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl_tot - cost_tot) / CAPITAL
    return pd.Series(out).sort_index(), pd.DataFrame(rows)


def cell_table(pos: pd.DataFrame) -> pd.DataFrame:
    """Monthly P&L per cell, then a t-test on each cell's monthly series."""
    rows = []
    for (side, state), g in pos.groupby(["side", "state"]):
        mth = g.groupby("ym")["pnl"].sum()
        # return per unit of capital deployed in that cell, so cells are comparable
        cap = g.groupby("ym")["notional"].sum()
        ret = (mth / cap.replace(0, np.nan)).dropna()
        if len(ret) < 48:
            continue
        se = ret.std(ddof=1) / np.sqrt(len(ret))
        rows.append(dict(side=side, state=state, months=len(ret),
                         mean_bp=ret.mean() * 1e4, t=ret.mean() / se if se > 0 else np.nan,
                         share_of_pnl=mth.sum() / pos["pnl"].sum()
                         if pos["pnl"].sum() != 0 else np.nan,
                         avg_notional=cap.mean()))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()

    m = load(a.prices)
    idm = idm_of(m)
    port, pos = run(m, idm)

    print("=" * 80)
    print("0. THE BOOK BEING DECOMPOSED — unchanged, frozen")
    print("=" * 80)
    yrs = len(port) / 12
    av = port.std(ddof=1) * np.sqrt(12)
    sr = (port.mean() * 12) / av
    print(f"  Sharpe {sr:+.3f}  t {sr*np.sqrt(yrs):+.2f}  "
          f"return {port.mean()*12*100:+.2f}%  vol {av*100:.1f}%")
    print(f"  {len(pos):,} position-months  "
          f"{pos['ym'].nunique()} months  {pos['symbol'].nunique()} instruments")
    print(f"  backwardated observations: {(m['basis'] > 0).mean():.1%}")
    print("\n  Nothing is re-optimised below. This is an accounting split of the P&L the")
    print("  strategy already produced, by position side and by curve state.")

    print("\n" + "=" * 80)
    print("1. THE FOUR CELLS")
    print("=" * 80)
    tab = cell_table(pos)
    if tab.empty:
        raise SystemExit("insufficient data for the decomposition")
    piv = tab.pivot(index="side", columns="state", values="mean_bp")
    tpv = tab.pivot(index="side", columns="state", values="t")
    print("  mean monthly return in basis points, per unit of capital in that cell:\n")
    print(f"  {'':8s} {'backwardated':>16s} {'contangoed':>16s}")
    for side in ("long", "short"):
        if side not in piv.index:
            continue
        b = piv.loc[side].get("backwardated", np.nan)
        c = piv.loc[side].get("contangoed", np.nan)
        tb = tpv.loc[side].get("backwardated", np.nan)
        tc = tpv.loc[side].get("contangoed", np.nan)
        print(f"  {side:8s} {b:>+10.1f} (t{tb:>+5.2f}) {c:>+10.1f} (t{tc:>+5.2f})")
    print()
    print(tab.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    print("\n" + "=" * 80)
    print("2. WHICH EXPLANATION DOES THE PATTERN MATCH?")
    print("=" * 80)
    def cell(side, state):
        r = tab[(tab["side"] == side) & (tab["state"] == state)]
        return float(r["mean_bp"].iloc[0]) if len(r) else np.nan
    lb, lc = cell("long", "backwardated"), cell("long", "contangoed")
    sb, sc = cell("short", "backwardated"), cell("short", "contangoed")
    cells = {"long/back": lb, "long/cont": lc, "short/back": sb, "short/cont": sc}
    weakest = min((v, k) for k, v in cells.items() if np.isfinite(v))[1]
    print(f"  weakest cell: {weakest}")
    print()
    print(f"  STORAGE BOUND predicts short/cont is weakest      "
          f"{'MATCH' if weakest == 'short/cont' else 'no'}")
    n_cells = sum(np.isfinite(v) for v in cells.values())
    if n_cells < 4:
        print(f"  WARNING: only {n_cells} of 4 cells populated. The decomposition needs")
        print("  all four to discriminate; with a missing cell the comparisons below are")
        print("  degenerate and should not be read as evidence.")
    long_avg = np.nanmean([lb, lc]); short_avg = np.nanmean([sb, sc])
    back_avg = np.nanmean([lb, sb]); cont_avg = np.nanmean([lc, sc])
    spread_cells = np.nanmax(list(cells.values())) - np.nanmin(list(cells.values()))
    print(f"  SYMMETRIC CURVE predicts no cell structure         "
          f"spread across cells {spread_cells:.1f}bp")
    print(f"  MARKET BETA predicts both long cells beat shorts   "
          f"long {long_avg:+.1f}bp vs short {short_avg:+.1f}bp  "
          f"{'MATCH' if long_avg > short_avg and min(lb, lc) > max(sb, sc) else 'no'}")
    print(f"\n  by state:  backwardated {back_avg:+.1f}bp   contangoed {cont_avg:+.1f}bp")
    print(f"  by side:   long {long_avg:+.1f}bp   short {short_avg:+.1f}bp")

    print("\n" + "=" * 80)
    print("3. IS THE ASYMMETRY SIGNIFICANT? — paired monthly test")
    print("=" * 80)
    print("  short-in-contango against the average of the other three cells, month by")
    print("  month, so common shocks cancel.\n")
    per = {}
    for (side, state), g in pos.groupby(["side", "state"]):
        mth = g.groupby("ym")["pnl"].sum()
        cap = g.groupby("ym")["notional"].sum()
        per[f"{side}/{state.replace('backwardated','back').replace('contangoed','cont')}"] = \
            (mth / cap.replace(0, np.nan))
    dfp = pd.DataFrame(per).dropna()
    if "short/cont" in dfp.columns and len(dfp) > 60:
        others = dfp[[c for c in dfp.columns if c != "short/cont"]].mean(axis=1)
        d = dfp["short/cont"] - others
        se = d.std(ddof=1) / np.sqrt(len(d))
        t_asym = d.mean() / se if se > 0 else np.nan
        print(f"  short/cont minus the other three: {d.mean()*1e4:+.1f}bp per month  "
              f"t {t_asym:+.2f}   n={len(d)}")
        print(f"  storage theory predicts this is NEGATIVE")
        asym_ok = np.isfinite(t_asym) and t_asym < -2
    else:
        t_asym, asym_ok = np.nan, False
        print("  insufficient overlap")

    print("\n" + "=" * 80)
    print("4. DOSE RESPONSE — is it stronger where the wall is CLOSER?")
    print("=" * 80)
    print("  Within contangoed shorts only, split by how deep the contango is. The bound")
    print("  binds harder the closer the basis sits to full carry, so the effect should")
    print("  strengthen monotonically with contango depth. A generic short-leg weakness")
    print("  predicts no gradient at all.\n")
    sc_pos = pos[(pos["side"] == "short") & (pos["state"] == "contangoed")].copy()
    if len(sc_pos) > 200:
        sc_pos["depth"] = -sc_pos["basis"]         # positive = deeper contango
        qs = sc_pos["depth"].quantile([0, 1/3, 2/3, 1.0]).to_numpy()
        labels = ["shallow contango", "medium contango", "deep contango"]
        for i, lab in enumerate(labels):
            seg = sc_pos[(sc_pos["depth"] >= qs[i]) & (sc_pos["depth"] <= qs[i+1])]
            mth = seg.groupby("ym")["pnl"].sum()
            cap = seg.groupby("ym")["notional"].sum()
            ret = (mth / cap.replace(0, np.nan)).dropna()
            if len(ret) < 36:
                continue
            se = ret.std(ddof=1) / np.sqrt(len(ret))
            print(f"    {lab:18s} {ret.mean()*1e4:>+8.1f}bp  "
                  f"t {ret.mean()/se if se>0 else np.nan:>+5.2f}  n={len(ret)}")
        print("\n  monotone deterioration with depth supports the bound. Flatness does not.")

    print("\n" + "=" * 80)
    print("5. PLACEBO — shuffle the curve-state tag across instruments each month")
    print("=" * 80)
    print("  Same book, same positions, same P&L. Only WHICH instrument is labelled")
    print("  backwardated is scrambled. If the cell structure survives that, it is not")
    print("  about the curve state.\n")
    rng = np.random.default_rng(0)
    dif = []
    for sd in range(a.seeds):
        _, pp = run(m, idm, shuffle_tag_seed=sd)
        tt = cell_table(pp)
        if tt.empty:
            continue
        def c2(side, state):
            r = tt[(tt["side"] == side) & (tt["state"] == state)]
            return float(r["mean_bp"].iloc[0]) if len(r) else np.nan
        v = c2("short", "contangoed")
        others = np.nanmean([c2("long", "backwardated"), c2("long", "contangoed"),
                             c2("short", "backwardated")])
        if np.isfinite(v) and np.isfinite(others):
            dif.append(v - others)
    real_diff = sc - np.nanmean([lb, lc, sb])
    placebo_ok = False
    if dif:
        dif = np.array(dif)
        z = (real_diff - dif.mean()) / max(dif.std(ddof=1), 1e-9)
        print(f"  placebo gap {dif.mean():+.1f} +/- {dif.std(ddof=1):.1f} bp "
              f"over {len(dif)} shuffles")
        print(f"  real gap {real_diff:+.1f}bp sits {z:+.1f} sd out   "
              f"beats {(dif > real_diff).mean():.0%} of shuffles in the predicted direction")
        placebo_ok = z < -2

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    checks = [
        ("short-in-contango is the weakest cell", weakest == "short/cont"),
        ("the asymmetry is significant (t < -2)", asym_ok),
        ("survives the curve-state placebo", placebo_ok),
    ]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    if all(v for _, v in checks):
        print("  THE STORAGE BOUND IS SUPPORTED. This is a claim about the SHAPE of the")
        print("  factor rather than its size, derived from an arbitrage that either exists")
        print("  or does not, and tested on a pattern no symmetric or beta story predicts.")
        print("  It also has a direct implementation consequence worth stating: the short")
        print("  leg is structurally weaker where the bound binds, which is an argument")
        print("  for asymmetric position limits rather than a symmetric long-short book.")
    elif weakest == "short/cont":
        print("  DIRECTIONALLY SUPPORTED, NOT SIGNIFICANT. The predicted cell is weakest")
        print("  but the test cannot separate it from noise. Report the table and the")
        print("  t-statistic exactly, and say the sample cannot resolve it. Given that the")
        print("  minimum detectable Sharpe difference here is 0.50, being unable to resolve")
        print("  an effect is the expected outcome, not a surprising one.")
    else:
        print("  NOT SUPPORTED. The predicted cell is not the weakest, so the storage-bound")
        print("  asymmetry does not appear in this sample. Report it as a fourth")
        print("  tested-and-rejected extension.")


if __name__ == "__main__":
    main()