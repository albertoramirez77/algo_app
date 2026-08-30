"""
neutrality.py — what kind of neutral is this book, exactly?

    python neutrality.py --prices px_clean.parquet

THE CLAIM UNDER TEST

The pitch says the book is "dollar-neutral by construction". Rank weights are demeaned and
do sum to zero, but they are then divided by each instrument's dollar volatility before
becoming positions. Converting back to notional leaves w_i / vol_i, which sums to zero only
if every instrument has the same volatility. They do not.

So the construction is INVERSE-VOLATILITY neutral - equal risk contribution per name - and
not dollar-neutral. That is the right construction for a cross-sectional relative-value
book, because equalising dollars would over-weight the quiet instruments and under-weight
the volatile ones, which is the opposite of what risk budgeting is for. But it is not what
the pitch says, and the pitch has to say the true thing.

This measures three quantities so the claim can be restated precisely:

    net dollar exposure     as a share of gross, month by month
    net RISK exposure       each position's notional times its own volatility, summed
    market beta             the outcome that actually matters to an allocator

A book can carry non-trivial net notional and still be market-neutral in the sense that
matters, if the long and short legs move together. Beta is the test of that, and it is
already measured at -0.06.
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
COST_MULTIPLE = 3.0


def load(path: str):
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
    med = df.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]
        n = med[s] * i.dollar_price_mult
        cost[s] = 1.5 * (i.tick_value / n * 1e4) + i.commission / n * 1e4
    cs = pd.Series(cost)
    df = df[~df["symbol"].isin(set(cs[cs > COST_MULTIPLE * cs.median()].index))].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    a = ap.parse_args()
    m = load(a.prices)

    rows, rets = [], {}
    prev = {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < 6:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        net_n = gross_n = net_r = gross_r = pnl = cost = 0.0
        held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * IDM / den))
            held[sym] = n
            notional = n * dpm * px
            net_n += notional; gross_n += abs(notional)
            net_r += notional * vol; gross_r += abs(notional * vol)
            pnl += notional * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * 3.0 / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        if gross_n > 0:
            rows.append(dict(ym=ym, net_n=net_n / CAPITAL, gross_n=gross_n / CAPITAL,
                             ratio_n=net_n / gross_n,
                             ratio_r=net_r / gross_r if gross_r > 0 else np.nan))
            rets[ym] = (pnl - cost) / CAPITAL

    D = pd.DataFrame(rows).set_index("ym")
    R = pd.Series(rets).sort_index()

    print("=" * 78)
    print("1. WEIGHTS BEFORE SIZING")
    print("=" * 78)
    print("  Rank weights are demeaned, so they sum to exactly zero every month. That is")
    print("  the property the pitch was describing. It is preserved only until the")
    print("  weights are divided by each instrument's dollar volatility.")

    print("\n" + "=" * 78)
    print("2. NET DOLLAR EXPOSURE AFTER SIZING")
    print("=" * 78)
    print(f"  {'':28s} {'mean':>10s} {'sd':>9s} {'min':>9s} {'max':>9s}")
    print(f"  {'net notional / capital':28s} {D['net_n'].mean():>+10.3f} "
          f"{D['net_n'].std():>9.3f} {D['net_n'].min():>+9.3f} {D['net_n'].max():>+9.3f}")
    print(f"  {'gross notional / capital':28s} {D['gross_n'].mean():>10.3f} "
          f"{D['gross_n'].std():>9.3f} {D['gross_n'].min():>9.3f} {D['gross_n'].max():>9.3f}")
    print(f"  {'net as share of gross':28s} {D['ratio_n'].mean():>+10.1%} "
          f"{D['ratio_n'].std():>9.1%} {D['ratio_n'].min():>+9.1%} "
          f"{D['ratio_n'].max():>+9.1%}")
    print(f"\n  |net| / gross averages {D['ratio_n'].abs().mean():.1%}. The book is NOT")
    print("  dollar-neutral, and the pitch should not say that it is.")

    print("\n" + "=" * 78)
    print("3. NET RISK EXPOSURE AFTER SIZING")
    print("=" * 78)
    print("  Each position's notional weighted by its own volatility, which is what the")
    print("  inverse-volatility construction is actually equalising.\n")
    print(f"  {'net risk as share of gross risk':34s} mean {D['ratio_r'].mean():>+7.1%}   "
          f"|mean| {D['ratio_r'].abs().mean():>6.1%}   sd {D['ratio_r'].std():>6.1%}")
    print("\n  If this is materially smaller than the dollar figure above, the book is")
    print("  balanced in risk rather than in dollars - which is the correct construction")
    print("  for a cross-sectional relative-value strategy and the accurate description.")

    print("\n" + "=" * 78)
    print("4. THE OUTCOME THAT ACTUALLY MATTERS")
    print("=" * 78)
    mkt = m.groupby("ym")["fwd"].mean().dropna()
    j = pd.concat([R.rename("s"), mkt.rename("m")], axis=1).dropna()
    X = np.column_stack([np.ones(len(j)), j["m"].to_numpy()])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ j["s"].to_numpy())
    e = j["s"].to_numpy() - X @ b
    se = e.std(ddof=2) / np.sqrt(len(j))
    r2 = 1 - e.var() / j["s"].var()
    print(f"  market beta                    {b[1]:+.3f}")
    print(f"  R-squared against the market   {r2:.4f}")
    print(f"  alpha                          {b[0]*12*100:+.2f}%/yr (t {b[0]/se:+.2f})")
    print("\n  A book can carry net notional and still be market-neutral in the sense an")
    print("  allocator cares about, if the long and short legs move together. Beta is the")
    print("  test of that, and beta is what should be quoted.")

    print("\n" + "=" * 78)
    print("HOW TO SAY IT IN THE PITCH")
    print("=" * 78)
    print("  Wrong:  'weights are scaled to unit gross, which makes the book")
    print("           dollar-neutral by construction'")
    print()
    print("  Right:  'Ranks are demeaned so the signal weights sum to zero, then divided")
    print(f"           by each instrument's dollar volatility so every name contributes")
    print(f"           equal risk. The book is therefore balanced in RISK rather than in")
    print(f"           dollars: net notional averages {D['ratio_n'].abs().mean():.0%} of gross, while the")
    print(f"           realised market beta is {b[1]:+.2f} and the complex explains {r2:.0%} of")
    print("           variance. Equalising dollars instead would over-weight the quiet")
    print("           instruments and under-weight the volatile ones, which is the")
    print("           opposite of what a risk budget is for.'")


if __name__ == "__main__":
    main()