"""
netcap.py — bounding the one uncontrolled tail left in the book.

    python netcap.py --prices px_clean.parquet

THE PROBLEM

Signal weights are demeaned and sum to zero. They are then divided by each instrument's
dollar volatility, which balances RISK across names but leaves dollar exposure unconstrained.
Measured on the real book, net notional averages a harmless 3% of gross but ranges from
-46% to +49% in individual months - at the extremes, roughly 1.3 to 1.9 times capital in
outright directional exposure. That is the only genuinely uncontrolled tail found in this
project, and "small but not zero" understates it.

THE FIX, AND WHY IT IS EXACT RATHER THAN APPROXIMATE

Position i receives notional w_i x K / vol_i, where K is the common capital-and-target
scalar. So net notional is proportional to the sum of w_i / vol_i, not to the sum of w_i.
Writing v_i = 1 / vol_i, subtracting a single constant c from every weight gives

    sum (w_i - c) v_i  =  sum w_i v_i  -  c sum v_i

which is exactly zero at c = (sum w_i v_i) / (sum v_i). One constant, applied to every
instrument, removes net dollar exposure completely without reordering anything: every
position moves by the same amount, so the ranking the signal produced is untouched.

A partial version scales that constant so the book lands on a chosen cap rather than at
zero, which keeps more of the original weights while bounding the tail.

WHAT WOULD MAKE THIS NOT WORTH ADOPTING

If capping costs meaningful Sharpe, it is buying a tail that never actually cost anything.
The test is whether the drawdown and the exposure range improve by more than the return
gives up - and if they do not, the honest report is that the cap was tested and rejected.
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
CAPS = [None, 0.40, 0.25, 0.15, 0.0]


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
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    df = df[~df["symbol"].isin(drop)].copy()
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
    return m, drop


def apply_cap(w: np.ndarray, vol: np.ndarray, cap: float | None) -> np.ndarray:
    """
    Subtract a single constant from every weight so that net notional lands on the cap.
    Every position moves by the same amount, so the signal's ordering is preserved.
    """
    if cap is None:
        return w
    v = 1.0 / vol
    net, gross = float(np.sum(w * v)), float(np.sum(np.abs(w) * v))
    if gross <= 0:
        return w
    if abs(net) / gross <= cap:
        return w
    c_full = net / np.sum(v)            # lambda = 1 zeroes net exposure exactly
    lo, hi = 0.0, 1.0
    for _ in range(40):                 # bisection on the fraction of the constant applied
        mid = (lo + hi) / 2
        wm = w - mid * c_full
        nm, gm = float(np.sum(wm * v)), float(np.sum(np.abs(wm) * v))
        if gm > 0 and abs(nm) / gm > cap:
            lo = mid
        else:
            hi = mid
    return w - hi * c_full


def book(m: pd.DataFrame, cap=None, bps=3.0, drop=None, min_n=6):
    if drop:
        m = m[m["symbol"] != drop]
    prev, out, diag = {}, {}, []
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy().astype(float)
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        w = apply_cap(w, s["vol"].to_numpy(), cap)
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0
        held = {}
        net_n = gross_n = 0.0
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
            pnl += notional * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
        if gross_n > 0:
            diag.append(dict(ym=ym, ratio=net_n / gross_n, net=net_n / CAPITAL,
                             gross=gross_n / CAPITAL))
    return pd.Series(out).sort_index(), pd.DataFrame(diag).set_index("ym")


def st(r):
    r = r.dropna()
    if len(r) < 24:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12, vol=av,
                dd=float((eq / eq.cummax() - 1).min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    a = ap.parse_args()
    m, dropped = load(a.prices)
    syms = sorted(m["symbol"].unique())
    print(f"  universe rule excluded {sorted(dropped)}; {len(syms)} instruments\n")

    print("=" * 80)
    print("1. THE CAP LADDER")
    print("=" * 80)
    print("  A single constant is subtracted from every weight, so no position is")
    print("  reordered relative to any other - only the whole book shifts.\n")
    print(f"  {'cap':>8s} {'Sharpe':>8s} {'return':>8s} {'vol':>7s} {'maxDD':>8s} "
          f"{'|net|/gross':>12s} {'worst month':>12s}")
    res = {}
    for cap in CAPS:
        r, d = book(m, cap=cap)
        s = st(r)
        res[cap] = (s, d, r)
        lab = "none" if cap is None else ("full" if cap == 0.0 else f"{cap:.0%}")
        print(f"  {lab:>8s} {s['sharpe']:>8.3f} {s['ann']*100:>7.2f}% "
              f"{s['vol']*100:>6.1f}% {s['dd']*100:>7.1f}% "
              f"{d['ratio'].abs().mean():>11.1%} {d['ratio'].abs().max():>11.1%}")

    base_s, base_d, base_r = res[None]
    print("\n  The 'worst month' column is the point of the exercise: uncapped, the book")
    print("  reaches a net directional position it never intended to take.")

    print("\n" + "=" * 80)
    print("2. WHAT THE CAP COSTS AND WHAT IT BUYS")
    print("=" * 80)
    print(f"  {'cap':>8s} {'d Sharpe':>10s} {'d maxDD':>10s} {'d worst net':>13s} "
          f"{'jackknife':>10s}")
    for cap in CAPS[1:]:
        s, d, r = res[cap]
        jk = [st(book(m, cap=cap, drop=x)[0])["sharpe"] for x in syms]
        jk = [v for v in jk if np.isfinite(v)]
        lab = "full" if cap == 0.0 else f"{cap:.0%}"
        print(f"  {lab:>8s} {s['sharpe']-base_s['sharpe']:>+10.3f} "
              f"{(s['dd']-base_s['dd'])*100:>+9.1f}pp "
              f"{(d['ratio'].abs().max()-base_d['ratio'].abs().max())*100:>+12.1f}pp "
              f"{min(jk) if jk else np.nan:>10.3f}")
    print(f"\n  uncapped jackknife worst case for reference: ", end="")
    jk0 = [st(book(m, drop=x)[0])["sharpe"] for x in syms]
    jk0 = [v for v in jk0 if np.isfinite(v)]
    print(f"{min(jk0):.3f}")

    print("\n" + "=" * 80)
    print("3. EXPOSURE DISTRIBUTION, UNCAPPED VERSUS CAPPED")
    print("=" * 80)
    print(f"  {'':16s} {'mean':>9s} {'sd':>9s} {'5th pct':>9s} {'95th pct':>9s} "
          f"{'worst':>9s}")
    for cap in (None, 0.25, 0.0):
        d = res[cap][1]["ratio"]
        lab = "uncapped" if cap is None else ("fully neutral" if cap == 0.0
                                              else f"capped {cap:.0%}")
        print(f"  {lab:16s} {d.mean():>+9.1%} {d.std():>9.1%} "
              f"{d.quantile(0.05):>+9.1%} {d.quantile(0.95):>+9.1%} "
              f"{d.abs().max():>9.1%}")
    over = (base_d["ratio"].abs() > 0.25).mean()
    print(f"\n  months where uncapped net exposure exceeds 25% of gross: {over:.0%}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    s25 = res[0.25][0]
    d25 = res[0.25][1]
    d_sr = s25["sharpe"] - base_s["sharpe"]
    d_dd = (s25["dd"] - base_s["dd"]) * 100
    print(f"  at a 25% cap:  Sharpe {d_sr:+.3f}   drawdown {d_dd:+.1f}pp   "
          f"worst net {base_d['ratio'].abs().max():.0%} -> {d25['ratio'].abs().max():.0%}")
    print()
    if d_sr > -0.05:
        print("  ADOPT. The cap bounds a tail the strategy never chose to take, at a cost")
        print("  well inside the noise of this sample. Report it as a measured control")
        print("  rather than a proposed one, and quote both the average exposure and the")
        print("  bounded worst case.")
    elif d_sr > -0.12:
        print("  DEFENSIBLE EITHER WAY. The cap costs real return but removes a genuine")
        print("  tail. State the trade-off and the choice made; a reader can disagree with")
        print("  the choice but not with the disclosure.")
    else:
        print("  DO NOT ADOPT at this level. The cap costs more than the tail is worth on")
        print("  this sample. Report that it was tested and rejected, and disclose the")
        print("  exposure range instead of controlling it.")


if __name__ == "__main__":
    main()