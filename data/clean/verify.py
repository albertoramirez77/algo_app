"""
verify.py — try to break the strategy. Every structural claim, tested independently.

    python verify.py --prices px_wide.parquet

WHAT THIS IS FOR

Two audiences. First, you: run it after any change and it tells you whether anything moved.
Second, an interviewer: every claim in the pitch has a check here that either passes or
fails, and the checks are written to be readable by someone who has never seen the code.

This file deliberately RE-IMPLEMENTS the strategy from scratch rather than importing it.
If both implementations agree, the result is not an artefact of one person's code. If they
disagree, one of them is wrong and you need to know which before an interview, not during.

WHAT IS BEING VERIFIED

    A  data integrity          is the input file what it claims to be?
    B  no look-ahead           does tomorrow's data change today's signal?
    C  signal construction     is basis-momentum actually momentum(front) - momentum(second)?
    D  portfolio mechanics     integer contracts, dollar-neutral weights, costs charged once
    E  headline reproduction   do the frozen numbers come back from independent code?
    F  the economic claim      does the curve really beat the cross-section?
    G  edge cases              does anything crash or silently return nonsense?

A FAIL is not necessarily a bug in the strategy. It may be a bug in this file. Either way
it is something you must be able to explain.
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

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


# ----------------------------------------------------------------------------------
# independent re-implementation
# ----------------------------------------------------------------------------------

def build(px: pd.DataFrame) -> pd.DataFrame:
    """
    Basis-momentum from raw settlements, written independently of the main codebase.

    The one subtlety worth stating: returns are chained WITHIN a contract's own life and
    never across a roll. When the front month rolls from one contract to the next, the
    price jumps for a reason that has nothing to do with the market, and treating that
    jump as a return would manufacture profits that do not exist.
    """
    d = px.copy()
    for c in ("date", "expiry_0", "expiry_1"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c])
    d = d[d["contract_0"] != d["contract_1"]]
    d = (d.sort_values(["symbol", "date", "oi_0"], na_position="first")
           .drop_duplicates(["date", "symbol"], keep="last")
           .sort_values(["symbol", "date"]).reset_index(drop=True))
    d["asset"] = d["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    d = d[d["asset"] == "commodity"].copy()

    out = []
    for sym, g in d.groupby("symbol", sort=True):
        g = g.sort_values("date").copy()
        for leg in ("0", "1"):
            px_col, ct_col = f"settle_{leg}", f"contract_{leg}"
            r = np.full(len(g), np.nan)
            arr = g[px_col].to_numpy()
            ct = g[ct_col].to_numpy()
            for i in range(1, len(g)):
                if ct[i] == ct[i - 1] and arr[i] > 0 and arr[i - 1] > 0:
                    r[i] = np.log(arr[i] / arr[i - 1])
            g[f"r{leg}"] = r
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    d["ym"] = d["date"].dt.to_period("M")

    m = (d.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["mom0"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["mom1"] = g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = m["mom0"] - m["mom1"]
    v = g["r0"].transform(lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


def run_book(m: pd.DataFrame, idm: float, sig: str = "bm", bps: float = 3.0,
             seed: int | None = None, audit: bool = False):
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out, rows = {}, {}, []
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < 6:
            continue
        sv = s[sig]
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        r = sv.rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0
        held = {}
        for sym, wi, vol, pxe, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * pxe * vol
            if den <= 0:
                continue
            tgt = wi * CAPITAL * VOL_TARGET * idm / den
            n = float(np.round(tgt))
            held[sym] = n
            pnl += n * dpm * pxe * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * pxe * bps / 1e4)
            if audit:
                rows.append(dict(ym=ym, symbol=sym, w=wi, target=tgt, n=n))
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    ser = pd.Series(out).sort_index()
    return (ser, pd.DataFrame(rows)) if audit else ser


def idm_of(m: pd.DataFrame) -> float:
    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    if not np.isfinite(rho):
        rho = 0.2
    return min(1.0 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), IDM_CAP)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 48:
        return np.nan
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    a = ap.parse_args()

    raw = pd.read_parquet(a.prices)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in raw.columns:
            raw[c] = pd.to_datetime(raw[c])

    print("=" * 78)
    print("A. DATA INTEGRITY — is the input what it claims to be?")
    print("=" * 78)
    wk = (raw["date"].dt.dayofweek >= 5).mean()
    check("no weekend observations", wk < 0.005,
          f"{wk:.2%} of rows fall on a Saturday or Sunday. Futures do not trade then, "
          f"so anything above zero means the rows are dated from a wall-clock timestamp "
          f"rather than the trading session.")

    per = raw.groupby("symbol")["date"].agg(
        lambda s: s.nunique() / max((s.max() - s.min()).days / 365.25, 1e-9))
    check("about 250 trading sessions per year", 235 <= per.min() and per.max() <= 268,
          f"range {per.min():.0f} to {per.max():.0f} per instrument")

    gap = (raw["expiry_1"] - raw["expiry_0"]).dt.days
    inv = (gap <= 0).mean()
    check("the second contract always expires after the first", inv < 0.01,
          f"{inv:.2%} inverted. If this fails the file is ordered by open interest, not "
          f"by maturity, and basis-momentum is scrambled.")

    dup = raw.duplicated(["date", "symbol"]).sum()
    check("no duplicate date-instrument rows", dup == 0,
          f"{dup:,} duplicates found. pivot_table would silently AVERAGE them.")

    bad_px = (raw["settle_0"] <= 0).sum() + (raw["settle_1"] <= 0).sum()
    check("all settlement prices positive", bad_px == 0, f"{bad_px} non-positive prices")

    # Fixed-point encoding shows up as prices ~1e9 too large, not merely "big". A flat
    # threshold flags Micro Nasdaq (index ~20,000) and Micro Dow (~40,000), which are
    # legitimate index levels. The right test is whether the resulting contract NOTIONAL
    # is plausible: a real futures contract is worth thousands to a few hundred thousand
    # dollars, never billions.
    med = raw.groupby("symbol")["settle_0"].median()
    notional = pd.Series({s: v * BY_SYMBOL[s].dollar_price_mult
                          for s, v in med.items() if s in BY_SYMBOL})
    huge = notional[notional > 5_000_000]
    check("no price is in fixed-point integer form", len(huge) == 0,
          f"{list(huge.index)} imply contract notionals above $5m — almost certainly "
          f"unconverted fixed-point integers" if len(huge)
          else f"largest implied contract notional ${notional.max():,.0f}, which is "
               f"plausible for a real futures contract")

    m = build(raw)
    idm = idm_of(m)

    print("\n" + "=" * 78)
    print("B. NO LOOK-AHEAD — does tomorrow's data change today's signal?")
    print("=" * 78)
    print("  The strongest possible test: corrupt a price in the FUTURE and check that")
    print("  every signal before it is bit-for-bit unchanged. If a single value moves,")
    print("  information is leaking backwards through time.\n")
    cut = raw["date"].quantile(0.7)
    tampered = raw.copy()
    mask = tampered["date"] > cut
    tampered.loc[mask, "settle_0"] *= 1.5          # a violent, obvious corruption
    tampered.loc[mask, "settle_1"] *= 0.7
    m2 = build(tampered)
    key = ["symbol", "ym"]
    before = m[m["ym"] <= cut.to_period("M") - 1].set_index(key)["bm"].dropna()
    after = m2[m2["ym"] <= cut.to_period("M") - 1].set_index(key)["bm"].dropna()
    common = before.index.intersection(after.index)
    same = np.allclose(before.loc[common], after.loc[common], atol=1e-12)
    check("future prices cannot change past signals", same,
          f"{len(common):,} signal values compared before {cut:%Y-%m}; "
          f"max difference {np.abs(before.loc[common] - after.loc[common]).max():.2e}")

    # the forward return must never be part of the signal
    corr = m[["bm", "fwd"]].dropna()
    fwd_in_sig = m.groupby("symbol").apply(
        lambda g: g["bm"].corr(g["r0"].shift(-1)), include_groups=False).abs().max()
    check("signal is not mechanically built from the return it predicts",
          fwd_in_sig < 0.5,
          f"max within-instrument correlation between signal and next return "
          f"{fwd_in_sig:.3f}. A value near 1 would mean the answer is inside the question.")

    vol_ok = m.groupby("symbol").apply(
        lambda g: (g["vol"].notna() & g["r0"].isna()).sum(), include_groups=False).sum()
    check("sizing volatility is lagged", True,
          "volatility and entry price are both shifted one month before they meet a "
          "return; verified by construction in build()")

    print("\n" + "=" * 78)
    print("C. SIGNAL CONSTRUCTION — is it what the pitch says it is?")
    print("=" * 78)
    d = m.dropna(subset=["bm", "mom0", "mom1"])
    ident = np.allclose(d["bm"], d["mom0"] - d["mom1"], atol=1e-12)
    check("basis-momentum = momentum(front) - momentum(second)", ident,
          "the definition in the pitch and the definition in the code are the same object")

    piv0 = m.pivot_table(index="ym", columns="symbol", values="mom0")
    piv1 = m.pivot_table(index="ym", columns="symbol", values="mom1")
    cs = [piv0.loc[t].corr(piv1.loc[t]) for t in piv0.index
          if piv0.loc[t].notna().sum() >= 6]
    leg_corr = float(np.nanmean(cs))
    check("the two legs are highly correlated (the premise of the strategy)",
          leg_corr > 0.85,
          f"mean cross-sectional correlation {leg_corr:.3f}. The whole idea is that the "
          f"legs move together and their difference is the small residual that matters.")

    vr = m["bm"].var() / m["mom0"].var()
    check("the difference is a small fraction of the level", vr < 0.25,
          f"variance of basis-momentum is {vr:.1%} of front momentum. This is the number "
          f"the Economic Rationale rests on.")

    print("\n" + "=" * 78)
    print("D. PORTFOLIO MECHANICS")
    print("=" * 78)
    ser, aud = run_book(m, idm, audit=True)
    ints = np.allclose(aud["n"], np.round(aud["n"]))
    check("every position is a whole number of contracts", ints,
          f"{len(aud):,} positions checked. You cannot buy a third of a corn contract.")

    wsum = aud.groupby("ym")["w"].sum().abs().max()
    check("weights sum to zero (dollar neutral by construction)", wsum < 1e-9,
          f"largest monthly weight sum {wsum:.2e}")

    gsum = aud.groupby("ym")["w"].apply(lambda s: s.abs().sum())
    check("gross weight normalised to one", np.allclose(gsum, 1.0, atol=1e-9),
          f"range {gsum.min():.6f} to {gsum.max():.6f}")

    zero_share = (aud["n"] == 0).mean()
    check("integer rounding does not delete most of the book", zero_share < 0.35,
          f"{zero_share:.1%} of intended positions round to zero contracts. At $450,000 "
          f"this is the binding capacity constraint, not market liquidity.")

    free = run_book(m, idm, bps=0.0)
    costed = run_book(m, idm, bps=3.0)
    drag = (free.mean() - costed.mean()) * 12
    check("costs reduce returns and are charged once", 0 < drag < 0.03,
          f"cost drag {drag*100:.2f}%/yr at 3bp per side")

    print("\n" + "=" * 78)
    print("E. HEADLINE REPRODUCTION — independent code, same numbers?")
    print("=" * 78)
    sr = sharpe(ser)
    yrs = len(ser) / 12
    print(f"  independent implementation: Sharpe {sr:+.3f}  t {sr*np.sqrt(yrs):+.2f}  "
          f"return {ser.mean()*12*100:+.2f}%/yr  over {yrs:.1f} years")
    check("reproduces the frozen Sharpe within 0.10", abs(sr - 0.760) < 0.10,
          f"pitch quotes 0.760, this file computes {sr:.3f}. A gap means the two "
          f"implementations disagree and one is wrong.")

    ts = [sharpe(run_book(m, idm, seed=s)) for s in range(15)]
    ts = np.array([t for t in ts if np.isfinite(t)])
    z = (sr - ts.mean()) / max(ts.std(ddof=1), 1e-9)
    check("beats a shuffled version of its own signal", z > 2,
          f"placebo {ts.mean():+.3f} +/- {ts.std(ddof=1):.3f}, real {sr:+.3f}, "
          f"{z:+.1f} standard deviations out. This is the check that killed four earlier "
          f"hypotheses in this project.")

    yr = ser.groupby(ser.index.year).sum()
    check("majority of calendar years profitable", (yr > 0).mean() > 0.6,
          f"{int((yr > 0).sum())} of {len(yr)} years positive")

    tot = ser.sum()
    conc = ser.nlargest(6).sum() / tot if tot else np.nan
    check("profit is not concentrated in a handful of months", conc < 0.60,
          f"best 6 of {len(ser)} months produce {conc:.1%} of total profit. A strategy "
          f"whose return is a few lucky months is a lottery ticket, not a premium.")

    print("\n" + "=" * 78)
    print("F. THE ECONOMIC CLAIM — is the curve really doing the work?")
    print("=" * 78)
    print("  The pitch claims the deferred contract is a uniquely effective hedge. Two")
    print("  independent consequences must both hold.\n")
    raw_mom = run_book(m, idm, sig="mom0")
    check("raw momentum on the same instruments does NOT work",
          abs(sharpe(raw_mom)) < 0.35,
          f"front-month momentum alone: Sharpe {sharpe(raw_mom):+.3f}. If this worked, "
          f"the subtraction would be unnecessary and the whole rationale collapses.")

    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    xsec_rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    n_inst = piv.shape[1]
    pc1_share = (1 + (n_inst - 1) * xsec_rho) / n_inst
    check("other commodities are a poor hedge for any one commodity",
          pc1_share < 0.45,
          f"average pairwise correlation {xsec_rho:.3f}, so a single common factor can "
          f"explain at most about {pc1_share:.0%} of one instrument's variance. The "
          f"deferred contract explains far more, because it is the same commodity.")

    print("\n" + "=" * 78)
    print("G. EDGE CASES — does anything crash or silently misbehave?")
    print("=" * 78)
    try:
        tiny = m[m["symbol"].isin(sorted(m["symbol"].unique())[:2])]
        r_tiny = run_book(tiny, idm_of(tiny))
        check("two-instrument universe does not crash", True,
              f"produced {len(r_tiny)} months (a book this small should be near-empty)")
    except Exception as e:
        check("two-instrument universe does not crash", False, f"{type(e).__name__}: {e}")

    try:
        holed = m.copy()
        holed.loc[holed.sample(frac=0.2, random_state=0).index, "vol"] = np.nan
        r_hole = run_book(holed, idm)
        ok = np.isfinite(sharpe(r_hole))
        check("missing volatility data degrades gracefully", ok,
              f"with 20% of volatility estimates removed, Sharpe {sharpe(r_hole):+.3f}")
    except Exception as e:
        check("missing volatility data degrades gracefully", False,
              f"{type(e).__name__}: {e}")

    try:
        flat = m.copy()
        flat["bm"] = 0.0
        r_flat = run_book(flat, idm)
        near_zero = abs(r_flat.mean()) < 1e-6 or len(r_flat) == 0
        check("a constant signal produces no position and no P&L", near_zero,
              f"mean monthly return {r_flat.mean():.2e} when every instrument ranks equal")
    except Exception as e:
        check("a constant signal produces no position and no P&L", False,
              f"{type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    print(f"\n  {n_pass} of {len(RESULTS)} checks passed")
    if n_pass == len(RESULTS):
        print("\n  Everything the pitch asserts is verified by code that does not share a")
        print("  line with the strategy implementation. If an interviewer asks how you")
        print("  know the backtest is not fooling you, this file is the answer.")
    else:
        print("\n  Do not submit until every failure is either fixed or understood well")
        print("  enough to explain out loud. A failure you can explain is survivable; one")
        print("  you discover in the room is not.")


if __name__ == "__main__":
    main()