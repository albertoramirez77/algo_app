"""
repair.py — fix the two data defects properly rather than explaining them away.

    python repair.py --in px_wide.parquet --out px_clean.parquet

THE ROOT CAUSE — both symptoms, one bug

fetch_curve.py assembles the wide file with

    w = a.merge(b, on="date", how="inner")

where `a` is the front leg and `b` the deferred leg. On a roll date Databento emits records
for BOTH the outgoing and incoming contract. Merging on date alone then produces a
CARTESIAN PRODUCT: two front records times two deferred records is four rows, and three of
those four pair contracts that never coexisted in the series.

That single bug produces both verification failures.

    6,007 duplicate (date, symbol) rows   - 100% on dates where contract_0 differs
    2 non-positive settlements            - both MCL, both showing -37.63

The second one is worth reading closely. WTI genuinely settled at -$37.63 on 2020-04-20 -
that part is history, not a data error. But the file also shows -37.63 on 2020-04-17, when
the May contract actually settled near $18. The same expiring contract appears on both
dates because the duplicate resolution picked the wrong row. A calendar-rolled continuous
series should never have been holding the May contract into its final settlement at all.

So this is not "two rows out of 130,864 that we can explain." It is a resolution rule
picking the wrong contract, and the negative price is how it announced itself.

THE FIX

For each (symbol, date) with more than one record, choose the contract the continuous
series actually resolved to, using continuity with the surrounding days:

    1  if one candidate matches the contract held on the PREVIOUS trading day, and the
       series has not yet reached a roll, prefer it
    2  otherwise prefer the candidate that matches the NEXT day's contract, since a roll
       that has happened is a roll that stays happened
    3  never select a contract with a non-positive settlement while a positive alternative
       exists
    4  break any remaining tie on open interest

Then verify what the repair claims: no duplicates, expiry never runs backwards within an
instrument, no front contract held past its own expiry, and the strategy's headline numbers
either survive or the change is quantified.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")


# ----------------------------------------------------------------------------------

def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One record per (symbol, date), chosen by continuity rather than by a heuristic.

    Returns the repaired frame and an audit trail of every decision, so each choice can be
    inspected rather than trusted.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    keep_idx, audit = [], []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values(["date"])
        by_date = {d: sub for d, sub in g.groupby("date", sort=True)}
        dates = sorted(by_date)
        chosen_contract: dict[pd.Timestamp, str] = {}

        # first pass: every date with a single record is unambiguous
        for d in dates:
            sub = by_date[d]
            if len(sub) == 1:
                chosen_contract[d] = sub["contract_0"].iloc[0]

        # second pass: resolve the ambiguous dates using their neighbours
        for i, d in enumerate(dates):
            sub = by_date[d]
            if len(sub) == 1:
                keep_idx.append(sub.index[0])
                continue

            prev_c = next((chosen_contract[dates[j]] for j in range(i - 1, -1, -1)
                           if dates[j] in chosen_contract), None)
            next_c = next((chosen_contract[dates[j]] for j in range(i + 1, len(dates))
                           if dates[j] in chosen_contract), None)

            cand = sub.copy()
            pos = cand["settle_0"] > 0
            reason = ""
            if pos.any():
                cand = cand[pos]                       # rule 3, applied first
                if len(cand) < len(sub):
                    reason = "dropped non-positive settle; "

            pick = None
            if next_c is not None and (cand["contract_0"] == next_c).any():
                pick = cand[cand["contract_0"] == next_c].iloc[0]
                reason += "matches next day (roll already happened)"
            elif prev_c is not None and (cand["contract_0"] == prev_c).any():
                pick = cand[cand["contract_0"] == prev_c].iloc[0]
                reason += "matches previous day (roll not yet)"
            else:
                cand = cand.sort_values("oi_0", na_position="first")
                pick = cand.iloc[-1]
                reason += "no neighbour match; highest open interest"

            chosen_contract[d] = pick["contract_0"]
            keep_idx.append(pick.name)
            audit.append(dict(symbol=sym, date=d, n_candidates=len(sub),
                              chosen=pick["contract_0"], settle=pick["settle_0"],
                              rejected="|".join(
                                  sub.loc[sub.index != pick.name, "contract_0"].astype(str)),
                              rejected_settles="|".join(
                                  sub.loc[sub.index != pick.name, "settle_0"]
                                  .round(2).astype(str)),
                              reason=reason))

    return df.loc[sorted(keep_idx)].reset_index(drop=True), pd.DataFrame(audit)


def sanity(df: pd.DataFrame, label: str) -> dict:
    d = df.copy()
    dup = d.duplicated(["date", "symbol"]).sum()
    neg = ((d["settle_0"] <= 0) | (d["settle_1"] <= 0)).sum()
    gap = (d["expiry_1"] - d["expiry_0"]).dt.days
    inv = (gap <= 0).sum()
    past = (d["expiry_0"] < d["date"]).sum()
    back = 0
    for _, g in d.sort_values(["symbol", "date"]).groupby("symbol"):
        back += (g["expiry_0"].diff().dt.days < 0).sum()
    print(f"  {label}")
    print(f"    rows {len(d):,}   duplicates {dup:,}   non-positive settles {neg}")
    print(f"    inverted legs {inv}   front expiry already past {past}   "
          f"expiry runs backwards {back}")
    return dict(rows=len(d), dup=dup, neg=neg, inv=inv, past=past, back=back)


def quick_backtest(df: pd.DataFrame) -> dict:
    """Minimal reimplementation, just enough to confirm the headline numbers move or not."""
    d = df.copy()
    d["asset"] = d["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    d = d[(d["asset"] == "commodity") & (d["contract_0"] != d["contract_1"])]
    d = d.sort_values(["symbol", "date"])
    for leg in ("0", "1"):
        blk = d.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = d.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d[f"r{leg}"] = np.log(d[f"settle_{leg}"] / prev)
        d.loc[~np.isfinite(d[f"r{leg}"]), f"r{leg}"] = np.nan
    d["ym"] = d["date"].dt.to_period("M")
    m = (d.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"])
    g = m.groupby("symbol")
    m["bm"] = (g["r0"].transform(lambda s: s.rolling(12, min_periods=12).sum())
               - g["r1"].transform(lambda s: s.rolling(12, min_periods=12).sum()))
    m["vol"] = (g["r0"].transform(lambda s: s.rolling(6, min_periods=3).std())
                * np.sqrt(12)).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)

    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    idm = min(1 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), 2.5)

    prev_pos, out = {}, {}
    for ym, gg in m.groupby("ym"):
        s = gg[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < 6:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0
        held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            npos = float(np.round(wi * 450_000 * 0.20 * idm / den))
            held[sym] = npos
            pnl += npos * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(npos - prev_pos.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * 3 / 1e4)
        for sym in set(prev_pos) - set(held):
            cost += abs(prev_pos[sym]) * BY_SYMBOL[sym].commission
        prev_pos = held
        out[ym] = (pnl - cost) / 450_000
    ser = pd.Series(out).sort_index()
    yrs = len(ser) / 12
    av = ser.std(ddof=1) * np.sqrt(12)
    sr = (ser.mean() * 12) / av if av > 0 else np.nan
    return dict(sharpe=sr, t=sr * np.sqrt(yrs), ann=ser.mean() * 12, months=len(ser))


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="px_wide.parquet")
    ap.add_argument("--out", default="px_clean.parquet")
    ap.add_argument("--audit", default="repair_audit.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

    print("=" * 80)
    print("1. BEFORE")
    print("=" * 80)
    b4 = sanity(df, a.src)

    print("\n" + "=" * 80)
    print("2. THE MCL RECORDS THAT EXPOSED THE BUG")
    print("=" * 80)
    win = df[(df["symbol"] == "MCL") &
             (df["date"].between("2020-04-14", "2020-04-23"))]
    if len(win):
        cols = ["date", "contract_0", "settle_0", "expiry_0", "contract_1",
                "settle_1", "oi_0"]
        print(win[[c for c in cols if c in win.columns]]
              .sort_values(["date", "contract_0"]).to_string(index=False))
        print("\n  WTI really did settle at -$37.63 on 2020-04-20. It did NOT settle there")
        print("  on 2020-04-17. If the same contract shows the same negative price on both")
        print("  dates, the resolution picked a stale record, and a calendar-rolled series")
        print("  should not have been holding that contract into settlement at all.")

    print("\n" + "=" * 80)
    print("3. REPAIRING")
    print("=" * 80)
    fixed, audit = resolve_duplicates(df)
    print(f"  {len(audit):,} ambiguous (symbol, date) pairs resolved")
    if len(audit):
        print("\n  by reason:")
        for reason, g in audit.groupby("reason"):
            print(f"    {len(g):>6,}  {reason}")
        print(f"\n  sample of the decisions:")
        print(audit.head(8).to_string(index=False))
        audit.to_csv(a.audit, index=False)
        print(f"\n  full audit trail written to {a.audit} — every choice is inspectable")

    print("\n" + "=" * 80)
    print("4. AFTER")
    print("=" * 80)
    af = sanity(fixed, a.out)

    still_neg = fixed[(fixed["settle_0"] <= 0) | (fixed["settle_1"] <= 0)]
    if len(still_neg):
        print(f"\n  {len(still_neg)} non-positive settlements remain:")
        print(still_neg[["date", "symbol", "contract_0", "settle_0",
                         "settle_1"]].to_string(index=False))
        print("\n  These have no positive alternative on that date, so the record is the")
        print("  market's own. A log return is undefined against a negative price, so the")
        print("  affected instrument-month leaves the cross-section. That is the correct")
        print("  treatment: the strategy cannot express a position it could not have held.")
        fixed = fixed[~((fixed["settle_0"] <= 0) | (fixed["settle_1"] <= 0))]
        print(f"  {len(still_neg)} rows removed; {len(fixed):,} remain")

    print("\n" + "=" * 80)
    print("5. DID THE REPAIR MOVE THE STRATEGY?")
    print("=" * 80)
    print("  A repair that silently changes the headline is not a repair, it is a new")
    print("  result. Both are computed here so the difference is explicit.\n")
    before = quick_backtest(df)
    after = quick_backtest(fixed)
    print(f"  {'':10s} {'Sharpe':>9s} {'t':>7s} {'return':>9s} {'months':>8s}")
    print(f"  {'before':10s} {before['sharpe']:>+9.3f} {before['t']:>+7.2f} "
          f"{before['ann']*100:>+8.2f}% {before['months']:>8d}")
    print(f"  {'after':10s} {after['sharpe']:>+9.3f} {after['t']:>+7.2f} "
          f"{after['ann']*100:>+8.2f}% {after['months']:>8d}")
    delta = after["sharpe"] - before["sharpe"]
    print(f"  {'change':10s} {delta:>+9.3f}")
    print()
    if abs(delta) < 0.05:
        print("  The headline is unchanged. The defects were real but immaterial to the")
        print("  result, which is the outcome you want: the fix is correct AND the earlier")
        print("  numbers were not being propped up by bad rows.")
    else:
        print("  The headline MOVED. Report the repaired number and say why it changed.")
        print("  A Sharpe that depends on a duplicate-resolution rule is a Sharpe that")
        print("  needs that rule stated in the pitch.")

    fixed.to_parquet(a.out, index=False)
    print(f"\n  -> {a.out}   {len(fixed):,} rows, {fixed['symbol'].nunique()} instruments")
    print("\n  Next: python verify.py --prices " + a.out)
    print("  All 26 checks should now pass. If any still fail, the repair is incomplete.")


if __name__ == "__main__":
    main()