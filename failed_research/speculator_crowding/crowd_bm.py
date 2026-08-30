"""
crowd_bm.py — does basis-momentum pay more when speculators are crowded ELSEWHERE?

    python crowd_bm.py --cot cot.parquet --prices px_wide.parquet

THE RESEARCH THAT SHAPES THIS TEST

Boos & Grob (2023), "Tracking speculative trading": trend signals explain speculators'
position CHANGES in commodity futures with average R-squared above 40% across 23
commodities, and - decisively for this design - "the basis and other popular trading
signals do not improve the position change forecast." Managed money funds are predominantly
trend-followers. Producers act as contrarians who mirror momentum traders.

Uhl (2025, Review of Financial Economics) asks the analogous question for TREND: does
speculative crowding affect trend-following performance? He measures crowding as the
ALIGNMENT of net speculative open interest with a generic trend-following position, rather
than as a generic ratio of speculators to total open interest. Nobody has run that test on
basis-momentum.

THE HYPOTHESIS

Speculators crowd into TREND. Basis-momentum is a signal they demonstrably do NOT trade.
When speculator positioning is heavily aligned with the trend signal, they are demanding
liquidity for reasons unrelated to the curve, and someone must absorb it. Basis-momentum
should be compensated more richly in exactly those states - not because BM is crowded, but
because the crowding is happening elsewhere in the same contract.

This is the mechanism Boons & Prado assert but never test with positioning data. Every
proxy tried in this project so far - volatility, basis, turnover - measures market STATE.
Positioning identifies AGENTS.

THREE DISCRIMINATING PREDICTIONS

  P1  the conditioning works through TREND-alignment crowding, not BM-alignment crowding.
      If speculators were crowded into basis-momentum itself, the sign would flip: crowding
      into a signal DECAYS it. Opposite predictions from the same data.

  P2  it works through MONEY MANAGERS specifically. Producers are contrarian and swap
      dealers intermediate index flow. If all three categories condition equally, the
      effect is generic activity and the mechanism claim fails.

  P3  producer crowding and money-manager crowding carry OPPOSITE signs, because producers
      mirror momentum traders as contrarians.

No generic-activity story predicts all three. That is what makes this testable rather than
a narrative.

WHAT KILLED THE EARLIER POSITIONING TEST, AND WHY THIS IS DIFFERENT

Hypothesis 1 asked whether positioning PREDICTS returns. It does not: slope 0.003, t 0.08,
with power to detect one sixtieth of the published effect. This asks whether positioning
CONDITIONS a price-based signal. Different question, different estimand, and the first
result does not settle it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL, UNIVERSE
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL = 450_000.0
VOL_TARGET = 0.20
IDM_CAP = 2.5
J = 12
VOL_WINDOW = 6
TREND_LOOKBACK = 12          # months, for the generic trend position
CROWD_WINDOW = 36            # months, for standardising alignment within instrument


# ----------------------------------------------------------------------------------
# prices
# ----------------------------------------------------------------------------------

def load_prices(path: str) -> pd.DataFrame:
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
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["spread"] = m["r0"] - m["r1"]
    m["bm"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["mom"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    # generic trend position: sign of trailing return, the simplest defensible TSMOM
    m["trend_pos"] = np.sign(
        g["r0"].transform(lambda s: s.rolling(TREND_LOOKBACK, min_periods=TREND_LOOKBACK).sum()))
    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


# ----------------------------------------------------------------------------------
# positioning
# ----------------------------------------------------------------------------------

def load_cot(path: str, verify_only: bool = False) -> pd.DataFrame:
    """
    Monthly positioning per instrument, with the publication lag enforced.

    The CFTC measures positions at Tuesday's close and publishes Friday 15:30 ET, which is
    after settlement for every contract here. So a report measured in month M is only fully
    actionable from the following month. Every positioning column is therefore lagged one
    month before it touches a return.
    """
    c = pd.read_parquet(path)
    c["report_date"] = pd.to_datetime(c["report_date"])
    c["ym"] = c["report_date"].dt.to_period("M")

    have = set(c["symbol"].unique())
    want = {i.symbol for i in UNIVERSE if i.asset == "commodity"}
    missing = sorted(want - have)
    print("=" * 82)
    print("0. POSITIONING DATA — coverage gate")
    print("=" * 82)
    print(f"  COT rows {len(c):,}   instruments {len(have)}   "
          f"{c['report_date'].min():%Y-%m} to {c['report_date'].max():%Y-%m}")
    if missing:
        print(f"  NOT PRESENT in the COT file: {missing}")
        print("  Those contract codes were never verified against a live CFTC report — only")
        print("  the original 13 were. They are dropped rather than silently mismatched.")
    for col in ("mm_long", "mm_short"):
        if col not in c.columns:
            raise SystemExit(
                f"'{col}' missing. Re-run fetch_cot.py: the money-manager columns are "
                "required and the parser already extracts them.")

    # month-end observation per instrument
    c = (c.sort_values(["symbol", "report_date"])
           .groupby(["symbol", "ym"]).tail(1)
           .set_index(["symbol", "ym"]).sort_index())

    out = pd.DataFrame(index=c.index)
    oi = c["open_interest"].replace(0, np.nan)
    out["mm_net"] = (c["mm_long"] - c["mm_short"]) / oi
    out["prod_net"] = (c["prod_long"] - c["prod_short"]) / oi
    if {"swap_long", "swap_short"}.issubset(c.columns):
        out["swap_net"] = (c["swap_long"] - c["swap_short"]) / oi
    out["oi"] = c["open_interest"]
    return out.reset_index()


def build_crowding(m: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """
    Uhl's alignment measure, applied per instrument-month.

        alignment = sign(trend position) x net positioning, standardised within instrument

    High when speculators are positioned in the same direction as a generic trend follower
    and positioned heavily. That is what "crowded into trend" means, and it is more
    informative than the ratio of speculators to open interest, which says nothing about
    WHAT they are crowded into.

    Everything is lagged one month before it meets a return.
    """
    d = m.merge(pos, on=["symbol", "ym"], how="left").sort_values(["symbol", "ym"])
    g = d.groupby("symbol")
    for cat in ("mm", "prod", "swap"):
        col = f"{cat}_net"
        if col not in d.columns:
            continue
        # crowding into TREND: alignment of positioning with the trend position
        d[f"{cat}_trendalign"] = d["trend_pos"] * d[col]
        # crowding into BASIS-MOMENTUM: alignment with the BM position, for P1
        d[f"{cat}_bmalign"] = np.sign(d["bm"]) * d[col]
        for suf in ("trendalign", "bmalign", "net"):
            c2 = f"{cat}_{suf}" if suf != "net" else col
            z = d.groupby("symbol")[c2].transform(
                lambda s: (s - s.rolling(CROWD_WINDOW, min_periods=12).mean())
                / s.rolling(CROWD_WINDOW, min_periods=12).std())
            # LAG: the report is not actionable within its own month
            d[f"{c2}_z"] = z.groupby(d["symbol"]).shift(1).clip(-3, 3)
    return d


# ----------------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------------

def zc(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def fm(panel: pd.DataFrame, y: str, xs: list[str], min_n: int = 8) -> pd.DataFrame:
    coefs = []
    for _, g in panel.groupby("ym"):
        s = g[[y] + xs].dropna()
        if len(s) < min_n:
            continue
        X = np.column_stack([np.ones(len(s))] + [s[x].to_numpy() for x in xs])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        coefs.append(np.linalg.pinv(X.T @ X) @ (X.T @ s[y].to_numpy()))
    if len(coefs) < 48:
        return pd.DataFrame()
    C = np.array(coefs)
    rows = []
    for i, nm in enumerate(["const"] + xs):
        cc = C[:, i]
        se = cc.std(ddof=1) / np.sqrt(len(cc))
        rows.append(dict(term=nm, coef=cc.mean(),
                         t=cc.mean() / se if se > 0 else np.nan, n=len(cc)))
    return pd.DataFrame(rows)


def show(title: str, tab: pd.DataFrame, focus: str | None = None) -> None:
    if tab.empty:
        print(f"  {title}: too few cross-sections"); return
    print(f"\n  {title}")
    for _, r in tab.iterrows():
        star = " *" if abs(r["t"]) > 2 else ""
        mark = "   <--" if focus and r["term"] == focus else ""
        print(f"    {r['term']:22s} {r['coef']*100:>+8.4f}%  t {r['t']:>+6.2f}{star}{mark}")


def idm_of(m: pd.DataFrame) -> float:
    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    if not np.isfinite(rho):
        rho = 0.2
    return min(1.0 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), IDM_CAP)


def portfolio(m: pd.DataFrame, idm: float, sig: str = "bm", tilt: str | None = None,
              bps: float = 3.0, min_n: int = 6) -> pd.Series:
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        cols = ["symbol", sig, "vol", "px_entry", "fwd"] + ([tilt] if tilt else [])
        s = g[cols].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s[sig].rank()
        w = (r - r.mean()).to_numpy()
        if tilt:
            w = w * np.clip(1.0 + zc(s[tilt]).to_numpy(), 0.0, None)
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
            n = float(np.round(wi * CAPITAL * VOL_TARGET * idm / den))
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


def stat(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 48:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12,
                dd=float((eq / eq.cummax() - 1).min()))


def line(lbl: str, s: dict, base: float | None = None) -> None:
    if not np.isfinite(s["sharpe"]):
        print(f"  {lbl:38s} n={s['n']}"); return
    d = f"  {s['sharpe']-base:+6.3f}" if base is not None else "        "
    star = " *" if abs(s["t"]) > 2 else ""
    print(f"  {lbl:38s} SR {s['sharpe']:>+6.3f}{d}  t {s['t']:>+5.2f}  "
          f"ret {s['ann']*100:>+6.2f}%  dd {s['dd']*100:>+6.1f}%{star}")


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", default="cot.parquet")
    ap.add_argument("--prices", default="px_wide.parquet")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()

    m = load_prices(a.prices)
    pos = load_cot(a.cot)
    d = build_crowding(m, pos)

    cats = [c for c in ("mm", "prod", "swap") if f"{c}_net_z" in d.columns]
    print(f"\n  trader categories available: {cats}")
    cov = d.groupby("symbol")["mm_net_z"].apply(lambda s: s.notna().mean())
    print(f"  money-manager crowding coverage per instrument: "
          f"{cov.min():.0%} to {cov.max():.0%}")
    usable = d.dropna(subset=["bm", "fwd", "mm_trendalign_z"])
    print(f"  usable instrument-months: {len(usable):,} across "
          f"{usable['symbol'].nunique()} instruments, {usable['ym'].nunique()} months")

    print("\n" + "=" * 82)
    print("1. POWER, STATED BEFORE THE RESULT")
    print("=" * 82)
    d["bm_z"] = d.groupby("ym")["bm"].transform(zc)
    base_tab = fm(d.dropna(subset=["bm_z", "fwd"]), "fwd", ["bm_z"])
    if not base_tab.empty:
        r = base_tab[base_tab["term"] == "bm_z"].iloc[0]
        se = abs(r["coef"] / r["t"]) if r["t"] else np.nan
        print(f"  unconditional basis-momentum slope {r['coef']*100:+.4f}%  t {r['t']:+.2f}")
        print(f"  standard error {se*100:.4f}%  -> minimum detectable interaction at t=2 "
              f"is {2*se*100:.4f}%")
        print("  An interaction must be roughly this size to be resolvable. Stating it")
        print("  first is what four earlier extensions in this project failed to do.")
        print()
        print("  VALIDATION CAVEAT, STATED HONESTLY. On synthetic data this test could not")
        print("  distinguish embedded conditioning (interaction t -0.16) from none")
        print("  (t -0.25). Either the simulator failed to produce a detectable effect or")
        print("  the test is underpowered for realistic effect sizes; that was not")
        print("  resolved. CONSEQUENCE: a null below is UNINFORMATIVE unless the minimum")
        print("  detectable effect above is small relative to the unconditional slope. If")
        print("  the MDE is a large fraction of that slope, this test could not have found")
        print("  the effect and the result says nothing either way. Read the MDE first.")

    print("\n" + "=" * 82)
    print("2. P1 — TREND CROWDING vs BM CROWDING (opposite predictions)")
    print("=" * 82)
    print("  Boos & Grob show speculators trade TREND and that the basis does NOT improve")
    print("  the forecast of their position changes. So:")
    print("    crowding into TREND   -> liquidity demand unrelated to the curve -> BM pays MORE")
    print("    crowding into BM      -> the signal itself is crowded          -> BM pays LESS")
    print("  Same data, opposite signs. A generic-activity story predicts neither.\n")
    for tag, lab in (("trendalign", "crowded into TREND"), ("bmalign", "crowded into BM")):
        col = f"mm_{tag}_z"
        if col not in d.columns:
            continue
        sub = d.dropna(subset=["bm_z", "fwd", col]).copy()
        sub[f"{col}_x"] = sub["bm_z"] * sub.groupby("ym")[col].transform(zc)
        sub[f"{col}_c"] = sub.groupby("ym")[col].transform(zc)
        show(f"money managers {lab}",
             fm(sub, "fwd", ["bm_z", f"{col}_c", f"{col}_x"]), f"{col}_x")

    print("\n" + "=" * 82)
    print("3. P2 and P3 — WHICH TRADER CATEGORY, AND WITH WHAT SIGN?")
    print("=" * 82)
    print("  Money managers are trend followers. Producers are contrarians who mirror")
    print("  momentum traders. Swap dealers intermediate index flow. If all three condition")
    print("  identically, the effect is generic activity and the mechanism claim fails.\n")
    inter = {}
    for cat in cats:
        col = f"{cat}_trendalign_z"
        sub = d.dropna(subset=["bm_z", "fwd", col]).copy()
        sub["cz"] = sub.groupby("ym")[col].transform(zc)
        sub["ix"] = sub["bm_z"] * sub["cz"]
        tab = fm(sub, "fwd", ["bm_z", "cz", "ix"])
        if tab.empty:
            continue
        row = tab[tab["term"] == "ix"].iloc[0]
        inter[cat] = (row["coef"], row["t"])
        nm = {"mm": "money managers", "prod": "producers", "swap": "swap dealers"}[cat]
        print(f"    {nm:16s} interaction {row['coef']*100:>+8.4f}%  t {row['t']:>+6.2f}"
              f"{' *' if abs(row['t']) > 2 else ''}")
    mm_t = inter.get("mm", (np.nan, np.nan))[1]
    pr_t = inter.get("prod", (np.nan, np.nan))[1]
    p2 = np.isfinite(mm_t) and abs(mm_t) > 2 and (
        not np.isfinite(pr_t) or abs(mm_t) > abs(pr_t))
    p3 = (np.isfinite(mm_t) and np.isfinite(pr_t) and
          np.sign(inter["mm"][0]) != np.sign(inter["prod"][0]))
    print(f"\n  P2 money managers dominate: {'PASS' if p2 else 'FAIL'}")
    print(f"  P3 producers carry the opposite sign: {'PASS' if p3 else 'FAIL'}")

    print("\n" + "=" * 82)
    print("4. IS IT TRADEABLE?")
    print("=" * 82)
    idm = idm_of(d)
    base = portfolio(d, idm, "bm")
    sr0 = stat(base)["sharpe"]
    line("plain basis-momentum", stat(base))
    for cat in cats:
        col = f"{cat}_trendalign_z"
        line(f"conviction tilted by {cat} crowding",
             stat(portfolio(d, idm, "bm", tilt=col)), sr0)
    print("\n  A significant interaction that does not survive being traded is a")
    print("  statistical fact, not a strategy. Report both.")

    print("\n" + "=" * 82)
    print("5. PLACEBO — shuffle crowding across instruments within each month")
    print("=" * 82)
    col = "mm_trendalign_z"
    real = inter.get("mm", (np.nan, np.nan))[1]
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(a.seeds):
        sub = d.dropna(subset=["bm_z", "fwd", col]).copy()
        sub[col] = sub.groupby("ym")[col].transform(
            lambda s: rng.permutation(s.to_numpy()))
        sub["cz"] = sub.groupby("ym")[col].transform(zc)
        sub["ix"] = sub["bm_z"] * sub["cz"]
        tab = fm(sub, "fwd", ["bm_z", "cz", "ix"])
        if not tab.empty:
            ts.append(tab[tab["term"] == "ix"]["t"].iloc[0])
    placebo_ok = False
    if ts and np.isfinite(real):
        ts = np.array(ts)
        z = (real - ts.mean()) / max(ts.std(ddof=1), 1e-9)
        print(f"  placebo interaction t {ts.mean():+.2f} +/- {ts.std(ddof=1):.2f} "
              f"over {len(ts)} shuffles")
        print(f"  real {real:+.2f} sits {z:+.1f} sd out   "
              f"{'PASS' if abs(z) > 2 else 'FAIL'}")
        placebo_ok = abs(z) > 2

    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    p1 = np.isfinite(mm_t) and abs(mm_t) > 2
    checks = [("P1 trend-crowding interaction significant", p1),
              ("P2 money managers dominate the effect", p2),
              ("P3 producers carry the opposite sign", p3),
              ("survives the crowding placebo", placebo_ok)]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    if p1 and placebo_ok and (p2 or p3):
        print("  SUPPORTED. Basis-momentum is compensated more richly when speculators are")
        print("  crowded into TREND — a signal they demonstrably trade — while being a")
        print("  signal they demonstrably do not. That is Boons & Prado's stated mechanism,")
        print("  tested with positioning data for the first time, using a measure adapted")
        print("  from Uhl (2025) that has only ever been applied to trend-following. It is")
        print("  also the Economic Rationale this strategy has been missing.")
    elif p1:
        print("  PARTIAL. The interaction is significant but the discriminating predictions")
        print("  do not all hold, so the mechanism is not pinned down. Report exactly which")
        print("  passed and which did not.")
    else:
        print("  NOT SUPPORTED. Positioning does not condition basis-momentum in this")
        print("  sample. Combined with hypothesis 1 — positioning does not predict returns")
        print("  either, with power to detect one sixtieth of the published effect — the")
        print("  honest conclusion is that CFTC positioning carries no usable information")
        print("  for this strategy, as a predictor OR as a conditioner. That is a real")
        print("  finding about a widely used dataset.")


if __name__ == "__main__":
    main()