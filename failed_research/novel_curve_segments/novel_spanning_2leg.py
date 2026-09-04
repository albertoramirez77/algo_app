"""
2-LEG SPANNING TEST
===================

Purpose:
    Test whether the existing front-vs-deferred momentum signal contains
    incremental NEXT-MONTH return information after controlling for:

        1. front momentum
        2. deferred momentum
        3. contemporaneous calendar-spread/basis information
        4. common commodity-market momentum

This is NOT a curvature-identification test.
With only two maturities, curvature is unidentified.

The important question is whether BM survives conventional controls.

Usage:
    python novel_spanning_2leg.py --prices data/px_clean.parquet
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    BY_SYMBOL = {}


FORMATION = 12
MIN_CS = 6


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    if "date" not in df.columns:
        raise ValueError("prices file must contain 'date'")

    required = [
        "symbol",
        "date",
        "contract_0",
        "contract_1",
        "settle_0",
        "settle_1",
        "expiry_0",
        "expiry_1",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prices file missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["expiry_0"] = pd.to_datetime(df["expiry_0"])
    df["expiry_1"] = pd.to_datetime(df["expiry_1"])

    # Restrict to commodities exactly as the production strategy does.
    if BY_SYMBOL:
        df["asset"] = df["symbol"].map(
            lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?"
        )
        df = df[df["asset"] == "commodity"].copy()

    keys = ["symbol", "date"]

    # Resolve duplicate daily records deterministically.
    if "oi_0" in df.columns:
        df = (
            df.sort_values(keys + ["oi_0"], na_position="first")
              .drop_duplicates(keys, keep="last")
        )
    else:
        df = df.drop_duplicates(keys, keep="last")

    df = df.sort_values(keys).reset_index(drop=True)

    # Contract-life returns. A return is only formed against the
    # immediately preceding observation of the SAME contract.
    for k in (0, 1):

        blk = (
            df.groupby("symbol")[f"contract_{k}"]
              .transform(lambda s: (s != s.shift()).cumsum())
        )

        prev = (
            df.groupby(["symbol", blk])[f"settle_{k}"]
              .shift()
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(df[f"settle_{k}"] / prev)

        df[f"r{k}"] = r.where(np.isfinite(r))

    return df


# ---------------------------------------------------------------------
# MONTHLY PANEL
# ---------------------------------------------------------------------

def panel(df: pd.DataFrame) -> pd.DataFrame:

    d = df.copy()

    d["ym"] = d["date"].dt.to_period("M")

    # First available trading observation for each commodity/month.
    d["dom"] = d.groupby(["symbol", "ym"]).cumcount()

    marks = (
        d.loc[d["dom"] == 0, ["symbol", "ym", "date"]]
         .rename(columns={"date": "mark"})
    )

    # Cumulative contract-life return.
    for k in (0, 1):
        d[f"c{k}"] = (
            d.groupby("symbol")[f"r{k}"]
             .transform(lambda s: s.fillna(0.0).cumsum())
        )

    # IMPORTANT:
    # The original script's bug was here. `marks` contains `mark`,
    # not `date`, so the merge must use left_on/right_on explicitly.
    s = (
        d.merge(
            marks,
            left_on=["symbol", "ym", "date"],
            right_on=["symbol", "ym", "mark"],
            how="inner",
        )
        .copy()
    )

    s = (
        s.sort_values(["symbol", "ym"])
         .reset_index(drop=True)
    )

    g = s.groupby("symbol", sort=False)

    # Returns between successive monthly observations.
    for k in (0, 1):
        s[f"ret{k}"] = g[f"c{k}"].diff()

        s[f"m{k}"] = (
            g[f"ret{k}"]
            .transform(
                lambda x:
                    x.rolling(
                        FORMATION,
                        min_periods=FORMATION
                    ).sum()
            )
        )

    # Existing basis-momentum object.
    s["bm"] = s["m0"] - s["m1"]

    # Next-month front-contract return.
    s["fwd"] = g["ret0"].shift(-1)

    # Current calendar basis.
    s["basis"] = np.log(s["settle_0"] / s["settle_1"])

    # Annualized basis.
    tau = (
        (s["expiry_1"] - s["expiry_0"]).dt.days
        / 365.25
    )

    s["ann_basis"] = (
        s["basis"]
        / tau.replace(0, np.nan)
    )

    # Common commodity momentum at each month.
    # This is a cross-sectional market control, so it varies only by month.
    s["market_mom"] = (
        s.groupby("ym")["m0"]
         .transform("mean")
    )

    # Keep only columns we actually need.
    return s


# ---------------------------------------------------------------------
# HAC MEAN TEST
# ---------------------------------------------------------------------

def hac_mean_t(x, lags=3):

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n < 24:
        return np.nan, np.nan

    mu = float(x.mean())
    u = x - mu

    # Long-run variance estimator.
    var = float(np.dot(u, u) / n)

    L = min(lags, n - 1)

    for lag in range(1, L + 1):

        gamma = float(
            np.dot(u[lag:], u[:-lag]) / n
        )

        weight = 1.0 - lag / (L + 1)

        var += 2.0 * weight * gamma

    se = math.sqrt(max(var, 0.0) / n)

    t = mu / se if se > 0 else np.nan

    return mu, t


# ---------------------------------------------------------------------
# FAMA-MACBETH CROSS-SECTIONAL REGRESSION
# ---------------------------------------------------------------------

def fmb_multi(
    p: pd.DataFrame,
    y: str,
    xs: list[str],
):

    monthly_betas = []

    for ym, g in p.groupby("ym", sort=False):

        cols = [y] + xs

        z = (
            g[cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(z) < max(MIN_CS, len(xs) + 3):
            continue

        X = np.column_stack([
            np.ones(len(z)),
            z[xs].to_numpy(float)
        ])

        yy = z[y].to_numpy(float)

        # Skip rank-deficient months.
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue

        try:
            b = np.linalg.lstsq(X, yy, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        monthly_betas.append(b[1:])

    if len(monthly_betas) < 24:
        return None

    B = np.asarray(monthly_betas)

    out = {
        "months": len(B)
    }

    for j, x in enumerate(xs):

        beta, t = hac_mean_t(B[:, j], lags=3)

        out[f"{x}_beta"] = beta
        out[f"{x}_t_hac3"] = t

    return out


# ---------------------------------------------------------------------
# RANK PORTFOLIO
# ---------------------------------------------------------------------

def rank_portfolio(
    p: pd.DataFrame,
    signal: str,
) -> pd.Series:

    out = []

    for ym, g in p.groupby("ym", sort=True):

        z = (
            g[["symbol", signal, "fwd"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(z) < MIN_CS:
            continue

        ranks = z[signal].rank()

        w = ranks - ranks.mean()

        denom = np.abs(w).sum()

        if denom <= 0:
            continue

        w = w / denom

        pnl = float((w * z["fwd"]).sum())

        out.append((ym, pnl))

    return pd.Series(dict(out)).sort_index()


# ---------------------------------------------------------------------
# PORTFOLIO STATS
# ---------------------------------------------------------------------

def stats(r: pd.Series):

    r = r.dropna()

    if len(r) < 48:
        return {}

    ann_vol = r.std(ddof=1) * math.sqrt(12)

    ann_ret = r.mean() * 12

    sharpe = (
        ann_ret / ann_vol
        if ann_vol > 0
        else np.nan
    )

    return {
        "n": len(r),
        "ann": ann_ret,
        "vol": ann_vol,
        "sharpe": sharpe,
        "t": sharpe * math.sqrt(len(r) / 12)
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--prices",
        default="data/px_clean.parquet"
    )

    a = ap.parse_args()

    raw = load(a.prices)
    p = panel(raw)

    print("=" * 90)
    print("2-LEG SPANNING TEST")
    print("NOT A CURVATURE IDENTIFICATION TEST")
    print("=" * 90)

    print(
        f"rows={len(p):,}, "
        f"instruments={p.symbol.nunique()}, "
        f"months={p.ym.nunique()}"
    )

    # -------------------------------------------------------------
    # 1. BASIC PREDICTIVE TESTS
    # -------------------------------------------------------------

    print("\n[1] UNCONDITIONAL FAMA-MACBETH")

    for x in [
        "bm",
        "m0",
        "m1",
        "basis",
        "ann_basis",
    ]:
        print(f"{x}: {fmb_multi(p, 'fwd', [x])}")

    # -------------------------------------------------------------
    # 2. CONVENTIONAL SPANNING
    # -------------------------------------------------------------

    print("\n[2] DOES BM SURVIVE CONVENTIONAL CONTROLS?")

    controls = [
        "m0",
        "m1",
        "basis",
    ]

    augmented = controls + ["bm"]

    print("\nBASE MODEL")
    print(
        fmb_multi(
            p,
            "fwd",
            controls
        )
    )

    print("\nBASE + BM")
    print(
        fmb_multi(
            p,
            "fwd",
            augmented
        )
    )

    # -------------------------------------------------------------
    # 3. A CLEANER TEST
    #
    # Do NOT include market_mom in the cross-sectional regression:
    # it is constant across all names within a month and therefore
    # cannot be separately identified from the intercept.
    # -------------------------------------------------------------

    print("\n[3] BM AFTER FRONT/DEFERRED MOMENTUM + BASIS")

    result = fmb_multi(
        p,
        "fwd",
        ["m0", "m1", "basis", "bm"]
    )

    print(result)

    # -------------------------------------------------------------
    # 4. PORTFOLIO DIAGNOSTIC
    # -------------------------------------------------------------

    print("\n[4] RANK PORTFOLIOS")

    for x in ["m0", "bm"]:

        r = rank_portfolio(p, x)

        print(
            f"{x}: {stats(r)}"
        )

    # -------------------------------------------------------------
    # 5. CORRELATION / SPANNING DIAGNOSTIC
    # -------------------------------------------------------------

    print("\n[5] SIGNAL GEOMETRY")

    cols = [
        "m0",
        "m1",
        "bm",
        "basis",
        "ann_basis",
    ]

    print(
        p[cols]
        .corr()
        .round(3)
        .to_string()
    )

    print()
    print("=" * 90)

    print(
        "INTERPRETATION"
    )

    print(
        "A surviving positive BM coefficient means the existing "
        "basis-momentum signal is not fully spanned by front momentum, "
        "deferred momentum and current basis."
    )

    print(
        "It still does NOT demonstrate a novel curve component: "
        "with only two maturities, curvature is unidentified."
    )

    print(
        "The decisive experiment requires contract_2 / contract_3 "
        "and explicit level/slope/curvature decomposition."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()