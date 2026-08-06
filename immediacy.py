"""
immediacy.py — Selling Immediacy in Commodity Futures.

Backtest engine for a cross-sectional liquidity-provision strategy: take the side of
commercial hedgers' weekly net position changes, held ~3 weeks, in contracts where
hedgers are structurally net short.

Mechanism (Kang, Rouwenhorst & Tang, Journal of Finance 2020): speculators are momentum
traders and trade impatiently; hedgers are contrarians who absorb that flow and are
compensated by a price reversal. The compensation is concentrated in trading days 5-20
after the CFTC position measurement date, which is exactly the window that opens once
the report becomes public.

DESIGN RULES ENFORCED IN CODE
  1. Every input carries the timestamp it became knowable. Signals assert against it.
  2. Positions are integers times real multipliers. No fractional contracts anywhere.
  3. No back-adjusted price series is ever constructed.
  4. Out-of-sample window is walled off behind an explicit flag.

Run:
    python immediacy.py --smoke                 # synthetic data, verifies machinery
    python immediacy.py --run --oos-unlock      # real data, touches OOS. Once.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------

CAPITAL = 450_000.0

# The six free parameters. Justifications live in 01_specification.md section 3.
PARAMS = dict(
    hp_window_weeks=52,     # production cycle length
    hold_weeks=3,           # observed half-life of the imbalance
    stress_quantile=0.75,   # KRT's capital-loss dummy
    vol_span_days=60,       # hedger margin recalibration horizon
    vol_target=0.20,        # set by integer granularity, not by return
    buffer_frac=0.10,       # Carver no-trade band
)

# Fixed conventions. Never searched. Changing these is a spec change, not a tuning run.
IDM = 2.5
FORECAST_SCALAR = 10.0
FORECAST_CAP = 20.0
WINSOR = 3.0
BUFFER_FLOOR_CONTRACTS = 0.5
ROLL_OFFSET_BDAYS = 5

# OOS split: the most recent 25% of the sample is sealed.
OOS_FRACTION = 0.25


@dataclass(frozen=True)
class Contract:
    symbol: str                # what we TRADE (may be a micro)
    cftc_code: str             # CFTC contract market code, for the DCOT join
    sector: str
    multiplier: float          # contract units of the TRADED contract
    tick: float                # minimum price increment, in quote units
    commission_side: float     # all-in commission per side, USD
    price_root: str = ""       # Databento root to FETCH prices from (full-size)
    dataset: str = "GLBX.MDP3"

    @property
    def fetch_root(self) -> str:
        return self.price_root or self.symbol


# Prices are fetched from the FULL-SIZE contract even where we trade the micro. The micro
# is a fractional clone of the same underlying with an identical price series, and the
# full-size contract has history back to 2010 while the micro does not: Micro WTI listed
# in 2021, Micro Copper in 2022. Sizing applies the micro multiplier; the returns are the
# same series either way.
#
# LISTED_FROM is when the contract we actually TRADE became available. Before that date the
# strategy could not have held it, so the universe is time-varying and the engine must know.
# These are estimates — run dbn_diagnose.py and replace them with measured values.
LISTED_FROM = {
    "MCL": "2021-07-12",   # VERIFY
    "MHG": "2022-01-01",   # VERIFY
    "MGC": "2010-10-01",   # VERIFY
    "SIL": "2022-01-01",   # VERIFY
    "KE":  "2013-01-01",   # KCBT -> CME Globex migration. VERIFY
}


UNIVERSE = [
    #        symbol  cftc      sector       mult      tick     comm/side  price_root
    Contract("MCL", "067651", "energy",     100,      0.01,    0.75,      "CL"),
    Contract("QG",  "023651", "energy",     2_500,    0.005,   0.75,      "NG"),
    Contract("MGC", "088691", "metals",     10,       0.10,    0.75,      "GC"),
    Contract("SIL", "084691", "metals",     1_000,    0.005,   0.75,      "SI"),
    Contract("MHG", "085692", "metals",     2_500,    0.0005,  0.75,      "HG"),
    Contract("ZC",  "002602", "grains",     5_000,    0.0025,  1.50,      "ZC"),
    Contract("ZW",  "001602", "grains",     5_000,    0.0025,  1.50,      "ZW"),
    Contract("KE",  "001612", "grains",     5_000,    0.0025,  1.50,      "KE"),
    Contract("ZS",  "005602", "oilseeds",   5_000,    0.0025,  1.50,      "ZS"),
    Contract("ZM",  "026603", "oilseeds",   100,      0.10,    1.50,      "ZM"),
    Contract("ZL",  "007601", "oilseeds",   60_000,   0.0001,  1.50,      "ZL"),
    Contract("LE",  "057642", "livestock",  40_000,   0.00025, 1.50,      "LE"),
    Contract("HE",  "054642", "livestock",  40_000,   0.00025, 1.50,      "HE"),
]

BY_SYMBOL = {c.symbol: c for c in UNIVERSE}
N_UNIVERSE = len(UNIVERSE)

# Per-instrument risk budget, in annualised dollars of volatility. This one number
# determines the universe: a contract is tradeable only if one lot's annualised dollar
# volatility is comfortably below it.
RISK_BUDGET = CAPITAL * PARAMS["vol_target"] * IDM / N_UNIVERSE


# ----------------------------------------------------------------------------------
# POINT-IN-TIME GUARD
# ----------------------------------------------------------------------------------

class Clock:
    """
    Records, for every input, the timestamp at which it became knowable, and refuses
    to hand it back before then. Any look-ahead becomes an exception, not a silent
    improvement in the Sharpe ratio.
    """

    def __init__(self) -> None:
        self.log: list[dict] = []

    def register(self, name: str, as_of: pd.Timestamp, knowable: pd.Timestamp) -> None:
        if knowable < as_of:
            raise ValueError(f"{name}: knowable {knowable} precedes as-of {as_of}")
        self.log.append(dict(input=name, as_of=as_of, knowable=knowable))

    def assert_usable(self, name: str, knowable: pd.Timestamp,
                      decision_time: pd.Timestamp) -> None:
        if decision_time < knowable:
            raise ValueError(
                f"LOOK-AHEAD: {name} knowable {knowable} used at {decision_time}"
            )

    def report(self) -> pd.DataFrame:
        df = pd.DataFrame(self.log)
        if df.empty:
            return df
        df["lag_days"] = (df["knowable"] - df["as_of"]).dt.days
        return df.groupby("input").agg(
            n=("as_of", "size"),
            max_lag_days=("lag_days", "max"),
            min_lag_days=("lag_days", "min"),
        ).reset_index()


CLOCK = Clock()


# ----------------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------------

def cot_release_date(report_tuesday: pd.Timestamp) -> pd.Timestamp:
    """
    CFTC Disaggregated COT: positions measured Tuesday close, published Friday 15:30 ET.
    15:30 ET is after the settlement window of every contract in this universe, so the
    first executable settlement is the following Monday.
    """
    return report_tuesday + pd.Timedelta(days=3)


def first_tradeable_date(report_tuesday: pd.Timestamp,
                         trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Monday after the Friday release, or the next trading day if Monday is a holiday."""
    release = cot_release_date(report_tuesday)
    later = trading_days[trading_days > release]
    return later[0] if len(later) else None


def load_cot(path: str | None = None) -> pd.DataFrame:
    """
    Disaggregated COT, futures-only. Columns required:
        report_date, symbol, prod_long, prod_short, open_interest

    Production source, one of:
      - https://publicreporting.cftc.gov/resource/72hh-3qpy.json  (Socrata, DCOT futures-only)
      - the annual `f_year.zip` archives under cftc.gov/MarketReports/CommitmentsofTraders

    Join key is `cftc_code` (CFTC contract market code) -> Contract.cftc_code. Codes must
    be verified against the current report before first use; they change on exchange
    renames (MGEX -> MIAX Futures, November 2024).
    """
    if path is None:
        raise RuntimeError("No COT path supplied. Use --smoke for synthetic data.")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    for _, r in df.iterrows():
        CLOCK.register("cot", r["report_date"], cot_release_date(r["report_date"]))
    return df


def load_prices(path: str | None = None) -> pd.DataFrame:
    """
    Daily settlement prices and open interest per *individual contract* (not continuous).

    Production source: Databento GLBX.MDP3.
      - `statistics` schema for official settlement price and open interest
      - `definition` schema for expiration / first notice dates
      - `ohlcv-1d` ONLY for the volatility estimate

    ohlcv-1d is aggregated on UTC dates from electronic-session trades and is not the
    venue settlement. Do not use its close as a price. History begins 2010-06-06.

    Columns: date, symbol, contract, settle, volume, open_interest, expiry
    """
    if path is None:
        raise RuntimeError("No price path supplied. Use --smoke for synthetic data.")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


# ----------------------------------------------------------------------------------
# ROLL: deterministic calendar, never volume or open interest
# ----------------------------------------------------------------------------------

def build_front_series(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Select the held contract each day: the nearest expiry that is still more than
    ROLL_OFFSET_BDAYS business days out.

    A volume- or open-interest-based roll needs the current session's statistics to
    decide the current session's roll. That is a same-day leak unless carefully lagged.
    A fixed calendar offset cannot leak: the roll date is knowable years in advance.

    Returns per (date, symbol): the held contract, its settlement, and a roll flag.
    No back-adjustment is performed. Returns are chained within a contract's life.
    """
    out = []
    for sym, g in prices.groupby("symbol", sort=False):
        for date, day in g.groupby("date", sort=True):
            cutoff = date + pd.tseries.offsets.BDay(ROLL_OFFSET_BDAYS)
            live = day[day["expiry"] > cutoff].sort_values("expiry")
            if live.empty:
                continue
            r = live.iloc[0]
            out.append(dict(date=date, symbol=sym, contract=r["contract"],
                            settle=r["settle"], volume=r.get("volume", np.nan),
                            open_interest=r["open_interest"]))
    front = pd.DataFrame(out).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Return within a contract only. On a roll day the return is NaN and is treated as
    # zero P&L on the price leg; the roll's cost is charged separately in the cost model.
    front["prev_settle"] = front.groupby(["symbol", "contract"])["settle"].shift(1)
    front["ret"] = front["settle"] / front["prev_settle"] - 1.0
    front["is_roll"] = front.groupby("symbol")["contract"].transform(
        lambda s: s != s.shift(1)
    ).fillna(False)
    return front


# ----------------------------------------------------------------------------------
# SIGNAL
# ----------------------------------------------------------------------------------

def compute_signals(cot: pd.DataFrame, weekly_ret: pd.DataFrame,
                    p: dict = PARAMS) -> pd.DataFrame:
    """
    Q   : hedger net purchases this week, scaled by open interest      (the trade)
    HP̄  : 52-week mean hedging pressure                                (the eligibility)
    g   : stress gate, doubles weight after hedger capital losses      (the intensity)

    Returns one row per (report_date, symbol) with the final forecast F.
    """
    df = cot.sort_values(["symbol", "report_date"]).copy()
    df["net_long"] = df["prod_long"] - df["prod_short"]
    df["oi_lag"] = df.groupby("symbol")["open_interest"].shift(1)

    # --- Q: the liquidity-provision signal -------------------------------------
    df["Q"] = (df["net_long"] - df.groupby("symbol")["net_long"].shift(1)) / df["oi_lag"]

    # --- HP̄: 52-week smoothed hedging pressure ---------------------------------
    df["HP"] = (df["prod_short"] - df["prod_long"]) / df["open_interest"]
    df["HPbar"] = (df.groupby("symbol")["HP"]
                     .transform(lambda s: s.rolling(p["hp_window_weeks"],
                                                    min_periods=p["hp_window_weeks"]).mean()))

    # --- stress gate: hedger capital loss over the prior 4 weeks ----------------
    df = df.merge(weekly_ret, on=["symbol", "report_date"], how="left")
    df["cap_loss_1w"] = df.groupby("symbol")["HP"].shift(1) * df["week_ret"]
    df["cap_loss_4w"] = (df.groupby("symbol")["cap_loss_1w"]
                           .transform(lambda s: s.rolling(4, min_periods=4).sum()))
    thresh = (df.groupby("symbol")["cap_loss_4w"]
                .transform(lambda s: s.shift(1)
                                      .rolling(260, min_periods=104)
                                      .quantile(p["stress_quantile"])))
    df["g"] = np.where(df["cap_loss_4w"] >= thresh, 2.0, 1.0)

    # --- sector demeaning: the crush is one hedging event, not three ------------
    df["sector"] = df["symbol"].map(lambda s: BY_SYMBOL[s].sector)
    df["Q_tilde"] = df["Q"] - df.groupby(["report_date", "sector"])["Q"].transform("mean")

    # --- cross-sectional z, winsorised ------------------------------------------
    grp = df.groupby("report_date")["Q_tilde"]
    df["z"] = ((df["Q_tilde"] - grp.transform("mean")) / grp.transform("std")).clip(-WINSOR, WINSOR)

    # --- eligibility: only where hedgers are structurally net short -------------
    med = df.groupby("report_date")["HPbar"].transform("median")
    df["e"] = (df["HPbar"] > med).astype(float)

    df["s"] = df["z"].fillna(0.0) * df["e"].fillna(0.0) * df["g"]

    # --- holding horizon as a rolling mean of weekly scores ---------------------
    df["S"] = (df.groupby("symbol")["s"]
                 .transform(lambda x: x.rolling(p["hold_weeks"], min_periods=1).mean()))

    # --- forecast scalar on an EXPANDING window. A full-sample scalar is look-ahead.
    active = df["S"].abs() > 1e-12
    absS = (df.assign(_a=df["S"].abs().where(active))
              .groupby("report_date")["_a"].transform("mean"))
    expanding_absS = (absS.groupby(df["report_date"]).first()
                          .expanding().mean()
                          .reindex(df["report_date"]).to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        df["F"] = np.clip(df["S"] * FORECAST_SCALAR / expanding_absS,
                          -FORECAST_CAP, FORECAST_CAP)
    df["F"] = df["F"].fillna(0.0)

    df["release_date"] = df["report_date"].map(cot_release_date)
    return df[["report_date", "release_date", "symbol", "Q", "HPbar", "g", "z", "e", "S", "F"]]


# ----------------------------------------------------------------------------------
# SIZING: integer contracts, real multipliers
# ----------------------------------------------------------------------------------

def target_contracts(forecast: float, sigma_ann: float, price: float,
                     c: Contract, p: dict = PARAMS, n_active: int = N_UNIVERSE,
                     avg_corr: float = 0.15) -> tuple[float, float]:
    """
    Carver volatility targeting. Returns (fractional target, buffer half-width).

        N* = (F/10) * [Capital * tau * IDM * w] / [sigma * multiplier * price]
    """
    denom = sigma_ann * c.multiplier * price
    if denom <= 0 or not np.isfinite(denom):
        return 0.0, BUFFER_FLOOR_CONTRACTS
    # Risk budget is shared among the ACTIVE names. Roughly half the universe is
    # ineligible each week (hedging pressure below the cross-sectional median), so a
    # fixed 1/13 weight under-sizes the book and realised vol lands well under target.
    n = max(n_active, 1)
    # IDM from the realised average pairwise correlation of the active book, capped at
    # Carver's 2.5. A fixed IDM mis-scales badly when the eligible count moves between
    # 5 and 13, which it does every week.
    idm = min(1.0 / np.sqrt((1.0 / n) + (1.0 - 1.0 / n) * max(avg_corr, 0.01)), IDM)
    budget = CAPITAL * p["vol_target"] * idm / n
    unit = budget / denom                           # contracts at forecast = 10
    n_star = (forecast / FORECAST_SCALAR) * unit
    buffer = max(p["buffer_frac"] * unit, BUFFER_FLOOR_CONTRACTS)
    return n_star, buffer


def apply_buffer(held: int, n_star: float, buffer: float) -> int:
    """
    No-trade band. At $450k a 10% band is frequently narrower than one contract, so the
    0.5-contract floor is what actually stops a target oscillating around x.5 from
    churning every week for no change in exposure.
    """
    if held > n_star + buffer:
        return int(np.round(n_star + buffer))   # trade DOWN to the upper edge
    if held < n_star - buffer:
        return int(np.round(n_star - buffer))   # trade UP to the lower edge
    return held                                  # inside the band: do nothing


# ----------------------------------------------------------------------------------
# COSTS: commission + volatility-scaled spread + square-root impact
# ----------------------------------------------------------------------------------

def trade_cost(n_contracts: int, c: Contract, price: float,
               sigma_daily: float, sigma_ref: float, adv_contracts: float) -> dict:
    """
    Three components, reported separately so the dominant one is visible.

      commission : fixed per contract
      spread     : half a tick, widening linearly with realised volatility
      impact     : square-root law scaled by participation

    At $450k the impact term is ~4 orders of magnitude below the other two. That is the
    capacity finding, and it is the opposite of the one most applicants report.
    """
    q = abs(n_contracts)
    if q == 0:
        return dict(commission=0.0, spread=0.0, impact=0.0, total=0.0)

    commission = q * c.commission_side

    vol_ratio = 1.0 if sigma_ref <= 0 else max(1.0, sigma_daily / sigma_ref)
    half_spread_price = 0.5 * c.tick * vol_ratio
    spread = q * half_spread_price * c.multiplier

    notional = q * c.multiplier * price
    participation = 0.0 if adv_contracts <= 0 else q / adv_contracts
    # The square-root law is calibrated for meaningful participation. Below ~1bp of ADV
    # a trade rests inside the top of book and its cost IS the spread, already charged
    # above. We keep the term anyway rather than zero it, so the reported figure is
    # conservative and the model matches the stated cost framework.
    impact = notional * sigma_daily * np.sqrt(participation)

    return dict(commission=commission, spread=spread, impact=impact,
                total=commission + spread + impact)


# ----------------------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------------------

@dataclass
class Result:
    daily: pd.DataFrame
    trades: pd.DataFrame
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)

    def stats(self, net: bool = True) -> dict:
        col = "pnl_net" if net else "pnl_gross"
        r = self.daily[col] / CAPITAL
        if r.std() == 0 or len(r) < 60:
            return dict(n_days=len(r))
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        return dict(
            n_days=len(r),
            ann_return=ann_ret,
            ann_vol=ann_vol,
            sharpe=ann_ret / ann_vol,
            max_drawdown=dd,
            skew=r.skew(),
        )


def run_backtest(front: pd.DataFrame, signals: pd.DataFrame,
                 p: dict = PARAMS, start=None, end=None) -> Result:
    """Daily loop. Positions change only on rebalance days; P&L accrues every day."""
    front = front.sort_values(["date", "symbol"]).copy()
    if start is not None:
        front = front[front["date"] >= pd.Timestamp(start)]
    if end is not None:
        front = front[front["date"] <= pd.Timestamp(end)]

    dates = pd.DatetimeIndex(sorted(front["date"].unique()))

    # Rolling volatility from daily returns of the held contract.
    piv_ret = front.pivot_table(index="date", columns="symbol", values="ret")
    piv_px = front.pivot_table(index="date", columns="symbol", values="settle")
    piv_oi = front.pivot_table(index="date", columns="symbol", values="open_interest")
    sigma_d = piv_ret.ewm(span=p["vol_span_days"], min_periods=20).std()
    sigma_ref = piv_ret.ewm(span=252, min_periods=60).std()

    # ADV from realised volume of the held contract. Falls back to an open-interest
    # proxy only if volume is unavailable, which should never happen on real data.
    if "volume" in front.columns and front["volume"].notna().any():
        piv_vol = front.pivot_table(index="date", columns="symbol", values="volume")
        adv = piv_vol.rolling(20, min_periods=5).median()
    else:
        adv = piv_oi.rolling(20, min_periods=5).mean() * 0.25

    # Map each report week to the first date it can be traded on.
    signals = signals.copy()
    trade_map = {}
    for rd in signals["report_date"].unique():
        t = first_tradeable_date(pd.Timestamp(rd), dates)
        if t is not None:
            trade_map[pd.Timestamp(rd)] = t
    signals["trade_date"] = signals["report_date"].map(trade_map)
    signals = signals.dropna(subset=["trade_date"])

    held: dict[str, int] = {c.symbol: 0 for c in UNIVERSE}
    rows, trades = [], []

    fc_by_date = {d: g for d, g in signals.groupby("trade_date")}

    for i, dt in enumerate(dates):
        # ---- P&L on yesterday's book, using today's return ----------------------
        gross = 0.0
        for sym, n in held.items():
            if n == 0 or sym not in piv_ret.columns:
                continue
            r = piv_ret.at[dt, sym] if dt in piv_ret.index else np.nan
            px_prev = piv_px.at[dates[i - 1], sym] if i > 0 and dates[i - 1] in piv_px.index else np.nan
            if np.isfinite(r) and np.isfinite(px_prev):
                gross += n * BY_SYMBOL[sym].multiplier * px_prev * r

        # ---- rebalance, if a new report is tradeable today ----------------------
        cost_today = dict(commission=0.0, spread=0.0, impact=0.0, total=0.0)
        if dt in fc_by_date:
            g = fc_by_date[dt]
            tradeable = [s for s in g.loc[g["F"].abs() > 0, "symbol"]
                         if s in piv_px.columns and dt in piv_px.index
                         and np.isfinite(piv_px.at[dt, s])
                         and dt >= pd.Timestamp(LISTED_FROM.get(s, "1900-01-01"))]
            n_active = max(len(tradeable), 1)
            win = piv_ret.loc[:dt].tail(252)
            cm = win.corr().to_numpy() if win.shape[0] >= 60 else None
            rho_t = 0.15 if cm is None else float(
                np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
            if not np.isfinite(rho_t):
                rho_t = 0.15
            CLOCK.assert_usable("cot", pd.Timestamp(g["release_date"].iloc[0]), dt)
            for _, row in g.iterrows():
                sym = row["symbol"]
                if sym not in piv_px.columns or dt not in piv_px.index:
                    continue
                if dt < pd.Timestamp(LISTED_FROM.get(sym, "1900-01-01")):
                    continue          # contract not listed yet: cannot have held it
                px = piv_px.at[dt, sym]
                sd = sigma_d.at[dt, sym] if dt in sigma_d.index else np.nan
                if not (np.isfinite(px) and np.isfinite(sd)) or sd <= 0:
                    continue
                c = BY_SYMBOL[sym]
                n_star, buf = target_contracts(row["F"], sd * np.sqrt(252), px, c, p,
                                               n_active=n_active, avg_corr=rho_t)
                new_n = apply_buffer(held[sym], n_star, buf)
                delta = new_n - held[sym]
                if delta != 0:
                    tc = trade_cost(delta, c, px, sd,
                                    sigma_ref.at[dt, sym] if dt in sigma_ref.index else np.nan,
                                    adv.at[dt, sym] if dt in adv.index else 0.0)
                    for k in cost_today:
                        cost_today[k] += tc[k]
                    trades.append(dict(date=dt, symbol=sym, delta=delta,
                                       n_star=n_star, new_n=new_n, **tc))
                    held[sym] = new_n

        # ---- roll costs ---------------------------------------------------------
        day = front[front["date"] == dt]
        for _, r in day.iterrows():
            sym = r["symbol"]
            if r["is_roll"] and held.get(sym, 0) != 0:
                c = BY_SYMBOL[sym]
                sd = sigma_d.at[dt, sym] if dt in sigma_d.index else np.nan
                tc = trade_cost(held[sym], c, r["settle"], sd if np.isfinite(sd) else 0.01,
                                sigma_ref.at[dt, sym] if dt in sigma_ref.index else np.nan,
                                adv.at[dt, sym] if dt in adv.index else 0.0)
                for k in cost_today:
                    cost_today[k] += tc[k]

        rows.append(dict(date=dt, pnl_gross=gross,
                         pnl_net=gross - cost_today["total"], **cost_today,
                         gross_notional=sum(abs(held[s]) * BY_SYMBOL[s].multiplier *
                                            (piv_px.at[dt, s] if dt in piv_px.index and
                                             s in piv_px.columns and
                                             np.isfinite(piv_px.at[dt, s]) else 0.0)
                                            for s in held),
                         n_positions=sum(1 for v in held.values() if v != 0)))

    return Result(daily=pd.DataFrame(rows).set_index("date"),
                  trades=pd.DataFrame(trades),
                  audit=CLOCK.report())


# ----------------------------------------------------------------------------------
# BENCHMARK AND DIAGNOSTICS
# ----------------------------------------------------------------------------------

def tsmom_benchmark(front: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Deliberately naive 12-month time-series momentum on the same universe: sign of the
    trailing return, volatility-scaled, equal weighted. The comparison must be against a
    plain benchmark, not a tuned one, or the correlation number means nothing.
    """
    piv = front.pivot_table(index="date", columns="symbol", values="ret")
    sig = np.sign(piv.rolling(lookback, min_periods=lookback // 2).sum()).shift(1)
    vol = piv.ewm(span=60, min_periods=20).std().shift(1)
    w = sig / vol
    w = w.div(w.abs().sum(axis=1), axis=0)
    return (w * piv).sum(axis=1).rename("tsmom")


def correlation_profile(strategy: pd.Series, bench: pd.Series) -> dict:
    """
    Two numbers. The unconditional correlation is table stakes. The conditional one is
    the whole diversification argument: the mechanism predicts we earn MORE when trend
    does worst, because that is when trend followers unwind fastest and pay most for
    immediacy. If this is not true, the structural claim is falsified.
    """
    df = pd.concat([strategy.rename("strat"), bench], axis=1).dropna()
    if len(df) < 60:
        return dict(n=len(df))
    wk = df.resample("W-FRI").sum()
    worst = wk[wk["tsmom"] <= wk["tsmom"].quantile(0.20)]
    return dict(
        n_weeks=len(wk),
        corr_full=wk["strat"].corr(wk["tsmom"]),
        mean_weekly_uncond=wk["strat"].mean(),
        mean_weekly_worst_trend_quintile=worst["strat"].mean(),
        passes_structural_test=bool(worst["strat"].mean() >= wk["strat"].mean()),
    )


def sensitivity_grid(front, cot, weekly_ret, hold_grid=(2, 3, 4, 5),
                     vol_grid=(0.15, 0.20, 0.25), end=None) -> pd.DataFrame:
    """
    Report the surface, not the peak. A flat surface is evidence the result is not an
    artefact of one parameter cell. A spike is a warning, and should be reported as one.
    """
    out = []
    for h in hold_grid:
        for v in vol_grid:
            p = dict(PARAMS, hold_weeks=h, vol_target=v)
            global RISK_BUDGET
            RISK_BUDGET = CAPITAL * v * IDM / N_UNIVERSE
            s = compute_signals(cot, weekly_ret, p)
            st = run_backtest(front, s, p, end=end).stats(net=True)
            out.append(dict(hold_weeks=h, vol_target=v,
                            sharpe_net=round(st.get("sharpe", np.nan), 3),
                            ann_vol=round(st.get("ann_vol", np.nan), 3)))
    RISK_BUDGET = CAPITAL * PARAMS["vol_target"] * IDM / N_UNIVERSE
    return pd.DataFrame(out)


def split_oos(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """The most recent OOS_FRACTION of the sample is sealed until the end."""
    return dates[int(len(dates) * (1 - OOS_FRACTION))]


# ----------------------------------------------------------------------------------
# SYNTHETIC SMOKE TEST — machinery only. Produces no evidence about the strategy.
# ----------------------------------------------------------------------------------

def synthetic(seed: int = 7, years: int = 8):
    """
    Random-walk prices and random hedger flows with a deliberately WEAK embedded effect.
    Purpose: prove the engine runs, positions are integers, costs decompose, and the
    look-ahead guard fires. The resulting Sharpe is meaningless and is not reported.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=252 * years)
    tuesdays = pd.date_range(dates[0], dates[-1], freq="W-TUE")

    px_rows, cot_rows = [], []
    for c in UNIVERSE:
        lvl = {"MCL": 65, "QG": 4, "MGC": 3300, "SIL": 40, "MHG": 5.5, "ZC": 4.5,
               "ZW": 5.5, "KE": 5.5, "ZS": 11, "ZM": 320, "ZL": 0.5,
               "LE": 2.0, "HE": 0.9}[c.symbol]
        vol = 0.25 / np.sqrt(252)
        path = lvl * np.exp(np.cumsum(rng.normal(0, vol, len(dates))))
        # quarterly expiries so the roll logic is exercised
        for q_start in range(0, len(dates), 63):
            exp = dates[min(q_start + 75, len(dates) - 1)]
            seg = dates[q_start:q_start + 63]
            for d, pxv in zip(seg, path[q_start:q_start + 63]):
                px_rows.append(dict(date=d, symbol=c.symbol,
                                    contract=f"{c.symbol}{exp:%y%m}",
                                    settle=float(pxv), volume=120_000,
                                    open_interest=400_000, expiry=exp))
                CLOCK.register("settlement", d, d)
        base = 100_000
        for t in tuesdays:
            CLOCK.register("cot", t, cot_release_date(t))
            cot_rows.append(dict(report_date=t, symbol=c.symbol,
                                 prod_long=base + rng.normal(0, 8_000),
                                 prod_short=base * 1.4 + rng.normal(0, 8_000),
                                 open_interest=200_000 + rng.normal(0, 5_000)))
    return pd.DataFrame(px_rows), pd.DataFrame(cot_rows)


def weekly_returns_from_front(front: pd.DataFrame,
                              report_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Tuesday-to-Tuesday returns, aligned to the COT measurement date."""
    piv = front.pivot_table(index="date", columns="symbol", values="settle")
    wk = piv.reindex(piv.index.union(report_dates)).ffill().reindex(report_dates)
    ret = wk.pct_change()
    out = ret.stack().reset_index()
    out.columns = ["report_date", "symbol", "week_ret"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic data, machinery only")
    ap.add_argument("--run", action="store_true", help="real data")
    ap.add_argument("--cot", type=str, default=None)
    ap.add_argument("--prices", type=str, default=None)
    ap.add_argument("--oos-unlock", action="store_true",
                    help="touch the out-of-sample window. Once, at the end.")
    args = ap.parse_args()

    if args.smoke:
        prices, cot = synthetic()
        print(f"[smoke] synthetic: {len(prices):,} price rows, {len(cot):,} COT rows")
    elif args.run:
        prices, cot = load_prices(args.prices), load_cot(args.cot)
    else:
        ap.print_help()
        return

    front = build_front_series(prices)
    wk_ret = weekly_returns_from_front(front, pd.DatetimeIndex(sorted(cot["report_date"].unique())))
    signals = compute_signals(cot, wk_ret)

    dates = pd.DatetimeIndex(sorted(front["date"].unique()))
    cut = split_oos(dates)
    print(f"IS: {dates[0]:%Y-%m-%d} to {cut:%Y-%m-%d}  |  OOS sealed from {cut:%Y-%m-%d}")

    res_is = run_backtest(front, signals, end=cut)
    print("\n--- IN SAMPLE ---")
    print(" gross:", {k: round(v, 4) for k, v in res_is.stats(net=False).items()})
    print(" net  :", {k: round(v, 4) for k, v in res_is.stats(net=True).items()})

    # integer invariant
    if not res_is.trades.empty:
        assert (res_is.trades["new_n"] == res_is.trades["new_n"].round()).all(), \
            "fractional contract detected"
        cs = res_is.trades[["commission", "spread", "impact"]].sum()
        print(" cost mix:", {k: f"${v:,.0f}" for k, v in cs.items()})

    bench = tsmom_benchmark(front)
    print(" vs TSMOM:", {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in correlation_profile(res_is.daily["pnl_net"], bench).items()})

    print("\n--- LOOK-AHEAD AUDIT ---")
    print(res_is.audit.to_string(index=False) if not res_is.audit.empty else " (no registered inputs)")

    if args.oos_unlock:
        res_oos = run_backtest(front, signals, start=cut)
        print("\n--- OUT OF SAMPLE (touched once) ---")
        print(" net:", {k: round(v, 4) for k, v in res_oos.stats(net=True).items()})


if __name__ == "__main__":
    main()