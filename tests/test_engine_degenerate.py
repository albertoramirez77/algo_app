"""
Degenerate inputs through the execution engine: sizing, costs, and the cases where a
cross-section collapses. The standard being applied is the one in the brief — none of
these may raise, and none may silently produce NaN P&L.

Where the current code survives only by accident rather than by design, the accident
is named and pinned, because an accident is exactly what a later rewrite removes
without noticing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import immediacy as m
from conftest import make_cot, make_price_panel

ZC = m.BY_SYMBOL["ZC"]


# ----------------------------------------------------------------------------------
# integer positions — FROZEN
# ----------------------------------------------------------------------------------

def test_apply_buffer_always_returns_a_python_int():
    rng = np.random.default_rng(0)
    for _ in range(500):
        held = int(rng.integers(-50, 50))
        n_star = float(rng.normal(0, 20))
        buf = float(abs(rng.normal(0, 3))) + 0.5
        out = m.apply_buffer(held, n_star, buf)
        assert isinstance(out, (int, np.integer))
        assert out == int(out)


def test_a_target_inside_the_band_produces_no_trade():
    assert m.apply_buffer(5, 5.3, 0.5) == 5
    assert m.apply_buffer(5, 4.7, 0.5) == 5


def test_a_target_outside_the_band_trades_only_to_the_near_edge():
    """Carver's rule: close the gap to the edge of the band, not to the target."""
    assert m.apply_buffer(10, 2.0, 1.0) == 3      # trade DOWN to n_star + buffer
    assert m.apply_buffer(0, 8.0, 1.0) == 7       # trade UP to n_star - buffer


def test_a_nan_target_leaves_the_position_where_it_was():
    """No sizing information must mean no trade, never a trade to zero."""
    assert m.apply_buffer(4, np.nan, 0.5) == 4
    assert m.apply_buffer(4, 2.0, np.nan) == 4
    assert m.apply_buffer(0, np.nan, np.nan) == 0


def test_the_buffer_never_falls_below_half_a_contract():
    """At $450k a 10% band is often narrower than one lot; the floor is what bites."""
    _, buf = m.target_contracts(10.0, 0.20, 400.0, ZC)
    assert buf >= m.BUFFER_FLOOR_CONTRACTS


def test_every_trade_in_a_full_run_is_an_integer_number_of_contracts():
    front, signals = _small_universe_run()
    res = m.run_backtest(front, signals)
    if not res.trades.empty:
        assert (res.trades["new_n"] == res.trades["new_n"].round()).all()
        assert (res.trades["delta"] == res.trades["delta"].round()).all()


# ----------------------------------------------------------------------------------
# sizing guards
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("sigma,price", [(0.0, 400.0), (400.0, 0.0), (np.nan, 400.0),
                                         (0.20, np.nan), (-0.1, 400.0)])
def test_an_unusable_denominator_sizes_to_flat_rather_than_raising(sigma, price):
    n_star, buf = m.target_contracts(10.0, sigma, price, ZC)
    assert n_star == 0.0
    assert buf == m.BUFFER_FLOOR_CONTRACTS


def test_a_nan_forecast_produces_a_nan_target_which_the_buffer_absorbs():
    """
    The two guards compose: target_contracts passes NaN through, apply_buffer declines
    to trade on it. Neither alone is sufficient; both together mean no position moves.
    """
    n_star, buf = m.target_contracts(np.nan, 0.20, 400.0, ZC)
    assert np.isnan(n_star)
    assert m.apply_buffer(7, n_star, buf) == 7


def test_a_nan_correlation_poisons_the_size_and_the_engine_guards_it_upstream():
    """
    ACCIDENT, NAMED. target_contracts computes min(1/sqrt(...max(avg_corr, 0.01)...)).
    Python's max(nan, 0.01) returns nan, so a NaN correlation returns (nan, nan) and
    the buffer floor is lost too.

    Nothing downstream of target_contracts catches that; what saves the run is the
    explicit `if not np.isfinite(rho_t): rho_t = 0.15` in run_backtest. This test pins
    both halves so removing the upstream guard fails here.
    """
    n_star, buf = m.target_contracts(10.0, 0.20, 400.0, ZC, avg_corr=np.nan)
    assert np.isnan(n_star) and np.isnan(buf)

    import inspect
    src = inspect.getsource(m.run_backtest)
    assert "np.isfinite(rho_t)" in src, "the upstream NaN-correlation guard is gone"


def test_the_risk_budget_is_shared_among_the_active_names():
    """Halving the active count must double each name's target, not leave it fixed."""
    a, _ = m.target_contracts(10.0, 0.20, 400.0, ZC, n_active=12, avg_corr=0.15)
    b, _ = m.target_contracts(10.0, 0.20, 400.0, ZC, n_active=6, avg_corr=0.15)
    assert b > a


def test_the_diversification_multiplier_is_capped_at_carvers_value():
    """A near-zero correlation must not let IDM run away."""
    for corr in (0.0, 0.001, 0.01):
        n_star, _ = m.target_contracts(10.0, 0.20, 400.0, ZC, n_active=13,
                                       avg_corr=corr)
        capped, _ = m.target_contracts(10.0, 0.20, 400.0, ZC, n_active=13,
                                       avg_corr=1.0)
        assert n_star / capped <= m.IDM / 1.0 + 1e-9


def test_the_target_is_linear_in_the_forecast():
    """Doubling the forecast doubles the target; the scalar is not applied twice."""
    a, _ = m.target_contracts(5.0, 0.20, 400.0, ZC)
    b, _ = m.target_contracts(10.0, 0.20, 400.0, ZC)
    assert b == pytest.approx(2.0 * a, rel=1e-12)


# ----------------------------------------------------------------------------------
# cost model
# ----------------------------------------------------------------------------------

def test_no_trade_costs_nothing():
    c = m.trade_cost(0, ZC, 400.0, 0.01, 0.01, 1000.0)
    assert c == dict(commission=0.0, spread=0.0, impact=0.0, total=0.0)


def test_commission_is_charged_per_contract_per_side_and_is_sign_blind():
    up = m.trade_cost(3, ZC, 400.0, 0.01, 0.01, 1000.0)
    down = m.trade_cost(-3, ZC, 400.0, 0.01, 0.01, 1000.0)
    assert up["commission"] == pytest.approx(3 * ZC.commission_side)
    assert down["commission"] == up["commission"]


def test_the_three_components_sum_to_the_total():
    c = m.trade_cost(4, ZC, 400.0, 0.02, 0.01, 5000.0)
    assert c["total"] == pytest.approx(c["commission"] + c["spread"] + c["impact"])


def test_all_cost_components_are_non_negative():
    rng = np.random.default_rng(1)
    for _ in range(300):
        c = m.trade_cost(int(rng.integers(-20, 20)), ZC,
                         float(rng.uniform(1, 1000)), float(rng.uniform(0, 0.1)),
                         float(rng.uniform(0.001, 0.05)), float(rng.uniform(1, 1e5)))
        assert all(v >= 0 for v in c.values())


def test_the_spread_widens_with_volatility_and_never_narrows():
    calm = m.trade_cost(1, ZC, 400.0, 0.01, 0.01, 1000.0)["spread"]
    wild = m.trade_cost(1, ZC, 400.0, 0.05, 0.01, 1000.0)["spread"]
    quiet = m.trade_cost(1, ZC, 400.0, 0.001, 0.01, 1000.0)["spread"]
    assert wild > calm
    assert quiet == calm, "vol_ratio floors at 1.0; a calm day gets no discount"


def test_a_nan_reference_volatility_falls_back_to_a_ratio_of_one():
    """
    ACCIDENT, NAMED. `max(1.0, sigma_daily / sigma_ref)` with a NaN reference relies on
    Python's max returning its first argument when the comparison is False. Rewritten
    as np.maximum this returns NaN and poisons the whole day's cost.

    sigma_ref is NaN for the first 60 sessions of every symbol, so this path is real.
    """
    ref_nan = m.trade_cost(2, ZC, 400.0, 0.01, np.nan, 1000.0)
    baseline = m.trade_cost(2, ZC, 400.0, 0.01, 0.01, 1000.0)
    assert np.isfinite(ref_nan["total"])
    assert ref_nan["spread"] == pytest.approx(baseline["spread"])
    assert np.isnan(np.maximum(1.0, 0.01 / np.nan)), "the idiom that would break it"


def test_a_zero_reference_volatility_also_falls_back_to_one():
    c = m.trade_cost(2, ZC, 400.0, 0.01, 0.0, 1000.0)
    assert np.isfinite(c["total"])


def test_zero_average_volume_charges_no_impact_rather_than_dividing_by_zero():
    c = m.trade_cost(2, ZC, 400.0, 0.01, 0.01, 0.0)
    assert c["impact"] == 0.0
    assert np.isfinite(c["total"])


def test_a_nan_average_volume_produces_a_nan_cost():
    """
    UNGUARDED PATH, pinned. `participation = 0.0 if adv <= 0 else q / adv` treats NaN
    as "not <= 0" and divides by it, so the day's total cost and therefore pnl_net
    become NaN.

    It does not fire today: no trade is taken until the 60-day volatility estimate
    exists, by which time the 20-day median volume does too. The first trade in the
    real run is 517 sessions into the sample. Nothing enforces that ordering, though,
    and this is the line that would absorb a reordering.
    """
    c = m.trade_cost(2, ZC, 400.0, 0.01, 0.01, np.nan)
    assert np.isnan(c["impact"])
    assert np.isnan(c["total"])


def test_the_impact_term_dominates_the_other_two_not_the_reverse():
    """
    CONTRADICTS A DOCSTRING — reported, not repaired.

    trade_cost's docstring says: "At $450k the impact term is ~4 orders of magnitude
    below the other two. That is the capacity finding." The code's own output says the
    opposite. On the in-sample run the decomposition is commission $4,750 (18.1%),
    spread $11,390 (43.4%), impact $10,102 (38.5%) — impact is 2.13x commission and
    0.89x spread, i.e. the same order of magnitude, not four below.

    The cause is visible in the second comment in the same function: the square-root
    law is applied with a coefficient of 1 at participation levels far below where it
    is calibrated, and is deliberately kept as a conservative overstatement. The two
    comments cannot both be true. This test pins the arithmetic so whichever one is
    corrected, it is corrected against a number.
    """
    c = m.trade_cost(2, ZC, 400.0, 0.012, 0.012, adv_contracts=50_000.0)
    assert c["impact"] > 20 * c["spread"]
    assert c["impact"] > 100 * c["commission"]


def test_impact_falls_as_the_square_root_of_participation():
    """The functional form itself is right; only its calibration is in question."""
    a = m.trade_cost(2, ZC, 400.0, 0.012, 0.012, 50_000.0)["impact"]
    b = m.trade_cost(2, ZC, 400.0, 0.012, 0.012, 200_000.0)["impact"]
    assert a / b == pytest.approx(2.0, rel=1e-9)     # 4x the ADV, half the impact


# ----------------------------------------------------------------------------------
# collapsing cross-sections
# ----------------------------------------------------------------------------------

def _signal_frame(symbols, n_weeks=140, oi=200_000.0, seed=0):
    rng = np.random.default_rng(seed)
    rd = pd.date_range("2016-01-05", periods=n_weeks, freq="W-TUE")
    cot = make_cot(symbols, rd, rng=rng)
    if oi != 200_000.0:
        cot["open_interest"] = oi
    wk = pd.DataFrame([dict(report_date=t, symbol=s, week_ret=float(rng.normal(0, 0.02)))
                       for t in rd for s in symbols])
    return cot, wk


def test_a_single_instrument_universe_can_never_take_a_position():
    """
    Same shape as the one-member sector already documented in compute_signals, one
    level up. With one name the cross-sectional median of HPbar IS that name, so
    `HPbar > median` is False forever and eligibility is identically zero. The
    cross-sectional standard deviation is also NaN, so z is zero as well.

    Two independent mechanisms both zero it. Neither reports anything.
    """
    cot, wk = _signal_frame(["ZC"])
    sig = m.compute_signals(cot, wk)
    assert (sig["e"] == 0.0).all()
    assert sig["z"].isna().all()
    assert (sig["F"] == 0.0).all()


def test_a_two_instrument_universe_makes_exactly_one_name_eligible():
    cot, wk = _signal_frame(["ZC", "ZW"])
    sig = m.compute_signals(cot, wk).dropna(subset=["HPbar"])
    per_week = sig.groupby("report_date")["e"].sum()
    assert (per_week == 1.0).all()


def test_a_zero_open_interest_observation_nans_the_whole_weeks_cross_section():
    """
    THE FINDING. Q divides by the previous week's open interest. One zero makes that
    name's Q infinite; the cross-sectional mean and standard deviation then become
    non-finite, so EVERY name's z that week is NaN, not just the affected one.

    `s = z.fillna(0) * e * g` turns that into a silent zero for the entire book. With
    hold_weeks=3 the rolling mean keeps a third of the signal missing for three
    consecutive weeks and nothing prints a warning.

    cot.parquet contains no zero open interest today — fetch_cot.build filters
    `open_interest > 0` — so this is latent, not live. It is the same defect class as
    the one-member sector: a single bad cell deletes a sleeve in silence.
    """
    symbols = ["ZC", "ZW", "ZS", "ZM", "ZL"]
    cot, wk = _signal_frame(symbols)
    bad_week = sorted(cot["report_date"].unique())[60]
    cot.loc[(cot["report_date"] == bad_week) & (cot["symbol"] == "ZC"),
            "open_interest"] = 0.0

    sig = m.compute_signals(cot, wk)
    hit = sorted(sig["report_date"].unique())[61]          # oi_lag is the zero
    week = sig[sig["report_date"] == hit]
    assert np.isinf(week["Q"]).sum() == 1                  # one name divides by zero
    assert week["z"].isna().all()                          # every name loses its z
    assert (week["s"].abs() < 1e-12).all() if "s" in week else True

    # and the hole is invisible downstream: S is a 3-week mean, so the forecast that
    # week is carried by the two neighbouring weeks and never reads as missing.
    assert sig.loc[sig["report_date"] == hit, "F"].notna().all()


def test_a_week_with_no_volatility_estimate_takes_no_trades():
    """An all-NaN volatility week must skip, not size on stale or zero risk."""
    front, signals = _small_universe_run()
    front = front.copy()
    front["ret"] = np.nan                                   # no vol can be estimated
    res = m.run_backtest(front, signals)
    assert res.trades.empty
    assert res.daily["pnl_net"].notna().all()
    assert (res.daily["pnl_net"] == 0.0).all()


def test_a_run_with_no_eligible_names_produces_zero_pnl_and_no_nan():
    front, signals = _small_universe_run()
    flat = signals.copy()
    flat["F"] = 0.0
    res = m.run_backtest(front, flat)
    assert res.trades.empty
    assert res.daily["pnl_net"].notna().all()
    assert res.daily["pnl_net"].abs().sum() == 0.0


def test_a_single_instrument_backtest_runs_without_raising():
    front, signals = _small_universe_run(symbols=("ZC",))
    res = m.run_backtest(front, signals)
    assert len(res.daily) > 0
    assert res.daily["pnl_net"].notna().all()


# ----------------------------------------------------------------------------------
# reported statistics
# ----------------------------------------------------------------------------------

def _result(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return m.Result(daily=pd.DataFrame({"pnl_net": values, "pnl_gross": values},
                                       index=idx),
                    trades=pd.DataFrame())


def test_a_short_sample_reports_only_its_length():
    assert _result([1.0] * 59).stats() == dict(n_days=59)


def test_an_empty_result_reports_only_its_length():
    assert _result([]).stats() == dict(n_days=0)


def test_an_exactly_constant_pnl_is_caught_by_the_zero_variance_guard():
    assert _result([0.0] * 300).stats() == dict(n_days=300)


def test_a_constant_nonzero_pnl_slips_past_the_guard_and_reports_an_absurd_sharpe():
    """
    CHARACTERISATION OF A KNOWN DEFECT — deliberately left unfixed.

    Result.stats guards with `r.std() == 0`. A constant nonzero daily P&L has a
    standard deviation of floating-point dust rather than exactly zero, so the guard
    misses and the reported Sharpe is of order 1e17.

    Same shape as the `se > 0` guard in cluster_ols: an exact-equality test standing in
    for a tolerance. The real run's P&L is never constant, so nothing on the record
    depends on it. The fix is a relative tolerance, which is a behaviour change and the
    author's call.
    """
    st = _result([100.0] * 300).stats()
    assert st["ann_vol"] < 1e-15
    assert st["sharpe"] > 1e15


# ----------------------------------------------------------------------------------
# helper: a small end-to-end run
# ----------------------------------------------------------------------------------

def _small_universe_run(symbols=("ZC", "ZW", "ZS", "ZM", "ZL", "MGC")):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-04", periods=600)
    prices = make_price_panel(
        symbols, dates,
        settle=lambda s, c, d: 100.0 * float(
            np.exp(rng.normal(0, 0.01))))
    front = m.build_front_series(prices)
    rd = pd.DatetimeIndex(sorted(pd.date_range(dates[0], dates[-1], freq="W-TUE")))
    cot, wk = _signal_frame(list(symbols), n_weeks=len(rd))
    cot["report_date"] = np.tile(rd, len(symbols))[:len(cot)]
    wk["report_date"] = np.tile(rd, len(symbols))[:len(wk)]
    signals = m.compute_signals(cot, wk)
    return front, signals
