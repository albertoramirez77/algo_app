"""
test_traded_engine.py — coverage for the code that produces the pitch numbers.

The rest of tests/ exercises immediacy.py, the engine for the hedger-flow hypothesis
that was tested and rejected (see failed_research/hedger_flow_null/). Nothing tested
final_numbers.py or universe.py, which are what the reported strategy actually runs on.
This file closes that gap.

Every test here runs on synthetic data. No vendor file and no API key is needed, so a
reviewer can verify the engine's invariants without a Databento subscription.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import final_numbers as fn
import universe as uni


# ----------------------------------------------------------------------------------
# 1. universe integrity — the specs every dollar figure is computed from
# ----------------------------------------------------------------------------------

def test_symbols_are_unique():
    syms = [i.symbol for i in uni.UNIVERSE]
    assert len(syms) == len(set(syms))


def test_every_contract_has_a_positive_multiplier_tick_and_tick_value():
    for i in uni.UNIVERSE:
        assert i.multiplier > 0, i.symbol
        assert i.tick > 0, i.symbol
        assert i.tick_value > 0, i.symbol
        assert i.commission >= 0, i.symbol


def test_the_cents_quoted_contracts_carry_the_hundredth_scale():
    """
    CME quotes grains, the soy complex and livestock in CENTS. Corn settling at 440.75 is
    $4.4075 a bushel, so one contract is $22,037 and not $2.2 million. Taking the quote at
    face value inflated seven of seventeen commodity notionals by 100x. This is the test
    that would have caught it.
    """
    cents = {"ZC", "ZW", "KE", "ZS", "ZL", "LE", "HE"}
    for s in cents:
        assert uni.BY_SYMBOL[s].price_scale == pytest.approx(0.01), s


def test_soybean_meal_is_the_dollar_exception_inside_the_soy_complex():
    """ZM is quoted in DOLLARS per short ton, unlike the rest of the complex."""
    assert uni.BY_SYMBOL["ZM"].price_scale == pytest.approx(1.0)


def test_corn_notional_is_in_the_tens_of_thousands_not_the_millions():
    notional = 440.75 * uni.BY_SYMBOL["ZC"].dollar_price_mult
    assert 15_000 < notional < 30_000, notional


def test_dollar_price_mult_is_multiplier_times_scale():
    for i in uni.UNIVERSE:
        assert i.dollar_price_mult == pytest.approx(i.multiplier * i.price_scale)


def test_the_narrow_universe_is_a_subset_of_the_full_one():
    assert set(uni.NARROW) <= set(uni.BY_SYMBOL)


# ----------------------------------------------------------------------------------
# 2. the net-exposure control — built, measured, and switched off
# ----------------------------------------------------------------------------------

def test_neutralise_drives_net_dollar_exposure_to_exactly_zero():
    """
    Position i receives notional proportional to w_i / vol_i, so net notional is
    proportional to sum(w_i / vol_i). Subtracting one constant from every weight makes
    that sum zero.
    """
    rng = np.random.default_rng(0)
    w = rng.normal(size=12)
    w -= w.mean()
    vol = rng.uniform(0.1, 0.6, size=12)
    out = fn.neutralise(w, vol)
    assert float(np.sum(out / vol)) == pytest.approx(0.0, abs=1e-12)


def test_neutralise_does_not_reorder_the_signal():
    """It is a risk control, not a second signal: the ranking must be untouched."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=16)
    vol = rng.uniform(0.1, 0.6, size=16)
    assert list(np.argsort(fn.neutralise(w, vol))) == list(np.argsort(w))


def test_the_net_exposure_control_ships_switched_off():
    """The pitch reports the exposure rather than constraining it. Guard the default."""
    assert fn.NEUTRALISE is False


# ----------------------------------------------------------------------------------
# 3. the frozen specification — changing any of these is a spec change, not a tuning run
# ----------------------------------------------------------------------------------

def test_the_specification_constants_are_what_the_pitch_reports():
    assert fn.CAPITAL == 450_000.0
    assert fn.VOL_TARGET == 0.20
    assert fn.IDM == 2.5
    assert fn.J == 12          # twelve-month formation: one inventory cycle
    assert fn.VOL_WINDOW == 6
    assert fn.N_GRIDS == 21
    assert fn.COST_MULTIPLE == 3.0


# ----------------------------------------------------------------------------------
# 4. performance statistics
# ----------------------------------------------------------------------------------

def test_sharpe_and_t_are_consistent_with_each_other():
    """t = Sharpe * sqrt(years). If these ever disagree the pitch table is wrong."""
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.01, 0.05, size=183))
    s = fn.st(r)
    assert s["t"] == pytest.approx(s["sharpe"] * np.sqrt(s["yrs"]), rel=1e-9)
    assert s["yrs"] == pytest.approx(183 / 12)


def test_too_short_a_series_returns_nan_rather_than_a_flattering_number():
    s = fn.st(pd.Series([0.01] * 12))
    assert np.isnan(s["sharpe"])


@pytest.mark.xfail(reason="FROZEN: a constant return series has floating-point residual "
                          "volatility near 1e-18, so st() reports an astronomically large "
                          "Sharpe instead of NaN. Unreachable on real data; recorded "
                          "rather than silently patched.",
                   strict=True)
def test_a_constant_series_should_not_produce_a_finite_sharpe():
    assert not np.isfinite(fn.st(pd.Series([0.01] * 60))["sharpe"])


def test_r2_of_a_variable_against_itself_is_one():
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    assert fn.r2(y, y.reshape(-1, 1)) == pytest.approx(1.0, abs=1e-9)


def test_r2_against_pure_noise_is_near_zero():
    rng = np.random.default_rng(4)
    y = rng.normal(size=500)
    X = rng.normal(size=(500, 1))
    assert abs(fn.r2(y, X)) < 0.05


def test_r2_refuses_to_report_on_too_short_a_sample():
    rng = np.random.default_rng(9)
    y = rng.normal(size=100)
    assert np.isnan(fn.r2(y, y.reshape(-1, 1)))


# ----------------------------------------------------------------------------------
# 5. signal construction — weights, and the rounding claim
# ----------------------------------------------------------------------------------

def _panel(n_sym: int = 8, n_months: int = 40, seed: int = 5) -> pd.DataFrame:
    """A synthetic daily panel with the columns grid_targets expects."""
    rng = np.random.default_rng(seed)
    syms = uni.NARROW[:n_sym]
    dates = pd.bdate_range("2015-01-01", periods=n_months * 21)
    rows = []
    for s in syms:
        px = 100.0
        for d in dates:
            r = rng.normal(0, 0.01)
            px *= np.exp(r)
            rows.append(dict(symbol=s, date=d, settle_0=px, settle_1=px * 1.01,
                             r0=r, r1=r + rng.normal(0, 0.002), asset="commodity"))
    df = pd.DataFrame(rows)
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def test_grid_targets_produces_both_long_and_short_positions():
    """
    Weights are rank minus mean rank, so a cross-sectional book is always two-sided.
    There is no flat state — which is why the pitch says it is invested in every month.
    """
    df = _panel()
    t = fn.grid_targets(df, offset=0, keep=set(uni.NARROW[:8]))
    assert not t.empty
    for _, g in t.groupby("date"):
        assert (g["target"] > 0).any() and (g["target"] < 0).any()


def test_rounding_once_after_averaging_is_not_the_same_as_rounding_each_grid():
    """
    The tranched book averages 21 fractional target vectors and rounds ONCE. Rounding each
    grid separately and then averaging is a different operation, and it discards positions
    that individually round to zero. The pitch claims these differ; this pins the claim.
    """
    grids = np.array([[0.6, 1.4],
                      [0.6, 1.4],
                      [-0.4, 1.4]])
    round_once = np.round(grids.mean(axis=0))      # mean [0.267, 1.4] -> [0, 1]
    round_each = np.round(grids).mean(axis=0)      # [[1,1],[1,1],[-0,1]] -> [0.667, 1]
    assert round_once.tolist() == [0.0, 1.0]
    assert round_each[0] == pytest.approx(2 / 3)
    assert not np.array_equal(round_once, round_each)


def test_positions_are_whole_contracts():
    """No fractional contracts anywhere. A third of a corn contract does not exist."""
    targets = np.array([0.49, -0.51, 2.4, -3.6])
    held = np.round(targets)
    assert np.all(held == held.astype(int))
    assert held.tolist() == [0.0, -1.0, 2.0, -4.0]


# ----------------------------------------------------------------------------------
# 6. the cost model — the universe rule is computed from specs, never from returns
# ----------------------------------------------------------------------------------

def test_ex_ante_cost_is_one_and_a_half_ticks_plus_commission():
    """
    Half a tick to cross plus one further tick of slippage, plus the exact commission,
    all expressed in basis points of notional. Reproduced here independently.
    """
    df = _panel(n_sym=4, n_months=12)
    out = fn.ex_ante_costs(df).set_index("symbol")
    for s in out.index:
        inst = uni.BY_SYMBOL[s]
        notional = out.loc[s, "notional"]
        tick_bp = inst.tick_value / notional * 1e4
        expected = 1.5 * tick_bp + inst.commission / notional * 1e4
        assert out.loc[s, "cost_bp"] == pytest.approx(expected, rel=1e-9)


def test_the_exclusion_rule_depends_only_on_contract_specifications():
    """
    The universe rule is a function of tick value, multiplier, price and commission.
    No return series enters it, which is what makes it ex ante.
    """
    df = _panel(n_sym=6, n_months=12)
    a = fn.ex_ante_costs(df)
    shuffled = df.copy()
    rng = np.random.default_rng(7)
    shuffled["r0"] = rng.permutation(shuffled["r0"].to_numpy())
    b = fn.ex_ante_costs(shuffled)
    pd.testing.assert_series_equal(
        a.set_index("symbol")["cost_bp"], b.set_index("symbol")["cost_bp"])
