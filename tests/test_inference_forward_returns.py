"""
forward_returns is the left-hand side of every cross-sectional regression in the
project. If its window is off by one day in either direction the entire mechanism
test measures something other than what it claims.

The method here is to put a single known return on a single known day and then assert
exactly which forward observations contain it. That pins the window by construction
rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mechanism_test import forward_returns

H = 5          # small enough to enumerate the whole window by hand


def one_spike(n_days=40, spike_at=20, spike=0.10, symbol="A"):
    """A return series that is zero everywhere except one day."""
    dates = pd.bdate_range("2015-01-05", periods=n_days)
    ret = np.zeros(n_days)
    ret[spike_at] = spike
    return pd.DataFrame(dict(date=dates, symbol=symbol, ret=ret)), dates


# ----------------------------------------------------------------------------------
# the window is t+1 .. t+h
# ----------------------------------------------------------------------------------

def test_the_spike_appears_in_exactly_the_h_days_before_it():
    """
    A return on day k shows up in the forward return of days k-h .. k-1 and nowhere
    else. That is the definition of a window covering t+1 .. t+h.
    """
    k, spike = 20, 0.10
    front, dates = one_spike(spike_at=k, spike=spike)
    fwd = forward_returns(front, H)["A"]

    carrying = [i for i in range(len(dates)) if np.isclose(fwd.iloc[i], spike)]
    assert carrying == list(range(k - H, k))


def test_the_spike_day_itself_does_not_contain_its_own_return():
    """Day t's forward return excludes day t. This is the look-ahead boundary."""
    k, spike = 20, 0.10
    front, _ = one_spike(spike_at=k, spike=spike)
    fwd = forward_returns(front, H)["A"]
    assert fwd.iloc[k] == pytest.approx(0.0, abs=1e-12)


def test_the_day_h_before_the_spike_does_contain_it():
    """The far edge of the window is inclusive: t+h is in, t+h+1 is out."""
    k, spike = 20, 0.10
    front, _ = one_spike(spike_at=k, spike=spike)
    fwd = forward_returns(front, H)["A"]
    assert fwd.iloc[k - H] == pytest.approx(spike, rel=1e-12)      # t+h  -> in
    assert fwd.iloc[k - H - 1] == pytest.approx(0.0, abs=1e-12)    # t+h+1 -> out


def test_the_last_h_rows_are_nan_because_the_window_runs_off_the_end():
    """No forward window may be silently truncated into a shorter one."""
    front, dates = one_spike()
    fwd = forward_returns(front, H)["A"]
    assert fwd.iloc[-H:].isna().all()
    assert fwd.iloc[:-H].notna().all()


@pytest.mark.parametrize("h", [1, 2, 5, 15, 20])
def test_window_length_equals_h_for_every_horizon(h):
    """The number of observations carrying the spike is exactly h, at any horizon."""
    k, spike = 30, 0.07
    front, _ = one_spike(n_days=60, spike_at=k, spike=spike)
    fwd = forward_returns(front, h)["A"]
    carrying = [i for i in range(60) if np.isclose(fwd.iloc[i], spike)]
    assert carrying == list(range(k - h, k))


def test_returns_compound_within_the_window_rather_than_summing():
    """Two spikes inside one window must multiply, not add."""
    dates = pd.bdate_range("2015-01-05", periods=30)
    ret = np.zeros(30)
    ret[10], ret[11] = 0.10, 0.20
    front = pd.DataFrame(dict(date=dates, symbol="A", ret=ret))
    fwd = forward_returns(front, H)["A"]
    assert fwd.iloc[9] == pytest.approx(1.10 * 1.20 - 1.0, rel=1e-12)


# ----------------------------------------------------------------------------------
# no look-ahead, structurally
# ----------------------------------------------------------------------------------

def test_a_value_outside_the_window_cannot_change_the_answer():
    """
    Make a future return knowable only after the decision date's window closes, and
    the forward return on that date must be bit-for-bit unchanged.
    """
    front, dates = one_spike(n_days=40, spike_at=20, spike=0.0)
    base = forward_returns(front, H)["A"].copy()

    tampered = front.copy()
    tampered.loc[tampered.index[30], "ret"] = 0.99      # far beyond day 20's window
    after = forward_returns(tampered, H)["A"]

    decision_rows = slice(0, 30 - H)                    # windows that close before day 30
    pd.testing.assert_series_equal(base.iloc[decision_rows], after.iloc[decision_rows])


def test_every_row_depends_only_on_strictly_later_rows():
    """
    Sweep the whole series: perturbing day j must change the forward return of day i
    only when i < j <= i+h. Anything else would be a leak or a dropped observation.
    """
    n, h = 25, 4
    dates = pd.bdate_range("2015-01-05", periods=n)
    front = pd.DataFrame(dict(date=dates, symbol="A", ret=np.zeros(n)))
    base = forward_returns(front, h)["A"].to_numpy()

    for j in range(n):
        t = front.copy()
        t.loc[t.index[j], "ret"] = 0.05
        after = forward_returns(t, h)["A"].to_numpy()
        changed = {i for i in range(n)
                   if not np.allclose(np.nan_to_num(base[i], nan=-999),
                                      np.nan_to_num(after[i], nan=-999))}
        expected = {i for i in range(n) if i < j <= i + h and i < n - h}
        assert changed == expected, f"perturbing day {j} changed rows {sorted(changed)}"


# ----------------------------------------------------------------------------------
# assumptions the implementation rests on
# ----------------------------------------------------------------------------------

def test_the_shift_counts_rows_of_the_shared_calendar_not_a_symbols_own_days():
    """
    DISCLOSURE TEST. `cum.shift(-h)` moves h rows down the pivot index, and that index
    is the UNION of every symbol's dates. A symbol that does not trade on a date still
    occupies a row, so its h-day window spans h shared-calendar days, not h of its own
    trading days.

    With one symbol missing two days in the middle, its 5-day window covers only three
    of its own observations. Every product in the current universe reports ~252 days a
    year, so the calendars coincide and this costs nothing today. It would start to
    matter the moment a product with a different holiday calendar joins the universe.
    """
    dates = pd.bdate_range("2015-01-05", periods=20)
    a = pd.DataFrame(dict(date=dates, symbol="A", ret=0.0))
    b_dates = dates.delete([10, 11])                      # B misses two sessions
    b = pd.DataFrame(dict(date=b_dates, symbol="B", ret=0.0))
    b.loc[b.index[-1], "ret"] = 0.0
    front = pd.concat([a, b], ignore_index=True)

    fwd = forward_returns(front, H)
    assert list(fwd.index) == list(dates)                 # union calendar, 20 rows
    # B has no observation on those dates, and fillna(0) makes its window read as a
    # zero return there rather than as missing.
    assert fwd.loc[dates[6], "B"] == pytest.approx(0.0)


def test_missing_observations_are_treated_as_zero_return_not_as_missing():
    """
    DISCLOSURE TEST. `piv.fillna(0.0)` is what makes a roll day contribute no price
    change, which is the documented intent. The same line also gives a symbol that has
    not listed yet a flat zero-return history rather than NaN, so a forward window
    straddling a listing date is a real number instead of missing.

    LISTED_FROM is applied in the backtest but not in the cross-sectional panels, so
    this is the mechanism by which pre-listing rows can enter a regression.
    """
    dates = pd.bdate_range("2015-01-05", periods=20)
    a = pd.DataFrame(dict(date=dates, symbol="A", ret=0.01))
    late = pd.DataFrame(dict(date=dates[15:], symbol="LATE", ret=0.01))
    fwd = forward_returns(pd.concat([a, late], ignore_index=True), H)

    assert fwd.loc[dates[0], "LATE"] == pytest.approx(0.0)   # not NaN
    assert fwd.loc[dates[0], "A"] == pytest.approx(1.01 ** H - 1, rel=1e-12)


def test_duplicate_date_symbol_rows_are_averaged_silently():
    """
    DISCLOSURE TEST. pivot_table's default aggfunc is 'mean'. Two rows for the same
    (date, symbol) are averaged rather than raising, so a duplicate that survived
    upstream hygiene becomes a quiet halving of that day's return.

    test_curve.load() de-duplicates explicitly; build_front_series does not, and this
    is the line that would absorb the mistake.
    """
    dates = pd.bdate_range("2015-01-05", periods=10)
    front = pd.DataFrame(dict(date=list(dates) + [dates[3]],
                              symbol="A",
                              ret=[0.0] * 10 + [0.20]))
    front.loc[3, "ret"] = 0.0
    fwd = forward_returns(front, H)["A"]
    # day 3 carries mean(0.0, 0.20) = 0.10, not 0.20 and not an error
    assert fwd.iloc[2] == pytest.approx(0.10, rel=1e-12)
