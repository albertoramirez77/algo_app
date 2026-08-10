"""
The point-in-time clock. FROZEN — every test here is a characterisation of what the
code does today. Nothing in this file changes behaviour, and one property that the
author may want is recorded as an xfail rather than repaired.

The rule being pinned: COT positions are measured at Tuesday's close, published Friday
15:30 ET, and first executable at the following Monday's settlement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import immediacy as m
from immediacy import Clock, cot_release_date, first_tradeable_date


# ----------------------------------------------------------------------------------
# the guard raises; it does not warn
# ----------------------------------------------------------------------------------

def test_assert_usable_raises_on_look_ahead():
    """A decision taken before the input was knowable must be an exception."""
    c = Clock()
    with pytest.raises(ValueError, match="LOOK-AHEAD"):
        c.assert_usable("cot", pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-09"))


def test_assert_usable_permits_the_exact_boundary():
    """Deciding at the instant of knowability is allowed; the guard is strict-before."""
    c = Clock()
    c.assert_usable("cot", pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-10"))


def test_assert_usable_returns_none_rather_than_a_boolean():
    """
    It must not be usable as a condition. A function returning False would let a caller
    write `if not clock.assert_usable(...)` and carry on; raising cannot be ignored by
    accident.
    """
    c = Clock()
    assert c.assert_usable("cot", pd.Timestamp("2020-01-01"),
                           pd.Timestamp("2020-01-02")) is None


def test_the_engine_never_wraps_the_guard_in_a_try_block():
    """
    FROZEN: assert_usable must never be caught. Read the source of run_backtest and
    assert no `try` is open at the point the call is made.
    """
    import inspect

    src = inspect.getsource(m.run_backtest).splitlines()
    call = next(i for i, ln in enumerate(src) if "assert_usable" in ln)
    depth = 0
    for ln in src[:call]:
        stripped = ln.strip()
        if stripped.startswith("try:"):
            depth += 1
        elif stripped.startswith(("except", "finally")):
            depth = max(depth - 1, 0)
    assert depth == 0, "assert_usable is inside a try block"


def test_register_rejects_an_input_knowable_before_it_was_measured():
    c = Clock()
    with pytest.raises(ValueError, match="precedes as-of"):
        c.register("cot", pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-09"))


def test_the_audit_reports_the_lag_it_actually_saw():
    c = Clock()
    for t in pd.date_range("2020-01-07", periods=5, freq="W-TUE"):
        c.register("cot", t, cot_release_date(t))
    rep = c.report().set_index("input")
    assert rep.loc["cot", "n"] == 5
    assert rep.loc["cot", "min_lag_days"] == 3
    assert rep.loc["cot", "max_lag_days"] == 3


# ----------------------------------------------------------------------------------
# the release calendar
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("report,expected,why", [
    ("2024-01-02", "2024-01-05", "Tuesday measurement, Friday release"),
    ("2024-01-01", "2024-01-05", "Monday measurement still publishes that Friday"),
    ("2024-01-03", "2024-01-06", "Wednesday: +3 days lands past Friday"),
    ("2024-01-04", "2024-01-07", "Thursday: +3 days lands past Friday"),
])
def test_release_date_for_each_measurement_weekday(report, expected, why):
    assert cot_release_date(pd.Timestamp(report)) == pd.Timestamp(expected), why


def test_a_monday_report_is_not_released_a_day_early():
    """
    The correction the docstring is about. Without the Friday anchor a Monday-measured
    report would release on Thursday and be tradeable Friday — one session early.
    """
    monday = pd.Timestamp("2024-01-01")
    assert monday.dayofweek == 0
    naive = monday + pd.Timedelta(days=3)
    assert naive == pd.Timestamp("2024-01-04")             # Thursday: too early
    assert cot_release_date(monday) == pd.Timestamp("2024-01-05")


def test_release_is_never_earlier_than_three_days_after_measurement():
    """Sweep every weekday: the +3 floor always holds."""
    for d in pd.date_range("2024-01-01", periods=400, freq="D"):
        assert cot_release_date(d) >= d + pd.Timedelta(days=3)


def test_every_measurement_weekday_monday_to_thursday_first_trades_the_next_monday():
    """
    The property that actually matters downstream: whichever weekday the CFTC measured
    on, the first executable settlement is the following Monday.
    """
    days = pd.bdate_range("2024-01-01", periods=60)
    for d in pd.date_range("2024-01-01", periods=28, freq="D"):
        if d.dayofweek > 3:
            continue
        t = first_tradeable_date(d, days)
        assert t is not None and t.dayofweek == 0, f"{d:%Y-%m-%d} ({d.day_name()})"


# ----------------------------------------------------------------------------------
# the 2025 shutdown correction
# ----------------------------------------------------------------------------------

def test_reports_measured_inside_the_shutdown_are_held_until_the_backlog_cleared():
    for s in ("2025-10-07", "2025-10-28", "2025-11-10", "2025-12-23"):
        assert cot_release_date(pd.Timestamp(s)) == m.SHUTDOWN_CLEARED


def test_a_report_measured_after_the_backlog_cleared_is_released_normally():
    assert cot_release_date(pd.Timestamp("2025-12-30")) == pd.Timestamp("2026-01-02")


def test_a_held_report_first_trades_the_day_after_the_backlog_cleared():
    """
    SHUTDOWN_CLEARED is a Monday and the reports went out at 15:30 ET, after every
    settlement in this universe. First executable settlement is Tuesday.
    """
    days = pd.bdate_range("2025-12-01", periods=40)
    t = first_tradeable_date(pd.Timestamp("2025-11-04"), days)
    assert m.SHUTDOWN_CLEARED.dayofweek == 0
    assert t == pd.Timestamp("2025-12-30")


def test_the_window_starts_at_the_first_report_that_was_actually_withheld():
    """
    SHUTDOWN_START is 2025-09-30, not 2025-10-01. The suspension applied to PUBLICATION
    dates, and the report measured 2025-09-30 was scheduled to publish 2025-10-03 —
    inside the suspension — so it was the first one withheld.

    CFTC press release 9138-25: publication resumed 2025-11-19 with "data from the
    first missed report, which would have published Oct. 3".
    """
    assert m.SHUTDOWN_START == pd.Timestamp("2025-09-30")
    assert cot_release_date(pd.Timestamp("2025-09-30")) == m.SHUTDOWN_CLEARED


def test_the_last_report_published_on_schedule_is_left_alone():
    """
    The report measured 2025-09-23 went out on 2025-09-26, before the lapse. Moving the
    window start one week further back would wrongly withhold it.
    """
    assert cot_release_date(pd.Timestamp("2025-09-23")) == pd.Timestamp("2025-09-26")


def test_the_2025_09_30_report_is_never_released_before_it_was_public():
    """The regression this window exists to prevent: a 47-day look-ahead on one week."""
    assert cot_release_date(pd.Timestamp("2025-09-30")) >= pd.Timestamp("2025-11-19")


def test_no_other_report_in_the_sample_is_released_before_it_was_public():
    """
    The companion assertion: apart from the 2025-09-30 boundary case above, every
    report measured from 2025-09-23 onward gets a release date consistent with the
    CFTC's published record.
    """
    assert cot_release_date(pd.Timestamp("2025-09-23")) == pd.Timestamp("2025-09-26")
    for s in pd.date_range("2025-10-07", "2025-12-23", freq="W-TUE"):
        assert cot_release_date(s) == m.SHUTDOWN_CLEARED


# ----------------------------------------------------------------------------------
# first_tradeable_date edge cases (previously untested)
# ----------------------------------------------------------------------------------

def test_a_holiday_monday_pushes_execution_to_tuesday():
    """Martin Luther King Jr Day 2024 falls on Monday 15 January."""
    days = pd.bdate_range("2024-01-01", periods=40).drop(pd.Timestamp("2024-01-15"))
    t = first_tradeable_date(pd.Timestamp("2024-01-09"), days)
    assert t == pd.Timestamp("2024-01-16")


def test_a_run_of_closed_sessions_is_skipped_entirely():
    days = pd.bdate_range("2024-01-01", periods=40).drop(
        pd.date_range("2024-01-15", "2024-01-18"))
    t = first_tradeable_date(pd.Timestamp("2024-01-09"), days)
    assert t == pd.Timestamp("2024-01-19")


def test_the_year_boundary_is_crossed_without_special_casing():
    """A report measured in late December first trades in January."""
    days = pd.bdate_range("2025-12-01", "2026-02-01").drop(pd.Timestamp("2026-01-01"))
    t = first_tradeable_date(pd.Timestamp("2025-12-30"), days)
    assert t == pd.Timestamp("2026-01-05")          # release Fri 2026-01-02, next is Mon


def test_a_new_year_holiday_on_the_release_day_does_not_pull_execution_earlier():
    days = pd.bdate_range("2025-12-01", "2026-02-01").drop(
        [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05")])
    t = first_tradeable_date(pd.Timestamp("2025-12-30"), days)
    assert t == pd.Timestamp("2026-01-06")


def test_no_trading_day_after_the_release_returns_none():
    """The sample ends before the report could be traded: no position, not a guess."""
    days = pd.bdate_range("2024-01-01", periods=5)
    assert first_tradeable_date(pd.Timestamp("2024-01-09"), days) is None


def test_an_empty_calendar_returns_none():
    assert first_tradeable_date(pd.Timestamp("2024-01-09"),
                                pd.DatetimeIndex([])) is None


def test_the_release_day_itself_is_never_tradeable():
    """
    Strictly greater than, because publication is 15:30 ET — after the settlement
    window of every contract in this universe.
    """
    days = pd.bdate_range("2024-01-01", periods=40)
    rep = pd.Timestamp("2024-01-02")
    rel = cot_release_date(rep)
    assert rel in days
    assert first_tradeable_date(rep, days) > rel


def test_execution_is_always_strictly_after_the_release_across_the_whole_sample():
    """The end-to-end point-in-time property, swept over every week for four years."""
    days = pd.bdate_range("2022-01-03", "2026-01-01")
    for rep in pd.date_range("2022-01-04", "2025-09-23", freq="W-TUE"):
        t = first_tradeable_date(rep, days)
        assert t is not None and t > cot_release_date(rep)


def test_an_unsorted_calendar_gives_the_wrong_answer():
    """
    DISCLOSURE TEST. `trading_days[trading_days > release][0]` takes the first element
    in INDEX ORDER, not the earliest date. Every caller sorts before calling, so the
    pipeline is correct; the precondition is undocumented and unenforced.
    """
    days = pd.bdate_range("2024-01-01", periods=40)
    shuffled = pd.DatetimeIndex(np.random.default_rng(0).permutation(days.to_numpy()))
    good = first_tradeable_date(pd.Timestamp("2024-01-09"), days)
    bad = first_tradeable_date(pd.Timestamp("2024-01-09"), shuffled)
    assert good == pd.Timestamp("2024-01-15")
    assert bad != good
