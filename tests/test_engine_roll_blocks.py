"""
FROZEN INVARIANT: returns are never chained across a contract block.

The distinction that matters is block versus name. With an open-interest-ranked
continuous series the front month can revert A -> B -> A when open interest oscillates
near the roll. Grouping on the contract NAME stitches the two separate stints in A
together across the gap in between and manufactures a return that never existed.

These tests construct that exact A -> B -> A sequence and assert both seams are cut.
They also compute what name-grouping would have produced, so the size of the error
being avoided is on the record rather than asserted in the abstract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import immediacy as m


FAR = pd.Timestamp("2030-01-01")     # expiry beyond the sample: the roll filter is inert,
                                     # which is the shape px.parquet actually has


def series(contracts, settles, symbol="ZC", start="2015-01-05"):
    """One row per date, the contract name changing as given."""
    dates = pd.bdate_range(start, periods=len(contracts))
    return pd.DataFrame(dict(date=dates, symbol=symbol, contract=list(contracts),
                             settle=[float(s) for s in settles],
                             volume=1000.0, open_interest=5000.0, expiry=FAR))


# ----------------------------------------------------------------------------------
# the A -> B -> A seam
# ----------------------------------------------------------------------------------

CONTRACTS = ["A", "A", "A", "B", "B", "A", "A", "A"]
SETTLES = [100, 101, 102, 200, 202, 110, 111, 112]


def test_no_return_is_computed_across_either_seam():
    """
    Two contract changes, two NaN returns. Days 3 (A->B) and 5 (B->A) must both be
    blank; nothing else may be.
    """
    front = m.build_front_series(series(CONTRACTS, SETTLES))
    ret = front["ret"].to_numpy()
    assert np.isnan(ret[0])                     # first observation, no predecessor
    assert np.isnan(ret[3]), "A -> B seam was chained"
    assert np.isnan(ret[5]), "B -> A seam was chained"
    assert np.isfinite(ret[[1, 2, 4, 6, 7]]).all()


def test_the_second_stint_in_a_is_not_stitched_to_the_first():
    """
    The decisive case. Grouping by name would compute 110/102 - 1 on day 5 by pairing
    the second stint in A with the last day of the FIRST stint in A. That price move
    never happened to any position.
    """
    df = series(CONTRACTS, SETTLES)
    front = m.build_front_series(df)

    name_grouped = df["settle"] / df.groupby(["symbol", "contract"])["settle"].shift(1) - 1.0
    phantom = name_grouped.iloc[5]

    assert phantom == pytest.approx(110 / 102 - 1.0)     # what the bug would produce
    assert abs(phantom) > 0.07                           # ~7.8%, not a rounding artefact
    assert np.isnan(front["ret"].iloc[5])                # what the code actually produces


def test_returns_inside_a_block_are_chained_normally():
    """The invariant cuts seams; it must not also cut ordinary days."""
    front = m.build_front_series(series(CONTRACTS, SETTLES))
    assert front["ret"].iloc[1] == pytest.approx(101 / 100 - 1.0)
    assert front["ret"].iloc[2] == pytest.approx(102 / 101 - 1.0)
    assert front["ret"].iloc[4] == pytest.approx(202 / 200 - 1.0)
    assert front["ret"].iloc[6] == pytest.approx(111 / 110 - 1.0)


def test_the_invariant_holds_for_every_contract_change_in_a_long_sequence():
    """
    Sweep rather than spot-check: wherever the contract name differs from the previous
    row, the return must be NaN, and wherever it does not, the return must be finite.
    """
    rng = np.random.default_rng(0)
    names, prices, cur = [], [], "A"
    px = 100.0
    for i in range(200):
        if rng.random() < 0.12:
            cur = rng.choice(["A", "B", "C"])
            px *= rng.uniform(1.5, 2.5)          # a jump only a chained return would show
        px *= 1.0 + rng.normal(0, 0.01)
        names.append(cur); prices.append(px)

    front = m.build_front_series(series(names, prices))
    changed = front["contract"].ne(front["contract"].shift(1))
    assert front.loc[changed, "ret"].isna().all()
    assert front.loc[~changed, "ret"].notna().all()


def test_two_symbols_do_not_share_a_block_counter():
    """A contract change in one product must not blank a return in another."""
    a = series(["A"] * 6, [100, 101, 102, 103, 104, 105], symbol="ZC")
    b = series(["X", "X", "Y", "Y", "Y", "Y"], [50, 51, 90, 91, 92, 93], symbol="ZW")
    front = m.build_front_series(pd.concat([a, b], ignore_index=True))

    zc = front[front["symbol"] == "ZC"].reset_index(drop=True)
    zw = front[front["symbol"] == "ZW"].reset_index(drop=True)
    assert zc["ret"].iloc[1:].notna().all()          # ZC never rolls
    assert np.isnan(zw["ret"].iloc[2])               # ZW's seam is its own


# ----------------------------------------------------------------------------------
# the roll flag
# ----------------------------------------------------------------------------------

def test_the_roll_flag_marks_exactly_the_contract_changes():
    front = m.build_front_series(series(CONTRACTS, SETTLES))
    assert list(front["is_roll"]) == [True, False, False, True, False, True, False, False]


def test_the_first_day_of_a_symbol_is_flagged_as_a_roll():
    """
    DISCLOSURE TEST. `s != s.shift(1)` compares against NaN on the first row, which is
    True, so day one of every symbol carries is_roll=True. The subsequent .fillna(False)
    never fires because the comparison already produced a bool.

    It costs nothing today: the roll cost is charged only when a position is already
    held, and on day one the book is empty. It would start costing something if a roll
    cost were ever charged unconditionally.
    """
    front = m.build_front_series(series(["A"] * 5, [100, 101, 102, 103, 104]))
    assert bool(front["is_roll"].iloc[0]) is True
    assert not front["is_roll"].iloc[1:].any()


# ----------------------------------------------------------------------------------
# no back-adjusted series is constructed
# ----------------------------------------------------------------------------------

def test_settles_are_passed_through_unmodified():
    """
    FROZEN: no back-adjustment anywhere. The settlement column out must be the
    settlement column in, including across the price jump at a roll.
    """
    df = series(CONTRACTS, SETTLES)
    front = m.build_front_series(df)
    assert list(front["settle"]) == [float(s) for s in SETTLES]


def test_the_roll_selector_is_deterministic_and_ignores_volume_and_open_interest():
    """
    FROZEN: the roll is a fixed calendar offset. Feeding wildly different volume and
    open-interest columns must not change which contract is selected on any day.
    """
    dates = pd.bdate_range("2015-01-05", periods=40)
    rows = []
    for d in dates:
        for name, exp in (("NEAR", pd.Timestamp("2015-02-20")),
                          ("FARC", pd.Timestamp("2015-05-20"))):
            rows.append(dict(date=d, symbol="ZC", contract=name, settle=100.0,
                             volume=1.0, open_interest=1.0, expiry=exp))
    base = m.build_front_series(pd.DataFrame(rows))

    rng = np.random.default_rng(1)
    noisy = pd.DataFrame(rows)
    noisy["volume"] = rng.uniform(1, 1e6, len(noisy))
    noisy["open_interest"] = rng.uniform(1, 1e6, len(noisy))
    other = m.build_front_series(noisy)

    pd.testing.assert_series_equal(base["contract"], other["contract"])


def test_the_roll_happens_five_business_days_before_expiry():
    """ROLL_OFFSET_BDAYS is the whole rule; pin the boundary it implies."""
    dates = pd.bdate_range("2015-01-05", periods=40)
    expiry = pd.Timestamp("2015-02-20")          # a Friday
    rows = []
    for d in dates:
        for name, exp in (("NEAR", expiry), ("FARC", pd.Timestamp("2015-05-20"))):
            rows.append(dict(date=d, symbol="ZC", contract=name, settle=100.0,
                             volume=1.0, open_interest=1.0, expiry=exp))
    front = m.build_front_series(pd.DataFrame(rows))

    switch = front.loc[front["contract"].eq("FARC"), "date"].min()
    cutoff_at_switch = switch + pd.tseries.offsets.BDay(m.ROLL_OFFSET_BDAYS)
    prev = front.loc[front["date"] < switch, "date"].max()
    assert cutoff_at_switch >= expiry            # NEAR is inside the band on switch day
    assert prev + pd.tseries.offsets.BDay(m.ROLL_OFFSET_BDAYS) < expiry   # not the day before
