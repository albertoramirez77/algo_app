"""
monthly_panel builds the non-overlapping observations behind the basis control. The
claim is: the basis at the end of month M predicts the return of month M+1, with no
observation sharing a day with any other.

A period-shift join is easy to get one month off in either direction and the result
still looks plausible, so the tests below tag each month with a value that identifies
it and then assert the pairing arithmetic directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_curve import monthly_panel


def monthly_tagged(symbols=("A",), n_months=12, start="2015-01-01"):
    """
    One symbol-month is tagged so its identity survives the join:

        basis  during month j  ==  j / 100
        ret_0  during month j  compounds to exactly  j / 1000

    So if the join is correct, every output row satisfies mret == (basis*10 + 1)/1000,
    i.e. the return month is the basis month plus one.
    """
    rows = []
    months = pd.period_range(start, periods=n_months, freq="M")
    for s in symbols:
        for j, per in enumerate(months):
            days = pd.bdate_range(per.start_time, per.end_time)
            # one day carries the whole month's return, the rest are flat
            step = (1.0 + j / 1000.0) ** (1.0 / 1.0) - 1.0
            for i, d in enumerate(days):
                rows.append(dict(date=d, symbol=s, basis=j / 100.0,
                                 ret_0=step if i == 0 else 0.0))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# the pairing is basis month M -> return month M+1
# ----------------------------------------------------------------------------------

def test_basis_month_is_exactly_one_month_before_the_return_month():
    """The headline property, asserted on the tags rather than on the dates."""
    m = monthly_panel(monthly_tagged())
    assert len(m) > 0
    basis_month = (m["basis"] * 100).round().astype(int)
    return_month = (m["mret"] * 1000).round().astype(int)
    assert (return_month == basis_month + 1).all()


def test_the_pairing_is_not_contemporaneous():
    """Guards the M -> M direction, which would be look-ahead."""
    m = monthly_panel(monthly_tagged())
    basis_month = (m["basis"] * 100).round().astype(int)
    return_month = (m["mret"] * 1000).round().astype(int)
    assert not (return_month == basis_month).any()


def test_the_pairing_is_not_backwards():
    """Guards the M -> M-1 direction, which would predict the past."""
    m = monthly_panel(monthly_tagged())
    basis_month = (m["basis"] * 100).round().astype(int)
    return_month = (m["mret"] * 1000).round().astype(int)
    assert not (return_month == basis_month - 1).any()


def test_the_period_column_arithmetic_agrees_with_the_tags():
    """`ym` is the basis month and `ym_prev` its successor's shift; check them too."""
    m = monthly_panel(monthly_tagged())
    assert (m["ym_prev"] == m["ym"]).all()
    # ym is a monthly Period; the return it is paired with belongs to ym + 1
    tagged_basis_month = (m["basis"] * 100).round().astype(int)
    ordinal = m["ym"].apply(lambda p: p.ordinal)
    assert (ordinal - ordinal.min() == tagged_basis_month).all()


# ----------------------------------------------------------------------------------
# the basis is observed at the end of its month, before the return period opens
# ----------------------------------------------------------------------------------

def test_the_basis_date_is_the_last_trading_day_of_its_month():
    """Anything earlier would discard information; anything later cannot exist."""
    df = monthly_tagged()
    m = monthly_panel(df)
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    last_day = d.groupby(["symbol", "ym"])["date"].max().rename("last")
    j = m.merge(last_day, on=["symbol", "ym"], how="left")
    assert (j["date"] == j["last"]).all()


def test_the_basis_observation_precedes_every_day_of_the_return_it_predicts():
    """
    The return of month M+1 is compounded from days whose first entry is the move from
    the month-M close. The basis is read at that same close, so it is known before any
    of the return has happened.
    """
    df = monthly_tagged()
    m = monthly_panel(df)
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    for _, row in m.iterrows():
        nxt = d[(d["symbol"] == row["symbol"]) & (d["ym"] == row["ym"] + 1)]
        assert (nxt["date"] > row["date"]).all()


def test_the_first_day_of_the_return_month_is_inside_the_return():
    """
    The move from the month-M close to the first close of month M+1 belongs to M+1.
    Dropping it would leave a one-day hole between the signal and the return.
    """
    rows = []
    for j, per in enumerate(pd.period_range("2015-01-01", periods=4, freq="M")):
        for i, d in enumerate(pd.bdate_range(per.start_time, per.end_time)):
            # only the FIRST day of each month carries a return
            rows.append(dict(date=d, symbol="A", basis=j / 100.0,
                             ret_0=0.05 if i == 0 else 0.0))
    m = monthly_panel(pd.DataFrame(rows))
    assert len(m) >= 2
    assert np.allclose(m["mret"].to_numpy(), 0.05)


# ----------------------------------------------------------------------------------
# non-overlap
# ----------------------------------------------------------------------------------

def test_every_observation_is_a_distinct_symbol_month():
    """No symbol-month may appear twice; that is what non-overlapping means here."""
    m = monthly_panel(monthly_tagged(symbols=("A", "B", "C")))
    assert not m.duplicated(["symbol", "ym"]).any()


def test_no_two_return_windows_share_a_day():
    """
    Reconstruct the day set behind each observation's return and assert the sets are
    pairwise disjoint within a symbol. This is the property that stops 21-day forward
    returns sampled daily from inflating every t-statistic.
    """
    df = monthly_tagged(symbols=("A", "B"))
    m = monthly_panel(df)
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")

    for sym, g in m.groupby("symbol"):
        seen: set = set()
        for _, row in g.iterrows():
            days = set(d.loc[(d["symbol"] == sym) & (d["ym"] == row["ym"] + 1),
                             "date"])
            assert not (days & seen), f"{sym}: return windows overlap"
            seen |= days


def test_a_missing_month_breaks_the_chain_rather_than_bridging_it():
    """
    If a symbol has no data in month M, month M+1 must not be paired with month M-1.
    A period-shift join gets this right; a positional shift would not.
    """
    df = monthly_tagged(n_months=6)
    df = df[df["date"].dt.month != 3]                       # delete March entirely
    m = monthly_panel(df)
    months = sorted(m["ym"].astype(str))
    assert "2015-02" not in months     # Feb has no March return to pair with
    assert "2015-03" not in months     # March itself is gone
    assert "2015-01" in months and "2015-04" in months


# ----------------------------------------------------------------------------------
# assumptions the implementation rests on
# ----------------------------------------------------------------------------------

def test_the_month_end_basis_depends_on_the_input_being_sorted_by_date():
    """
    DISCLOSURE TEST. `groupby(...).tail(1)` takes the last row in FRAME ORDER, not the
    latest date. test_curve.load() sorts by (symbol, date) and enrich() preserves that,
    so the pipeline is correct today. Fed an unsorted frame the function silently reads
    the basis off whatever row happens to be last.
    """
    df = monthly_tagged(n_months=4)
    sorted_panel = monthly_panel(df)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    shuffled_panel = monthly_panel(shuffled)

    a = sorted_panel.set_index(["symbol", "ym"])["date"].sort_index()
    b = shuffled_panel.set_index(["symbol", "ym"])["date"].sort_index()
    assert not a.equals(b), (
        "tail(1) has become order-independent; the sortedness precondition is no "
        "longer load-bearing and this test should be replaced by a stronger one"
    )


def test_days_with_a_missing_basis_are_dropped_from_the_return_as_well():
    """
    DISCLOSURE TEST. monthly_panel starts from `df.dropna(subset=['basis'])`, so a day
    whose basis is unusable also loses its RETURN. The monthly return is therefore
    compounded only over days that had a valid basis, not over the whole month.

    In the current data the basis is dropped when the two legs are out of order or the
    maturity gap is implausible, which are exactly roll-adjacent days.
    """
    rows = []
    for j, per in enumerate(pd.period_range("2015-01-01", periods=3, freq="M")):
        for i, d in enumerate(pd.bdate_range(per.start_time, per.end_time)):
            rows.append(dict(date=d, symbol="A", basis=j / 100.0, ret_0=0.01))
    df = pd.DataFrame(rows)

    full = monthly_panel(df)
    holed = df.copy()
    feb = holed["date"].dt.month == 2
    holed.loc[feb & (holed.groupby(holed["date"].dt.month).cumcount() < 5),
              "basis"] = np.nan
    holed_panel = monthly_panel(holed)

    r_full = full.loc[full["ym"].astype(str) == "2015-01", "mret"].iloc[0]
    r_hole = holed_panel.loc[holed_panel["ym"].astype(str) == "2015-01", "mret"].iloc[0]
    assert r_hole < r_full          # five days of the February return went missing


def test_a_roll_day_nan_return_is_treated_as_zero_price_change():
    """
    `ret_0.fillna(0.0)` inside the monthly compounding is the documented convention:
    a roll starts a new contract's series and contributes no price change.
    """
    rows = []
    for j, per in enumerate(pd.period_range("2015-01-01", periods=3, freq="M")):
        for i, d in enumerate(pd.bdate_range(per.start_time, per.end_time)):
            rows.append(dict(date=d, symbol="A", basis=j / 100.0,
                             ret_0=np.nan if i == 0 else 0.0))
    m = monthly_panel(pd.DataFrame(rows))
    assert (m["mret"].abs() < 1e-12).all()
