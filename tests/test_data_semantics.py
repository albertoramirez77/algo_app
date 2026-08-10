"""
Data semantics. Every failure in this project so far started here, so each vendor
assumption is checked against something authoritative rather than against memory.

The authority used, in order of preference:
  1. the installed databento SDK's own enums and constants  (databento 0.64.0)
  2. the raw batch CSV that Databento actually shipped, on disk under batch_*/
  3. CFTC's published record

Where a shipped .parquet violates an invariant, the test is marked xfail with the
measurement in the reason. Nothing here rewrites a data file.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

import fetch_curve
import fetch_prices_batch
import immediacy as m

pytestmark = pytest.mark.filterwarnings("ignore")


def _parquet(name):
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return pd.read_parquet(p)


def _a_batch_csv():
    hits = sorted(glob.glob(str(ROOT / "batch_*" / "*" / "*.statistics.csv.zst")))
    if not hits:
        pytest.skip("no batch statistics CSV on disk")
    return hits[0]


# ----------------------------------------------------------------------------------
# 1. constants, against the SDK
# ----------------------------------------------------------------------------------

def test_stat_type_constants_match_the_sdk_enum():
    """3, 6 and 9 are hardcoded in two fetchers. Check them, do not trust them."""
    from databento_dbn import StatType

    assert fetch_curve.STAT_SETTLEMENT == int(StatType.SETTLEMENT_PRICE) == 3
    assert fetch_curve.STAT_VOLUME == int(StatType.CLEARED_VOLUME) == 6
    assert fetch_curve.STAT_OI == int(StatType.OPEN_INTEREST) == 9


def test_the_volume_stat_is_cleared_volume_not_traded_volume():
    """
    DISCLOSURE. stat_type 6 is CLEARED_VOLUME — the exchange's official cleared figure
    published with the settlement, not intraday traded volume. It is what the ADV
    estimate in run_backtest is built from, and it is a next-session number.
    """
    from databento_dbn import StatType

    assert StatType(6).name == "CLEARED_VOLUME"


def test_the_fixed_point_scale_matches_the_sdk():
    """One price unit is 1e-9. Both fetchers hardcode 1e9; the SDK is the authority."""
    from databento_dbn import FIXED_PRICE_SCALE

    assert FIXED_PRICE_SCALE == 1_000_000_000
    assert fetch_curve.FIXED_POINT == float(FIXED_PRICE_SCALE)
    assert fetch_prices_batch.FIXED_POINT == float(FIXED_PRICE_SCALE)


def test_the_undefined_sentinels_are_what_the_validators_must_reject():
    """
    The sentinel for an undefined stat quantity is version-dependent: DBN v3 widened it
    to int64, but the CSVs on disk carry the int32 value. A validator has to reject
    both, which is why this is written down rather than inferred.
    """
    from databento_dbn import UNDEF_PRICE, UNDEF_TIMESTAMP

    assert UNDEF_PRICE == 2 ** 63 - 1
    assert UNDEF_TIMESTAMP == 2 ** 64 - 1
    assert fetch_curve.UNDEF_INT32 == 2 ** 31 - 1
    assert fetch_curve.UNDEF_INT64 == 2 ** 63 - 1


# ----------------------------------------------------------------------------------
# 2. what the shipped batch CSV actually contains
# ----------------------------------------------------------------------------------

def test_the_batch_csv_has_the_columns_both_fetchers_reach_for():
    df = pd.read_csv(_a_batch_csv(), nrows=5)
    for c in ("ts_recv", "ts_event", "ts_ref", "price", "quantity",
              "stat_type", "instrument_id", "symbol"):
        assert c in df.columns, f"batch CSV is missing {c}"


def test_settlement_records_always_carry_a_reference_timestamp():
    """
    stat_type 3 has ts_ref populated on every row. That is what makes ts_ref a safe
    session key for settlements.
    """
    from databento_dbn import UNDEF_TIMESTAMP

    df = pd.read_csv(_a_batch_csv(), usecols=["stat_type", "ts_ref"])
    s3 = df.loc[df["stat_type"] == fetch_curve.STAT_SETTLEMENT, "ts_ref"]
    assert len(s3) > 1000
    assert (s3 == UNDEF_TIMESTAMP).sum() == 0


def test_volume_and_open_interest_often_have_no_reference_timestamp():
    """
    THE REASON oi_0 IS 35% NaN IN px_curve.parquet. stat_type 6 and 9 leave ts_ref
    undefined on roughly 42% of rows. fetch_curve.shape_stats picks ONE timestamp
    column for the whole frame and prefers ts_ref, so those rows date to NaT, are
    dropped by the groupby, and come back as NaN after the left merge.
    """
    from databento_dbn import UNDEF_TIMESTAMP

    df = pd.read_csv(_a_batch_csv(), usecols=["stat_type", "ts_ref"])
    for st in (fetch_curve.STAT_VOLUME, fetch_curve.STAT_OI):
        share = (df.loc[df["stat_type"] == st, "ts_ref"] == UNDEF_TIMESTAMP).mean()
        assert share > 0.30, f"stat_type {st}: {share:.1%} undefined"


def test_prices_are_fixed_point_integers_in_the_batch_csv():
    df = pd.read_csv(_a_batch_csv(), usecols=["stat_type", "price"])
    s3 = df.loc[df["stat_type"] == fetch_curve.STAT_SETTLEMENT, "price"]
    assert s3.median() > 1e6, "settlement prices are not fixed point"
    assert 0.01 < s3.median() / 1e9 < 1e5, "1e-9 scaling gives an implausible level"


def test_normalize_price_detects_rather_than_assumes():
    """Re-running with pretty_px=True must not double-scale."""
    raw = pd.Series([17_420_000_000.0] * 10)
    pretty = pd.Series([17.42] * 10)
    assert fetch_prices_batch.normalize_price(raw).median() == pytest.approx(17.42)
    assert fetch_prices_batch.normalize_price(pretty).median() == pytest.approx(17.42)


def test_normalize_price_has_a_blind_spot_between_the_two_scales():
    """
    DISCLOSURE. The detector is a median > 1e6 test. A fixed-point price whose real
    level is below 0.001 has a median under 1e6 and is left unscaled. No product in
    this universe trades there — ZL is the lowest at ~0.05 — so the heuristic holds,
    but it is a heuristic and not a check of the encoding.
    """
    tiny_fixed_point = pd.Series([500_000.0] * 10)          # real level 0.0005
    assert fetch_prices_batch.normalize_price(tiny_fixed_point).median() == 500_000.0


# ----------------------------------------------------------------------------------
# 3. the schema the analysis expects
# ----------------------------------------------------------------------------------

PX_COLUMNS = ["date", "symbol", "contract", "settle", "volume", "open_interest", "expiry"]


def test_px_parquet_has_the_columns_load_prices_documents():
    px = _parquet("px.parquet")
    for c in PX_COLUMNS:
        assert c in px.columns


def test_px_curve_parquet_has_the_columns_test_curve_expects():
    cur = _parquet("px_curve.parquet")
    for c in ("date", "symbol", "contract_0", "contract_1", "settle_0", "settle_1",
              "expiry_0"):
        assert c in cur.columns


def test_cot_parquet_has_the_columns_load_cot_documents():
    cot = _parquet("cot.parquet")
    for c in ("report_date", "symbol", "prod_long", "prod_short", "open_interest"):
        assert c in cot.columns


def test_cot_open_interest_is_strictly_positive():
    """
    The guard that keeps the zero-open-interest cross-section wipeout latent rather
    than live. fetch_cot.build filters `open_interest > 0`; this asserts it held.
    """
    cot = _parquet("cot.parquet")
    assert (cot["open_interest"] > 0).all()


def test_cot_report_dates_are_overwhelmingly_tuesdays():
    cot = _parquet("cot.parquet")
    assert (cot["report_date"].dt.dayofweek == 1).mean() > 0.90


def test_cot_hedging_pressure_is_positive_on_average():
    """Producers are structurally net short; a negative mean means the legs are swapped."""
    cot = _parquet("cot.parquet")
    hp = ((cot["prod_short"] - cot["prod_long"]) / cot["open_interest"]).mean()
    assert hp > 0


def test_no_settlement_carries_an_undefined_sentinel():
    for name, cols in (("px.parquet", ["settle"]),
                       ("px_curve.parquet", ["settle_0", "settle_1"])):
        d = _parquet(name)
        for c in cols:
            assert (d[c] == 2 ** 63 - 1).sum() == 0
            assert (d[c] == 2 ** 31 - 1).sum() == 0
            assert (d[c] > 0).all(), f"{name}:{c} has a non-positive settlement"


# ----------------------------------------------------------------------------------
# 4. calendar integrity — the invariant px.parquet violates
# ----------------------------------------------------------------------------------

def test_px_curve_has_no_weekend_sessions():
    """px_curve.parquet is built from ts_ref and is clean. This is the control."""
    cur = _parquet("px_curve.parquet")
    assert (pd.to_datetime(cur["date"]).dt.dayofweek >= 5).sum() == 0


def test_px_curve_reports_about_252_sessions_a_year():
    """
    Count DISTINCT sessions, not rows: px_curve carries duplicate (date, symbol) pairs
    on roll dates, which test_curve.load resolves separately.
    """
    cur = _parquet("px_curve.parquet")
    d = pd.to_datetime(cur["date"])
    per = cur.assign(d=d).groupby("symbol")["d"].agg(
        n="nunique", first="min", last="max")
    rate = per["n"] / ((per["last"] - per["first"]).dt.days / 365.25)
    assert rate.between(235, 265).all(), rate.round(0).to_dict()


@pytest.mark.xfail(reason=(
    "px.parquet is dated from ts_recv (wall-clock receipt) instead of ts_ref (the "
    "session the statistic refers to). fetch_prices_batch.download picks ts_recv "
    "first; fetch_curve.shape_stats picks ts_ref first. Measured on the shipped file: "
    "10,658 of 62,939 rows (16.9%) fall on a Sunday, build_front_series reports 305 "
    "sessions a year instead of 252, and 95.3% of COT trade dates land on a Sunday. "
    "stat_type 3 has ts_ref populated on 100% of rows, so the fix is one column name. "
    "Refetching moves every hypothesis-1 number, so it is the author's call."
), strict=False)
def test_px_parquet_has_no_weekend_sessions():
    px = _parquet("px.parquet")
    assert (pd.to_datetime(px["date"]).dt.dayofweek >= 5).sum() == 0


@pytest.mark.xfail(reason="same root cause: ts_recv dating inflates the session count",
                   strict=False)
def test_px_parquet_reports_about_252_sessions_a_year():
    px = _parquet("px.parquet")
    d = pd.to_datetime(px["date"])
    span = (d.max() - d.min()).days / 365.25
    assert 235 <= d.nunique() / span <= 265


@pytest.mark.xfail(reason=(
    "consequence of the ts_recv dating above: first_tradeable_date picks the first "
    "entry in the session index after the Friday release, and that index contains "
    "Sundays, so 824 of 865 report weeks execute on a Sunday."
), strict=False)
def test_every_cot_trade_date_falls_on_a_weekday():
    px = _parquet("px.parquet")
    cot = _parquet("cot.parquet")
    px["date"] = pd.to_datetime(px["date"])
    px["expiry"] = pd.to_datetime(px["expiry"])
    front = m.build_front_series(px)
    dates = pd.DatetimeIndex(sorted(front["date"].unique()))
    trade = [m.first_tradeable_date(pd.Timestamp(r), dates)
             for r in sorted(cot["report_date"].unique())]
    trade = [t for t in trade if t is not None]
    assert all(t.dayofweek < 5 for t in trade)


# ----------------------------------------------------------------------------------
# 5. the continuous-contract expiry placeholder
# ----------------------------------------------------------------------------------

def test_the_continuous_mode_expiry_is_a_placeholder_not_a_real_expiration():
    """
    DISCLOSURE. With MODE='continuous', fetch_prices_batch sets expiry = date + 400
    days for every row. immediacy.load_prices documents its input as being "per
    *individual contract* (not continuous)", and build_front_series selects "the
    nearest expiry that is still more than ROLL_OFFSET_BDAYS out".

    Against px.parquet both statements are inert: there is exactly one row per
    (date, symbol) and its expiry is always 400 days away, so the roll selector never
    chooses anything and the roll is whatever .n.0 resolved to server-side. That is an
    open-interest-ranked roll, which is the same ordering defect already identified on
    the curve data.
    """
    px = _parquet("px.parquet")
    gap = (pd.to_datetime(px["expiry"]) - pd.to_datetime(px["date"])).dt.days
    assert (gap == 400).mean() > 0.99
    assert not px.duplicated(["date", "symbol"]).any()


# ----------------------------------------------------------------------------------
# 6. the duplicate tie-break in test_curve hygiene
# ----------------------------------------------------------------------------------

@pytest.mark.xfail(reason=(
    "test_curve.load resolves duplicate (date, symbol) rows with "
    "sort_values([...,'oi_0']).drop_duplicates(keep='last'), commented as keeping "
    "'the higher-OI leg'. pandas sorts NaN LAST, and oi_0 is NaN on 35% of rows "
    "because stat_type 9 leaves ts_ref undefined. In the 356 duplicate groups where "
    "one leg has open interest and the other does not, keep='last' therefore selects "
    "the leg WITHOUT open interest -- the opposite of the stated rule. These are roll "
    "dates, which the delivery test depends on. Fixing it moves the delivery-profile "
    "numbers, so it is the author's call."
), strict=False)
def test_the_duplicate_tie_break_keeps_the_higher_open_interest_leg():
    cur = _parquet("px_curve.parquet")
    dup = cur[cur.duplicated(["date", "symbol"], keep=False)]
    if dup.empty:
        pytest.skip("no duplicates in this file")
    kept = (dup.sort_values(["symbol", "date", "oi_0"])
               .drop_duplicates(["date", "symbol"], keep="last"))
    mixed = dup.groupby(["date", "symbol"])["oi_0"].apply(
        lambda s: s.isna().any() and s.notna().any())
    assert int(mixed.sum()) == 0 or kept["oi_0"].isna().sum() == 0


def test_the_curve_file_still_carries_the_open_interest_ranked_ordering_defect():
    """
    Characterisation of the state the pending --roll c refetch is meant to fix: the
    shipped px_curve.parquet has no expiry_1 at all, so the basis is a raw log ratio
    rather than an annualised rate, and 1,129 rows resolve both legs to the same
    instrument.
    """
    cur = _parquet("px_curve.parquet")
    assert "expiry_1" not in cur.columns
    assert (cur["contract_0"] == cur["contract_1"]).sum() > 0
