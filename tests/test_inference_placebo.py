"""
The placebos are load-bearing: they are what stands in for a fresh holdout now that
the out-of-sample window is spent. A placebo that quietly fails to shuffle, or that
shuffles across groups instead of within them, would make a dead signal look alive.

Three permutations exist in the project and all three use the same idiom:

    frame[col] = frame.groupby(KEY)[col].transform(lambda s: rng.permutation(s.values))

test_curve.basis_control   shuffles `basis` in time, within symbol
test_curve.delivery_profile shuffles `dte`  in time, within symbol
attribute.variant           shuffles `z`    across names, within report week

The properties asserted here are: the multiset within each group survives, nothing
crosses a group boundary, and the assignment actually changes.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

import attribute
import immediacy as m


def permute_within(frame: pd.DataFrame, key: str, col: str, seed: int) -> pd.DataFrame:
    """The exact idiom used in the three call sites, isolated so it can be tested."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out[col] = out.groupby(key)[col].transform(lambda s: rng.permutation(s.to_numpy()))
    return out


# ----------------------------------------------------------------------------------
# the permutation idiom itself
# ----------------------------------------------------------------------------------

def test_the_multiset_within_each_group_is_preserved():
    """A permutation may reorder values; it may not create, destroy or alter them."""
    df = pd.DataFrame(dict(symbol=["A"] * 20 + ["B"] * 20,
                           basis=np.arange(40.0)))
    out = permute_within(df, "symbol", "basis", seed=0)
    for s in ("A", "B"):
        assert (Counter(df.loc[df.symbol == s, "basis"])
                == Counter(out.loc[out.symbol == s, "basis"]))


def test_no_value_crosses_a_group_boundary():
    """
    Values are tagged by group so a leak is detectable: every value seen under symbol
    A after the shuffle must have started under symbol A.
    """
    df = pd.DataFrame(dict(symbol=["A"] * 20 + ["B"] * 20,
                           basis=list(np.arange(20.0)) + list(np.arange(100.0, 120.0))))
    out = permute_within(df, "symbol", "basis", seed=1)
    assert out.loc[out.symbol == "A", "basis"].max() < 100
    assert out.loc[out.symbol == "B", "basis"].min() >= 100


def test_the_assignment_actually_changes():
    """A permutation that happened to be the identity would be a silent no-op."""
    df = pd.DataFrame(dict(symbol=["A"] * 30 + ["B"] * 30, basis=np.arange(60.0)))
    out = permute_within(df, "symbol", "basis", seed=2)
    assert (df["basis"].to_numpy() != out["basis"].to_numpy()).sum() > 30


def test_interleaved_groups_are_realigned_to_the_right_rows():
    """
    The lambda hands back a bare numpy array and pandas puts it back positionally. If
    the group's rows are NOT contiguous in the frame, that realignment is the step
    that could scramble values across symbols. The real frames are sorted by symbol,
    so this is the case the pipeline never hits and therefore never checks.
    """
    n = 60
    df = pd.DataFrame(dict(symbol=["A", "B"] * (n // 2),          # fully interleaved
                           basis=[float(i) for i in range(n)]))
    df.loc[df.symbol == "B", "basis"] += 1000.0
    out = permute_within(df, "symbol", "basis", seed=3)
    assert out.loc[out.symbol == "A", "basis"].max() < 1000
    assert out.loc[out.symbol == "B", "basis"].min() >= 1000


def test_different_seeds_give_different_shuffles():
    """Five placebo draws must be five draws, not the same draw five times."""
    df = pd.DataFrame(dict(symbol=["A"] * 50, basis=np.arange(50.0)))
    draws = [permute_within(df, "symbol", "basis", seed=s)["basis"].to_numpy()
             for s in range(5)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])


def test_a_single_member_group_cannot_be_shuffled():
    """
    A group of one permutes to itself. Any symbol with a single observation therefore
    contributes an unshuffled row to every placebo draw, which biases the placebo
    toward the real signal. Harmless at 190+ observations per symbol; recorded because
    it is the mechanism by which a placebo becomes conservative without saying so.
    """
    df = pd.DataFrame(dict(symbol=["A"] * 3 + ["SOLO"], basis=[1.0, 2.0, 3.0, 9.0]))
    out = permute_within(df, "symbol", "basis", seed=4)
    assert out.loc[out.symbol == "SOLO", "basis"].iloc[0] == 9.0


# ----------------------------------------------------------------------------------
# attribute.variant — the cross-sectional placebo behind the "signal is not the
# source" conclusion
# ----------------------------------------------------------------------------------

def _sig_frame(n_weeks=30, symbols=("MCL", "QG", "MGC", "SIL", "ZC", "ZW")):
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2016-01-05", periods=n_weeks, freq="W-TUE")
    rows = []
    for t in weeks:
        for s in symbols:
            rows.append(dict(report_date=t, symbol=s,
                             z=rng.normal(), e=float(rng.integers(0, 2)),
                             g=float(rng.choice([1.0, 2.0])), s=0.0))
    return pd.DataFrame(rows)


def test_variant_preserves_the_within_week_multiset_of_z():
    """The cross-sectional ranking is destroyed; the distribution is not."""
    sig = _sig_frame()
    out = attribute.variant(sig, m.PARAMS, permute_seed=0)
    for t, g in sig.groupby("report_date"):
        got = out.loc[out["report_date"] == t, "z"]
        assert Counter(np.round(g["z"], 12)) == Counter(np.round(got, 12))


KEYS = ["report_date", "symbol"]


def _aligned(a: pd.DataFrame, b: pd.DataFrame):
    """
    variant() returns a frame re-sorted by (symbol, report_date) because
    rescale_forecast sorts before the rolling mean. Compare on the keys, never on
    position.
    """
    return (a.sort_values(KEYS).reset_index(drop=True),
            b.sort_values(KEYS).reset_index(drop=True))


def test_variant_leaves_the_eligibility_filter_and_stress_gate_untouched():
    """
    The whole point of the placebo is that ONLY the ranking changes. If `e` or `g`
    moved too, a null result would no longer isolate the signal's contribution.
    """
    sig = _sig_frame()
    a, b = _aligned(sig, attribute.variant(sig, m.PARAMS, permute_seed=0))
    pd.testing.assert_series_equal(a["e"], b["e"])
    pd.testing.assert_series_equal(a["g"], b["g"])


def test_variant_actually_reassigns_z_across_names():
    """A placebo that returned the input unchanged would look like a real signal."""
    sig = _sig_frame()
    out = attribute.variant(sig, m.PARAMS, permute_seed=0)
    a = sig.sort_values(["report_date", "symbol"])["z"].to_numpy()
    b = out.sort_values(["report_date", "symbol"])["z"].to_numpy()
    assert (a != b).mean() > 0.5


def test_variant_drop_filter_sets_eligibility_to_one_for_everybody():
    """The ablation is what it says: the filter is removed, not merely relaxed."""
    sig = _sig_frame()
    a, b = _aligned(sig, attribute.variant(sig, m.PARAMS, drop_filter=True))
    expected = a["z"].fillna(0.0).to_numpy() * 1.0 * a["g"].to_numpy()
    assert np.allclose(b["s"].to_numpy(), expected)


def test_variant_drop_stress_sets_the_gate_to_one_for_everybody():
    sig = _sig_frame()
    a, b = _aligned(sig, attribute.variant(sig, m.PARAMS, drop_stress=True))
    expected = a["z"].fillna(0.0).to_numpy() * a["e"].fillna(0.0).to_numpy() * 1.0
    assert np.allclose(b["s"].to_numpy(), expected)


def test_variant_does_not_mutate_the_frame_it_was_given():
    """
    Five placebo seeds are run in a loop off one base frame. If variant() mutated its
    argument, draw 2 would be a shuffle of draw 1 and the placebo spread would
    collapse toward zero.
    """
    sig = _sig_frame()
    before = sig.copy(deep=True)
    for seed in range(3):
        attribute.variant(sig, m.PARAMS, permute_seed=seed)
    pd.testing.assert_frame_equal(sig, before)


# ----------------------------------------------------------------------------------
# the delivery placebo shuffles the AXIS, not the outcome
# ----------------------------------------------------------------------------------

def test_shuffling_days_to_expiry_keeps_the_residual_pool_intact():
    """
    delivery_profile shuffles `dte` and re-reads the SAME residuals. The near and far
    buckets therefore draw from one unchanged pool, which is the correct null: it
    destroys the association with the calendar and nothing else.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(dict(symbol=["A"] * 200,
                           dte=rng.integers(0, 91, 200).astype(float),
                           resid=rng.normal(size=200)))
    out = permute_within(df, "symbol", "dte", seed=5)
    assert Counter(np.round(df["resid"], 12)) == Counter(np.round(out["resid"], 12))
    assert Counter(df["dte"]) == Counter(out["dte"])
    assert (df["dte"].to_numpy() != out["dte"].to_numpy()).sum() > 50
