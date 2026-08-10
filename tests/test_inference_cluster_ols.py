"""
cluster_ols is the estimator behind the headline number in the basis control. It is
hand-rolled and had never been checked against anything. These tests pin down what it
computes, what convention it uses, and where it stops being trustworthy.

Nothing here asserts that any real-data result should be significant. The inputs are
synthetic and the right answer is known before the code runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference as ref
from conftest import clustered_sample
from test_curve import cluster_ols, demean, monthly_panel  # noqa: F401  (demean used below)


# ----------------------------------------------------------------------------------
# 1. the point estimate is ordinary least squares, exactly
# ----------------------------------------------------------------------------------

def test_point_estimate_matches_ols_exactly():
    """The slope cluster_ols reports is the plain OLS slope, to floating-point noise."""
    y, x, cl = clustered_sample(seed=1)
    b, t, n = cluster_ols(y, x, cl)
    beta_ref, _, _ = ref.ols(y, x)
    assert n == len(y)
    assert b == pytest.approx(beta_ref[1], rel=1e-12, abs=1e-12)


def test_point_estimate_is_unaffected_by_the_cluster_labels():
    """Clustering changes the standard error and nothing else."""
    y, x, cl = clustered_sample(seed=2)
    b_true, _, _ = cluster_ols(y, x, cl)
    b_one, _, _ = cluster_ols(y, x, np.array(["all"] * len(y)))
    b_single, _, _ = cluster_ols(y, x, np.arange(len(y)).astype(str))
    assert b_true == pytest.approx(b_one, rel=1e-12)
    assert b_true == pytest.approx(b_single, rel=1e-12)


# ----------------------------------------------------------------------------------
# 2. the standard error is the uncorrected cluster sandwich
# ----------------------------------------------------------------------------------

def _se_from(b, t):
    """cluster_ols returns (slope, slope/se, n); recover the se it used."""
    return abs(b / t)


def test_clustered_se_matches_hand_computed_sandwich():
    """The reported t implies exactly the sandwich standard error computed by hand."""
    y, x, cl = clustered_sample(seed=3)
    b, t, _ = cluster_ols(y, x, cl)
    V = ref.var_cluster(y, x, cl, correction="uncorrected")
    assert _se_from(b, t) == pytest.approx(np.sqrt(V[1, 1]), rel=1e-10)


def test_singleton_clusters_reduce_to_white_hc0():
    """
    With one observation per cluster the sandwich collapses to White's HC0. That is a
    closed-form identity, so it checks the meat is being accumulated per cluster and
    not per observation by accident.
    """
    y, x, _ = clustered_sample(seed=4)
    cl = np.arange(len(y)).astype(str)
    b, t, _ = cluster_ols(y, x, cl)
    V = ref.var_hc0(y, x)
    assert _se_from(b, t) == pytest.approx(np.sqrt(V[1, 1]), rel=1e-10)


def test_no_finite_sample_correction_is_applied():
    """
    DISCLOSURE TEST, not a bug report. cluster_ols omits the [G/(G-1)][(n-1)/(n-k)]
    factor that Stata's vce(cluster) applies by default, so its t-statistics are
    slightly LARGER in absolute value than Stata would report on the same data.

    With 40 clusters the gap is about 1.3%. This test fails the moment somebody adds
    or removes the correction, which is the point: the convention becomes a decision
    on the record rather than an accident.
    """
    y, x, cl = clustered_sample(n_clusters=40, per_cluster=10, seed=5)
    b, t, _ = cluster_ols(y, x, cl)
    se_used = _se_from(b, t)

    se_uncorrected = np.sqrt(ref.var_cluster(y, x, cl, "uncorrected")[1, 1])
    se_stata = np.sqrt(ref.var_cluster(y, x, cl, "stata")[1, 1])

    assert se_used == pytest.approx(se_uncorrected, rel=1e-10)
    assert se_used < se_stata

    n_obs, n_par, n_grp = 400, 2, 40
    expected = np.sqrt((n_grp / (n_grp - 1)) * ((n_obs - 1) / (n_obs - n_par)))
    assert se_stata / se_uncorrected == pytest.approx(expected, rel=1e-10)
    assert expected == pytest.approx(1.0140, abs=1e-4)


def test_clustered_se_exceeds_classical_when_the_regressor_is_clustered_too():
    """
    Clustering only bites when the REGRESSOR is correlated within the cluster. A
    common shock in the error alone leaves the sandwich almost unchanged, because an
    idiosyncratic x is orthogonal to it.

    That is the case the basis control is actually in: the basis moves together across
    commodities within a month. So this is the configuration that has to inflate.
    """
    y, x, cl = clustered_sample(cluster_sd=2.0, noise_sd=0.2, x_cluster_sd=2.0, seed=6)
    b, t, _ = cluster_ols(y, x, cl)
    se_cluster = _se_from(b, t)
    se_classical = np.sqrt(ref.var_classical(y, x)[1, 1])
    assert se_cluster > 1.5 * se_classical


def test_a_common_error_shock_alone_barely_moves_the_clustered_se():
    """
    The companion fact, recorded so the previous test is not read as saying more than
    it does: with x idiosyncratic, clustering the error changes very little.
    """
    y, x, cl = clustered_sample(cluster_sd=2.0, noise_sd=0.2, x_cluster_sd=0.0, seed=6)
    b, t, _ = cluster_ols(y, x, cl)
    se_cluster = _se_from(b, t)
    se_classical = np.sqrt(ref.var_classical(y, x)[1, 1])
    assert se_cluster == pytest.approx(se_classical, rel=0.25)


def test_clustered_se_approaches_classical_without_common_shocks():
    """The mirror image: no common shock anywhere, and the two converge."""
    y, x, cl = clustered_sample(n_clusters=300, per_cluster=10,
                                cluster_sd=0.0, noise_sd=1.0, seed=7)
    b, t, _ = cluster_ols(y, x, cl)
    se_cluster = _se_from(b, t)
    se_classical = np.sqrt(ref.var_classical(y, x)[1, 1])
    assert se_cluster == pytest.approx(se_classical, rel=0.10)


# ----------------------------------------------------------------------------------
# 3. recovery — both directions
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("true_slope", [-3.0, -0.25, 0.0, 0.25, 4.77])
def test_recovers_an_embedded_slope(true_slope):
    """Embed a known slope, get it back. 4.77 is KRT's published number, on purpose."""
    y, x, cl = clustered_sample(n_clusters=200, per_cluster=10, slope=true_slope,
                                cluster_sd=0.2, noise_sd=0.5, seed=8)
    b, t, n = cluster_ols(y, x, cl)
    assert b == pytest.approx(true_slope, abs=0.05)
    assert n == 2000


def test_zero_slope_returns_zero_and_an_insignificant_t():
    """Embed nothing, and the estimator must report nothing."""
    y, x, cl = clustered_sample(n_clusters=200, per_cluster=10, slope=0.0,
                                cluster_sd=0.2, noise_sd=0.5, seed=9)
    b, t, _ = cluster_ols(y, x, cl)
    assert b == pytest.approx(0.0, abs=0.05)
    assert abs(t) < 2.0


def test_a_real_slope_is_detected_and_a_shuffled_one_is_not():
    """
    The estimator separates signal from a permutation of itself. This is the property
    the whole placebo argument rests on, stated at the level of the estimator.
    """
    y, x, cl = clustered_sample(n_clusters=200, per_cluster=10, slope=1.0,
                                cluster_sd=0.2, noise_sd=0.5, seed=10)
    _, t_real, _ = cluster_ols(y, x, cl)
    rng = np.random.default_rng(11)
    ts = [cluster_ols(y, rng.permutation(x), cl)[1] for _ in range(20)]
    assert t_real > 5.0
    assert np.mean(np.abs(ts)) < 2.0


# ----------------------------------------------------------------------------------
# 4. degradation and guards
# ----------------------------------------------------------------------------------

def test_below_fifty_usable_rows_returns_nan_rather_than_a_number():
    """The 50-row floor returns NaN, so a thin bucket cannot produce a headline."""
    y, x, cl = clustered_sample(n_clusters=7, per_cluster=7, seed=12)
    b, t, n = cluster_ols(y, x, cl)
    assert n == 49
    assert np.isnan(b) and np.isnan(t)


def test_the_fifty_row_floor_counts_usable_rows_not_supplied_rows():
    """NaNs are dropped BEFORE the floor is applied, which is the safe direction."""
    y, x, cl = clustered_sample(n_clusters=11, per_cluster=5, seed=13)   # 55 rows
    y = y.copy()
    y[:10] = np.nan                                                      # 45 usable
    b, t, n = cluster_ols(y, x, cl)
    assert n == 45
    assert np.isnan(b)


def test_nan_rows_are_dropped_pairwise_and_the_cluster_labels_follow():
    """
    A misaligned mask here would silently pair each residual with another row's
    cluster. Dropping the same rows from y, x and cl by hand must give the identical
    answer.
    """
    y, x, cl = clustered_sample(seed=14)
    y = y.copy()
    y[::7] = np.nan
    b, t, n = cluster_ols(y, x, cl)

    keep = np.isfinite(y) & np.isfinite(x)
    b2, t2, n2 = cluster_ols(y[keep], x[keep], cl[keep])
    assert (b, t, n) == pytest.approx((b2, t2, n2), rel=1e-12)


def test_a_single_cluster_gives_a_rank_one_meat_and_no_usable_t():
    """
    One cluster means the sandwich has rank one, the slope variance collapses, and the
    t-statistic is not interpretable. It must not come back as a large finite number.
    """
    y, x, _ = clustered_sample(seed=15)
    b, t, n = cluster_ols(y, x, np.array(["only"] * len(y)))
    assert np.isfinite(b)
    assert np.isnan(t) or abs(t) > 1e6


def test_a_constant_regressor_produces_no_slope_information():
    """
    The vestigial loop in delivery_profile regresses on a constant 1e-12. A constant
    regressor is collinear with the intercept, so pinv splits the fit arbitrarily and
    the 'slope' it returns carries no information about the data.

    This test documents that the value is meaningless; delivery_profile discards it
    and mean_t() does the real work.
    """
    y, x, cl = clustered_sample(seed=16)
    const = np.full(len(y), 1e-12)
    b, t, n = cluster_ols(y, const, cl)
    assert n == len(y)
    assert abs(b) < 1e-9          # a "slope" at the scale of the regressor, not the data
    # The same y against a different constant gives a different answer, which is the
    # tell that nothing here is a property of y.
    b2, _, _ = cluster_ols(y, np.full(len(y), 1e-6), cl)
    assert not np.isclose(b, b2, rtol=1e-3)


def test_a_near_perfect_fit_reports_an_absurd_t_rather_than_nan():
    """
    CHARACTERISATION OF A KNOWN DEFECT — deliberately left unfixed.

    cluster_ols guards with `se > 0`. On an exactly-collinear fit the residuals are
    not exactly zero, they are floating-point dust around 1e-14, so `se > 0` is True
    and the function returns a t-statistic of order 1e14 instead of NaN.

    No bucket in the current pipeline is collinear, so nothing on the record depends
    on this. It is pinned here so the hazard is visible and so a future guard change
    is caught. The fix would be a relative test (se > tol * |beta|), which is a
    behaviour change and is the author's call, not this test's.
    """
    x = np.arange(60.0)
    y = 3.0 * x + 1.0                       # exact linear relation
    b, t, n = cluster_ols(y, x, (x // 10).astype(str))
    assert b == pytest.approx(3.0)
    assert np.isfinite(t) and abs(t) > 1e6      # <- the defect, recorded not repaired


def test_an_exactly_zero_standard_error_does_report_nan():
    """The guard that does exist works when the residuals are bit-for-bit zero."""
    n = 60
    x = np.zeros(n)
    y = np.zeros(n)
    b, t, _ = cluster_ols(y, x, (np.arange(n) // 10).astype(str))
    assert np.isnan(t)


# ----------------------------------------------------------------------------------
# 5. the within transform used by the basis control
# ----------------------------------------------------------------------------------

def test_demean_then_ols_equals_least_squares_dummy_variables():
    """
    basis_control demeans by symbol and then regresses. That is supposed to be the
    fixed-effects estimator; here it is checked against the dummy-variable regression
    that defines it.
    """
    import pandas as pd

    rng = np.random.default_rng(17)
    n_sym, n_month = 8, 40
    rows = []
    for s in range(n_sym):
        alpha = rng.normal(0, 3.0)                     # symbol fixed effect
        for mth in range(n_month):
            bx = rng.normal()
            rows.append(dict(symbol=f"S{s}", ym=mth, basis=bx,
                             mret=alpha + 0.7 * bx + rng.normal(0, 0.4)))
    m = pd.DataFrame(rows)

    w = demean(m, ["basis", "mret"])
    b, _, _ = cluster_ols(w["mret"].to_numpy(), w["basis"].to_numpy(),
                          w["ym"].astype(str).to_numpy())
    b_ref = ref.lsdv_slope(m["mret"], m["basis"], m["symbol"])
    assert b == pytest.approx(b_ref, rel=1e-9)


def test_demean_does_not_mutate_its_input():
    """demean returns a copy; the caller's frame must be untouched."""
    import pandas as pd

    m = pd.DataFrame(dict(symbol=["A"] * 5 + ["B"] * 5,
                          basis=np.arange(10.0), mret=np.arange(10.0)))
    before = m.copy(deep=True)
    demean(m, ["basis", "mret"])
    pd.testing.assert_frame_equal(m, before)


def test_within_transform_leaves_degrees_of_freedom_uncounted():
    """
    DISCLOSURE TEST. Demeaning by symbol consumes one degree of freedom per symbol,
    but cluster_ols is handed the demeaned data and counts only two parameters. The
    standard error is therefore very slightly optimistic relative to a dummy-variable
    regression that counts them.

    With clustering by date and no finite-sample correction the effect is nil in this
    implementation — the uncorrected sandwich has no (n-k) term at all — so this test
    records that the omission costs nothing HERE, and will fail if a correction is
    added without also counting the absorbed effects.
    """
    import pandas as pd

    rng = np.random.default_rng(18)
    rows = []
    for s in range(6):
        alpha = rng.normal(0, 2.0)
        for mth in range(50):
            bx = rng.normal()
            rows.append(dict(symbol=f"S{s}", ym=mth, basis=bx,
                             mret=alpha + 0.5 * bx + rng.normal(0, 0.5)))
    m = pd.DataFrame(rows)
    w = demean(m, ["basis", "mret"])
    b, t, _ = cluster_ols(w["mret"].to_numpy(), w["basis"].to_numpy(),
                          w["ym"].astype(str).to_numpy())
    V = ref.var_cluster(w["mret"].to_numpy(), w["basis"].to_numpy(),
                        w["ym"].astype(str).to_numpy(), "uncorrected")
    assert abs(b / t) == pytest.approx(np.sqrt(V[1, 1]), rel=1e-10)
