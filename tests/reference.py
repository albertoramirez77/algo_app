"""
Independent reference implementations of the estimators the project rolls by hand.

These are deliberately written a different way from the code under test: `lstsq` and
`inv` rather than `pinv`, per-row Python accumulation rather than boolean masking. If
both arrive at the same number, the agreement means something. If they were the same
code twice, it would not.

Formulas, so they can be checked against a textbook rather than against this file:

    OLS                 b = (X'X)^-1 X'y
    classical Var(b)    s2 (X'X)^-1,           s2 = e'e / (n - k)
    HC0 (White)         (X'X)^-1 [ X' diag(e^2) X ] (X'X)^-1
    CRVE (cluster)      (X'X)^-1 [ sum_g (X_g' e_g)(X_g' e_g)' ] (X'X)^-1

The CRVE above carries NO finite-sample correction. Stata's `vce(cluster)` multiplies
it by  [G/(G-1)] * [(n-1)/(n-k)]. Both conventions appear in the literature; which one
is in use has to be stated, not assumed. `uncorrected` vs `stata` selects it here.
"""

from __future__ import annotations

import numpy as np


def design(x: np.ndarray) -> np.ndarray:
    """Intercept plus one regressor."""
    return np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])


def ols(y: np.ndarray, x: np.ndarray):
    """Point estimate by least squares. Returns (beta, residuals, X)."""
    X = design(x)
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    e = np.asarray(y, dtype=float) - X @ beta
    return beta, e, X


def var_classical(y, x):
    """s2 (X'X)^-1 — the homoskedastic textbook covariance."""
    beta, e, X = ols(y, x)
    n, k = X.shape
    s2 = float(e @ e) / (n - k)
    return s2 * np.linalg.inv(X.T @ X)


def var_hc0(y, x):
    """White's heteroskedasticity-consistent covariance, written as the sandwich."""
    beta, e, X = ols(y, x)
    bread = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(e ** 2) @ X
    return bread @ meat @ bread


def var_cluster(y, x, cl, correction: str = "uncorrected"):
    """
    Cluster-robust covariance, accumulated one row at a time so that no vectorised
    step can hide an indexing mistake.
    """
    beta, e, X = ols(y, x)
    n, k = X.shape
    cl = list(cl)
    bread = np.linalg.inv(X.T @ X)

    groups: dict[object, list[int]] = {}
    for i, g in enumerate(cl):
        groups.setdefault(g, []).append(i)

    meat = np.zeros((k, k))
    for idx in groups.values():
        score = np.zeros(k)
        for i in idx:
            score = score + X[i] * e[i]
        meat = meat + np.outer(score, score)

    V = bread @ meat @ bread
    if correction == "stata":
        G = len(groups)
        V = V * (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    elif correction != "uncorrected":
        raise ValueError(f"unknown correction {correction!r}")
    return V


def lsdv_slope(y, x, unit):
    """
    Fixed-effects slope by least-squares dummy variables: regress y on x plus one
    dummy per unit. This is the definition the within/demeaning transform is supposed
    to reproduce, so it is the right thing to check `demean` against.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    units = sorted(set(unit))
    D = np.column_stack([np.asarray([1.0 if u == g else 0.0 for u in unit])
                         for g in units])
    X = np.column_stack([np.asarray(x), D])       # no separate intercept: dummies span it
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0])
