"""
Shared fixtures. Nothing here touches real data: every test builds its own inputs so
the answer is known before the code runs.

The project is a flat set of modules at the repository root, so the root has to be on
sys.path before `import immediacy` will work from inside tests/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------

def make_price_panel(symbols, dates, *, expiries=None, settle=None,
                     volume=120_000.0, open_interest=400_000.0) -> pd.DataFrame:
    """
    A price frame in the shape immediacy.load_prices documents:
        date, symbol, contract, settle, volume, open_interest, expiry

    `expiries` maps symbol -> list of (contract_name, expiry, [dates it trades on]).
    When omitted, each symbol gets a single contract expiring well after the sample,
    which is the simplest case that still exercises the roll selector.
    """
    rows = []
    for s in symbols:
        if expiries is None:
            spec = [(f"{s}_C1", dates[-1] + pd.Timedelta(days=365), list(dates))]
        else:
            spec = expiries[s]
        for contract, expiry, cdates in spec:
            for d in cdates:
                px = settle(s, contract, d) if callable(settle) else (
                    100.0 if settle is None else settle)
                rows.append(dict(date=d, symbol=s, contract=contract,
                                 settle=float(px), volume=float(volume),
                                 open_interest=float(open_interest),
                                 expiry=pd.Timestamp(expiry)))
    return pd.DataFrame(rows)


def make_cot(symbols, report_dates, *, rng=None) -> pd.DataFrame:
    """COT frame in the shape immediacy.load_cot documents."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for s in symbols:
        for t in report_dates:
            rows.append(dict(report_date=pd.Timestamp(t), symbol=s,
                             prod_long=100_000.0 + rng.normal(0, 8_000),
                             prod_short=140_000.0 + rng.normal(0, 8_000),
                             open_interest=200_000.0 + rng.normal(0, 5_000)))
    return pd.DataFrame(rows)


def clustered_sample(n_clusters=40, per_cluster=10, slope=2.0, intercept=0.5,
                     cluster_sd=1.0, noise_sd=0.3, x_cluster_sd=0.0, seed=0):
    """
    y = intercept + slope*x + u_g + e, with u_g common to every member of cluster g,
    and x = v_g + w, where v_g is likewise common to the cluster.

    A common shock in the ERROR alone does not inflate a clustered standard error: if
    x is idiosyncratic within the cluster it is orthogonal to u_g and the sandwich
    barely moves. The clustered error separates from the classical one only when the
    REGRESSOR is also correlated within the cluster. `x_cluster_sd` controls that, and
    it is the case that matters here — the basis moves together across commodities
    within a month, which is exactly why the basis control clusters on the month.
    """
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_clusters), per_cluster)
    v = rng.normal(0, x_cluster_sd, n_clusters)[g] if x_cluster_sd else 0.0
    x = v + rng.normal(size=len(g))
    u = rng.normal(0, cluster_sd, n_clusters)[g]
    e = rng.normal(0, noise_sd, len(g))
    y = intercept + slope * x + u + e
    return y, x, g.astype(str)


@pytest.fixture
def bdays():
    """A clean business-day index, long enough for the 50-observation floor."""
    return pd.bdate_range("2015-01-05", periods=400)
