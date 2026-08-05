"""
fetch_prices.py — build px.parquet from Databento CME data.

THIS ONE COSTS MONEY. Read the whole docstring before running it.

You have $125 of free credit. The download you need is small, but it is easy to
accidentally request something enormous. There are three modes, and you should run them
in this order:

    1.  python fetch_prices.py --inspect --symbol ZC
        Pulls ONE day for ONE product. Costs pennies. Prints the statistics schema's
        stat_type values so you can confirm which code is settlement and which is open
        interest. Do this first.

    2.  python fetch_prices.py --estimate
        Prices the full request WITHOUT downloading anything. Free. If the number
        surprises you, stop and work out why before continuing.

    3.  python fetch_prices.py --run --out px.parquet
        The actual download.

A NOTE ON WHICH SCHEMA WE USE, because it differs from the specification document.

The spec says to use `ohlcv-1d` for the volatility estimate. Having built the engine, that
turns out to be unnecessary: volatility is computed from settlement-to-settlement returns,
and the statistics schema carries settlement price, open interest AND cleared volume. So we
pull `statistics` only, plus `definition` once for contract expiries. One schema, cheaper,
and it sidesteps the UTC-date boundary problem in `ohlcv-1d` entirely.

If asked about this in interview, that is the right answer: we do not use daily bars because
their close is the last electronic trade on a UTC day, not the venue's settlement, and
settlement is what the position is marked and rolled at.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

DATASET = "GLBX.MDP3"
START = "2010-06-06"          # Databento's CME history begins here
END = None                    # None = today

# COT-parent product roots. We read the signal on the parent and trade the micro, so the
# PRICE we need is the one we actually trade. Both are listed; map is in immediacy.py.
PRODUCTS = {
    # traded symbol -> Databento product root
    "MCL": "MCL", "QG": "QG", "MGC": "MGC", "SIL": "SIL", "MHG": "MHG",
    "ZC": "ZC", "ZW": "ZW", "KE": "KE", "ZS": "ZS", "ZM": "ZM", "ZL": "ZL",
    "LE": "LE", "HE": "HE",
}

# Databento statistics stat_type codes. VERIFY THESE WITH --inspect BEFORE TRUSTING THEM.
# These are my best recollection and are exactly the kind of thing that is quietly wrong.
STAT_SETTLEMENT = 3
STAT_OPEN_INTEREST = 9
STAT_CLEARED_VOLUME = 6


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit(
            "Set your key first:\n"
            "    export DATABENTO_API_KEY='db-xxxxxxxx'\n"
            "(get it from databento.com -> Portal -> API keys)"
        )
    return db.Historical(key)


def inspect(symbol: str) -> None:
    """One day, one product. Confirms the stat_type codes without spending real money."""
    c = client()
    print(f"Pulling one day of statistics for {symbol}...\n")
    data = c.timeseries.get_range(
        dataset=DATASET, schema="statistics",
        symbols=[f"{symbol}.FUT"], stype_in="parent",
        start="2024-03-14", end="2024-03-15",
    )
    df = data.to_df()
    if df.empty:
        print("Empty. Try a different date (must be a trading day).")
        return
    print("stat_type distribution:")
    print(df.groupby("stat_type").agg(n=("price", "size"),
                                      example_price=("price", "first"),
                                      example_qty=("quantity", "first")).to_string())
    print("\nColumns available:", list(df.columns))
    print("\nMatch these against the guesses at the top of this file:")
    print(f"  STAT_SETTLEMENT     = {STAT_SETTLEMENT}   <- should look like a real price")
    print(f"  STAT_OPEN_INTEREST  = {STAT_OPEN_INTEREST}   <- should be a large integer in quantity")
    print(f"  STAT_CLEARED_VOLUME = {STAT_CLEARED_VOLUME}   <- also a large integer in quantity")
    print("\nIf they don't match, edit the constants and re-run this.")


def estimate() -> None:
    """Price the full request without downloading. Free."""
    c = client()
    total = 0.0
    for sym, root in PRODUCTS.items():
        for schema in ("statistics", "definition"):
            try:
                cost = c.metadata.get_cost(
                    dataset=DATASET, schema=schema,
                    symbols=[f"{root}.FUT"], stype_in="parent",
                    start=START, end=END,
                )
                total += float(cost)
                print(f"  {sym:5s} {schema:11s} ${float(cost):8.2f}")
            except Exception as e:
                print(f"  {sym:5s} {schema:11s} estimate failed: {e}")
    print(f"\nESTIMATED TOTAL: ${total:,.2f}   (you have $125 of credit)")
    if total > 100:
        print("\nThat is most of your credit. Consider shortening the history to 2015+ ")
        print("by setting START, or dropping to a smaller universe for the first pass.")


def fetch_expiries(c, root: str) -> pd.DataFrame:
    d = c.timeseries.get_range(
        dataset=DATASET, schema="definition",
        symbols=[f"{root}.FUT"], stype_in="parent", start=START, end=END,
    ).to_df()
    if d.empty:
        return pd.DataFrame(columns=["contract", "expiry"])
    col = "expiration" if "expiration" in d.columns else "expiration_date"
    out = (d[["raw_symbol", col]].dropna().drop_duplicates("raw_symbol")
             .rename(columns={"raw_symbol": "contract", col: "expiry"}))
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.tz_localize(None).dt.normalize()
    return out


def fetch_stats(c, root: str) -> pd.DataFrame:
    s = c.timeseries.get_range(
        dataset=DATASET, schema="statistics",
        symbols=[f"{root}.FUT"], stype_in="parent", start=START, end=END,
    ).to_df()
    if s.empty:
        return pd.DataFrame()
    s = s.reset_index()
    tscol = "ts_ref" if "ts_ref" in s.columns else "ts_recv"
    s["date"] = pd.to_datetime(s[tscol]).dt.tz_localize(None).dt.normalize()
    sym = "symbol" if "symbol" in s.columns else "raw_symbol"

    def pick(stat, value_col):
        x = s[s["stat_type"] == stat][["date", sym, value_col]].copy()
        x.columns = ["date", "contract", "v"]
        # keep the LAST record per contract-day: CME publishes a preliminary settlement
        # and then a final one, and we want the final.
        return x.groupby(["date", "contract"], as_index=False)["v"].last()

    settle = pick(STAT_SETTLEMENT, "price").rename(columns={"v": "settle"})
    oi = pick(STAT_OPEN_INTEREST, "quantity").rename(columns={"v": "open_interest"})
    vol = pick(STAT_CLEARED_VOLUME, "quantity").rename(columns={"v": "volume"})

    out = settle.merge(oi, on=["date", "contract"], how="left") \
                .merge(vol, on=["date", "contract"], how="left")
    return out


def run(out_path: str) -> None:
    c = client()
    frames = []
    for sym, root in PRODUCTS.items():
        print(f"{sym}: ", end="", flush=True)
        try:
            stats = fetch_stats(c, root)
            exp = fetch_expiries(c, root)
            if stats.empty:
                print("NO DATA")
                continue
            df = stats.merge(exp, on="contract", how="left")
            df["symbol"] = sym
            missing = df["expiry"].isna().mean()
            frames.append(df)
            print(f"{len(df):>8,} rows   {df['date'].min():%Y-%m-%d} to "
                  f"{df['date'].max():%Y-%m-%d}   missing expiry {missing:.1%}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    if not frames:
        raise SystemExit("Nothing fetched.")
    px = pd.concat(frames, ignore_index=True)
    px = px.dropna(subset=["settle", "expiry"])
    px = px[["date", "symbol", "contract", "settle", "volume", "open_interest", "expiry"]]
    px.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}   {len(px):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--symbol", default="ZC")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default="px.parquet")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.symbol)
    elif a.estimate:
        estimate()
    elif a.run:
        run(a.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()