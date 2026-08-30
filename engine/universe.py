"""
universe.py — the wide research universe.

35 contracts across six asset classes, all on CME (dataset GLBX.MDP3), so this needs no
new vendor subscription and no new cost.

WHY 35 AND NOT 13

The power audit showed the old universe could not answer a cross-sectional question. A
within-instrument monthly basis regression on 13 commodities had an expected t of 0.79
under the published effect size — it could not have passed regardless of what carry does.
Breadth is not a nicety here, it is the difference between a test and a ritual.

RESEARCH UNIVERSE vs TRADEABLE UNIVERSE — these are different and the distinction matters.

We TEST on all 35. We report honestly which subset is tradeable at $450,000 after integer
contract granularity. Excluding a contract from the test because we cannot trade one lot
of it would be crippling the science to fit the account, and it is the wrong order: first
establish whether the effect exists, then establish what you can express.

`tradeable_450k` flags the second question. It is documentation, not a filter.

CAPACITY NOTE ON EQUITY INDEX
One E-mini S&P contract is roughly $350,000 of notional — most of the account in a single
lot. The micros (MES, MNQ, MYM, M2K) are tenth-size clones with identical price series, so
the signal is read and the exposure taken on the same underlying at a size that fits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Inst:
    symbol: str            # what we trade
    root: str              # Databento product root to fetch
    asset: str             # asset class
    sector: str            # finer grouping, for neutralisation
    multiplier: float      # contract units
    tick: float            # minimum price increment in quote units
    tick_value: float      # dollars per tick
    commission: float      # per side, all-in
    physical: bool         # physically delivered? drives the delivery-cycle work
    tradeable_450k: bool   # can one lot be sized sensibly at $450k?
    cftc: str = ""         # COT code where one exists
    price_scale: float = 1.0   # multiply the quoted settle by this to get DOLLARS

    @property
    def dollar_price_mult(self) -> float:
        """Contract notional = quoted settle x this."""
        return self.multiplier * self.price_scale


# PRICE QUOTATION UNITS — this caused a real error and is worth stating plainly.
#
# CME quotes the grains, the soy complex and the livestock contracts in CENTS, not dollars.
# Corn settling at 440.75 is $4.4075 per bushel, so one contract is 440.75 x 0.01 x 5,000 =
# $22,037, NOT $2.2 million. Taking the quote at face value inflated seven of seventeen
# commodity contracts by a factor of 100 and made the entire book look untradeable.
#
# Soybean meal is the exception inside the complex: it is quoted in DOLLARS per short ton.
#
# Anything not carrying an explicit scale below is quoted in dollars and takes 1.0.


UNIVERSE = [
    # ---- energy ---------------------------------------------------------
    Inst("MCL", "CL",  "commodity", "energy",    100,     0.01,    1.00, 0.75, True,  True,  "067651"),
    Inst("QG",  "NG",  "commodity", "energy",    2_500,   0.005,  12.50, 0.75, True,  True,  "023651"),
    Inst("HO",  "HO",  "commodity", "energy",    42_000,  0.0001,  4.20, 1.50, True,  False, "022651"),
    Inst("RB",  "RB",  "commodity", "energy",    42_000,  0.0001,  4.20, 1.50, True,  False, "111659"),
    # ---- metals ---------------------------------------------------------
    Inst("MGC", "GC",  "commodity", "metals",    10,      0.10,    1.00, 0.75, True,  True,  "088691"),
    Inst("SIL", "SI",  "commodity", "metals",    1_000,   0.005,   5.00, 0.75, True,  True,  "084691"),
    Inst("MHG", "HG",  "commodity", "metals",    2_500,   0.0005,  1.25, 0.75, True,  True,  "085692"),
    Inst("PL",  "PL",  "commodity", "metals",    50,      0.10,    5.00, 1.50, True,  True,  "076651"),
    Inst("PA",  "PA",  "commodity", "metals",    100,     0.05,    5.00, 1.50, True,  False, "075651"),
    # ---- grains and oilseeds --------------------------------------------
    Inst("ZC",  "ZC",  "commodity", "grains",    5_000,   0.0025, 12.50, 1.50, True,  True,  "002602", 0.01),
    Inst("ZW",  "ZW",  "commodity", "grains",    5_000,   0.0025, 12.50, 1.50, True,  True,  "001602", 0.01),
    Inst("KE",  "KE",  "commodity", "grains",    5_000,   0.0025, 12.50, 1.50, True,  True,  "001612", 0.01),
    Inst("ZS",  "ZS",  "commodity", "oilseeds",  5_000,   0.0025, 12.50, 1.50, True,  True,  "005602", 0.01),
    Inst("ZM",  "ZM",  "commodity", "oilseeds",  100,     0.10,   10.00, 1.50, True,  True,  "026603"),
    Inst("ZL",  "ZL",  "commodity", "oilseeds",  60_000,  0.0001,  6.00, 1.50, True,  True,  "007601", 0.01),
    # ---- livestock ------------------------------------------------------
    Inst("LE",  "LE",  "commodity", "livestock", 40_000,  0.00025,10.00, 1.50, True,  True,  "057642", 0.01),
    Inst("HE",  "HE",  "commodity", "livestock", 40_000,  0.00025,10.00, 1.50, True,  True,  "054642", 0.01),
    # ---- equity index (micros: tenth-size clones, same underlying) -------
    Inst("MES", "MES", "equity",    "eq_us_lg",  5,       0.25,    1.25, 0.35, False, True,  "13874A"),
    Inst("MNQ", "MNQ", "equity",    "eq_us_tech",2,       0.25,    0.50, 0.35, False, True,  "209742"),
    Inst("MYM", "MYM", "equity",    "eq_us_lg",  0.50,    1.0,     0.50, 0.35, False, True,  "12460+"),
    Inst("M2K", "M2K", "equity",    "eq_us_sm",  5,       0.10,    0.50, 0.35, False, True,  "239742"),
    # ---- rates ----------------------------------------------------------
    Inst("ZT",  "ZT",  "rates",     "rates_st",  2_000,   0.00390625, 7.8125, 0.75, False, True, "042601"),
    Inst("ZF",  "ZF",  "rates",     "rates_mid", 1_000,   0.0078125,  7.8125, 0.75, False, True, "044601"),
    Inst("ZN",  "ZN",  "rates",     "rates_mid", 1_000,   0.015625,  15.625,  0.75, False, True, "043602"),
    Inst("ZB",  "ZB",  "rates",     "rates_lg",  1_000,   0.03125,   31.25,   0.75, False, True, "020601"),
    Inst("UB",  "UB",  "rates",     "rates_lg",  1_000,   0.03125,   31.25,   0.75, False, True, "020604"),
    Inst("SR3", "SR3", "rates",     "rates_sofr",2_500,   0.0025,     6.25,   0.75, False, True, "134742"),
    # ---- FX -------------------------------------------------------------
    Inst("6E",  "6E",  "fx",        "fx_eur",    125_000, 0.00005, 6.25, 1.00, False, True,  "099741"),
    Inst("6J",  "6J",  "fx",        "fx_jpy",    12_500_000, 0.0000005, 6.25, 1.00, False, True, "097741"),
    Inst("6B",  "6B",  "fx",        "fx_gbp",    62_500,  0.0001,  6.25, 1.00, False, True,  "096742"),
    Inst("6A",  "6A",  "fx",        "fx_com",    100_000, 0.0001, 10.00, 1.00, False, True,  "232741"),
    Inst("6C",  "6C",  "fx",        "fx_com",    100_000, 0.00005, 5.00, 1.00, False, True,  "090741"),
    Inst("6S",  "6S",  "fx",        "fx_chf",    125_000, 0.0001, 12.50, 1.00, False, True,  "092741"),
    Inst("6N",  "6N",  "fx",        "fx_com",    100_000, 0.0001, 10.00, 1.00, False, True,  "112741"),
    Inst("6M",  "6M",  "fx",        "fx_em",     500_000, 0.00001, 5.00, 1.00, False, True,  "095741"),
]

BY_SYMBOL = {i.symbol: i for i in UNIVERSE}
PHYSICAL = [i for i in UNIVERSE if i.physical]
ASSETS = sorted({i.asset for i in UNIVERSE})
SECTORS = sorted({i.sector for i in UNIVERSE})

# The original 13. Kept so results can be compared like for like.
NARROW = ["MCL", "QG", "MGC", "SIL", "MHG", "ZC", "ZW", "KE", "ZS", "ZM", "ZL", "LE", "HE"]


def summary() -> None:
    print(f"{len(UNIVERSE)} instruments, {len(ASSETS)} asset classes, "
          f"{len(SECTORS)} sectors")
    for a in ASSETS:
        m = [i.symbol for i in UNIVERSE if i.asset == a]
        print(f"  {a:10s} {len(m):>2d}  {' '.join(m)}")
    print(f"\n  physically delivered: {len(PHYSICAL)} "
          f"(these carry a First Notice Day and a delivery constraint)")
    nt = [i.symbol for i in UNIVERSE if not i.tradeable_450k]
    print(f"  flagged not tradeable at $450k: {nt}")
    print("\n  CFTC codes for the financial contracts are UNVERIFIED — they are only")
    print("  needed if a hypothesis uses positioning data, and the current one does not.")
    print("  Verify with fetch_cot.py --list-markets before relying on any of them.")


if __name__ == "__main__":
    summary()