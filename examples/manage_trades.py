#!/usr/bin/env python3
"""
manage_trades.py  —  Multi-instrument Pricer and Trade Manager
=============================================================
  Instruments : VanillaSwap (IRS) | Bond | OptionTrade (swaptions)
              | EquitySwap (TRS)  | CreditDefaultSwap (CDS)
              | EquityOption
  Framework   : QuantLib 1.41 + PySpark 4.0 (pandas UDF)
  Curve       : Single-curve SOFR  (discount & forward)
  Greeks      : DV01, Duration, PV01, Convexity, Vega, Theta, Delta, CR01, JTD
  Convergence : Iterative re-pricing until max|ΔNPV| < tol
  Persistence : TradeRepository (SQLite, class-table-inheritance) + CSV outputs

  Workflow
  --------
  RUN-1  Generate 100 trades per instrument type, price, save to DB + CSVs.
  RUN-2  Reload all trades from DB, re-price, compare NPVs against RUN-1.
"""

import os, sys, logging, time
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

# ── path must come first so both driver and workers can find models/db ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── environment (MUST precede PySpark import) ──────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_HOME"]            = "/opt/anaconda3/lib/python3.13/site-packages/pyspark"
os.environ.setdefault("JAVA_HOME",   "/usr/local/opt/openjdk@17")
os.environ["PYSPARK_SUBMIT_ARGS"]   = (
    "--conf spark.driver.extraJavaOptions=-Dlog4j2.rootLogger.level=OFF "
    "--conf spark.executor.extraJavaOptions=-Dlog4j2.rootLogger.level=OFF "
    "pyspark-shell"
)
logging.getLogger("py4j").setLevel(logging.CRITICAL)
logging.getLogger("pyspark").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import QuantLib as ql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import pandas_udf
from tabulate import tabulate

from models import (TradeBase, BaseLeg, FixedLeg, FloatLeg, OptionLeg, EquityLeg,
                    CreditLeg, CDSPremiumLeg, CDSProtectionLeg, EquityOptionLeg,
                    VanillaSwap, InterestRateSwap, InterestRateSwaption, Bond, CallableBond,
                    OptionTrade, EquitySwap,
                    CreditDefaultSwap,
                    EquityOptionTrade, PricingResult,
                    TradeDirection, MarketDataSnapshot, MarketDataCache, make_default_snapshot)
from db import TradeRepository, MarketDataRepository


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

VALUATION_DATE = date(2025, 1, 15)

# Module-level cached snapshot — loaded once at startup
_mkt_cache = MarketDataCache()
MKT = _mkt_cache.get_or_create(VALUATION_DATE)

_EQ_OPTION_VOLS = {
    "SPY":   {0.25: 0.14, 0.5: 0.15, 1: 0.16, 2: 0.18},
    "QQQ":   {0.25: 0.18, 0.5: 0.19, 1: 0.20, 2: 0.22},
    "AAPL":  {0.25: 0.25, 0.5: 0.26, 1: 0.27, 2: 0.28},
    "MSFT":  {0.25: 0.22, 0.5: 0.23, 1: 0.24, 2: 0.25},
    "GOOGL": {0.25: 0.24, 0.5: 0.25, 1: 0.26, 2: 0.27},
    "SX5E":  {0.25: 0.16, 0.5: 0.17, 1: 0.18, 2: 0.20},
    "NKY":   {0.25: 0.18, 0.5: 0.19, 1: 0.20, 2: 0.22},
    "IWM":   {0.25: 0.20, 0.5: 0.21, 1: 0.22, 2: 0.24},
    "FTSE":  {0.25: 0.15, 0.5: 0.16, 1: 0.17, 2: 0.19},
}


def make_curve_df() -> pd.DataFrame:
    """Return sorted SOFR zero curve DataFrame for Spark broadcast."""
    return MKT.to_curve_df()


def build_sofr_curve(val_date: ql.Date, curve_df: pd.DataFrame) -> ql.YieldTermStructureHandle:
    """Build QuantLib yield curve from broadcast curve_df."""
    dc = ql.Actual365Fixed()
    dates = [val_date]
    rates = [float(curve_df["zero_rate"].iloc[0])]
    for _, row in curve_df.iterrows():
        months = max(1, int(round(float(row["tenor_y"]) * 12)))
        dates.append(val_date + ql.Period(months, ql.Months))
        rates.append(float(row["zero_rate"]))
    curve = ql.ZeroCurve(dates, rates, dc, ql.NullCalendar(),
                         ql.Linear(), ql.Continuous, ql.Annual)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRADE DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def make_irs_data(n: int = 50) -> list:
    """Generate n IRS trades covering FIXED_FLOAT, FLOAT_FIXED, FIXED_FIXED, FLOAT_FLOAT."""
    import random
    random.seed(42)
    base = VALUATION_DATE
    eff  = base + timedelta(days=2)

    tenors  = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    nots    = [1e6, 2e6, 5e6, 10e6, 20e6, 50e6]
    spreads = [-0.015, -0.010, -0.0075, -0.005, -0.0025,
                0.000,  0.0025,  0.005,  0.0075,  0.010, 0.015, 0.020]
    books   = ["IRD-NY", "IRD-LDN", "IRD-ASIA", "MACRO"]
    cptys   = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]

    subtypes = (["FIXED_FLOAT"] * 20 + ["FLOAT_FIXED"] * 15 +
                ["FIXED_FIXED"] * 8   + ["FLOAT_FLOAT"] * 7)
    random.shuffle(subtypes)
    while len(subtypes) < n:
        extra = subtypes[:]
        random.shuffle(extra)
        subtypes += extra

    trades = []
    for i in range(n):
        tenor_y    = random.choice(tenors)
        notl       = float(random.choice(nots))
        par_rate   = MKT.get_par_rate(tenor_y)
        fixed_rate = round(max(0.005, par_rate + random.choice(spreads)), 4)
        subtype    = subtypes[i]

        try:
            mat = date(eff.year + tenor_y, eff.month, eff.day)
        except ValueError:
            mat = date(eff.year + tenor_y, eff.month, eff.day - 1)

        if subtype in ("FIXED_FLOAT", "FLOAT_FIXED"):
            direction = TradeDirection.PAYER if subtype == "FIXED_FLOAT" else TradeDirection.RECEIVER
            legs = [
                FixedLeg("FIXED", notl, eff, mat, coupon_rate=fixed_rate),
                FloatLeg("FLOAT", notl, eff, mat, frequency="QUARTERLY", day_count="ACT/360"),
            ]
        elif subtype == "FIXED_FIXED":
            rate2 = round(max(0.005, par_rate + random.choice(spreads)), 4)
            direction = TradeDirection.PAYER
            legs = [
                FixedLeg("FIXED", notl, eff, mat, coupon_rate=fixed_rate, frequency="ANNUAL"),
                FixedLeg("FIXED", notl, eff, mat, coupon_rate=rate2, frequency="SEMIANNUAL"),
            ]
        else:  # FLOAT_FLOAT
            basis_spread_bps = random.uniform(0.0005, 0.0025)
            direction = TradeDirection.PAYER
            legs = [
                FloatLeg("FLOAT", notl, eff, mat, index_tenor_m=1,
                        day_count="ACT/360", frequency="MONTHLY", spread=basis_spread_bps),
                FloatLeg("FLOAT", notl, eff, mat, index_tenor_m=3,
                        day_count="ACT/360", frequency="QUARTERLY"),
            ]

        trades.append(InterestRateSwap(
            trade_id=f"IRS-{i+1:04d}", book=random.choice(books),
            counterparty=random.choice(cptys), trader=random.choice(traders),
            valuation_date=base, direction=direction, tenor_y=tenor_y,
            swap_subtype=subtype, legs=legs,
        ))
    return trades


def make_bond_data(n: int = 50) -> List[Bond]:
    """Generate n seasoned USD fixed-rate bonds (Treasury-like)."""
    import random
    random.seed(99)
    base = VALUATION_DATE
    if base.month > 6:
        issued = date(base.year, base.month - 6, base.day)
    else:
        issued = date(base.year - 1, base.month + 6, base.day)

    tenors  = [2, 3, 5, 7, 10, 20, 30]
    faces   = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
    books   = ["GOVTBOND-NY", "GOVTBOND-LDN", "RATES-ASIA"]
    cptys   = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]

    bonds: List[Bond] = []
    for i in range(n):
        tenor_y = random.choice(tenors)
        face    = float(random.choice(faces))
        par_coupon = MKT.get_par_rate(tenor_y)
        coupon  = round(par_coupon + random.uniform(-0.005, 0.005), 4)
        try:
            mat = date(issued.year + tenor_y, issued.month, issued.day)
        except ValueError:
            mat = date(issued.year + tenor_y, issued.month, issued.day - 1)
        direction = TradeDirection.LONG if random.random() < 0.5 else TradeDirection.SHORT
        bonds.append(Bond(
            trade_id=f"BOND-{i+1:04d}", book=random.choice(books),
            counterparty=random.choice(cptys), trader=random.choice(traders),
            valuation_date=base, direction=direction, tenor_y=tenor_y,
            legs=[BaseLeg(
                "BOND", face, issued, mat,
                coupon_rate=coupon, day_count="ACT/ACT",
                frequency="SEMIANNUAL", redemption=100.0, settlement_days=2,
                issue_date=issued,
            )],
            isin=f"US{i+1:09d}",
        ))
    return bonds


def make_option_data(n: int = 50) -> List[OptionTrade]:
    """Generate n vanilla SOFR swaptions (Payer/Receiver, Buy/Sell)."""
    import random
    random.seed(77)
    base        = VALUATION_DATE
    exp_tenors  = [1, 2, 3, 5]
    swap_tenors = [5, 10]
    nots        = [5e6, 10e6, 20e6, 50e6]
    books       = ["OPTIONS-NY", "OPTIONS-LDN", "MACRO"]
    cptys       = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]
    opt_types   = ["PAYER_SWAPTION", "RECEIVER_SWAPTION"]
    directions  = [TradeDirection.BUY, TradeDirection.SELL]

    options: List[OptionTrade] = []
    for i in range(n):
        exp_y    = random.choice(exp_tenors)
        swap_y   = random.choice(swap_tenors)
        notl     = float(random.choice(nots))
        opt_t    = random.choice(opt_types)
        direct   = random.choice(directions)
        base_vol = MKT.get_swaption_vol(exp_y, swap_y)
        vol      = round(base_vol + random.uniform(-0.05, 0.05), 4)
        atm_approx = MKT.get_par_rate(swap_y)
        strike   = round(atm_approx + random.uniform(-0.010, 0.010), 4)
        try:
            expiry = date(base.year + exp_y, base.month, base.day)
        except ValueError:
            expiry = date(base.year + exp_y, base.month, base.day - 1)
        try:
            mat = date(expiry.year + swap_y, expiry.month, expiry.day)
        except ValueError:
            mat = date(expiry.year + swap_y, expiry.month, expiry.day - 1)
        options.append(OptionTrade(
            trade_id=f"OPT-{i+1:04d}", book=random.choice(books),
            counterparty=random.choice(cptys), trader=random.choice(traders),
            valuation_date=base, direction=direct, tenor_y=exp_y, underlying_tenor_y=swap_y,
            legs=[OptionLeg(
                "OPTION", notl, expiry, mat,
                strike=strike, option_type=opt_t,
                exercise_type="EUROPEAN", vol=vol, vol_type="LOGNORMAL",
                vol_shift=0.01, underlying_tenor_m=swap_y * 12,
            )],
        ))
    return options


def make_irs_swaption_data(
    n: int = 10,
    book: str = "OPT-NY",
    trader: str = "T001",
    counterparty: str = "CPTY-01",
    valuation_date=None,
) -> "List[InterestRateSwaption]":
    """
    Generate n InterestRateSwaption trades (3-leg: fixed + float + option).

    Each trade has:
      legs[0] — FixedLeg (underlying swap's fixed coupon stream)
      legs[1] — FloatLeg (underlying swap's floating SOFR stream)
      legs[2] — OptionLeg (expiry, strike, vol — the swaption optionality)

    Alternates PAYER_SWAPTION / RECEIVER_SWAPTION with BUY/SELL directions.
    """
    import random
    random.seed(77)
    if valuation_date is None:
        valuation_date = VALUATION_DATE

    from datetime import timedelta
    from models.interest_rate_swaption import InterestRateSwaption

    expiry_tenors   = [1, 1, 2, 2, 3, 5, 1, 2, 3, 5]        # option expiry (y)
    undly_tenors    = [5, 10, 5, 10, 7, 10, 7, 7, 5, 10]     # underlying swap (y)
    option_types    = ["PAYER_SWAPTION", "RECEIVER_SWAPTION"] * 5
    directions      = [TradeDirection.BUY, TradeDirection.SELL] * 5
    coupons         = [0.040, 0.045, 0.042, 0.048, 0.043,
                       0.046, 0.044, 0.047, 0.041, 0.049]
    vols            = [0.35, 0.38, 0.36, 0.40, 0.37,
                       0.39, 0.34, 0.36, 0.38, 0.41]
    notionals       = [10_000_000, 15_000_000, 20_000_000, 10_000_000, 25_000_000,
                       15_000_000, 20_000_000, 10_000_000, 15_000_000, 20_000_000]

    trades = []
    for i in range(min(n, 10)):
        exp_y   = expiry_tenors[i]
        und_y   = undly_tenors[i]
        notl    = notionals[i]
        cpn     = coupons[i]
        vol     = vols[i]
        otype   = option_types[i]
        dirn    = directions[i]

        # Dates
        expiry_dt = valuation_date.replace(year=valuation_date.year + exp_y)
        maturity_dt = expiry_dt.replace(year=expiry_dt.year + und_y)
        swap_subtype = "FIXED_FLOAT" if otype == "PAYER_SWAPTION" else "FLOAT_FIXED"

        fixed_leg = FixedLeg(
            leg_type="FIXED", notional=notl,
            start_date=expiry_dt, end_date=maturity_dt,
            coupon_rate=cpn, currency="USD",
            day_count="30/360", frequency="SEMIANNUAL",
        )
        float_leg = FloatLeg(
            leg_type="FLOAT", notional=notl,
            start_date=expiry_dt, end_date=maturity_dt,
            spread=0.0, index_name="SOFR3M", currency="USD",
            day_count="ACT/360", frequency="QUARTERLY",
        )
        option_leg = OptionLeg(
            leg_type="OPTION", notional=notl,
            start_date=expiry_dt, end_date=maturity_dt,
            strike=cpn, option_type=otype,
            vol=vol, vol_type="LOGNORMAL", vol_shift=0.03,
            exercise_type="EUROPEAN",
            underlying_tenor_m=und_y * 12,
        )

        trade_id = f"IRS-SWPTN-{i + 1:04d}"
        trades.append(InterestRateSwaption(
            trade_id=trade_id,
            book=book,
            counterparty=counterparty,
            trader=trader,
            valuation_date=valuation_date,
            direction=dirn,
            legs=[fixed_leg, float_leg, option_leg],
            tenor_y=exp_y,
            underlying_tenor_y=und_y,
            swap_subtype=swap_subtype,
        ))
    return trades


def make_equity_data(n: int = 50) -> List[EquitySwap]:
    """Generate n equity total-return swaps across major indices."""
    import random
    random.seed(55)
    base    = VALUATION_DATE
    eff     = base + timedelta(days=2)
    tickers = ["SPY", "QQQ", "IWM", "EFA", "SX5E", "NKY", "FTSE"]
    tenors  = [1, 2, 3, 5]
    nots    = [5e6, 10e6, 20e6, 50e6]
    books   = ["EQUITY-NY", "EQUITY-LDN", "MACRO"]
    cptys   = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]
    result  = []
    for i in range(n):
        ticker  = random.choice(tickers)
        tenor_y = random.choice(tenors)
        notl    = float(random.choice(nots))
        direction = TradeDirection.LONG if random.random() < 0.5 else TradeDirection.SHORT
        s0 = MKT.equity_prices[ticker]
        q  = MKT.equity_divylds[ticker]
        try:
            mat = date(eff.year + tenor_y, eff.month, eff.day)
        except ValueError:
            mat = date(eff.year + tenor_y, eff.month, eff.day - 1)
        result.append(EquitySwap(
            trade_id=f"EQ-{i+1:04d}", book=random.choice(books),
            counterparty=random.choice(cptys), trader=random.choice(traders),
            valuation_date=base, direction=direction, tenor_y=tenor_y, underlying_ticker=ticker,
            legs=[EquityLeg(
                "EQUITY", notl, eff, mat,
                underlying_ticker=ticker, initial_price=s0,
                dividend_yield=q, equity_return_type="TOTAL",
                funding_spread=0.005,
            )],
        ))
    return result


def make_cds_data(n: int = 50) -> List[CreditDefaultSwap]:
    """Generate n single-name CDS across IG/HY reference entities."""
    import random
    random.seed(33)
    base = VALUATION_DATE
    eff  = base + timedelta(days=2)
    entities = [
        ("FORD MOTOR CO",      MKT.credit_spreads["FORD MOTOR CO"],      0.40, "SENIOR_UNSECURED", "CR"),
        ("GENERAL MOTORS CO",  MKT.credit_spreads["GENERAL MOTORS CO"],  0.40, "SENIOR_UNSECURED", "CR"),
        ("AT&T INC",           MKT.credit_spreads["AT&T INC"],           0.40, "SENIOR_UNSECURED", "CR"),
        ("VERIZON COMMS",      MKT.credit_spreads["VERIZON COMMS"],      0.40, "SENIOR_UNSECURED", "CR"),
        ("BOEING CO",          MKT.credit_spreads["BOEING CO"],          0.40, "SENIOR_UNSECURED", "CR"),
        ("NETFLIX INC",        MKT.credit_spreads["NETFLIX INC"],        0.35, "SENIOR_UNSECURED", "XR"),
        ("TESLA INC",          MKT.credit_spreads["TESLA INC"],          0.35, "SENIOR_UNSECURED", "XR"),
        ("IBM CORP",           MKT.credit_spreads["IBM CORP"],           0.40, "SENIOR_UNSECURED", "CR"),
        ("COMCAST CORP",       MKT.credit_spreads["COMCAST CORP"],       0.40, "SENIOR_UNSECURED", "CR"),
        ("AMAZON COM INC",     MKT.credit_spreads["AMAZON COM INC"],     0.40, "SENIOR_UNSECURED", "CR"),
    ]
    tenors = [1, 3, 5, 7, 10]
    nots   = [5e6, 10e6, 20e6]
    books  = ["CREDIT-NY", "CREDIT-LDN", "MACRO"]
    cptys  = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]
    result = []
    for i in range(n):
        name, spread, rec, seniority, doc = random.choice(entities)
        tenor_y = random.choice(tenors)
        notl    = float(random.choice(nots))
        direction = TradeDirection.BUY if random.random() < 0.5 else TradeDirection.SELL
        spread_bumped = round(spread + random.uniform(-0.003, 0.003), 4)
        hazard = spread_bumped / (1.0 - rec)
        try:
            mat = date(eff.year + tenor_y, eff.month, eff.day)
        except ValueError:
            mat = date(eff.year + tenor_y, eff.month, eff.day - 1)
        result.append(CreditDefaultSwap(
            trade_id=f"CDS-{i+1:04d}", book=random.choice(books),
            counterparty=random.choice(cptys), trader=random.choice(traders),
            valuation_date=base, direction=direction, tenor_y=tenor_y,
            legs=[
                CDSProtectionLeg(
                    "CDS_PROTECTION", notl, eff, mat,
                    reference_entity=name, credit_spread=spread_bumped,
                    recovery_rate=rec, hazard_rate=hazard,
                    seniority=seniority, doc_clause=doc,
                ),
                CDSPremiumLeg(
                    "CDS_PREMIUM", notl, eff, mat,
                    coupon_rate=spread_bumped,
                ),
            ],
        ))
    return result


def make_callable_bond_data(n: int = 100) -> List[CallableBond]:
    """Generate n callable/putable bonds with embedded HullWhite options."""
    import random
    random.seed(55)
    base = VALUATION_DATE

    tenors     = [5, 7, 10, 15, 20, 30]
    faces      = [5_000_000, 10_000_000, 20_000_000, 50_000_000]
    books      = ["BOND-NY", "BOND-LDN", "BOND-ASIA", "RATES-NY"]
    cptys      = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders    = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]
    call_types = (["CALL"] * 6 + ["PUT"] * 4)   # 60% CALLABLE / 40% PUTABLE
    ex_types   = (["BERMUDAN"] * 7 + ["AMERICAN"] * 3)   # 70% BERMUDAN / 30% AMERICAN

    trades: List[CallableBond] = []
    for i in range(n):
        tenor_y    = random.choice(tenors)
        face       = float(random.choice(faces))
        opt_t      = random.choice(call_types)
        ex_t       = random.choice(ex_types)
        direction  = TradeDirection.LONG if random.random() < 0.55 else TradeDirection.SHORT

        # Coupon near-ATM
        par_coupon = MKT.get_par_rate(tenor_y)
        coupon     = round(par_coupon + random.uniform(-0.005, 0.005), 4)
        coupon     = max(0.040, min(0.055, coupon))  # clamp 4.0–5.5%

        # HullWhite vol
        hw_vol = round(random.uniform(0.008, 0.015), 4)

        # Dates
        issued = base
        try:
            mat = date(base.year + tenor_y, base.month, base.day)
        except ValueError:
            mat = date(base.year + tenor_y, base.month, base.day - 1)

        # First call: 2-5 years from today
        nc_y = random.randint(2, min(5, tenor_y - 1))
        try:
            first_call = date(base.year + nc_y, base.month, base.day)
        except ValueError:
            first_call = date(base.year + nc_y, base.month, base.day - 1)

        trades.append(CallableBond(
            trade_id=f"CBOND-{i+1:04d}",
            book=random.choice(books),
            counterparty=random.choice(cptys),
            trader=random.choice(traders),
            valuation_date=base,
            direction=direction,
            tenor_y=tenor_y,
            isin=f"US{i+1001:09d}",
            legs=[
                BaseLeg(
                    "BOND", face, issued, mat,
                    coupon_rate=coupon, day_count="30/360",
                    frequency="SEMIANNUAL", redemption=100.0,
                    settlement_days=2, issue_date=issued,
                ),
                OptionLeg(
                    "OPTION", face, first_call, mat,
                    strike=1.00,                  # par call/put
                    option_type=opt_t,
                    exercise_type=ex_t,
                    vol=hw_vol,
                    vol_type="NORMAL",
                    vol_shift=0.0,
                    underlying_tenor_m=tenor_y * 12,
                    underlying_type="BOND",
                ),
            ],
        ))
    return trades


def make_equity_option_data(n: int = 100) -> List[EquityOptionTrade]:
    """Generate n vanilla equity options (calls and puts, European and American)."""
    import random
    from datetime import timedelta
    random.seed(88)
    base = VALUATION_DATE

    tickers   = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "SX5E", "NKY", "IWM", "FTSE"]
    expiries  = [0.25, 0.5, 1.0, 2.0]
    moneyness = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    nots      = [1e6, 2e6, 5e6, 10e6]
    books     = ["EQ-DERIV-NY", "EQ-DERIV-LDN", "EQ-DERIV-ASIA", "MACRO"]
    cptys     = [f"CPTY-{c:02d}" for c in range(1, 11)]
    traders = ["T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"]
    opt_types = ["CALL", "PUT"]
    ex_types  = ["EUROPEAN", "AMERICAN"]

    trades: List[EquityOptionTrade] = []
    for i in range(n):
        ticker   = random.choice(tickers)
        exp_y    = random.choice(expiries)
        s0       = MKT.equity_prices.get(ticker, 100.0)
        q        = MKT.equity_divylds.get(ticker, 0.013)
        mon      = random.choice(moneyness)
        strike   = round(s0 * mon, 2)
        notl     = float(random.choice(nots))
        opt_type = random.choice(opt_types)
        ex_type  = random.choice(ex_types)
        direction = TradeDirection.BUY if random.random() < 0.55 else TradeDirection.SELL
        pricing_model = "BLACK_SCHOLES" if ex_type == "EUROPEAN" else "BINOMIAL"

        vols_by_expiry = _EQ_OPTION_VOLS.get(ticker, {1: 0.25})
        expiry_keys = sorted(vols_by_expiry.keys())
        base_vol = vols_by_expiry.get(exp_y, vols_by_expiry[expiry_keys[-1]])
        smile_adj = abs(mon - 1.0) * 0.10
        vol = round(base_vol + smile_adj + random.uniform(-0.02, 0.02), 4)

        tenor_y = max(1, int(round(exp_y)))
        exp_days = int(exp_y * 365)
        expiry_date = base + timedelta(days=exp_days)

        trades.append(EquityOptionTrade(
            trade_id=f"EQOPT-{i+1:04d}",
            book=random.choice(books),
            counterparty=random.choice(cptys),
            trader=random.choice(traders),
            valuation_date=base,
            direction=direction,
            tenor_y=tenor_y,
            legs=[EquityOptionLeg(
                leg_type="EQUITY_OPTION",
                notional=notl,
                start_date=base,
                end_date=expiry_date,
                underlying_ticker=ticker,
                initial_price=s0,
                strike=strike,
                option_type=opt_type,
                exercise_type=ex_type,
                vol=vol,
                dividend_yield=q,
                risk_free_rate=0.0,
                pricing_model=pricing_model,
                n_steps=100,
            )],
        ))
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# 2b. POPULATE TRADES — parameterised bulk generator
# ══════════════════════════════════════════════════════════════════════════════

# ── Currency pairs for cross-currency IRS ─────────────────────────────────────
_XCCY_PAIRS = [
    ("USD", "EUR", 0.93),
    ("USD", "GBP", 0.79),
    ("USD", "JPY", 149.5),
    ("USD", "CHF", 0.89),
    ("USD", "AUD", 1.54),
    ("USD", "CAD", 1.36),
    ("USD", "CNH", 7.24),
    ("USD", "SGD", 1.34),
]

def make_xccy_irs_data(n: int = 50, valuation_date=None) -> list:
    """
    Generate cross-currency InterestRateSwap trades.
    Each trade has two legs with DIFFERENT currencies.  The pay leg is USD
    (FIXED or FLOAT) and the receive leg is a foreign currency (FIXED or FLOAT).
    The trade's fx_rate stores the spot receive_ccy / pay_ccy rate.
    """
    import random
    from datetime import date, timedelta
    from models.interest_rate_swap import InterestRateSwap
    from models.leg import BaseLeg, FixedLeg, FloatLeg

    if valuation_date is None:
        valuation_date = VALUATION_DATE

    random.seed(77)
    trades = []
    traders = [f"T{i:03d}" for i in range(1, 6)]
    cptys   = [f"CPTY-{i:02d}" for i in range(1, 11)]
    SUBTYPES = ["FIXED_FLOAT", "FLOAT_FIXED", "FIXED_FIXED", "FLOAT_FLOAT"]

    for i in range(1, n + 1):
        pay_ccy, rcv_ccy, spot = random.choice(_XCCY_PAIRS)
        # Small FX vol: spot +/- 3 %
        fx = round(spot * random.uniform(0.97, 1.03), 4)

        tenor = random.choice([2, 3, 5, 7, 10])
        notional_usd = random.choice([5, 10, 20, 25, 50, 75, 100]) * 1_000_000
        notional_fgn = round(notional_usd * fx, 0)
        stype = random.choice(SUBTYPES)

        val  = valuation_date
        base = val
        end  = date(base.year + tenor, base.month, base.day)

        freq_usd = "SEMIANNUAL"
        freq_fgn = "ANNUAL"

        def pay_leg(ccy, notl):
            leg_t = "FIXED" if stype.startswith("FIXED") else "FLOAT"
            if leg_t == "FIXED":
                cpn = round(random.uniform(0.03, 0.065), 4)
                return FixedLeg(
                    leg_type="FIXED", notional=float(notl),
                    start_date=base, end_date=end,
                    currency=ccy, day_count="30/360",
                    frequency=freq_usd, coupon_rate=cpn,
                )
            else:
                sprd = round(random.uniform(-0.002, 0.004), 4)
                return FloatLeg(
                    leg_type="FLOAT", notional=float(notl),
                    start_date=base, end_date=end,
                    currency=ccy, day_count="ACT/360",
                    frequency=freq_usd, spread=sprd,
                    index_name="SOFR3M", index_tenor_m=3,
                )

        def rcv_leg(ccy, notl):
            leg_t = "FIXED" if stype.endswith("FIXED") else "FLOAT"
            if leg_t == "FIXED":
                cpn = round(random.uniform(0.025, 0.055), 4)
                return FixedLeg(
                    leg_type="FIXED", notional=float(notl),
                    start_date=base, end_date=end,
                    currency=ccy, day_count="30/360",
                    frequency=freq_fgn, coupon_rate=cpn,
                )
            else:
                idx = {"EUR": "EURIBOR3M", "GBP": "SONIA3M"}.get(ccy, "SOFR3M")
                sprd = round(random.uniform(-0.003, 0.005), 4)
                return FloatLeg(
                    leg_type="FLOAT", notional=float(notl),
                    start_date=base, end_date=end,
                    currency=ccy, day_count="ACT/365",
                    frequency=freq_fgn, spread=sprd,
                    index_name=idx, index_tenor_m=3,
                )

        direction = (TradeDirection.PAYER
                     if stype.startswith("FIXED")
                     else TradeDirection.RECEIVER)
        tid = f"XCCY-{i:04d}"

        trades.append(InterestRateSwap(
            trade_id=tid,
            book="XCCY-LDN",
            trader=random.choice(traders),
            counterparty=random.choice(cptys),
            valuation_date=val,
            direction=direction,
            tenor_y=tenor,
            swap_subtype=stype,
            fx_rate=fx,
            legs=[pay_leg(pay_ccy, notional_usd), rcv_leg(rcv_ccy, notional_fgn)],
        ))

    return trades

def populate_trades(
    n_trades: int = 2000,
    n_traders: int = 10,
    n_books: int = 20,
    valuation_date=None,
) -> List[TradeBase]:
    """
    Generate n_trades evenly split across 6 instrument types, with configurable
    traders/books/counterparties.  Uses random.seed(42) for reproducibility.
    """
    import random
    random.seed(42)

    if valuation_date is None:
        valuation_date = VALUATION_DATE

    # ── Pools ─────────────────────────────────────────────────────────────────
    all_books_by_type = {
        "IRS":   ["IRD-NY", "IRD-LDN", "IRD-ASIA", "IRD-TOKYO"],
        "BOND":  ["BOND-NY", "BOND-LDN", "BOND-ASIA"],
        "OPT":   ["OPT-NY", "OPT-LDN", "OPT-ASIA"],
        "EQ":    ["EQ-NY", "EQ-LDN", "EQ-ASIA"],
        "CDS":   ["CDS-NY", "CDS-LDN", "CDS-ASIA"],
        "EQOPT": ["EQOPT-NY", "EQOPT-LDN", "EQOPT-ASIA"],
        "MACRO": ["MACRO"],
    }
    traders = [f"T{i:03d}" for i in range(1, n_traders + 1)]
    cptys   = [f"CPTY-{i:02d}" for i in range(1, 11)]

    # ── Per-type counts (remainder distributed to first types) ────────────────
    base   = n_trades // 6
    counts = [base] * 6
    for i in range(n_trades - base * 6):
        counts[i] += 1
    n_irs, n_bond, n_opt, n_eq, n_cds, n_eqopt = counts

    # ── Generate using existing functions ─────────────────────────────────────
    irs_trades   = make_irs_data(n_irs)
    bond_trades  = make_bond_data(n_bond)
    opt_trades   = make_option_data(n_opt)
    eq_trades    = make_equity_data(n_eq)
    cds_trades   = make_cds_data(n_cds)
    eqopt_trades = make_equity_option_data(n_eqopt)
    xccy_trades  = make_xccy_irs_data(50, valuation_date)  # XCCY-LDN book

    # ── Patch book / trader / counterparty for each type ──────────────────────
    macro = all_books_by_type["MACRO"][0]

    def _patch(trade_list, type_books):
        for t in trade_list:
            t.book         = random.choice(type_books)
            t.trader       = random.choice(traders)
            t.counterparty = random.choice(cptys)

    _patch(irs_trades,   all_books_by_type["IRS"]   + [macro])
    _patch(bond_trades,  all_books_by_type["BOND"]  + [macro])
    _patch(opt_trades,   all_books_by_type["OPT"]   + [macro])
    _patch(eq_trades,    all_books_by_type["EQ"]    + [macro])
    _patch(cds_trades,   all_books_by_type["CDS"]   + [macro])
    _patch(eqopt_trades, all_books_by_type["EQOPT"] + [macro])
    # xccy_trades already have book/trader/counterparty set in make_xccy_irs_data

    return irs_trades + bond_trades + opt_trades + eq_trades + cds_trades + eqopt_trades + xccy_trades


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

TRADE_SCHEMA = StructType([
    StructField("trade_id",     StringType()),
    StructField("instrument",   StringType()),   # "IRS"|"BOND"|"SWAPTION"|"EQ_SWAP"|"CDS"|"EQ_OPT"
    StructField("book",         StringType()),
    StructField("counterparty", StringType()),
    StructField("trader",       StringType()),
    StructField("direction",    StringType()),
    StructField("tenor_y",      IntegerType()),
    StructField("notional",     DoubleType()),
    StructField("coupon_rate",  DoubleType()),   # strike for opts, credit_spread for CDS
    StructField("swap_subtype", StringType()),   # IRS variant: FIXED_FLOAT|FLOAT_FIXED|FIXED_FIXED|FLOAT_FLOAT
    StructField("trade_json",   StringType()),
])

RESULT_SCHEMA = StructType([
    StructField("fixed_npv",       DoubleType()),
    StructField("float_npv",       DoubleType()),
    StructField("swap_npv",        DoubleType()),
    StructField("par_rate",        DoubleType()),
    StructField("clean_price",     DoubleType()),
    StructField("accrued",         DoubleType()),
    # greeks
    StructField("dv01",            DoubleType()),
    StructField("duration",        DoubleType()),
    StructField("pv01",            DoubleType()),
    StructField("convexity",       DoubleType()),
    StructField("vega",            DoubleType()),
    StructField("theta",           DoubleType()),
    StructField("delta",           DoubleType()),
    StructField("gamma",           DoubleType()),
    StructField("rho",             DoubleType()),
    StructField("cr01",            DoubleType()),
    StructField("jump_to_default", DoubleType()),
    StructField("error",           StringType()),
])

_NAN = float("nan")
_NAN_ROW: Dict[str, Any] = dict(
    fixed_npv=_NAN, float_npv=_NAN, swap_npv=_NAN, par_rate=_NAN,
    clean_price=_NAN, accrued=_NAN,
    dv01=_NAN, duration=_NAN, pv01=_NAN, convexity=_NAN,
    vega=_NAN, theta=_NAN, delta=_NAN,
    gamma=_NAN, rho=_NAN,
    cr01=_NAN, jump_to_default=_NAN, error="",
)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PRICING CORE  (runs on PySpark workers — no module-level QL objects)
# ══════════════════════════════════════════════════════════════════════════════

def _ql_maps():
    """Return local QL convention dicts (no module-level QL objects — not picklable)."""
    freq = {
        "SEMIANNUAL": ql.Semiannual, "QUARTERLY": ql.Quarterly,
        "ANNUAL": ql.Annual, "MONTHLY": ql.Monthly,
    }
    dc = {
        "30/360":   ql.Thirty360(ql.Thirty360.BondBasis),
        "ACT/360":  ql.Actual360(),
        "ACT/365F": ql.Actual365Fixed(),
        "ACT/ACT":  ql.ActualActual(ql.ActualActual.ISDA),
    }
    bdc = {
        "MOD_FOLLOWING": ql.ModifiedFollowing,
        "FOLLOWING":     ql.Following,
        "PRECEDING":     ql.Preceding,
    }
    cal = {
        "US_GOVT":       ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "UK_SETTLEMENT": ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
        "TARGET":        ql.TARGET(),
    }
    return freq, dc, bdc, cal


# ─── IRS ──────────────────────────────────────────────────────────────────────

def _price_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Dispatch IRS pricing based on swap_subtype."""
    subtype = getattr(trade, 'swap_subtype', 'FIXED_FLOAT')
    if subtype == "FIXED_FIXED":
        return _price_fixed_fixed(trade, curve_df)
    if subtype == "FLOAT_FLOAT":
        return _price_float_float(trade, curve_df)
    return _price_fixed_float(trade, curve_df)


def _price_xccy(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a cross-currency swap by discounting each leg independently in its
    own currency then converting to USD.

    pay_leg  (legs[0]): cash flows you pay out → negative contribution to NPV
    recv_leg (legs[1]): cash flows you receive  → positive contribution to NPV
    NPV = recv_npv_usd - pay_npv_usd
    DV01 = NPV(r+1bp) - NPV(r)  → sign falls out naturally from bump.
    """
    try:
        T   = max(float(trade.tenor_y or 5), 0.25)
        r   = 0.05  # flat discount rate (market proxy)
        n   = max(1, int(round(T)))

        def _annuity(rate):
            return sum(1.0 / (1.0 + rate) ** t for t in range(1, n + 1))

        def _disc(rate):
            return 1.0 / (1.0 + rate) ** T

        pay_ccy  = str(getattr(trade, "pay_currency",     "USD") or "USD")
        recv_ccy = str(getattr(trade, "receive_currency", "EUR") or "EUR")
        fx = max(float(getattr(trade, "fx_rate", 1.0) or 1.0), 1e-9)

        def _leg_pv(leg, rate, ann, disc):
            notl = float(getattr(leg, "notional", 0) or 0)
            lt   = str(getattr(leg, "leg_type", "FIXED"))
            if lt == "FIXED":
                cpn = float(getattr(leg, "coupon_rate", 0.05) or 0.05)
                pv  = notl * (cpn * ann + disc - 1.0)
            elif lt == "FLOAT":
                sprd = float(getattr(leg, "spread", 0.0) or 0.0)
                pv   = notl * sprd * ann
            else:
                pv = 0.0
            ccy = str(getattr(leg, "currency", pay_ccy) or pay_ccy)
            if ccy == recv_ccy and ccy != pay_ccy:
                pv /= fx
            return pv

        def _npv(rate):
            ann  = _annuity(rate)
            disc = _disc(rate)
            pay  = sum(_leg_pv(l, rate, ann, disc) for l in trade.legs[:1])
            recv = sum(_leg_pv(l, rate, ann, disc) for l in trade.legs[1:])
            return recv - pay

        ann  = _annuity(r)
        disc = _disc(r)
        npv      = _npv(r)
        npv_bump = _npv(r + 0.0001)
        total_dv01 = npv_bump - npv   # correct sign: +ve for PAYER, -ve for RECEIVER

        return {
            "trade_id":        trade.trade_id,
            "swap_npv":        npv,
            "fixed_npv":       sum(_leg_pv(l, r, ann, disc) for l in trade.legs[1:]),
            "float_npv":      -sum(_leg_pv(l, r, ann, disc) for l in trade.legs[:1]),
            "dv01":            total_dv01,
            "duration":        round(T / 2.0, 4),
            "par_rate":        r,
            "pv01":            total_dv01,
            "cr01":            0.0,
            "jump_to_default": 0.0,
            "delta":           0.0,
            "gamma":           0.0,
            "vega":            0.0,
            "theta":           0.0,
            "rho":             total_dv01,
            "convexity":       0.0,
            "clean_price":     0.0,
            "accrued":         0.0,
        }
    except Exception as e:
        return {"trade_id": trade.trade_id, "error": str(e), "swap_npv": None}


def _price_fixed_float(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a standard FIXED_FLOAT or FLOAT_FIXED IRS using QuantLib VanillaSwap."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    fl  = trade.fixed_leg
    flt = trade.float_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal[flt.calendar]
    dc_fixed = ql_dc[fl.day_count]
    dc_float = ql_dc[flt.day_count]
    bdc_f    = ql_bdc[fl.bdc]
    bdc_flt  = ql_bdc[flt.bdc]

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    eff = to_ql(fl.start_date); mat = to_ql(fl.end_date)

    fixed_sch = ql.Schedule(eff, mat, ql.Period(ql_freq[fl.frequency]),
                            calendar, bdc_f, bdc_f, ql.DateGeneration.Forward, False)
    float_sch = ql.Schedule(eff, mat, ql.Period(ql_freq[flt.frequency]),
                            calendar, bdc_flt, bdc_flt, ql.DateGeneration.Forward, False)
    sofr_idx  = ql.IborIndex(flt.index_name, ql.Period(flt.index_tenor_m, ql.Months),
                             flt.fixing_lag, ql.USDCurrency(), calendar, bdc_flt,
                             False, dc_float, sofr)
    short_rate  = float(curve_df["zero_rate"].iloc[0])
    eff_for_fix = eff
    for _ in range(36):
        fix_dt = calendar.advance(eff_for_fix, -flt.fixing_lag, ql.Days, ql.Preceding)
        if fix_dt > ql_val: break
        sofr_idx.addFixing(fix_dt, short_rate, True)
        eff_for_fix = calendar.advance(eff_for_fix, ql.Period(ql_freq[flt.frequency]), bdc_flt)

    stype    = ql.VanillaSwap.Payer if getattr(trade.direction, 'value', trade.direction) == "PAYER" else ql.VanillaSwap.Receiver
    notional = fl.notional; fxd_rate = fl.coupon_rate
    swap = ql.VanillaSwap(stype, notional, fixed_sch, fxd_rate, dc_fixed,
                          float_sch, sofr_idx, flt.spread, dc_float)
    swap.setPricingEngine(ql.DiscountingSwapEngine(sofr))
    swap_npv  = swap.NPV(); fixed_npv = swap.fixedLegNPV()
    float_npv = swap.floatingLegNPV(); par_rate = swap.fairRate()

    # DV01: +1bp parallel shift
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    sofr_idx_b = ql.IborIndex(flt.index_name, ql.Period(flt.index_tenor_m, ql.Months),
                              flt.fixing_lag, ql.USDCurrency(), calendar, bdc_flt,
                              False, dc_float, sofr_b)
    eff_for_fix = eff
    for _ in range(36):
        fix_dt = calendar.advance(eff_for_fix, -flt.fixing_lag, ql.Days, ql.Preceding)
        if fix_dt > ql_val: break
        sofr_idx_b.addFixing(fix_dt, short_rate, True)
        eff_for_fix = calendar.advance(eff_for_fix, ql.Period(ql_freq[flt.frequency]), bdc_flt)
    swap_b = ql.VanillaSwap(stype, notional, fixed_sch, fxd_rate, dc_fixed,
                            float_sch, sofr_idx_b, flt.spread, dc_float)
    swap_b.setPricingEngine(ql.DiscountingSwapEngine(sofr_b))
    dv01     = swap_b.NPV() - swap_npv
    duration = dv01 / (notional * 1e-4)
    pv01     = dv01 / notional * 1_000_000

    return dict(fixed_npv=fixed_npv, float_npv=float_npv, swap_npv=swap_npv,
                par_rate=par_rate, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_fixed_fixed(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a FIXED_FIXED swap: PV(receive leg[1]) - PV(pay leg[0])."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val
    sofr = build_sofr_curve(ql_val, curve_df)

    def leg_pv(leg, crv):
        dc   = ql_dc.get(leg.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
        freq = ql_freq.get(leg.frequency, ql.Semiannual)
        bdc  = ql_bdc.get(leg.bdc, ql.ModifiedFollowing)
        cal  = ql_cal.get(leg.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
        eff  = ql.Date(leg.start_date.day, leg.start_date.month, leg.start_date.year)
        mat  = ql.Date(leg.end_date.day, leg.end_date.month, leg.end_date.year)
        sch  = ql.Schedule(eff, mat, ql.Period(freq), cal, bdc, bdc,
                           ql.DateGeneration.Forward, False)
        fl   = ql.FixedRateLeg(sch, dc, [leg.notional], [leg.coupon_rate])
        return ql.CashFlows.npv(fl, crv, True)

    leg0, leg1 = trade.legs[0], trade.legs[1]
    pv0 = leg_pv(leg0, sofr)   # pay leg
    pv1 = leg_pv(leg1, sofr)   # receive leg
    swap_npv = pv1 - pv0

    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    dv01 = (leg_pv(leg1, sofr_b) - leg_pv(leg0, sofr_b)) - swap_npv
    notional = leg0.notional
    duration = dv01 / (notional * 1e-4)
    pv01 = dv01 / notional * 1_000_000

    return dict(fixed_npv=pv0, float_npv=pv1, swap_npv=swap_npv,
                par_rate=_NAN, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_float_float(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a FLOAT_FLOAT basis swap: NPV is PV of spread on leg[0]."""
    import math
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val
    sofr = build_sofr_curve(ql_val, curve_df)

    leg0 = trade.legs[0]   # SOFR1M + spread (pay)
    leg1 = trade.legs[1]   # SOFR3M (receive)
    notional = leg0.notional

    # For a basis swap both float legs have near-equal PV at par.
    # The NPV is essentially PV of the spread payments on leg0.
    # Use annuity factor from SOFR discount curve.
    ql_mat = ql.Date(leg0.end_date.day, leg0.end_date.month, leg0.end_date.year)
    tenor_frac = (leg0.end_date - leg0.start_date).days / 365.0
    r_avg = sofr.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()
    if r_avg > 0 and tenor_frac > 0:
        annuity = notional * (1.0 - math.exp(-r_avg * tenor_frac)) / r_avg
    else:
        annuity = notional * tenor_frac

    swap_npv = leg0.spread * annuity  # PV of spread payments (positive = receiver benefits)

    # DV01: bump SOFR
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    r_b    = sofr_b.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()
    if r_b > 0 and tenor_frac > 0:
        annuity_b = notional * (1.0 - math.exp(-r_b * tenor_frac)) / r_b
    else:
        annuity_b = notional * tenor_frac
    dv01 = leg0.spread * annuity_b - swap_npv
    duration = dv01 / (notional * 1e-4)
    pv01 = dv01 / notional * 1_000_000

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=swap_npv,
                par_rate=_NAN, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


# ─── Bond ─────────────────────────────────────────────────────────────────────

def _price_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a FixedRateBond using conventions stored on bond_leg."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl = trade.bond_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal[bl.calendar]
    dc       = ql_dc[bl.day_count]
    bdc      = ql_bdc[bl.bdc]
    freq     = ql_freq[bl.frequency]

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    issue_dt = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt = to_ql(bl.start_date); mat_dt = to_ql(bl.end_date)

    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(
        bl.settlement_days, bl.notional, schedule,
        [bl.coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond.setPricingEngine(ql.DiscountingBondEngine(sofr))

    dirty_pct = bond.dirtyPrice()
    clean_pct = bond.cleanPrice()
    npv_usd   = bond.NPV()
    accrued   = (dirty_pct - clean_pct) / 100.0 * bl.notional
    dc_yield  = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = bond.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        ytm = _NAN

    # DV01: +1bp parallel shift
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    bond_p1 = ql.FixedRateBond(
        bl.settlement_days, bl.notional, schedule,
        [bl.coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond_p1.setPricingEngine(ql.DiscountingBondEngine(sofr_b))
    npv_p1   = bond_p1.NPV()
    raw_dv01 = npv_p1 - npv_usd
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    dv01     = sign * raw_dv01
    duration = dv01 / (bl.notional * 1e-4)
    pv01     = dv01 / bl.notional * 1_000_000

    # Convexity: central difference (P_up + P_down - 2P) / (P * Δr²)
    bumped_m1 = curve_df.copy(); bumped_m1["zero_rate"] -= 1e-4
    sofr_m1   = build_sofr_curve(ql_val, bumped_m1)
    bond_m1   = ql.FixedRateBond(
        bl.settlement_days, bl.notional, schedule,
        [bl.coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond_m1.setPricingEngine(ql.DiscountingBondEngine(sofr_m1))
    npv_m1    = bond_m1.NPV()
    convexity = (npv_m1 + npv_p1 - 2.0 * npv_usd) / (npv_usd * 1e-8) if npv_usd != 0.0 else _NAN

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=npv_usd,
                par_rate=ytm, clean_price=clean_pct, accrued=accrued,
                dv01=dv01, duration=duration, pv01=pv01, convexity=convexity,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


# ─── Callable Bond ────────────────────────────────────────────────────────────

def _price_callable_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a CallableFixedRateBond using HullWhite model + trinomial tree engine.

    legs[0] = BondLeg  (BOND)   — coupon schedule, notional, day count
    legs[1] = OptionLeg (OPTION) — option_type CALL/PUT, exercise_type, strike, vol

    Returns:
      swap_npv    = callable bond full (dirty) NPV in USD
      clean_price = clean price in USD (not % of par)
      par_rate    = YTM of the callable bond (from clean price)
      dv01        = dollar rate sensitivity to +1bp SOFR shift
      duration    = effective duration (years) = dv01 / (notional × 1e-4)
      delta       = embedded option value: bullet_npv − callable_npv for CALLABLE
                    callable_npv − bullet_npv for PUTABLE (always ≥ 0)
      vega        = d(NPV)/d(vol) via +1% HullWhite sigma bump
      convexity   = (P_up + P_down − 2P) / (P × (1bp)²)
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl = trade.bond_leg
    ol = trade.option_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(bl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc       = ql_dc.get(bl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    bdc      = ql_bdc.get(bl.bdc, ql.ModifiedFollowing)
    freq     = ql_freq.get(bl.frequency, ql.Semiannual)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    issue_dt     = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt     = to_ql(bl.start_date)
    mat_dt       = to_ql(bl.end_date)
    first_call_dt = to_ql(ol.start_date)

    notional    = bl.notional
    coupon_rate = bl.coupon_rate
    strike      = ol.strike      # % of par (1.00 = par)
    hw_vol      = ol.vol         # HullWhite sigma (normal vol)
    call_type   = trade.call_type  # "CALLABLE" or "PUTABLE"

    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)

    # ── Bullet (non-callable) bond for comparison ──────────────────────────
    bullet = ql.FixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bullet.setPricingEngine(ql.DiscountingBondEngine(sofr))
    bullet_npv   = bullet.NPV()
    bullet_clean = bullet.cleanPrice()

    # ── Build call/put schedule ────────────────────────────────────────────
    call_schedule = ql.CallabilitySchedule()
    call_price    = ql.BondPrice(strike * 100.0, ql.BondPrice.Clean)
    ql_call_type  = ql.Callability.Call if call_type == "CALLABLE" else ql.Callability.Put

    for sch_dt in schedule:
        if sch_dt >= first_call_dt and sch_dt <= mat_dt:
            call_schedule.append(ql.Callability(call_price, ql_call_type, sch_dt))

    # ── Callable bond ──────────────────────────────────────────────────────
    callable_bond = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )

    hw_model = ql.HullWhite(sofr, a=0.1, sigma=hw_vol)
    engine   = ql.TreeCallableFixedRateBondEngine(hw_model, 40)
    callable_bond.setPricingEngine(engine)

    npv_usd   = callable_bond.NPV()
    dirty_pct = callable_bond.dirtyPrice()
    clean_pct = callable_bond.cleanPrice()
    clean_usd = clean_pct / 100.0 * notional
    accrued   = (dirty_pct - clean_pct) / 100.0 * notional

    # YTM of callable bond (uses clean price from callable pricer)
    dc_yield = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = callable_bond.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        try:
            ytm = bullet.bondYield(dc_yield, ql.Compounded, freq)
        except Exception:
            ytm = _NAN

    # ── Embedded option value ──────────────────────────────────────────────
    # For CALLABLE: option hurts holder (reduces price) → delta > 0 for issuer
    # For PUTABLE:  option helps holder (increases price) → delta > 0 for holder
    if call_type == "CALLABLE":
        option_value = bullet_npv - npv_usd   # positive: call reduces bond value
    else:
        option_value = npv_usd - bullet_npv   # positive: put increases bond value

    # ── DV01 via +1bp parallel shift ──────────────────────────────────────
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)

    hw_b   = ql.HullWhite(sofr_b, a=0.1, sigma=hw_vol)
    eng_b  = ql.TreeCallableFixedRateBondEngine(hw_b, 40)
    cb_b   = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )
    cb_b.setPricingEngine(eng_b)
    npv_p1   = cb_b.NPV()
    raw_dv01 = npv_p1 - npv_usd
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    dv01     = sign * raw_dv01
    duration = dv01 / (notional * 1e-4)
    pv01     = dv01 / notional * 1_000_000

    # ── Convexity ─────────────────────────────────────────────────────────
    bumped_m1 = curve_df.copy(); bumped_m1["zero_rate"] -= 1e-4
    sofr_m1   = build_sofr_curve(ql_val, bumped_m1)
    hw_m1     = ql.HullWhite(sofr_m1, a=0.1, sigma=hw_vol)
    eng_m1    = ql.TreeCallableFixedRateBondEngine(hw_m1, 40)
    cb_m1     = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )
    cb_m1.setPricingEngine(eng_m1)
    npv_m1    = cb_m1.NPV()
    convexity = (npv_m1 + npv_p1 - 2.0 * npv_usd) / (npv_usd * 1e-8) if npv_usd != 0.0 else _NAN

    # ── Vega: d(NPV)/d(sigma) via +1% sigma bump ──────────────────────────
    vol_bump = hw_vol + 0.01
    hw_vg    = ql.HullWhite(sofr, a=0.1, sigma=vol_bump)
    eng_vg   = ql.TreeCallableFixedRateBondEngine(hw_vg, 40)
    cb_vg    = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )
    cb_vg.setPricingEngine(eng_vg)
    vega = cb_vg.NPV() - npv_usd   # per 1% vol increase

    return dict(fixed_npv=bullet_npv, float_npv=_NAN, swap_npv=npv_usd,
                par_rate=ytm, clean_price=clean_usd, accrued=accrued,
                dv01=dv01, duration=duration, pv01=pv01, convexity=convexity,
                vega=vega, theta=_NAN, delta=option_value,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


# ─── Swaption ─────────────────────────────────────────────────────────────────

def _price_option(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a European swaption using Black / Bachelier flat vol engine.
    ol.vol_type: "LOGNORMAL" -> BlackSwaptionEngine  |  "NORMAL" -> BachelierSwaptionEngine
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()
    ol = trade.option_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(ol.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc_fixed = ql_dc.get(ol.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    dc_float = ql.Actual360()
    bdc      = ql_bdc.get(ol.bdc, ql.ModifiedFollowing)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    expiry_dt = to_ql(ol.start_date)
    mat_dt    = to_ql(ol.end_date)

    fixed_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Semiannual),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    float_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Quarterly),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    sofr_idx  = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                             ql.USDCurrency(), calendar, bdc, False, dc_float, sofr)

    is_payer  = (ol.option_type == "PAYER_SWAPTION")
    stype     = ql.VanillaSwap.Payer if is_payer else ql.VanillaSwap.Receiver
    underlying = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                                float_sch, sofr_idx, 0.0, dc_float)
    underlying.setPricingEngine(ql.DiscountingSwapEngine(sofr))
    try:
        atm_rate = underlying.fairRate()
    except Exception:
        atm_rate = _NAN

    exercise = ql.EuropeanExercise(expiry_dt)
    swaption = ql.Swaption(underlying, exercise)
    flat_vol = ql.QuoteHandle(ql.SimpleQuote(ol.vol))
    shift    = ol.vol_shift if ol.vol_shift else 0.0

    def _make_engine(crv, vol_qh):
        if ol.vol_type == "NORMAL":
            return ql.BachelierSwaptionEngine(crv, vol_qh)
        return ql.BlackSwaptionEngine(crv, vol_qh, ql.Actual365Fixed(), shift)

    swaption.setPricingEngine(_make_engine(sofr, flat_vol))
    premium = swaption.NPV()
    sign    = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SELL" else 1.0
    net_pnl = sign * premium

    # DV01: +1bp parallel SOFR shift
    bumped  = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b  = build_sofr_curve(ql_val, bumped)
    sofr_idx_b = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                              ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_b)
    under_b = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                             float_sch, sofr_idx_b, 0.0, dc_float)
    swptn_b = ql.Swaption(under_b, exercise)
    swptn_b.setPricingEngine(_make_engine(sofr_b, flat_vol))
    dv01     = sign * (swptn_b.NPV() - premium)
    duration = dv01 / (ol.notional * 1e-4)
    pv01     = dv01 / ol.notional * 1_000_000

    # Vega: +1% flat vol shift
    vol_up  = ql.QuoteHandle(ql.SimpleQuote(ol.vol + 0.01))
    under_v = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                             float_sch, sofr_idx, 0.0, dc_float)
    swptn_v = ql.Swaption(under_v, exercise)
    swptn_v.setPricingEngine(_make_engine(sofr, vol_up))
    vega = sign * (swptn_v.NPV() - premium)

    # Theta: reprice with evaluationDate advanced by 1 calendar day
    theta = _NAN
    try:
        ql_val_1d = ql_val + 1
        if expiry_dt.serialNumber() > ql_val_1d.serialNumber():
            ql.Settings.instance().evaluationDate = ql_val_1d
            sofr_1d     = build_sofr_curve(ql_val_1d, curve_df)
            sofr_idx_1d = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                                       ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_1d)
            under_1d = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                                      float_sch, sofr_idx_1d, 0.0, dc_float)
            swptn_1d = ql.Swaption(under_1d, exercise)
            swptn_1d.setPricingEngine(_make_engine(sofr_1d, flat_vol))
            premium_1d = swptn_1d.NPV()
            theta = sign * (premium_1d - premium)
    except Exception:
        pass
    finally:
        ql.Settings.instance().evaluationDate = ql_val  # always restore

    # Delta: d_premium / d_underlying_rate  (lower strike by 1bp)
    delta = _NAN
    try:
        strike_lo = ol.strike - 1e-4
        under_lo  = ql.VanillaSwap(stype, ol.notional, fixed_sch, strike_lo, dc_fixed,
                                   float_sch, sofr_idx, 0.0, dc_float)
        swptn_lo  = ql.Swaption(under_lo, exercise)
        swptn_lo.setPricingEngine(_make_engine(sofr, flat_vol))
        delta = sign * (swptn_lo.NPV() - premium) / 1e-4
    except Exception:
        pass

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=net_pnl,
                par_rate=atm_rate, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=vega, theta=theta, delta=delta,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


# ─── InterestRateSwaption ─────────────────────────────────────────────────────

def _price_irs_swaption(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price an InterestRateSwaption using the QuantLib Black/Bachelier swaption
    engine.  The underlying swap is constructed from legs[0] (FixedLeg) and
    legs[1] (FloatLeg); the option terms come from legs[2] (OptionLeg).

    Greeks computed:
      swap_npv  — option premium (signed by direction)
      fixed_npv — PV of underlying fixed leg (at expiry strike)
      float_npv — PV of underlying float leg (at current fwd rates)
      dv01      — parallel +1bp SOFR shift
      vega      — +1% flat vol shift
      theta     — 1-day time decay
      delta     — d(premium)/d(swap_rate) via strike -1bp
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    fl  = trade.fixed_leg     # legs[0]
    fll = trade.float_leg     # legs[1]
    ol  = trade.option_leg    # legs[2]

    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(fl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc_fixed = ql_dc.get(fl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    dc_float = ql.Actual360()
    bdc      = ql_bdc.get(fl.bdc, ql.ModifiedFollowing)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    expiry_dt = to_ql(ol.start_date)   # option expiry = underlying effective date
    mat_dt    = to_ql(fl.end_date)     # underlying swap maturity

    fixed_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Semiannual),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    float_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Quarterly),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    sofr_idx  = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                             ql.USDCurrency(), calendar, bdc, False, dc_float, sofr)

    is_payer  = (ol.option_type == "PAYER_SWAPTION")
    stype     = ql.VanillaSwap.Payer if is_payer else ql.VanillaSwap.Receiver
    notl      = fl.notional
    strike    = ol.strike

    underlying = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                                float_sch, sofr_idx, 0.0, dc_float)
    underlying.setPricingEngine(ql.DiscountingSwapEngine(sofr))

    try:
        atm_rate   = underlying.fairRate()
        fixed_npv_ = underlying.fixedLegNPV()
        float_npv_ = underlying.floatingLegNPV()
    except Exception:
        atm_rate = fixed_npv_ = float_npv_ = _NAN

    exercise  = ql.EuropeanExercise(expiry_dt)
    swaption  = ql.Swaption(underlying, exercise)
    flat_vol  = ql.QuoteHandle(ql.SimpleQuote(ol.vol))
    shift     = ol.vol_shift if ol.vol_shift else 0.0

    def _engine(crv, vqh):
        if ol.vol_type == "NORMAL":
            return ql.BachelierSwaptionEngine(crv, vqh)
        return ql.BlackSwaptionEngine(crv, vqh, ql.Actual365Fixed(), shift)

    swaption.setPricingEngine(_engine(sofr, flat_vol))
    premium  = swaption.NPV()
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SELL" else 1.0
    net_pnl  = sign * premium

    # DV01: +1bp SOFR curve shift
    bumped  = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b  = build_sofr_curve(ql_val, bumped)
    sofr_ib = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                           ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_b)
    under_b = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                             float_sch, sofr_ib, 0.0, dc_float)
    swptn_b = ql.Swaption(under_b, exercise)
    swptn_b.setPricingEngine(_engine(sofr_b, flat_vol))
    dv01     = sign * (swptn_b.NPV() - premium)
    duration = dv01 / (notl * 1e-4)
    pv01     = dv01 / notl * 1_000_000

    # Vega: +1% vol shift
    vol_up  = ql.QuoteHandle(ql.SimpleQuote(ol.vol + 0.01))
    under_v = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                             float_sch, sofr_idx, 0.0, dc_float)
    swptn_v = ql.Swaption(under_v, exercise)
    swptn_v.setPricingEngine(_engine(sofr, vol_up))
    vega = sign * (swptn_v.NPV() - premium)

    # Theta: 1-day time decay
    theta = _NAN
    try:
        ql_val_1d = ql_val + 1
        if expiry_dt.serialNumber() > ql_val_1d.serialNumber():
            ql.Settings.instance().evaluationDate = ql_val_1d
            sofr_1d = build_sofr_curve(ql_val_1d, curve_df)
            sofr_i1 = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                                   ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_1d)
            under_1d = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                                      float_sch, sofr_i1, 0.0, dc_float)
            swptn_1d = ql.Swaption(under_1d, exercise)
            swptn_1d.setPricingEngine(_engine(sofr_1d, flat_vol))
            theta = sign * (swptn_1d.NPV() - premium)
    except Exception:
        pass
    finally:
        ql.Settings.instance().evaluationDate = ql_val

    # Delta: d(premium)/d(swap_rate) via -1bp strike
    delta = _NAN
    try:
        under_lo = ql.VanillaSwap(stype, notl, fixed_sch, strike - 1e-4, dc_fixed,
                                  float_sch, sofr_idx, 0.0, dc_float)
        swptn_lo = ql.Swaption(under_lo, exercise)
        swptn_lo.setPricingEngine(_engine(sofr, flat_vol))
        delta = sign * (swptn_lo.NPV() - premium) / 1e-4
    except Exception:
        pass

    return dict(
        fixed_npv=fixed_npv_, float_npv=float_npv_, swap_npv=net_pnl,
        par_rate=atm_rate, clean_price=_NAN, accrued=_NAN,
        dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
        vega=vega, theta=theta, delta=delta,
        gamma=_NAN, rho=_NAN,
        cr01=_NAN, jump_to_default=_NAN, error="",
    )


# ─── Equity Swap ──────────────────────────────────────────────────────────────

def _price_equity_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analytical single-period equity total-return swap pricer.

    NPV = sign * (Notional*exp(-q*T) - Notional*exp(-(r+s)*T))
    sign = +1 if LONG (receive equity), -1 if SHORT.
    """
    import math

    el     = trade.equity_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr = build_sofr_curve(ql_val, curve_df)

    end_dt = el.end_date
    T = (date(end_dt.year, end_dt.month, end_dt.day) - val_dt).days / 365.0
    if T <= 0:
        return {**_NAN_ROW, "error": "Matured trade"}

    ql_mat = ql.Date(end_dt.day, end_dt.month, end_dt.year)
    r      = sofr.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()

    q        = el.dividend_yield
    s        = el.funding_spread
    notional = el.notional
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0

    pv_equity  = notional * math.exp(-q * T)
    pv_funding = notional * math.exp(-(r + s) * T)
    npv        = sign * (pv_equity - pv_funding)

    # Par rate: dividend yield (the equity yield component)
    par_rate = q

    # DV01: bump SOFR by +1bp
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    r_b    = sofr_b.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()
    pv_fund_b = notional * math.exp(-(r_b + s) * T)
    npv_b     = sign * (pv_equity - pv_fund_b)
    dv01      = npv_b - npv
    duration  = dv01 / (notional * 1e-4)
    pv01      = dv01 / notional * 1_000_000

    # Delta: $ per 1% move in underlying (TRS has 100% delta)
    delta = sign * notional * 0.01

    # Theta: reprice at T - 1 day
    T_1d      = max(1e-4, T - 1.0 / 365.0)
    pv_eq_1d  = notional * math.exp(-q * T_1d)
    pv_fun_1d = notional * math.exp(-(r + s) * T_1d)
    npv_1d    = sign * (pv_eq_1d - pv_fun_1d)
    theta     = npv_1d - npv

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=npv,
                par_rate=par_rate, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=theta, delta=delta,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


# ─── CDS ──────────────────────────────────────────────────────────────────────

def _price_cds(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    QuantLib CDS pricer: FlatHazardRate + MidPointCdsEngine.

    direction BUY  = protection buyer  (pays running spread)
    direction SELL = protection seller (receives running spread)
    """
    cl     = trade.protection_leg   # CDSProtectionLeg: reference entity, spread, recovery
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql.TARGET()
    dc       = ql.Actual360()

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    start_ql = to_ql(cl.start_date)
    mat_ql   = to_ql(cl.end_date)

    schedule = ql.Schedule(start_ql, mat_ql, ql.Period(ql.Quarterly),
                           calendar, ql.Following, ql.Following,
                           ql.DateGeneration.Forward, False)

    side     = ql.Protection.Buyer if getattr(trade.direction, 'value', trade.direction) == "BUY" else ql.Protection.Seller
    spread   = cl.credit_spread
    notional = cl.notional
    recovery = cl.recovery_rate
    hazard   = cl.hazard_rate if cl.hazard_rate > 0 else spread / (1.0 - recovery)

    dp_handle = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(ql_val, ql.QuoteHandle(ql.SimpleQuote(hazard)), ql.Actual365Fixed())
    )

    cds = ql.CreditDefaultSwap(side, notional, spread, schedule, ql.Following, dc)
    cds.setPricingEngine(ql.MidPointCdsEngine(dp_handle, recovery, sofr))
    npv = cds.NPV()
    try:
        fair_spread = cds.fairSpread()
    except Exception:
        fair_spread = _NAN

    # Premium leg NPV (spread payments) and protection leg NPV (contingent)
    try:
        premium_npv    = cds.premiumNPV()
        protection_npv = cds.protectionNPV()
    except Exception:
        premium_npv = protection_npv = _NAN

    # DV01: bump SOFR by +1bp
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    cds_b  = ql.CreditDefaultSwap(side, notional, spread, schedule, ql.Following, dc)
    cds_b.setPricingEngine(ql.MidPointCdsEngine(dp_handle, recovery, sofr_b))
    dv01     = cds_b.NPV() - npv
    duration = dv01 / (notional * 1e-4)
    pv01     = dv01 / notional * 1_000_000

    # CR01: bump credit spread by +1bp (implies new hazard rate)
    spread_b  = spread + 1e-4
    haz_b     = spread_b / (1.0 - recovery)
    dp_hand_b = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(ql_val, ql.QuoteHandle(ql.SimpleQuote(haz_b)), ql.Actual365Fixed())
    )
    cds_c = ql.CreditDefaultSwap(side, notional, spread, schedule, ql.Following, dc)
    cds_c.setPricingEngine(ql.MidPointCdsEngine(dp_hand_b, recovery, sofr))
    cr01 = cds_c.NPV() - npv

    # Jump-to-default: immediate default P&L
    lgd = (1.0 - recovery) * notional
    jump_to_default = lgd if getattr(trade.direction, 'value', trade.direction) == "BUY" else -lgd

    return dict(fixed_npv=premium_npv, float_npv=protection_npv, swap_npv=npv,
                par_rate=fair_spread, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=cr01, jump_to_default=jump_to_default, error="")


# ─── Equity Option ────────────────────────────────────────────────────────────

def _price_equity_option(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a vanilla equity option using Black-Scholes (European) or
    finite-difference PDE engine (American). Greeks: delta, gamma, vega, theta, rho.
    """
    import math as _math
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    ol = trade.option_leg
    if ol is None:
        return {**_NAN_ROW, "error": "No EquityOptionLeg found"}

    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    exp_dt = ol.end_date
    T_days = (exp_dt - val_dt).days
    T      = T_days / 365.0
    if T <= 0:
        return {**_NAN_ROW, "error": f"Expired option (T={T:.3f})"}

    sofr = build_sofr_curve(ql_val, curve_df)
    ql_exp = ql.Date(exp_dt.day, exp_dt.month, exp_dt.year)
    r = sofr.zeroRate(ql_exp, ql.Actual365Fixed(), ql.Continuous, ql.Annual).rate()

    S        = ol.initial_price
    K        = ol.strike
    q        = ol.dividend_yield
    vol      = ol.vol
    notional = ol.notional

    ql_opt  = ql.Option.Call if ol.option_type == "CALL" else ql.Option.Put
    payoff  = ql.PlainVanillaPayoff(ql_opt, K)

    if ol.exercise_type == "EUROPEAN":
        exercise = ql.EuropeanExercise(ql_exp)
    else:
        ql_start = ql_val + ql.Period(1, ql.Days)
        exercise = ql.AmericanExercise(ql_start, ql_exp)

    option = ql.VanillaOption(payoff, exercise)

    spot_q  = ql.SimpleQuote(S)
    spot_h  = ql.QuoteHandle(spot_q)
    rf_ts   = ql.FlatForward(ql_val, r, ql.Actual365Fixed())
    rf_q    = ql.YieldTermStructureHandle(rf_ts)
    div_ts  = ql.FlatForward(ql_val, q, ql.Actual365Fixed())
    div_q   = ql.YieldTermStructureHandle(div_ts)
    vol_ts  = ql.BlackConstantVol(ql_val, ql.NullCalendar(), vol, ql.Actual365Fixed())
    vol_q   = ql.BlackVolTermStructureHandle(vol_ts)
    process = ql.BlackScholesMertonProcess(spot_h, div_q, rf_q, vol_q)

    if ol.exercise_type == "EUROPEAN":
        engine = ql.AnalyticEuropeanEngine(process)
    else:
        engine = ql.FdBlackScholesVanillaEngine(process, 100, 200)

    option.setPricingEngine(engine)

    try:
        unit_premium = option.NPV()
    except Exception as e:
        return {**_NAN_ROW, "error": f"Pricing failed: {e}"}

    n_shares = notional / S
    sign = 1.0 if trade.direction == TradeDirection.BUY else -1.0
    premium = sign * unit_premium * n_shares

    def _greek(fn):
        try:
            v = fn()
            return v if v is not None and not _math.isnan(v) else _NAN
        except Exception:
            return _NAN

    unit_delta = _greek(option.delta)
    unit_gamma = _greek(option.gamma)
    unit_theta = _greek(option.theta)

    # vega / rho not always available for FD engine — fall back to bump-reprice
    unit_vega = _greek(option.vega)
    if _math.isnan(unit_vega):
        try:
            vol_bump = vol + 0.01
            vol_ts2  = ql.BlackConstantVol(ql_val, ql.NullCalendar(), vol_bump, ql.Actual365Fixed())
            vol_q2   = ql.BlackVolTermStructureHandle(vol_ts2)
            p2       = ql.BlackScholesMertonProcess(spot_h, div_q, rf_q, vol_q2)
            opt2     = ql.VanillaOption(payoff, exercise)
            if ol.exercise_type == "EUROPEAN":
                opt2.setPricingEngine(ql.AnalyticEuropeanEngine(p2))
            else:
                opt2.setPricingEngine(ql.FdBlackScholesVanillaEngine(p2, 100, 200))
            unit_vega = opt2.NPV() - unit_premium   # per 1% vol move
        except Exception:
            unit_vega = _NAN

    unit_rho = _greek(option.rho)
    if _math.isnan(unit_rho):
        try:
            r_bump  = r + 0.0001
            rf_q2   = ql.YieldTermStructureHandle(
                          ql.FlatForward(ql_val, r_bump, ql.Actual365Fixed()))
            p2      = ql.BlackScholesMertonProcess(spot_h, div_q, rf_q2, vol_q)
            opt2    = ql.VanillaOption(payoff, exercise)
            if ol.exercise_type == "EUROPEAN":
                opt2.setPricingEngine(ql.AnalyticEuropeanEngine(p2))
            else:
                opt2.setPricingEngine(ql.FdBlackScholesVanillaEngine(p2, 100, 200))
            unit_rho = (opt2.NPV() - unit_premium) / 0.0001   # per unit rate move
        except Exception:
            unit_rho = _NAN

    # delta fallback: finite difference on spot if FD engine didn't provide it
    if _math.isnan(unit_delta):
        try:
            h = S * 0.001
            def _npv_at(s):
                sq = ql.QuoteHandle(ql.SimpleQuote(s))
                pr = ql.BlackScholesMertonProcess(sq, div_q, rf_q, vol_q)
                o  = ql.VanillaOption(payoff, exercise)
                if ol.exercise_type == "EUROPEAN":
                    o.setPricingEngine(ql.AnalyticEuropeanEngine(pr))
                else:
                    o.setPricingEngine(ql.FdBlackScholesVanillaEngine(pr, 50, 100))
                return o.NPV()
            unit_delta = (_npv_at(S + h) - _npv_at(S - h)) / (2 * h)
        except Exception:
            unit_delta = _NAN

    def _safe(v):
        return v if not _math.isnan(v) else _NAN

    delta = sign * (_safe(unit_delta) * n_shares)         if not _math.isnan(unit_delta) else _NAN
    gamma = sign * (_safe(unit_gamma) * n_shares)         if not _math.isnan(unit_gamma) else _NAN
    vega  = sign * (_safe(unit_vega)  * n_shares * 0.01)  if not _math.isnan(unit_vega)  else _NAN
    theta = sign * (_safe(unit_theta) * n_shares)         if not _math.isnan(unit_theta) else _NAN
    rho   = sign * (_safe(unit_rho)   * n_shares * 0.0001) if not _math.isnan(unit_rho) else _NAN
    dv01  = rho

    return dict(
        fixed_npv=_NAN, float_npv=_NAN, swap_npv=premium,
        par_rate=_NAN, clean_price=_NAN, accrued=_NAN,
        dv01=dv01, duration=_NAN, pv01=_NAN, convexity=_NAN,
        vega=vega, theta=theta, delta=delta,
        gamma=gamma, rho=rho,
        cr01=_NAN, jump_to_default=_NAN, error="",
    )


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def _price_one(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Dispatch to the correct pricer based on trade type."""
    from models.interest_rate_swap import InterestRateSwap
    from models.cross_currency_swap import CrossCurrencySwap
    from models.interest_rate_swaption import InterestRateSwaption
    from models.callable_bond import CallableBond
    try:
        if isinstance(trade, EquityOptionTrade):
            return _price_equity_option(trade, curve_df)
        if isinstance(trade, EquitySwap):
            return _price_equity_swap(trade, curve_df)
        elif isinstance(trade, CreditDefaultSwap):
            return _price_cds(trade, curve_df)
        elif isinstance(trade, InterestRateSwaption):
            return _price_irs_swaption(trade, curve_df)
        elif isinstance(trade, OptionTrade):
            return _price_option(trade, curve_df)
        elif isinstance(trade, (VanillaSwap, InterestRateSwap)):
            # Multi-currency legs → cross-currency pricer
            ccys = {l.currency for l in trade.legs if l.currency}
            if len(ccys) > 1:
                return _price_xccy(trade, curve_df)
            return _price_swap(trade, curve_df)
        elif isinstance(trade, CrossCurrencySwap):
            return _price_xccy(trade, curve_df)
        elif isinstance(trade, CallableBond):
            return _price_callable_bond(trade, curve_df)
        elif isinstance(trade, Bond):
            return _price_bond(trade, curve_df)
        return {**_NAN_ROW, "error": f"Unknown: {type(trade).__name__}"}
    except Exception as exc:
        return {**_NAN_ROW, "error": str(exc)[:200]}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PANDAS UDF
# ══════════════════════════════════════════════════════════════════════════════

_curve_broadcast = None


def make_price_udf():
    bc = _curve_broadcast   # capture broadcast at call time

    @pandas_udf(RESULT_SCHEMA)
    def price_udf(trade_json: pd.Series) -> pd.DataFrame:
        import sys as _sys, os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from manage_trades import _price_one as _po   # import after path fix
        from models import TradeBase, EquityOptionTrade  # noqa: PLC0415
        curve_df = bc.value
        return pd.DataFrame([_po(TradeBase.fromJson(js), curve_df)
                              for js in trade_json])
    return price_udf


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SPARK RUNNER + ITERATIVE CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_pricing(n_irs: int = 50, n_bonds: int = 50, n_opts: int = 50,
                n_eq: int = 50, n_cds: int = 50, n_eq_opt: int = 100,
                tol: float = 1.0, max_iter: int = 5,
                trades: Optional[List] = None, run_id: str = "RUN-1"):
    """
    Price all instrument types in parallel with PySpark until convergence.
    Persists trades and results to SQLite via TradeRepository.

    Parameters
    ----------
    trades  : if provided, skip generation and use these trades directly
    run_id  : label for this pricing run stored in PricingResult table
    """
    global _curve_broadcast

    spark = (SparkSession.builder
             .appName("MultiInstrumentPricer")
             .master("local[*]")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.driver.extraJavaOptions",   "-Dlog4j2.rootLogger.level=OFF")
             .config("spark.executor.extraJavaOptions", "-Dlog4j2.rootLogger.level=OFF")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("OFF")
    n_cores = spark.sparkContext.defaultParallelism

    _real_stderr     = sys.stderr
    curve_df         = make_curve_df()
    _curve_broadcast = spark.sparkContext.broadcast(curve_df)

    if trades is not None:
        all_trades = trades
    else:
        swaps      = make_irs_data(n_irs)
        bonds      = make_bond_data(n_bonds)
        options    = make_option_data(n_opts)
        equities   = make_equity_data(n_eq)
        cdss       = make_cds_data(n_cds)
        eq_options = make_equity_option_data(n_eq_opt)
        all_trades = swaps + bonds + options + equities + cdss + eq_options

    def _instrument(t):
        if isinstance(t, EquityOptionTrade):    return "EQ_OPT"
        if isinstance(t, EquitySwap):           return "EQ_SWAP"
        if isinstance(t, CreditDefaultSwap):    return "CDS"
        if isinstance(t, InterestRateSwaption): return "IRS_SWPTN"
        if isinstance(t, OptionTrade):          return "SWAPTION"
        if isinstance(t, VanillaSwap) or isinstance(t, InterestRateSwap):  return "IRS"
        if isinstance(t, CallableBond):         return "CBOND"
        return "BOND"

    def _notional(t):
        return t.legs[0].notional

    def _coupon(t):
        if isinstance(t, EquityOptionTrade):
            ol = t.option_leg
            return ol.strike if ol else 0.0
        if isinstance(t, InterestRateSwaption): return t.fixed_leg.coupon_rate
        if isinstance(t, (VanillaSwap, InterestRateSwap)):
            fl = t.fixed_leg
            return fl.coupon_rate if fl is not None else (t.legs[0].spread if t.legs else 0.0)
        if isinstance(t, Bond):              return t.bond_leg.coupon_rate
        if isinstance(t, CallableBond):      return t.bond_leg.coupon_rate
        if isinstance(t, OptionTrade):       return t.option_leg.strike
        if isinstance(t, EquitySwap):        return t.equity_leg.initial_price
        if isinstance(t, CreditDefaultSwap): return t.credit_leg.credit_spread
        return 0.0

    def _swap_subtype(t):
        return getattr(t, 'swap_subtype', '') or ''

    rows = [(t.trade_id, _instrument(t), t.book, t.counterparty,
             getattr(t, 'trader', ''), str(t.direction.value), t.tenor_y,
             _notional(t), _coupon(t), _swap_subtype(t), t.toJson())
            for t in all_trades]
    trades_df = (spark.createDataFrame(rows, schema=TRADE_SCHEMA)
                 .repartitionByRange(n_cores, "trade_id"))
    price_udf = make_price_udf()

    def _execute() -> list:
        # Return plain Python list-of-dicts — completely avoids pandas BlockManager.
        # When PyCharm's display helper pre-imports NumPy, PySpark worker processes
        # reload it (double-import), which corrupts pandas' internal _NoValue
        # sentinels.  Row.asDict() and list comprehensions are pure Python; the
        # caller creates a fresh pd.DataFrame only where strictly needed.
        spark_df = (trades_df
                    .withColumn("r", price_udf("trade_json"))
                    .select(
                        "trade_id", "instrument", "book", "counterparty",
                        "direction", "tenor_y", "notional", "coupon_rate",
                        "swap_subtype",
                        F.col("r.fixed_npv").alias("fixed_npv"),
                        F.col("r.float_npv").alias("float_npv"),
                        F.col("r.swap_npv").alias("swap_npv"),
                        F.col("r.par_rate").alias("par_rate"),
                        F.col("r.clean_price").alias("clean_price"),
                        F.col("r.accrued").alias("accrued"),
                        F.col("r.dv01").alias("dv01"),
                        F.col("r.duration").alias("duration"),
                        F.col("r.pv01").alias("pv01"),
                        F.col("r.convexity").alias("convexity"),
                        F.col("r.vega").alias("vega"),
                        F.col("r.theta").alias("theta"),
                        F.col("r.delta").alias("delta"),
                        F.col("r.gamma").alias("gamma"),
                        F.col("r.rho").alias("rho"),
                        F.col("r.cr01").alias("cr01"),
                        F.col("r.jump_to_default").alias("jump_to_default"),
                        F.col("r.error").alias("error"),
                    ))
        return [row.asDict() for row in spark_df.collect()]

    n_total = len(all_trades)
    sep = "-" * 80
    print(sep)
    if trades is None:
        print(f"  Workers: {n_cores} cores  |  IRS:{n_irs}  Bonds:{n_bonds}  "
              f"Opts:{n_opts}  EQ:{n_eq}  CDS:{n_cds}  EqOpt:{n_eq_opt}  Total:{n_total}  |  Tol:${tol:.2f}")
    else:
        print(f"  Workers: {n_cores} cores  |  {n_total} trades (from DB)  |  run_id:{run_id}  |  Tol:${tol:.2f}")
    print(sep)

    prev_npv = None; result = None
    for it in range(1, max_iter + 1):
        sys.stderr = open(os.devnull, "w")
        t0 = time.time(); result = _execute(); elapsed = time.time() - t0
        sys.stderr.close(); sys.stderr = _real_stderr
        _v = lambda r: float(r["swap_npv"]) if r.get("swap_npv") is not None else float("nan")
        curr_npv = np.array([_v(r) for r in result], dtype=float)
        if prev_npv is not None:
            deltas = np.abs(curr_npv - prev_npv)
            max_d  = float(np.nanmax(deltas)); avg_d = float(np.nanmean(deltas))
            print(f"  Iter {it}  {elapsed:5.1f}s  max|dNPV|=${max_d:>10.4f}  "
                  f"mean|dNPV|=${avg_d:>10.4f}", flush=True)
            if max_d < tol:
                print(f"  Converged after {it} iterations\n", flush=True); break
        else:
            print(f"  Iter {it}  {elapsed:5.1f}s  {n_total} trades on "
                  f"{n_cores} parallel workers", flush=True)
        prev_npv = curr_npv.copy()

    # ── Build PricingResult objects ────────────────────────────────────────
    pricing_results_list = build_pricing_results(all_trades, result)

    # ── Persist to SQLite TradeRepository ─────────────────────────────────
    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir  = os.path.dirname(_examples_dir)
    db_path       = os.path.join(_project_dir, "trades.db")
    repo = TradeRepository(db_path)
    if trades is None:
        repo.upsert_many(all_trades)
    repo.save_results(pricing_results_list, run_id=run_id)
    repo.close()

    sys.stderr = open(os.devnull, "w"); spark.stop()
    sys.stderr.close(); sys.stderr = _real_stderr
    return result, all_trades, pricing_results_list


_NAN_F = float("nan")


def _fill_pricing_row(pr, row) -> None:
    """Fill PricingResult from a df row (pandas Series)."""
    import numpy as np
    def _g(col):
        v = row.get(col, _NAN_F) if hasattr(row, 'get') else getattr(row, col, _NAN_F)
        return _NAN_F if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    def _gs(col):
        v = row.get(col, '') if hasattr(row, 'get') else getattr(row, col, '')
        return '' if v is None else str(v)

    pr.npv         = _g('swap_npv')
    pr.fixed_npv   = _g('fixed_npv')
    pr.float_npv   = _g('float_npv')
    pr.par_rate    = _g('par_rate')
    pr.clean_price = _g('clean_price')
    pr.accrued     = _g('accrued')
    pr.premium     = _g('swap_npv') if pr.instrument in ('SWAPTION', 'EQ_OPT', 'IRS_SWPTN') else _NAN_F
    pr.dv01        = _g('dv01')
    pr.duration    = _g('duration')
    pr.pv01        = _g('pv01')
    pr.convexity   = _g('convexity')
    pr.vega        = _g('vega')
    pr.theta       = _g('theta')
    pr.delta       = _g('delta')
    pr.gamma       = _g('gamma')
    pr.rho         = _g('rho')
    pr.cr01        = _g('cr01')
    pr.jump_to_default = _g('jump_to_default')
    pr.error       = _gs('error')


def build_pricing_results(all_trades: list, records: list) -> List[PricingResult]:
    """Build flat PricingResult objects from trade objects + list-of-dicts result."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    # records is a plain Python list[dict] — no pandas involved.
    if records and isinstance(records, list) and isinstance(records[0], dict):
        df_idx = {r["trade_id"]: r for r in records}
    elif hasattr(records, "columns"):
        # Fallback: accept a DataFrame too (terminal/test usage)
        df_idx = {r["trade_id"]: r for r in records.to_dict("records")}
    else:
        df_idx = {}

    results = []
    for trade in all_trades:
        tid = trade.trade_id
        pr = PricingResult(
            trade_id       = tid,
            book           = trade.book,
            counterparty   = trade.counterparty,
            valuation_date = str(trade.valuation_date),
            direction      = str(trade.direction.value) if hasattr(trade.direction, 'value') else str(trade.direction),
            run_timestamp  = ts,
        )

        if isinstance(trade, EquityOptionTrade):
            pr.instrument = "EQ_OPT"
            pr.leg_count  = len(trade.legs)
            pr.leg_types  = ",".join(l.leg_type for l in trade.legs)
            ol = trade.option_leg
            if ol:
                pr.notional          = ol.notional
                pr.start_date        = str(ol.start_date)
                pr.end_date          = str(ol.end_date)
                pr.currency          = ol.currency
                pr.underlying_ticker = ol.underlying_ticker
                pr.initial_price     = ol.initial_price
                pr.dividend_yield    = ol.dividend_yield
                pr.strike            = ol.strike
                pr.option_type       = ol.option_type
                pr.exercise_type     = ol.exercise_type
                pr.vol               = ol.vol
                pr.coupon_rate       = ol.strike
            pr.tenor_y = trade.tenor_y

        elif isinstance(trade, EquitySwap):
            pr.instrument = "EQ_SWAP"
            pr.leg_count  = len(trade.legs)
            pr.leg_types  = ",".join(l.leg_type for l in trade.legs)
            el = next((l for l in trade.legs if l.leg_type == "EQUITY"), None)
            if el:
                pr.notional           = el.notional
                pr.start_date         = str(el.start_date)
                pr.end_date           = str(el.end_date)
                pr.currency           = el.currency
                pr.underlying_ticker  = getattr(el, 'underlying_ticker', '')
                pr.initial_price      = getattr(el, 'initial_price', _NAN_F)
                pr.dividend_yield     = getattr(el, 'dividend_yield', _NAN_F)
                pr.spread             = getattr(el, 'funding_spread', _NAN_F)
            pr.tenor_y = trade.tenor_y

        elif isinstance(trade, CreditDefaultSwap):
            pr.instrument = "CDS"
            pr.leg_count  = len(trade.legs)
            pr.leg_types  = ",".join(l.leg_type for l in trade.legs)
            cl = trade.protection_leg   # CDSProtectionLeg: reference entity, spread, recovery
            pr.notional          = cl.notional
            pr.start_date        = str(cl.start_date)
            pr.end_date          = str(cl.end_date)
            pr.currency          = cl.currency
            pr.reference_entity  = getattr(cl, 'reference_entity', '')
            pr.credit_spread     = getattr(cl, 'credit_spread', _NAN_F)
            pr.recovery_rate     = getattr(cl, 'recovery_rate', _NAN_F)
            pr.seniority         = getattr(cl, 'seniority', '')
            pr.doc_clause        = getattr(cl, 'doc_clause', '')
            pr.coupon_rate       = getattr(cl, 'credit_spread', _NAN_F)
            pr.tenor_y = trade.tenor_y

        elif isinstance(trade, InterestRateSwaption):
            pr.instrument    = "IRS_SWPTN"
            pr.swap_subtype  = getattr(trade, 'swap_subtype', 'FIXED_FLOAT')
            pr.tenor_y       = trade.tenor_y
            pr.leg_count     = len(trade.legs)
            pr.leg_types     = ",".join(l.leg_type for l in trade.legs)
            fl  = trade.fixed_leg
            ol  = trade.option_leg
            pr.notional      = fl.notional
            pr.start_date    = str(fl.start_date)
            pr.end_date      = str(fl.end_date)
            pr.currency      = fl.currency
            pr.coupon_rate   = fl.coupon_rate
            pr.strike        = getattr(ol, 'strike', _NAN_F)
            pr.option_type   = getattr(ol, 'option_type', '')
            pr.exercise_type = getattr(ol, 'exercise_type', '')
            pr.vol           = getattr(ol, 'vol', _NAN_F)

        elif isinstance(trade, OptionTrade):
            pr.instrument = "SWAPTION"
            pr.leg_count  = len(trade.legs)
            pr.leg_types  = ",".join(l.leg_type for l in trade.legs)
            ol = getattr(trade, 'option_leg', None)
            if ol is None and trade.legs:
                ol = trade.legs[0]
            if ol:
                pr.notional      = ol.notional
                pr.start_date    = str(ol.start_date)
                pr.end_date      = str(ol.end_date)
                pr.currency      = ol.currency
                pr.strike        = getattr(ol, 'strike', _NAN_F)
                pr.option_type   = getattr(ol, 'option_type', '')
                pr.exercise_type = getattr(ol, 'exercise_type', '')
                pr.vol           = getattr(ol, 'vol', _NAN_F)
            pr.tenor_y = trade.tenor_y

        elif isinstance(trade, (VanillaSwap, InterestRateSwap)):
            pr.instrument   = "IRS"
            pr.swap_subtype = getattr(trade, 'swap_subtype', 'FIXED_FLOAT')
            pr.tenor_y      = trade.tenor_y
            pr.leg_count    = len(trade.legs)
            pr.leg_types    = ",".join(l.leg_type for l in trade.legs)
            if trade.legs:
                l0 = trade.legs[0]
                pr.notional   = l0.notional
                pr.start_date = str(l0.start_date)
                pr.end_date   = str(l0.end_date)
                pr.currency   = l0.currency
            fl = getattr(trade, 'fixed_leg', None)
            if fl:
                pr.coupon_rate = fl.coupon_rate
            flt = getattr(trade, 'float_leg', None)
            if flt:
                pr.spread     = flt.spread
                pr.index_name = flt.index_name

        else:  # Bond or CallableBond
            is_callable = isinstance(trade, CallableBond)
            pr.instrument = "CBOND" if is_callable else "BOND"
            pr.leg_count  = len(trade.legs)
            pr.leg_types  = ",".join(l.leg_type for l in trade.legs)
            bl = trade.legs[0] if trade.legs else None
            if bl:
                pr.notional    = bl.notional
                pr.coupon_rate = getattr(bl, 'coupon_rate', _NAN_F)
                pr.start_date  = str(bl.start_date)
                pr.end_date    = str(bl.end_date)
                pr.currency    = bl.currency
            if is_callable and len(trade.legs) > 1:
                ol = trade.legs[1]
                pr.strike        = getattr(ol, 'strike', _NAN_F)
                pr.option_type   = getattr(ol, 'option_type', '')
                pr.exercise_type = getattr(ol, 'exercise_type', '')
                pr.vol           = getattr(ol, 'vol', _NAN_F)
                pr.swap_subtype  = trade.call_type   # "CALLABLE" or "PUTABLE"
            pr.tenor_y = trade.tenor_y

        if tid in df_idx:
            _fill_pricing_row(pr, df_idx[tid])

        results.append(pr)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DISPLAY + SANITY CHECKS + SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════════

def print_results(records_or_df, project_dir: str) -> None:
    # Accept list-of-dicts (from Spark) or DataFrame (from DB/tests).
    # ALL filtering and aggregation done in pure Python — no pandas boolean
    # indexing anywhere in this function (crashes in PyCharm due to NumPy
    # double-import corrupting pandas' BlockManager sentinels).
    if isinstance(records_or_df, list):
        all_recs = records_or_df
    else:
        all_recs = records_or_df.to_dict("records")

    # Sub-lists by instrument (pure Python filter — safe in all environments)
    irs_recs    = [r for r in all_recs if r.get("instrument") == "IRS"]
    bond_recs   = [r for r in all_recs if r.get("instrument") == "BOND"]
    cbond_recs  = [r for r in all_recs if r.get("instrument") == "CBOND"]
    opt_recs    = [r for r in all_recs if r.get("instrument") == "SWAPTION"]
    eq_recs     = [r for r in all_recs if r.get("instrument") == "EQ_SWAP"]
    cds_recs    = [r for r in all_recs if r.get("instrument") == "CDS"]
    eqopt_recs  = [r for r in all_recs if r.get("instrument") == "EQ_OPT"]
    err_recs    = [r for r in all_recs if len(str(r.get("error", ""))) > 0]

    # DataFrames for tabulate display ONLY — built fresh from filtered lists,
    # never from boolean indexing on an existing DataFrame.
    swaps   = pd.DataFrame(irs_recs)   if irs_recs   else pd.DataFrame()
    bonds   = pd.DataFrame(bond_recs)  if bond_recs  else pd.DataFrame()
    cbonds  = pd.DataFrame(cbond_recs) if cbond_recs else pd.DataFrame()
    opts    = pd.DataFrame(opt_recs)   if opt_recs   else pd.DataFrame()
    eqs     = pd.DataFrame(eq_recs)    if eq_recs    else pd.DataFrame()
    cdss    = pd.DataFrame(cds_recs)   if cds_recs   else pd.DataFrame()
    eq_opts = pd.DataFrame(eqopt_recs) if eqopt_recs else pd.DataFrame()

    sep  = "=" * 100
    sep2 = "-" * 100

    # ── SWAP TABLE ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  VANILLA IRS  --  Rates / NPV / Risk")
    print(sep)
    t_swap = pd.DataFrame({
        "Trade":    swaps["trade_id"],
        "Subtype":  swaps["swap_subtype"] if "swap_subtype" in swaps.columns else "",
        "Dir":      swaps["direction"],
        "Notl$M":   (swaps["notional"] / 1e6).map("{:.0f}".format),
        "Tnr":      swaps["tenor_y"].map("{}y".format),
        "FixRate":  swaps["coupon_rate"].map("{:.3%}".format),
        "ParRate":  swaps["par_rate"].map(lambda x: f"{x:.3%}" if pd.notna(x) else "N/A"),
        "NPV($)":   swaps["swap_npv"].map("${:>12,.0f}".format),
        "DV01($)":  swaps["dv01"].map("{:>8,.1f}".format),
        "Dur(yr)":  swaps["duration"].map("{:>+6.2f}".format),
    })
    print(tabulate(t_swap, headers="keys", tablefmt="simple", showindex=False))

    # ── BOND TABLE ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FIXED-RATE BONDS  --  Price / Yield / Risk")
    print(sep)
    t_bond = pd.DataFrame({
        "Trade":       bonds["trade_id"],
        "Dir":         bonds["direction"],
        "Face$M":      (bonds["notional"] / 1e6).map("{:.1f}".format),
        "Tnr":         bonds["tenor_y"].map("{}y".format),
        "Coupon":      bonds["coupon_rate"].map("{:.3%}".format),
        "CleanPx":     bonds["clean_price"].map("{:.4f}".format),
        "YTM":         bonds["par_rate"].map("{:.3%}".format),
        "DirtyNPV($)": bonds["swap_npv"].map("${:>11,.0f}".format),
        "Accrued($)":  bonds["accrued"].map("${:>8,.0f}".format),
        "DV01($)":     bonds["dv01"].map("{:>7,.1f}".format),
        "Dur(yr)":     bonds["duration"].map("{:>+6.2f}".format),
        "Convexity":   bonds["convexity"].map(
            lambda x: f"{x:>8.1f}" if pd.notna(x) else "NaN"),
    })
    print(tabulate(t_bond, headers="keys", tablefmt="simple", showindex=False))

    # ── CALLABLE BOND TABLE ────────────────────────────────────────────────
    if cbond_recs:
        print(f"\n{sep}")
        print("  CALLABLE / PUTABLE BONDS  --  Price / Greeks / Option Value")
        print(sep)
        t_cbond = pd.DataFrame({
            "Trade":     cbonds["trade_id"],
            "Type":      cbonds["option_type"].map(
                lambda x: "CALLABLE" if x == "CALL" else "PUTABLE"),
            "Dir":       cbonds["direction"],
            "Face$M":    (cbonds["notional"] / 1e6).map("{:.1f}".format),
            "Tnr":       cbonds["tenor_y"].map("{}y".format),
            "Coupon":    cbonds["coupon_rate"].map("{:.3%}".format),
            "ClnPx$M":   (cbonds["clean_price"] / 1e6).map("{:.3f}".format),
            "YTM":       cbonds["par_rate"].map(lambda x: f"{x:.3%}" if pd.notna(x) else "N/A"),
            "NPV($)":    cbonds["swap_npv"].map("${:>11,.0f}".format),
            "DV01($)":   cbonds["dv01"].map("{:>7,.1f}".format),
            "OptVal($)": cbonds["delta"].map("{:>8,.0f}".format),
            "Vega($)":   cbonds["vega"].map("{:>8,.1f}".format),
        })
        print(tabulate(t_cbond, headers="keys", tablefmt="simple", showindex=False))

    # ── OPTION TABLE ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SWAPTIONS  --  Premium / Greeks")
    print(sep)
    t_opt = pd.DataFrame({
        "Trade":    opts["trade_id"],
        "Dir":      opts["direction"],
        "Notl$M":   (opts["notional"] / 1e6).map("{:.0f}".format),
        "ExpTnr":   opts["tenor_y"].map("{}y".format),
        "Strike":   opts["coupon_rate"].map("{:.3%}".format),
        "ATM":      opts["par_rate"].map(lambda x: f"{x:.3%}" if pd.notna(x) else "nan"),
        "Premium":  opts["swap_npv"].map("${:>10,.0f}".format),
        "DV01($)":  opts["dv01"].map("{:>8,.1f}".format),
        "Vega($)":  opts["vega"].map("{:>8,.1f}".format),
        "Theta($)": opts["theta"].map(
            lambda x: f"{x:>8,.1f}" if pd.notna(x) else "NaN"),
        "Delta":    opts["delta"].map(
            lambda x: f"{x:>10,.1f}" if pd.notna(x) else "NaN"),
    })
    print(tabulate(t_opt, headers="keys", tablefmt="simple", showindex=False))

    # ── EQUITY SWAP TABLE ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  EQUITY SWAPS (TRS)  --  NPV / Greeks")
    print(sep)
    t_eq = pd.DataFrame({
        "Trade":    eqs["trade_id"],
        "Dir":      eqs["direction"],
        "Notl$M":   (eqs["notional"] / 1e6).map("{:.0f}".format),
        "Tnr":      eqs["tenor_y"].map("{}y".format),
        "DivYld":   eqs["par_rate"].map("{:.2%}".format),
        "NPV($)":   eqs["swap_npv"].map("${:>12,.0f}".format),
        "Delta($)": eqs["delta"].map("{:>10,.0f}".format),
        "DV01($)":  eqs["dv01"].map("{:>8,.1f}".format),
        "Theta($)": eqs["theta"].map("{:>8,.1f}".format),
    })
    print(tabulate(t_eq, headers="keys", tablefmt="simple", showindex=False))

    # ── CDS TABLE ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  CREDIT DEFAULT SWAPS  --  NPV / Credit Greeks")
    print(sep)
    t_cds = pd.DataFrame({
        "Trade":   cdss["trade_id"],
        "Dir":     cdss["direction"],
        "Notl$M":  (cdss["notional"] / 1e6).map("{:.0f}".format),
        "Tnr":     cdss["tenor_y"].map("{}y".format),
        "CdsSpr":  cdss["coupon_rate"].map("{:.3%}".format),
        "FairSpr": cdss["par_rate"].map(
            lambda x: f"{x:.3%}" if pd.notna(x) else "NaN"),
        "NPV($)":  cdss["swap_npv"].map("${:>12,.0f}".format),
        "CR01($)": cdss["cr01"].map("{:>8,.1f}".format),
        "JTD($)":  cdss["jump_to_default"].map("{:>12,.0f}".format),
    })
    print(tabulate(t_cds, headers="keys", tablefmt="simple", showindex=False))

    # ── PORTFOLIO SUMMARY ──────────────────────────────────────────────────
    # Use pure-Python counts from all_recs — avoids pandas boolean indexing
    # which crashes in PyCharm when NumPy is double-imported (BlockManager bug).
    _nan = float("nan")
    def _v(r, col):
        v = r.get(col, _nan)
        try: return float(v) if v is not None else _nan
        except (TypeError, ValueError): return _nan

    n_pay      = sum(1 for r in irs_recs   if r.get("direction") == "PAYER")
    n_rec      = sum(1 for r in irs_recs   if r.get("direction") == "RECEIVER")
    n_lng      = sum(1 for r in bond_recs  if r.get("direction") == "LONG")
    n_sht      = sum(1 for r in bond_recs  if r.get("direction") == "SHORT")
    n_cbl_lng  = sum(1 for r in cbond_recs if r.get("direction") == "LONG")
    n_cbl_sht  = sum(1 for r in cbond_recs if r.get("direction") == "SHORT")
    n_buy_opt  = sum(1 for r in opt_recs   if r.get("direction") == "BUY")
    n_sel_opt  = sum(1 for r in opt_recs   if r.get("direction") == "SELL")
    n_eq_l     = sum(1 for r in eq_recs    if r.get("direction") == "LONG")
    n_eq_s     = sum(1 for r in eq_recs    if r.get("direction") == "SHORT")
    n_cds_buy  = sum(1 for r in cds_recs   if r.get("direction") == "BUY")
    n_cds_sel  = sum(1 for r in cds_recs   if r.get("direction") == "SELL")
    n_eqo_buy  = sum(1 for r in eqopt_recs if r.get("direction") == "BUY")
    n_eqo_sel  = sum(1 for r in eqopt_recs if r.get("direction") == "SELL")

    print(f"\n{sep}")
    print("PORTFOLIO SUMMARY")
    print(f"  Total trades      : {len(all_recs):>5}  "
          f"(IRS:{len(irs_recs)}  Bonds:{len(bond_recs)}  CBonds:{len(cbond_recs)}  "
          f"Swaptions:{len(opt_recs)}  EqSwap:{len(eq_recs)}  CDS:{len(cds_recs)}  EqOpt:{len(eqopt_recs)})")
    print(f"  Pricing errors    : {len(err_recs):>5}")
    print(sep2)
    print(f"  -- IRS ---")
    print(f"  Payers / Receivers: {n_pay} / {n_rec}")
    print(f"  Total IRS Notional: ${sum(_v(r,'notional') for r in irs_recs):>22,.0f}")
    print(f"  Total IRS NPV     : ${sum(_v(r,'swap_npv') for r in irs_recs):>22,.2f}")
    print(f"  Total IRS DV01    : ${sum(_v(r,'dv01') for r in irs_recs):>22,.2f}")
    _pr = [_v(r,'par_rate') for r in irs_recs if _v(r,'par_rate') == _v(r,'par_rate')]
    print(f"  Avg Par Rate      : {(sum(_pr)/len(_pr) if _pr else 0):>21.4%}")
    print(f"  -- Bonds ---")
    print(f"  Long / Short      : {n_lng} / {n_sht}")
    print(f"  Total Face Value  : ${sum(_v(r,'notional') for r in bond_recs):>22,.0f}")
    print(f"  Total Dirty NPV   : ${sum(_v(r,'swap_npv') for r in bond_recs):>22,.2f}")
    print(f"  Total Bond DV01   : ${sum(_v(r,'dv01') for r in bond_recs):>22,.2f}")
    _cp = [_v(r,'clean_price') for r in bond_recs if _v(r,'clean_price') == _v(r,'clean_price')]
    print(f"  Avg Clean Price   : {(sum(_cp)/len(_cp) if _cp else 0):>21.4f} pct")
    _ytm = [_v(r,'par_rate') for r in bond_recs if _v(r,'par_rate') == _v(r,'par_rate')]
    print(f"  Avg YTM           : {(sum(_ytm)/len(_ytm) if _ytm else 0):>21.4%}")
    if opt_recs:
        print(f"  -- Swaptions ---")
        print(f"  Buy / Sell        : {n_buy_opt} / {n_sel_opt}")
        print(f"  Total Option Notl : ${sum(_v(r,'notional') for r in opt_recs):>22,.0f}")
        print(f"  Total Premium     : ${sum(_v(r,'swap_npv') for r in opt_recs):>22,.2f}")
        print(f"  Total Option DV01 : ${sum(_v(r,'dv01') for r in opt_recs):>22,.2f}")
        print(f"  Total Vega        : ${sum(_v(r,'vega') for r in opt_recs):>22,.2f}")
    if eq_recs:
        print(f"  -- Equity Swaps ---")
        print(f"  Long / Short      : {n_eq_l} / {n_eq_s}")
        print(f"  Total EQ Notional : ${sum(_v(r,'notional') for r in eq_recs):>22,.0f}")
        print(f"  Total EQ NPV      : ${sum(_v(r,'swap_npv') for r in eq_recs):>22,.2f}")
        print(f"  Total EQ Delta    : ${sum(_v(r,'delta') for r in eq_recs):>22,.2f}")
        print(f"  Total EQ DV01     : ${sum(_v(r,'dv01') for r in eq_recs):>22,.2f}")
    if cds_recs:
        print(f"  -- CDS ---")
        print(f"  Buy / Sell        : {n_cds_buy} / {n_cds_sel}")
        print(f"  Total CDS Notional: ${sum(_v(r,'notional') for r in cds_recs):>22,.0f}")
        print(f"  Total CDS NPV     : ${sum(_v(r,'swap_npv') for r in cds_recs):>22,.2f}")
        print(f"  Total CR01        : ${sum(_v(r,'cr01') for r in cds_recs):>22,.2f}")
        print(f"  Total JTD         : ${sum(_v(r,'jump_to_default') for r in cds_recs):>22,.2f}")
    if eqopt_recs:
        print(f"  -- Equity Options ---")
        print(f"  Buy / Sell        : {n_eqo_buy} / {n_eqo_sel}")
        print(f"  Total EqOpt Notl  : ${sum(_v(r,'notional') for r in eqopt_recs):>22,.0f}")
        print(f"  Total Premium     : ${sum(_v(r,'swap_npv') for r in eqopt_recs):>22,.2f}")
        print(f"  Total Delta       : ${sum(_v(r,'delta') for r in eqopt_recs):>22,.2f}")
        print(f"  Total Vega        : ${sum(_v(r,'vega') for r in eqopt_recs):>22,.2f}")
        print(f"  Total Gamma       : ${sum(_v(r,'gamma') for r in eqopt_recs):>22,.4f}")
    print(sep2)
    print(f"  Portfolio DV01    : ${sum(_v(r,'dv01') for r in all_recs):>22,.2f}")
    print(f"{sep}", flush=True)

    # ── SANITY CHECKS — all pure Python, no pandas boolean indexing ─────────
    print("\nSANITY CHECKS")
    checks = []

    if irs_recs:
        ff = [r for r in irs_recs if r.get("swap_subtype") in ("FIXED_FLOAT","FLOAT_FIXED")]
        if ff:
            resid = [abs(_v(r,'fixed_npv') + _v(r,'float_npv') - _v(r,'swap_npv')) for r in ff]
            checks.append(("IRS: fixed+float = swap_npv (FIXED_FLOAT/FLOAT_FIXED)",
                            max(resid) < 0.01, f"max residual ${max(resid):.5f}"))
            sign_ok = all(
                (_v(r,'dv01') > 0 if r.get("direction") == "PAYER" else _v(r,'dv01') < 0)
                for r in ff)
            checks.append(("IRS: DV01 sign FIXED_FLOAT/FLOAT_FIXED (payer>0, receiver<0)",
                            sign_ok, f"{len(ff)} trades"))
        par_vals = [_v(r,'par_rate') for r in irs_recs if _v(r,'par_rate') == _v(r,'par_rate')]
        if par_vals:
            checks.append(("IRS: par rates 3.5-6%",
                            min(par_vals) > 0.035 and max(par_vals) < 0.06,
                            f"range {min(par_vals):.3%}-{max(par_vals):.3%}"))
        from collections import Counter
        subtypes = Counter(r.get("swap_subtype","") for r in irs_recs)
        sub_str = ", ".join(f"{k}:{v}" for k, v in sorted(subtypes.items()))
        checks.append(("IRS: all 4 subtypes present", len(subtypes) == 4, sub_str))

    if bond_recs:
        sign_ok = all(
            (_v(r,'dv01') < 0 if r.get("direction") == "LONG" else _v(r,'dv01') > 0)
            for r in bond_recs)
        checks.append(("Bond: DV01 sign (long<0, short>0)", sign_ok, f"{len(bond_recs)} bonds"))
        px = [_v(r,'clean_price') for r in bond_recs]
        checks.append(("Bond: clean price 80-120", 80 < min(px) and max(px) < 120,
                        f"range {min(px):.2f}-{max(px):.2f}"))
        ytm = [_v(r,'par_rate') for r in bond_recs if _v(r,'par_rate') == _v(r,'par_rate')]
        if ytm:
            checks.append(("Bond: YTM 2-7%", min(ytm) > 0.02 and max(ytm) < 0.07,
                            f"range {min(ytm):.3%}-{max(ytm):.3%}"))
        acc = [_v(r,'accrued') for r in bond_recs]
        checks.append(("Bond: accrued >= 0", all(a >= 0 for a in acc),
                        f"avg ${sum(acc)/len(acc):,.0f}"))

    if opt_recs:
        buy_r = [r for r in opt_recs if r.get("direction") == "BUY"]
        sel_r = [r for r in opt_recs if r.get("direction") == "SELL"]
        buy_pos  = all(_v(r,'swap_npv') > 0 for r in buy_r) if buy_r else True
        sell_neg = all(_v(r,'swap_npv') < 0 for r in sel_r) if sel_r else True
        checks.append(("Option: BUY premium>0, SELL premium<0",
                        buy_pos and sell_neg, f"buy={len(buy_r)} sell={len(sel_r)}"))
        vega_ok = all(_v(r,'vega') > 0 for r in buy_r) if buy_r else True
        checks.append(("Option: BUY vega>0", vega_ok, f"{len(buy_r)} long options"))
        prem_pct = max(abs(_v(r,'swap_npv')) / _v(r,'notional') for r in opt_recs)
        checks.append(("Option: premium <20% notional",
                        prem_pct < 0.20, f"max {prem_pct:.2%} of notional"))
        atm = [_v(r,'par_rate') for r in opt_recs if _v(r,'par_rate') == _v(r,'par_rate')]
        if atm:
            checks.append(("Option: ATM rate 3.5-6%",
                            0.035 < min(atm) and max(atm) < 0.06,
                            f"range {min(atm):.3%}-{max(atm):.3%}"))

    if eq_recs:
        delta_ok = all(
            (_v(r,'delta') > 0 if r.get("direction") == "LONG" else _v(r,'delta') < 0)
            for r in eq_recs)
        checks.append(("EQ: LONG delta>0, SHORT delta<0", delta_ok, f"{len(eq_recs)} trades"))
        npv_pct = max(abs(_v(r,'swap_npv')) / _v(r,'notional') for r in eq_recs)
        checks.append(("EQ: |NPV| < 25% notional", npv_pct < 0.25, f"max {npv_pct:.2%}"))

    if cds_recs:
        jtd_ok = all(
            (_v(r,'jump_to_default') > 0 if r.get("direction") == "BUY"
             else _v(r,'jump_to_default') < 0)
            for r in cds_recs)
        checks.append(("CDS: BUY JTD>0, SELL JTD<0", jtd_ok, f"{len(cds_recs)} trades"))
        buy_c = [r for r in cds_recs if r.get("direction") == "BUY"]
        sel_c = [r for r in cds_recs if r.get("direction") == "SELL"]
        cr01_ok = (all(_v(r,'cr01') > 0 for r in buy_c) if buy_c else True) and \
                  (all(_v(r,'cr01') < 0 for r in sel_c) if sel_c else True)
        checks.append(("CDS: CR01 sign (BUY>0, SELL<0)", cr01_ok, "checked"))
        fair = [_v(r,'par_rate') for r in cds_recs if _v(r,'par_rate') == _v(r,'par_rate')]
        if fair:
            checks.append(("CDS: fair spread 25-500bps",
                            0.0025 < min(fair) and max(fair) < 0.05,
                            f"range {min(fair):.3%}-{max(fair):.3%}"))

    if eqopt_recs:
        buy_e = [r for r in eqopt_recs if r.get("direction") == "BUY"]
        sel_e = [r for r in eqopt_recs if r.get("direction") == "SELL"]
        buy_pos  = all(_v(r,'swap_npv') > 0 for r in buy_e) if buy_e else True
        sell_neg = all(_v(r,'swap_npv') < 0 for r in sel_e) if sel_e else True
        checks.append(("EqOpt: BUY premium>0, SELL premium<0",
                        buy_pos and sell_neg, f"buy={len(buy_e)} sell={len(sel_e)}"))
        delta_nz = all(abs(_v(r,'delta')) > 0 for r in eqopt_recs)
        checks.append(("EqOpt: |delta|>0 (calls +ve, puts -ve)", delta_nz,
                        f"{len(eqopt_recs)} options"))
        prem_pct = max(abs(_v(r,'swap_npv')) / _v(r,'notional') for r in eqopt_recs)
        checks.append(("EqOpt: premium <100% notional",
                        prem_pct < 1.0, f"max {prem_pct:.2%} of notional"))

    checks.append(("No pricing errors", len(err_recs) == 0,
                   f"{len(err_recs)} errors" if err_recs else "all clean"))

    for desc, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
        print(f"         [{detail}]")

    if err_recs:
        print("\n  Errors:")
        for r in err_recs:
            print(f"    {r.get('trade_id','?')}: {r.get('error','')}")

    db_path = os.path.join(project_dir, "trades.db")
    print(f"  Trade DB       -> {db_path}\n")


def _build_portfolio_summary(df: pd.DataFrame, inst_code: str, name: str) -> List[Dict]:
    """Build portfolio summary rows for one asset class."""
    rows: List[Dict] = []
    rows.append({"metric": "Instrument",    "value": name})
    rows.append({"metric": "Trade Count",   "value": len(df)})
    rows.append({"metric": "Error Count",   "value": int((df["error"] != "").sum())})
    rows.append({"metric": "Total Notional","value": f"${df['notional'].sum():,.0f}"})
    rows.append({"metric": "Total NPV",     "value": f"${df['swap_npv'].sum():,.2f}"})
    rows.append({"metric": "Total DV01",    "value": f"${df['dv01'].sum():,.2f}"})

    if inst_code == "IRS":
        rows.append({"metric": "Payers",    "value": int((df["direction"] == "PAYER").sum())})
        rows.append({"metric": "Receivers", "value": int((df["direction"] == "RECEIVER").sum())})
        if "swap_subtype" in df.columns:
            for sub in ["FIXED_FLOAT", "FLOAT_FIXED", "FIXED_FIXED", "FLOAT_FLOAT"]:
                cnt = int((df["swap_subtype"] == sub).sum())
                rows.append({"metric": f"  {sub}", "value": cnt})
        par_valid = df["par_rate"].dropna()
        if len(par_valid) > 0:
            rows.append({"metric": "Avg Par Rate", "value": f"{par_valid.mean():.4%}"})
    elif inst_code == "BOND":
        rows.append({"metric": "Long",              "value": int((df["direction"] == "LONG").sum())})
        rows.append({"metric": "Short",             "value": int((df["direction"] == "SHORT").sum())})
        rows.append({"metric": "Total Face",        "value": f"${df['notional'].sum():,.0f}"})
        rows.append({"metric": "Total Dirty NPV",   "value": f"${df['swap_npv'].sum():,.2f}"})
        rows.append({"metric": "Avg Clean Px",      "value": f"{df['clean_price'].mean():.4f}"})
        rows.append({"metric": "Avg YTM",           "value": f"{df['par_rate'].mean():.4%}"})
        rows.append({"metric": "Total Convexity",   "value": f"{df['convexity'].sum():,.1f}"})
    elif inst_code == "SWAPTION":
        rows.append({"metric": "Buy",              "value": int((df["direction"] == "BUY").sum())})
        rows.append({"metric": "Sell",             "value": int((df["direction"] == "SELL").sum())})
        rows.append({"metric": "Total Premium",    "value": f"${df['swap_npv'].sum():,.2f}"})
        rows.append({"metric": "Total Vega",       "value": f"${df['vega'].sum():,.2f}"})
        rows.append({"metric": "Total Theta",      "value": f"${df['theta'].sum():,.2f}"})
    elif inst_code == "EQ_SWAP":
        rows.append({"metric": "Long",             "value": int((df["direction"] == "LONG").sum())})
        rows.append({"metric": "Short",            "value": int((df["direction"] == "SHORT").sum())})
        rows.append({"metric": "Total EQ NPV",     "value": f"${df['swap_npv'].sum():,.2f}"})
        rows.append({"metric": "Total Delta",      "value": f"${df['delta'].sum():,.0f}"})
        rows.append({"metric": "Total EQ DV01",    "value": f"${df['dv01'].sum():,.2f}"})
    elif inst_code == "CDS":
        rows.append({"metric": "Buy",              "value": int((df["direction"] == "BUY").sum())})
        rows.append({"metric": "Sell",             "value": int((df["direction"] == "SELL").sum())})
        rows.append({"metric": "Total CR01",       "value": f"${df['cr01'].sum():,.2f}"})
        rows.append({"metric": "Total JTD",        "value": f"${df['jump_to_default'].sum():,.0f}"})

    return rows


def save_all_csvs(all_trades: list, records_or_df, out_dir: str) -> None:
    """Build PricingResult objects and write per-asset CSVs."""
    import csv as _csv
    results = build_pricing_results(all_trades, records_or_df)

    asset_map = {
        "IRS":      "IRS",
        "BOND":     "Bond",
        "SWAPTION": "Swaption",
        "EQ_SWAP":  "EquitySwap",
        "CDS":      "CDS",
        "EQ_OPT":   "EquityOption",
    }

    print("\nPer-asset CSV files:")
    for inst_code, name in asset_map.items():
        subset = [r for r in results if r.instrument == inst_code]
        if not subset:
            continue

        trade_csv = os.path.join(out_dir, f"{name}.csv")
        PricingResult.write_csv(subset, trade_csv)

        port_csv = os.path.join(out_dir, f"{name}portfolio.csv")
        summary  = PricingResult.portfolio_summary(subset)
        with open(port_csv, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(summary)

        print(f"  {name:<14} -> {trade_csv}  ({len(subset)} trades)")
        print(f"  {name+'portfolio':<14} -> {port_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd

# Numeric fields to compare across runs (all greeks + pricing outputs)
_COMPARE_FIELDS = [
    "npv", "fixed_npv", "float_npv", "par_rate", "clean_price", "accrued",
    "premium", "dv01", "duration", "pv01", "convexity",
    "vega", "theta", "delta", "gamma", "rho", "cr01", "jump_to_default",
]

_GREEK_THRESHOLDS = {
    # field       : (exact_tol, warn_tol)  — absolute values
    "npv"         : (0.01,   1.00),
    "fixed_npv"   : (0.01,   1.00),
    "float_npv"   : (0.01,   1.00),
    "par_rate"    : (1e-8,   1e-6),
    "clean_price" : (1e-6,   1e-4),
    "accrued"     : (0.01,   1.00),
    "premium"     : (0.01,   1.00),
    "dv01"        : (0.001,  0.10),
    "duration"    : (1e-6,   1e-4),
    "pv01"        : (0.001,  0.10),
    "convexity"   : (1e-6,   1e-4),
    "vega"        : (0.01,   1.00),
    "theta"       : (0.01,   1.00),
    "delta"       : (1e-6,   1e-4),
    "gamma"       : (1e-8,   1e-6),
    "rho"         : (0.01,   1.00),
    "cr01"        : (0.001,  0.10),
    "jump_to_default": (0.01, 1.00),
}


def _compare_runs(repo: "TradeRepository", run_a: str, run_b: str) -> bool:
    """
    Compare all numeric PricingResult fields between two run_ids.
    Pure Python — no pandas boolean indexing — safe in PyCharm.
    """
    import math

    recs_a = {r["trade_id"]: r for r in repo.get_results_df(run_id=run_a).to_dict("records")}
    recs_b = {r["trade_id"]: r for r in repo.get_results_df(run_id=run_b).to_dict("records")}
    common_ids = set(recs_a) & set(recs_b)

    rows = []
    all_ok = True

    for field in _COMPARE_FIELDS:
        pairs = []
        for tid in common_ids:
            va = recs_a[tid].get(field)
            vb = recs_b[tid].get(field)
            if va is not None and vb is not None:
                try:
                    fa, fb = float(va), float(vb)
                    if math.isfinite(fa) and math.isfinite(fb):
                        pairs.append(abs(fa - fb))
                except (TypeError, ValueError):
                    pass

        if not pairs:
            rows.append([field, 0, "—", "—", "—", "✓ (no data)"])
            continue

        n = len(pairs)
        mx   = max(pairs)
        mean = sum(pairs) / n
        exact_tol, warn_tol = _GREEK_THRESHOLDS.get(field, (0.01, 1.0))
        n_exact   = sum(1 for d in pairs if d < exact_tol)
        pct_exact = n_exact / n * 100

        if mx > warn_tol:
            status = f"⚠ WARN (>{warn_tol})"
            all_ok = False
        elif mx > exact_tol:
            status = f"~ close ({n_exact}/{n} exact)"
        else:
            status = "✓ exact"

        rows.append([field, n, f"{mx:,.6f}", f"{mean:,.6f}", f"{pct_exact:.1f}%", status])

    headers = ["field", "n", "max|diff|", "mean|diff|", "% exact", "status"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    return all_ok


def _sanity_check_results(df_or_records) -> None:
    """
    Run comprehensive sanity checks on pricing results (DataFrame or list[dict]).
    Uses pure Python throughout — no pandas boolean indexing — so it works even
    when pandas' BlockManager is corrupted by NumPy double-import in PyCharm.
    """
    import math
    from collections import Counter

    # Convert to list[dict] so all filtering is pure Python
    if isinstance(df_or_records, list):
        recs = df_or_records
    else:
        recs = df_or_records.to_dict("records")

    def _f(r, k, default=float("nan")):
        """Safe numeric field access."""
        v = r.get(k, default)
        return v if v is not None else default

    def _s(r, k, default=""):
        """Safe string field access."""
        v = r.get(k)
        return str(v) if v is not None else default

    checks: list[tuple[str, bool | None, str]] = []

    # ── Universal checks ──────────────────────────────────────────────────────
    npv_vals = [_f(r, "npv") for r in recs]
    finite_ok = all(math.isfinite(v) for v in npv_vals if v == v)
    n_valued  = sum(1 for v in npv_vals if v == v)
    checks.append(("All: NPV is finite", finite_ok, f"{n_valued}/{len(recs)} valued"))

    n_errors = sum(1 for r in recs if len(_s(r, "error")) > 0)
    checks.append(("All: no pricing errors", n_errors == 0,
                   f"{n_errors} errors" if n_errors else "clean"))

    notional_vals = [_f(r, "notional") for r in recs]
    checks.append(("All: notional > 0", all(v > 0 for v in notional_vals),
                   f"min {min(notional_vals):,.0f}"))

    tenor_vals = [_f(r, "tenor_y") for r in recs]
    checks.append(("All: tenor_y > 0", all(v > 0 for v in tenor_vals),
                   f"range {min(tenor_vals):.0f}–{max(tenor_vals):.0f}y"))

    # ── Sub-lists by instrument (pure Python) ────────────────────────────────
    irs_r    = [r for r in recs if _s(r, "instrument") == "IRS"]
    bond_r   = [r for r in recs if _s(r, "instrument") == "BOND"]
    cbond_r  = [r for r in recs if _s(r, "instrument") == "CBOND"]
    opt_r    = [r for r in recs if _s(r, "instrument") == "SWAPTION"]
    eq_r     = [r for r in recs if _s(r, "instrument") == "EQ_SWAP"]
    cds_r    = [r for r in recs if _s(r, "instrument") == "CDS"]
    eqopt_r  = [r for r in recs if _s(r, "instrument") == "EQ_OPT"]

    # ── IRS ───────────────────────────────────────────────────────────────────
    if irs_r:
        ff = [r for r in irs_r if _s(r, "swap_subtype") in ("FIXED_FLOAT", "FLOAT_FIXED")]
        if ff:
            resids = [abs(_f(r, "fixed_npv") + _f(r, "float_npv") - _f(r, "npv")) for r in ff]
            checks.append(("IRS: fixed_npv+float_npv=npv (FF/FxFl)",
                            max(resids) < 0.01, f"max residual ${max(resids):.5f}"))
            sign_ok = all(
                (_f(r, "dv01") > 0 if _s(r, "direction") == "PAYER" else _f(r, "dv01") < 0)
                for r in ff)
            checks.append(("IRS: DV01 sign (payer>0, receiver<0)",
                            sign_ok, f"{len(ff)} FIXED_FLOAT/FLOAT_FIXED"))
        par = [_f(r, "par_rate") for r in irs_r if _f(r, "par_rate") == _f(r, "par_rate")]
        if par:
            checks.append(("IRS: par rate 3.5–6.0%",
                            min(par) > 0.035 and max(par) < 0.06,
                            f"range {min(par):.3%}–{max(par):.3%}"))
        subtypes = Counter(_s(r, "swap_subtype") for r in irs_r)
        checks.append(("IRS: all 4 subtypes present", len(subtypes) == 4,
                        ", ".join(f"{k}:{v}" for k, v in sorted(subtypes.items()))))
        pv01_ok = all(_f(r, "pv01") * _f(r, "dv01") >= 0 for r in irs_r)
        checks.append(("IRS: PV01 sign matches DV01 sign", pv01_ok,
                        "pv01=dv01/notional×1M (signed sensitivity)"))

    # ── Bond ──────────────────────────────────────────────────────────────────
    if bond_r:
        sign_ok = all(
            (_f(r, "dv01") < 0 if _s(r, "direction") == "LONG" else _f(r, "dv01") > 0)
            for r in bond_r)
        checks.append(("Bond: DV01 sign (long<0, short>0)", sign_ok, f"{len(bond_r)} bonds"))
        px = [_f(r, "clean_price") for r in bond_r]
        checks.append(("Bond: clean price 80–130",
                        min(px) > 80 and max(px) < 130,
                        f"range {min(px):.2f}–{max(px):.2f}"))
        acc = [_f(r, "accrued") for r in bond_r]
        checks.append(("Bond: accrued >= 0", all(a >= 0 for a in acc),
                        f"avg ${sum(acc)/len(acc):,.0f}"))
        ytm = [_f(r, "par_rate") for r in bond_r if _f(r, "par_rate") == _f(r, "par_rate")]
        if ytm:
            checks.append(("Bond: YTM 2–7%",
                            min(ytm) > 0.02 and max(ytm) < 0.07,
                            f"range {min(ytm):.3%}–{max(ytm):.3%}"))
        dur = [_f(r, "duration") for r in bond_r if _f(r, "duration") == _f(r, "duration")]
        if dur:
            dur_ok = all(
                (_f(r, "duration") < 0 if _s(r, "direction") == "LONG" else _f(r, "duration") > 0)
                for r in bond_r if _f(r, "duration") == _f(r, "duration"))
            checks.append(("Bond: duration sign (LONG<0, SHORT>0)",
                            dur_ok, f"range {min(dur):.2f}–{max(dur):.2f}"))
        cvx = [_f(r, "convexity") for r in bond_r if _f(r, "convexity") == _f(r, "convexity")]
        if cvx:
            checks.append(("Bond: convexity > 0",
                            all(c > 0 for c in cvx),
                            f"range {min(cvx):.4f}–{max(cvx):.4f}"))

    # ── Callable Bond ─────────────────────────────────────────────────────────
    if cbond_r:
        # Price should be near par (70–110 % of par, using clean_price USD / notional)
        px_pct = [_f(r, "clean_price") / _f(r, "notional") * 100.0
                  for r in cbond_r
                  if _f(r, "clean_price") == _f(r, "clean_price")
                  and _f(r, "notional") > 0]
        if px_pct:
            checks.append(("CBOND: clean price 70–110 (% of par)",
                            min(px_pct) > 70 and max(px_pct) < 110,
                            f"range {min(px_pct):.2f}–{max(px_pct):.2f}"))

        # DV01 sign: LONG → rate rises → price falls → dv01 < 0
        dv01_ok = all(
            (_f(r, "dv01") < 0 if _s(r, "direction") == "LONG" else _f(r, "dv01") > 0)
            for r in cbond_r)
        checks.append(("CBOND: DV01 sign (LONG<0, SHORT>0)",
                        dv01_ok, f"{len(cbond_r)} callable bonds"))

        # Option value (delta) should be non-zero for all callable bonds
        opt_vals = [_f(r, "delta") for r in cbond_r if _f(r, "delta") == _f(r, "delta")]
        if opt_vals:
            checks.append(("CBOND: |option_value| > 0",
                            all(abs(v) > 0 for v in opt_vals),
                            f"min |delta| ${min(abs(v) for v in opt_vals):,.0f}"))

        ytm = [_f(r, "par_rate") for r in cbond_r if _f(r, "par_rate") == _f(r, "par_rate")]
        if ytm:
            checks.append(("CBOND: YTM 2–8%",
                            min(ytm) > 0.02 and max(ytm) < 0.08,
                            f"range {min(ytm):.3%}–{max(ytm):.3%}"))

    # ── Swaption ──────────────────────────────────────────────────────────────
    if opt_r:
        buy_o  = [r for r in opt_r if _s(r, "direction") == "BUY"]
        sell_o = [r for r in opt_r if _s(r, "direction") == "SELL"]
        prem_ok = ((not buy_o  or all(_f(r, "premium") > 0 for r in buy_o)) and
                   (not sell_o or all(_f(r, "premium") < 0 for r in sell_o)))
        checks.append(("Swaption: BUY premium>0, SELL premium<0",
                        prem_ok, f"buy={len(buy_o)} sell={len(sell_o)}"))
        vega_ok = all(_f(r, "vega") > 0 for r in buy_o) if buy_o else True
        checks.append(("Swaption: BUY vega > 0", vega_ok, f"{len(buy_o)} long swaptions"))
        prem_pct = max(abs(_f(r, "premium")) / _f(r, "notional") for r in opt_r)
        checks.append(("Swaption: premium < 20% notional",
                        prem_pct < 0.20, f"max {prem_pct:.2%}"))
        atm = [_f(r, "par_rate") for r in opt_r if _f(r, "par_rate") == _f(r, "par_rate")]
        if atm:
            checks.append(("Swaption: ATM rate 3.5–6%",
                            min(atm) > 0.035 and max(atm) < 0.06,
                            f"range {min(atm):.3%}–{max(atm):.3%}"))

    # ── IRS Swaption ─────────────────────────────────────────────────────────
    irs_swptn_r = [r for r in recs if _s(r, "instrument") == "IRS_SWPTN"]
    if irs_swptn_r:
        buy_s  = [r for r in irs_swptn_r if _s(r, "direction") == "BUY"]
        sell_s = [r for r in irs_swptn_r if _s(r, "direction") == "SELL"]
        prem_ok = ((not buy_s  or all(_f(r, "premium") > 0 for r in buy_s)) and
                   (not sell_s or all(_f(r, "premium") < 0 for r in sell_s)))
        checks.append(("IRSSwaption: BUY premium>0, SELL premium<0",
                        prem_ok, f"buy={len(buy_s)} sell={len(sell_s)}"))
        vega_ok = ((all(_f(r, "vega") > 0 for r in buy_s)  if buy_s  else True) and
                   (all(_f(r, "vega") < 0 for r in sell_s) if sell_s else True))
        checks.append(("IRSSwaption: vega sign (BUY>0, SELL<0)", vega_ok,
                        f"buy={len(buy_s)} sell={len(sell_s)}"))
        theta_ok = ((all(_f(r, "theta") < 0 for r in buy_s)  if buy_s  else True) and
                    (all(_f(r, "theta") > 0 for r in sell_s) if sell_s else True))
        checks.append(("IRSSwaption: theta sign (BUY<0, SELL>0)", theta_ok,
                        f"{len(irs_swptn_r)} IRS swaptions"))
        prem_pct = max(abs(_f(r, "premium")) / _f(r, "notional") for r in irs_swptn_r
                       if _f(r, "notional") > 0)
        checks.append(("IRSSwaption: premium < 20% notional",
                        prem_pct < 0.20, f"max {prem_pct:.2%}"))
        atm = [_f(r, "par_rate") for r in irs_swptn_r if _f(r, "par_rate") == _f(r, "par_rate")]
        if atm:
            checks.append(("IRSSwaption: ATM rate 3.5–6%",
                            min(atm) > 0.035 and max(atm) < 0.06,
                            f"range {min(atm):.3%}–{max(atm):.3%}"))

    # ── Equity Swap (TRS) ─────────────────────────────────────────────────────
    if eq_r:
        delta_ok = all(
            (_f(r, "delta") > 0 if _s(r, "direction") == "LONG" else _f(r, "delta") < 0)
            for r in eq_r)
        checks.append(("EQ_SWAP: LONG delta>0, SHORT delta<0", delta_ok, f"{len(eq_r)} trades"))
        npv_pct = max(abs(_f(r, "npv")) / _f(r, "notional") for r in eq_r)
        checks.append(("EQ_SWAP: |NPV| < 25% notional", npv_pct < 0.25, f"max {npv_pct:.2%}"))
        long_theta = [_f(r, "theta") for r in eq_r
                      if _s(r, "direction") == "LONG" and _f(r, "theta") == _f(r, "theta")]
        if long_theta:
            checks.append(("EQ_SWAP: LONG theta < 0 (carry cost)",
                            all(t < 0 for t in long_theta),
                            f"{len(long_theta)} LONG trades"))

    # ── CDS ───────────────────────────────────────────────────────────────────
    if cds_r:
        jtd_ok = all(
            (_f(r, "jump_to_default") > 0 if _s(r, "direction") == "BUY"
             else _f(r, "jump_to_default") < 0)
            for r in cds_r)
        checks.append(("CDS: BUY JTD>0, SELL JTD<0", jtd_ok, f"{len(cds_r)} trades"))
        cds_buy  = [r for r in cds_r if _s(r, "direction") == "BUY"]
        cds_sell = [r for r in cds_r if _s(r, "direction") == "SELL"]
        cr01_ok = ((not cds_buy  or all(_f(r, "cr01") > 0 for r in cds_buy)) and
                   (not cds_sell or all(_f(r, "cr01") < 0 for r in cds_sell)))
        checks.append(("CDS: CR01 sign (BUY>0, SELL<0)", cr01_ok, "checked"))
        fair = [_f(r, "par_rate") for r in cds_r if _f(r, "par_rate") == _f(r, "par_rate")]
        if fair:
            checks.append(("CDS: fair spread 25–500bps",
                            min(fair) > 0.0025 and max(fair) < 0.05,
                            f"range {min(fair):.3%}–{max(fair):.3%}"))
        npv_pct = max(abs(_f(r, "npv")) / _f(r, "notional") for r in cds_r)
        checks.append(("CDS: |NPV| < 15% notional", npv_pct < 0.15, f"max {npv_pct:.2%}"))

    # ── Equity Option ─────────────────────────────────────────────────────────
    if eqopt_r:
        eo_buy  = [r for r in eqopt_r if _s(r, "direction") == "BUY"]
        eo_sell = [r for r in eqopt_r if _s(r, "direction") == "SELL"]
        prem_ok = ((not eo_buy  or all(_f(r, "premium") > 0 for r in eo_buy)) and
                   (not eo_sell or all(_f(r, "premium") < 0 for r in eo_sell)))
        checks.append(("EqOpt: BUY premium>0, SELL premium<0",
                        prem_ok, f"buy={len(eo_buy)} sell={len(eo_sell)}"))
        delta_nz = all(abs(_f(r, "delta")) > 0 for r in eqopt_r)
        checks.append(("EqOpt: |delta| > 0 (non-zero sensitivity)",
                        delta_nz, f"{len(eqopt_r)} options"))
        buy_gamma  = [_f(r, "gamma") for r in eo_buy  if _f(r, "gamma") == _f(r, "gamma")]
        sell_gamma = [_f(r, "gamma") for r in eo_sell if _f(r, "gamma") == _f(r, "gamma")]
        gamma_ok = ((not buy_gamma  or all(g > 0 for g in buy_gamma)) and
                    (not sell_gamma or all(g < 0 for g in sell_gamma)))
        checks.append(("EqOpt: BUY gamma>0, SELL gamma<0", gamma_ok,
                        f"buy={len(buy_gamma)} sell={len(sell_gamma)}"))
        buy_vega  = [_f(r, "vega") for r in eo_buy  if _f(r, "vega") == _f(r, "vega")]
        sell_vega = [_f(r, "vega") for r in eo_sell if _f(r, "vega") == _f(r, "vega")]
        vega_ok = ((not buy_vega  or all(v > 0 for v in buy_vega)) and
                   (not sell_vega or all(v < 0 for v in sell_vega)))
        checks.append(("EqOpt: BUY vega>0, SELL vega<0", vega_ok,
                        f"buy={len(buy_vega)} sell={len(sell_vega)}"))
        buy_theta  = [_f(r, "theta") for r in eo_buy  if _f(r, "theta") == _f(r, "theta")]
        sell_theta = [_f(r, "theta") for r in eo_sell if _f(r, "theta") == _f(r, "theta")]
        n_buy_theta_pos = sum(1 for t in buy_theta if t >= 0)
        theta_sell_pos  = all(t > 0 for t in sell_theta) if sell_theta else True
        theta_ok = theta_sell_pos and (n_buy_theta_pos <= max(1, int(len(buy_theta) * 0.05)))
        checks.append(("EqOpt: theta sign (BUY<0 except deep-ITM American, SELL>0)",
                        theta_ok,
                        f"{n_buy_theta_pos} BUY with θ≥0; sell min {min(sell_theta):.0f}"
                        if sell_theta else "n/a"))
        prem_pct = max(abs(_f(r, "premium")) / _f(r, "notional") for r in eqopt_r)
        checks.append(("EqOpt: premium < 100% notional",
                        prem_pct < 1.0, f"max {prem_pct:.2%}"))
        calls_buy = [r for r in eqopt_r
                     if _s(r, "option_type") == "CALL" and _s(r, "direction") == "BUY"]
        puts_buy  = [r for r in eqopt_r
                     if _s(r, "option_type") == "PUT"  and _s(r, "direction") == "BUY"]
        call_d_ok = all(_f(r, "delta") > 0 for r in calls_buy) if calls_buy else True
        put_d_ok  = all(_f(r, "delta") < 0 for r in puts_buy)  if puts_buy  else True
        checks.append(("EqOpt: CALL delta>0 (BUY), PUT delta<0 (BUY)",
                        call_d_ok and put_d_ok,
                        f"buy_calls={len(calls_buy)} buy_puts={len(puts_buy)}"))

    # ── Print results ─────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in checks if ok is True)
    failed = sum(1 for _, ok, _ in checks if ok is False)
    total  = len(checks)
    print(f"\nRESULT SANITY CHECKS  ({passed}/{total} pass, {failed} fail)")
    print("─" * 72)
    for desc, ok, detail in checks:
        icon = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {icon}  {desc}")
        print(f"           [{detail}]")
    print("─" * 72)
    if failed:
        print(f"  ⚠  {failed} check(s) FAILED — review above.")
    else:
        print("  ✅  All sanity checks passed.")


def print_trade_tree(trades: list) -> None:
    """Print a trader → book → legs tree for all trades in the portfolio."""
    from collections import defaultdict

    # Build: trader → book → list of (trade_id, instrument, direction, legs)
    tree: dict = defaultdict(lambda: defaultdict(list))
    for t in trades:
        trader = getattr(t, "trader", "") or "(no trader)"
        book   = getattr(t, "book",   "") or "(no book)"
        inst   = type(t).__name__
        legs   = getattr(t, "legs", [])
        leg_descs = []
        for leg in legs:
            lt   = getattr(leg, "leg_type",  getattr(leg, "__class__", type(leg)).__name__)
            notl = getattr(leg, "notional",  0)
            leg_descs.append(f"{lt}  ${notl:,.0f}")
        tree[trader][book].append((t.trade_id, inst, str(t.direction.value), leg_descs))

    sep = "=" * 70
    print(f"\n{sep}")
    print("  TRADE TREE  —  Trader → Book → Trade → Legs")
    print(sep)
    for trader in sorted(tree):
        print(f"\n📋 Trader: {trader}")
        books = tree[trader]
        for book in sorted(books):
            entries = books[book]
            print(f"  └─ 📁 Book: {book}  ({len(entries)} trades)")
            for trade_id, inst, direction, leg_descs in sorted(entries):
                print(f"       └─ {trade_id:<18} {inst:<22} {direction}")
                for leg in leg_descs:
                    print(f"              └─ {leg}")
    print(f"\n{sep}\n")


def main() -> None:
    """
    Run the full price-and-persist workflow:
      1. Save market data snapshot to DB.
      2. RUN-1 — generate 600 synthetic trades, price them, persist to DB and CSVs.
      3. Retry loop — reload all trades from DB and re-price (RUN-N) until ALL
         numeric fields (NPV + all greeks) agree with RUN-1 within tolerances.
      4. Print a per-field comparison table after each reload run.
    """
    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir  = os.path.dirname(_examples_dir)

    db_path = os.path.join(_project_dir, "trades.db")

    # ── Persist market data snapshot ──────────────────────────────────────────
    mkt_repo = MarketDataRepository(db_path)
    mkt_repo.upsert(MKT)
    mkt_repo.close()

    repo = TradeRepository(db_path)

    # ── Clear previous pricing results ────────────────────────────────────────
    repo.clear_results()
    print("Cleared PricingResult table.")
    print("Multi-Instrument Pricer -- IRS+Bonds+Swaptions+EqSwaps+CDS+EqOptions+EquityOptions")
    print("=" * 80)

    # ── RUN 1: generate trades from code ──────────────────────────────────────
    print("\n[RUN 1] Generating trades and pricing...")
    all_trades = populate_trades()
    # Persist the newly generated trades before pricing
    repo.upsert_many(all_trades)
    result_df1, trades1, pr_list1 = run_pricing(trades=all_trades, run_id="RUN-1")
    print_results(result_df1, _project_dir)   # result_df1 is list[dict]
    print_trade_tree(trades1)
    save_all_csvs(trades1, result_df1, _project_dir)

    # ── Sanity check all result values from DB ─────────────────────────────────
    print("\n[SANITY] Checking RUN-1 result values from DB...")
    _sanity_check_results(repo.get_results_df(run_id="RUN-1"))

    # ── Print first 60 rows from PricingResult table ───────────────────────────
    print("\n[DB] First 60 rows from PricingResult table (RUN-1):")
    db_recs = repo.get_results_df(run_id="RUN-1").to_dict("records")
    display_cols = ["trade_id", "instrument", "direction", "tenor_y",
                    "notional", "npv", "dv01", "delta", "vega", "error"]
    # Build display table from list[dict] — avoids DataFrame column-selection take()
    display_rows = [{c: r.get(c) for c in display_cols if c in r} for r in db_recs[:60]]
    print(tabulate(display_rows, headers="keys",
                   tablefmt="simple", floatfmt=".2f", showindex=False))

    # ── Retry loop: reload from DB + re-price until values stabilise ──────────
    MAX_RETRIES = 5
    prev_run_id = "RUN-1"
    all_good    = False

    for attempt in range(2, MAX_RETRIES + 2):
        run_id = f"RUN-{attempt}"
        print(f"\n[{run_id}] Loading trades from DB and pricing...")
        db_trades = repo.list_all()
        print(f"  Loaded {len(db_trades)} trades from DB.")
        result_df_n, trades_n, pr_list_n = run_pricing(
            trades=db_trades,
            run_id=run_id
        )

        print(f"\n[COMPARE] {prev_run_id} vs {run_id} — all numeric fields:")
        all_good = _compare_runs(repo, prev_run_id, run_id)

        if all_good:
            print(f"\n✅  All fields within tolerances after {run_id}. Stopping.")
            break

        # Print worst offenders per instrument type for diagnosis
        print(f"\n  ⚠  Some fields outside tolerance — printing per-instrument NPV summary:")
        recs_a2 = repo.get_results_df(run_id=prev_run_id).to_dict("records")
        recs_b2 = repo.get_results_df(run_id=run_id).to_dict("records")
        a2_by_id = {r["trade_id"]: r for r in recs_a2}
        b2_by_id = {r["trade_id"]: r for r in recs_b2}
        import math as _math
        for inst in ["IRS", "BOND", "SWAPTION", "EQ_SWAP", "CDS", "EQ_OPT"]:
            pairs2 = []
            for tid, ra in a2_by_id.items():
                if ra.get("instrument") != inst:
                    continue
                rb = b2_by_id.get(tid)
                if rb is None:
                    continue
                va, vb = ra.get("npv"), rb.get("npv")
                if va is not None and vb is not None:
                    try:
                        fa, fb = float(va), float(vb)
                        if _math.isfinite(fa) and _math.isfinite(fb):
                            pairs2.append(abs(fa - fb))
                    except (TypeError, ValueError):
                        pass
            if not pairs2:
                continue
            n_ok2 = sum(1 for d in pairs2 if d < 0.01)
            print(f"    {inst:<10}  {n_ok2:>3}/{len(pairs2)} exact  "
                  f"  max|Δ|={max(pairs2):,.4f}  mean|Δ|={sum(pairs2)/len(pairs2):,.4f}")

        prev_run_id = run_id

    if not all_good:
        print(f"\n⚠  Stopped after {MAX_RETRIES} reload attempts — see comparison above.")

    # ── Final: print 20 rows side-by-side (RUN-1 vs latest) ──────────────────
    print(f"\n[DB] First 20 rows — RUN-1 vs {prev_run_id}:")
    all_recs_db = repo.get_results_df().to_dict("records")
    display_cols2 = ["trade_id", "run_id", "instrument", "direction", "tenor_y",
                     "notional", "npv", "dv01", "delta", "vega", "gamma",
                     "theta", "rho", "cr01", "error"]
    target_runs = {"RUN-1", prev_run_id}
    filtered_recs = [r for r in all_recs_db if r.get("run_id") in target_runs][:20]
    display_rows2 = [{c: r.get(c) for c in display_cols2 if c in r} for r in filtered_recs]
    print(tabulate(display_rows2, headers="keys",
                   tablefmt="simple", floatfmt=".4f", showindex=False))

    repo.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

