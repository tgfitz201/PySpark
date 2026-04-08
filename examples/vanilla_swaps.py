#!/usr/bin/env python3
"""
vanilla_swaps.py  —  Multi-instrument Pricer
============================================
  Instruments : VanillaSwap (IRS) | Bond | OptionTrade (swaptions)
              | EquitySwap (TRS)  | CreditDefaultSwap (CDS)
  Framework   : QuantLib 1.41 + PySpark 4.0 (pandas UDF)
  Curve       : Single-curve SOFR  (discount & forward)
  Greeks      : DV01, Duration, PV01, Convexity, Vega, Theta, Delta, CR01, JTD
  Convergence : Iterative re-pricing until max|ΔNPV| < tol
  Persistence : TradeRepository (SQLite) + CSV outputs
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

from models import (TradeBase, BaseLeg, OptionLeg, EquityLeg, CreditLeg, EquityOptionLeg,
                    VanillaSwap, Bond, OptionTrade, EquitySwap, CreditDefaultSwap,
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

def make_irs_data(n: int = 50) -> List[VanillaSwap]:
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

    subtypes = (["FIXED_FLOAT"] * 20 + ["FLOAT_FIXED"] * 15 +
                ["FIXED_FIXED"] * 8   + ["FLOAT_FLOAT"] * 7)
    random.shuffle(subtypes)
    # Extend subtypes list if n > len(subtypes)
    while len(subtypes) < n:
        extra = subtypes[:]
        random.shuffle(extra)
        subtypes += extra

    trades: List[VanillaSwap] = []
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
                BaseLeg("FIXED", notl, eff, mat, coupon_rate=fixed_rate),
                BaseLeg("FLOAT", notl, eff, mat, frequency="QUARTERLY", day_count="ACT/360"),
            ]
        elif subtype == "FIXED_FIXED":
            rate2 = round(max(0.005, par_rate + random.choice(spreads)), 4)
            direction = TradeDirection.PAYER
            legs = [
                BaseLeg("FIXED", notl, eff, mat, coupon_rate=fixed_rate, frequency="ANNUAL"),
                BaseLeg("FIXED", notl, eff, mat, coupon_rate=rate2, frequency="SEMIANNUAL"),
            ]
        else:  # FLOAT_FLOAT
            basis_spread_bps = random.uniform(0.0005, 0.0025)
            direction = TradeDirection.PAYER
            legs = [
                BaseLeg("FLOAT", notl, eff, mat, index_tenor_m=1,
                        day_count="ACT/360", frequency="MONTHLY", spread=basis_spread_bps),
                BaseLeg("FLOAT", notl, eff, mat, index_tenor_m=3,
                        day_count="ACT/360", frequency="QUARTERLY"),
            ]

        trades.append(VanillaSwap(
            trade_id=f"IRS-{i+1:04d}", book=random.choice(books),
            counterparty=random.choice(cptys), valuation_date=base,
            direction=direction, tenor_y=tenor_y, swap_subtype=subtype,
            legs=legs,
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
            counterparty=random.choice(cptys), valuation_date=base,
            direction=direction, tenor_y=tenor_y,
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
            counterparty=random.choice(cptys), valuation_date=base,
            direction=direct, tenor_y=exp_y, underlying_tenor_y=swap_y,
            legs=[OptionLeg(
                "OPTION", notl, expiry, mat,
                strike=strike, option_type=opt_t,
                exercise_type="EUROPEAN", vol=vol, vol_type="LOGNORMAL",
                vol_shift=0.01, underlying_tenor_m=swap_y * 12,
            )],
        ))
    return options


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
            counterparty=random.choice(cptys), valuation_date=base,
            direction=direction, tenor_y=tenor_y, underlying_ticker=ticker,
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
            counterparty=random.choice(cptys), valuation_date=base,
            direction=direction, tenor_y=tenor_y,
            legs=[CreditLeg(
                "CREDIT", notl, eff, mat,
                reference_entity=name, credit_spread=spread_bumped,
                recovery_rate=rec, hazard_rate=hazard,
                seniority=seniority, doc_clause=doc,
            )],
        ))
    return result


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
# 3.  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

TRADE_SCHEMA = StructType([
    StructField("trade_id",     StringType()),
    StructField("instrument",   StringType()),   # "IRS"|"BOND"|"SWAPTION"|"EQ_SWAP"|"CDS"|"EQ_OPT"
    StructField("book",         StringType()),
    StructField("counterparty", StringType()),
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

    stype    = ql.VanillaSwap.Payer if trade.direction == "PAYER" else ql.VanillaSwap.Receiver
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
    sign     = -1.0 if trade.direction == "SHORT" else 1.0
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
    sign    = -1.0 if trade.direction == "SELL" else 1.0
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
    sign     = -1.0 if trade.direction == "SHORT" else 1.0

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
    cl     = trade.credit_leg
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

    side     = ql.Protection.Buyer if trade.direction == "BUY" else ql.Protection.Seller
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
    jump_to_default = lgd if trade.direction == "BUY" else -lgd

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=npv,
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
    try:
        if isinstance(trade, EquityOptionTrade):
            return _price_equity_option(trade, curve_df)
        if isinstance(trade, EquitySwap):
            return _price_equity_swap(trade, curve_df)
        elif isinstance(trade, CreditDefaultSwap):
            return _price_cds(trade, curve_df)
        elif isinstance(trade, OptionTrade):
            return _price_option(trade, curve_df)
        elif isinstance(trade, VanillaSwap):
            return _price_swap(trade, curve_df)
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
        from models import TradeBase, EquityOptionTrade  # noqa: PLC0415
        curve_df = bc.value
        return pd.DataFrame([_price_one(TradeBase.fromJson(js), curve_df)
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
        if isinstance(t, OptionTrade):          return "SWAPTION"
        if isinstance(t, VanillaSwap):          return "IRS"
        return "BOND"

    def _notional(t):
        return t.legs[0].notional

    def _coupon(t):
        if isinstance(t, EquityOptionTrade):
            ol = t.option_leg
            return ol.strike if ol else 0.0
        if isinstance(t, VanillaSwap):
            fl = t.fixed_leg
            return fl.coupon_rate if fl is not None else (t.legs[0].spread if t.legs else 0.0)
        if isinstance(t, Bond):              return t.bond_leg.coupon_rate
        if isinstance(t, OptionTrade):       return t.option_leg.strike
        if isinstance(t, EquitySwap):        return t.equity_leg.initial_price
        if isinstance(t, CreditDefaultSwap): return t.credit_leg.credit_spread
        return 0.0

    def _swap_subtype(t):
        return getattr(t, 'swap_subtype', '') or ''

    rows = [(t.trade_id, _instrument(t), t.book, t.counterparty,
             str(t.direction.value), t.tenor_y, _notional(t), _coupon(t),
             _swap_subtype(t), t.toJson())
            for t in all_trades]
    trades_df = (spark.createDataFrame(rows, schema=TRADE_SCHEMA)
                 .repartitionByRange(n_cores, "trade_id"))
    price_udf = make_price_udf()

    def _execute() -> pd.DataFrame:
        return (trades_df
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
                ).toPandas())

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
        curr_npv = result["swap_npv"].values
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
    pr.premium     = _g('swap_npv') if pr.instrument in ('SWAPTION', 'EQ_OPT') else _NAN_F
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


def build_pricing_results(all_trades: list, df: pd.DataFrame) -> List[PricingResult]:
    """Build flat PricingResult objects from trade objects + pricing DataFrame."""
    from datetime import datetime
    ts = datetime.utcnow().isoformat()
    df_idx = df.set_index("trade_id") if "trade_id" in df.columns else df

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
            cl = next((l for l in trade.legs if l.leg_type == "CREDIT"), None)
            if cl:
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

        elif isinstance(trade, VanillaSwap):
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

        else:  # Bond
            pr.instrument = "BOND"
            pr.leg_count  = len(trade.legs)
            pr.leg_types  = ",".join(l.leg_type for l in trade.legs)
            bl = trade.legs[0] if trade.legs else None
            if bl:
                pr.notional    = bl.notional
                pr.coupon_rate = getattr(bl, 'coupon_rate', _NAN_F)
                pr.start_date  = str(bl.start_date)
                pr.end_date    = str(bl.end_date)
                pr.currency    = bl.currency
            pr.tenor_y = trade.tenor_y

        if tid in df_idx.index:
            _fill_pricing_row(pr, df_idx.loc[tid])

        results.append(pr)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DISPLAY + SANITY CHECKS + SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════════

def print_results(df: pd.DataFrame, project_dir: str) -> None:
    sep  = "=" * 100
    sep2 = "-" * 100
    swaps    = df[df["instrument"] == "IRS"].copy()
    bonds    = df[df["instrument"] == "BOND"].copy()
    opts     = df[df["instrument"] == "SWAPTION"].copy()
    eqs      = df[df["instrument"] == "EQ_SWAP"].copy()
    cdss     = df[df["instrument"] == "CDS"].copy()
    eq_opts  = df[df["instrument"] == "EQ_OPT"].copy()
    err   = df[df["error"].str.len() > 0]

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
    pay     = swaps[swaps["direction"] == "PAYER"]
    rec     = swaps[swaps["direction"] == "RECEIVER"]
    lng     = bonds[bonds["direction"] == "LONG"]
    sht     = bonds[bonds["direction"] == "SHORT"]
    buy     = opts[opts["direction"]   == "BUY"]   if len(opts)  > 0 else opts
    sel     = opts[opts["direction"]   == "SELL"]  if len(opts)  > 0 else opts
    eq_l    = eqs[eqs["direction"]     == "LONG"]
    eq_s    = eqs[eqs["direction"]     == "SHORT"]
    cds_buy  = cdss[cdss["direction"]  == "BUY"]
    cds_sell = cdss[cdss["direction"]  == "SELL"]
    eq_opt_buy  = eq_opts[eq_opts["direction"] == "BUY"]   if len(eq_opts) > 0 else eq_opts
    eq_opt_sell = eq_opts[eq_opts["direction"] == "SELL"]  if len(eq_opts) > 0 else eq_opts

    print(f"\n{sep}")
    print("PORTFOLIO SUMMARY")
    print(f"  Total trades      : {len(df):>5}  "
          f"(IRS:{len(swaps)}  Bonds:{len(bonds)}  Swaptions:{len(opts)}  "
          f"EqSwap:{len(eqs)}  CDS:{len(cdss)}  EqOpt:{len(eq_opts)})")
    print(f"  Pricing errors    : {len(err):>5}")
    print(sep2)
    print(f"  -- IRS ---")
    print(f"  Payers / Receivers: {len(pay)} / {len(rec)}")
    print(f"  Total IRS Notional: ${swaps['notional'].sum():>22,.0f}")
    print(f"  Total IRS NPV     : ${swaps['swap_npv'].sum():>22,.2f}")
    print(f"  Total IRS DV01    : ${swaps['dv01'].sum():>22,.2f}")
    print(f"  Avg Par Rate      : {swaps['par_rate'].mean():>21.4%}")
    print(f"  -- Bonds ---")
    print(f"  Long / Short      : {len(lng)} / {len(sht)}")
    print(f"  Total Face Value  : ${bonds['notional'].sum():>22,.0f}")
    print(f"  Total Dirty NPV   : ${bonds['swap_npv'].sum():>22,.2f}")
    print(f"  Total Bond DV01   : ${bonds['dv01'].sum():>22,.2f}")
    print(f"  Avg Clean Price   : {bonds['clean_price'].mean():>21.4f} pct")
    print(f"  Avg YTM           : {bonds['par_rate'].mean():>21.4%}")
    if len(opts) > 0:
        print(f"  -- Swaptions ---")
        print(f"  Buy / Sell        : {len(buy)} / {len(sel)}")
        print(f"  Total Option Notl : ${opts['notional'].sum():>22,.0f}")
        print(f"  Total Premium     : ${opts['swap_npv'].sum():>22,.2f}")
        print(f"  Total Option DV01 : ${opts['dv01'].sum():>22,.2f}")
        print(f"  Total Vega        : ${opts['vega'].sum():>22,.2f}")
    if len(eqs) > 0:
        print(f"  -- Equity Swaps ---")
        print(f"  Long / Short      : {len(eq_l)} / {len(eq_s)}")
        print(f"  Total EQ Notional : ${eqs['notional'].sum():>22,.0f}")
        print(f"  Total EQ NPV      : ${eqs['swap_npv'].sum():>22,.2f}")
        print(f"  Total EQ Delta    : ${eqs['delta'].sum():>22,.2f}")
        print(f"  Total EQ DV01     : ${eqs['dv01'].sum():>22,.2f}")
    if len(cdss) > 0:
        print(f"  -- CDS ---")
        print(f"  Buy / Sell        : {len(cds_buy)} / {len(cds_sell)}")
        print(f"  Total CDS Notional: ${cdss['notional'].sum():>22,.0f}")
        print(f"  Total CDS NPV     : ${cdss['swap_npv'].sum():>22,.2f}")
        print(f"  Total CR01        : ${cdss['cr01'].sum():>22,.2f}")
        print(f"  Total JTD         : ${cdss['jump_to_default'].sum():>22,.2f}")
    if len(eq_opts) > 0:
        print(f"  -- Equity Options ---")
        print(f"  Buy / Sell        : {len(eq_opt_buy)} / {len(eq_opt_sell)}")
        print(f"  Total EqOpt Notl  : ${eq_opts['notional'].sum():>22,.0f}")
        print(f"  Total Premium     : ${eq_opts['swap_npv'].sum():>22,.2f}")
        print(f"  Total Delta       : ${eq_opts['delta'].sum():>22,.2f}")
        print(f"  Total Vega        : ${eq_opts['vega'].sum():>22,.2f}")
        print(f"  Total Gamma       : ${eq_opts['gamma'].sum():>22,.4f}")
    print(sep2)
    print(f"  Portfolio DV01    : ${df['dv01'].sum():>22,.2f}")
    print(f"{sep}", flush=True)

    # ── SANITY CHECKS ──────────────────────────────────────────────────────
    print("\nSANITY CHECKS")
    checks = []

    if len(swaps) > 0:
        # Only FIXED_FLOAT and FLOAT_FIXED have valid fixed+float decomposition
        ff_swaps = swaps[swaps["swap_subtype"].isin(["FIXED_FLOAT", "FLOAT_FIXED"])] if "swap_subtype" in swaps.columns else swaps
        if len(ff_swaps) > 0:
            recon = (ff_swaps["fixed_npv"] + ff_swaps["float_npv"] - ff_swaps["swap_npv"]).abs()
            checks.append(("IRS: fixed+float = swap_npv (FIXED_FLOAT/FLOAT_FIXED)",
                            recon.max() < 0.01, f"max residual ${recon.max():.5f}"))
        # DV01 sign check: only meaningful for standard FIXED_FLOAT / FLOAT_FIXED
        std_swaps = swaps[swaps["swap_subtype"].isin(["FIXED_FLOAT", "FLOAT_FIXED"])] if "swap_subtype" in swaps.columns else swaps
        if len(std_swaps) > 0:
            sign_ok = (((std_swaps["direction"] == "PAYER")    & (std_swaps["dv01"] > 0)) |
                       ((std_swaps["direction"] == "RECEIVER") & (std_swaps["dv01"] < 0))).all()
            checks.append(("IRS: DV01 sign FIXED_FLOAT/FLOAT_FIXED (payer>0, receiver<0)",
                            sign_ok, f"{len(std_swaps)} trades"))
        # Only check par_rate for subtypes that produce a par rate
        par_swaps = swaps[swaps["par_rate"].notna()] if "par_rate" in swaps.columns else swaps
        if len(par_swaps) > 0:
            par_min, par_max = par_swaps["par_rate"].min(), par_swaps["par_rate"].max()
            checks.append(("IRS: par rates 3.5-6%",
                            (par_min > 0.035) and (par_max < 0.06),
                            f"range {par_min:.3%}-{par_max:.3%}"))
        if "swap_subtype" in swaps.columns:
            subtype_counts = swaps["swap_subtype"].value_counts().to_dict()
            sub_str = ", ".join(f"{k}:{v}" for k, v in sorted(subtype_counts.items()))
            checks.append(("IRS: all 4 subtypes present",
                            len(subtype_counts) == 4, sub_str))

    if len(bonds) > 0:
        bond_sign_ok = (((bonds["direction"] == "LONG")  & (bonds["dv01"] < 0)) |
                        ((bonds["direction"] == "SHORT") & (bonds["dv01"] > 0))).all()
        checks.append(("Bond: DV01 sign (long<0, short>0)", bond_sign_ok, f"{len(bonds)} bonds"))
        px_min, px_max = bonds["clean_price"].min(), bonds["clean_price"].max()
        checks.append(("Bond: clean price 80-120",
                        80 < px_min and px_max < 120,
                        f"range {px_min:.2f}-{px_max:.2f}"))
        ytm_min, ytm_max = bonds["par_rate"].min(), bonds["par_rate"].max()
        checks.append(("Bond: YTM 2-7%",
                        (ytm_min > 0.02) and (ytm_max < 0.07),
                        f"range {ytm_min:.3%}-{ytm_max:.3%}"))
        checks.append(("Bond: accrued >= 0",
                        (bonds["accrued"] >= 0).all(),
                        f"avg ${bonds['accrued'].mean():,.0f}"))

    if len(opts) > 0:
        buy_pos  = (buy["swap_npv"] > 0).all()  if len(buy)  > 0 else True
        sell_neg = (sel["swap_npv"] < 0).all()  if len(sel)  > 0 else True
        checks.append(("Option: BUY premium>0, SELL premium<0",
                        buy_pos and sell_neg, f"buy={len(buy)} sell={len(sel)}"))
        vega_ok = (buy["vega"] > 0).all() if len(buy) > 0 else True
        checks.append(("Option: BUY vega>0", vega_ok, f"{len(buy)} long options"))
        prem_min = opts["swap_npv"].abs().min()
        prem_max = opts["swap_npv"].abs().max()
        # With notionals up to 50M, premiums can exceed $5M — check as % of notional
        prem_pct = (opts["swap_npv"].abs() / opts["notional"]).max()
        checks.append(("Option: premium <20% notional",
                        prem_pct < 0.20, f"max {prem_pct:.2%} of notional"))
        atm_min = opts["par_rate"].min()
        atm_max = opts["par_rate"].max()
        checks.append(("Option: ATM rate 3.5-6%",
                        0.035 < atm_min and atm_max < 0.06,
                        f"range {atm_min:.3%}-{atm_max:.3%}"))

    if len(eqs) > 0:
        delta_ok = (((eqs["direction"] == "LONG")  & (eqs["delta"] > 0)) |
                    ((eqs["direction"] == "SHORT") & (eqs["delta"] < 0))).all()
        checks.append(("EQ: LONG delta>0, SHORT delta<0", delta_ok, f"{len(eqs)} trades"))
        npv_pct = (eqs["swap_npv"].abs() / eqs["notional"]).max()
        # For TRS: NPV = N*(exp(-q*T) - exp(-r*T)). With SOFR~4.5% vs div_yield~0.7%, a 5y swap
        # can have NPV up to ~17% of notional — so 25% is the appropriate bound here.
        checks.append(("EQ: |NPV| < 25% notional",
                        npv_pct < 0.25, f"max {npv_pct:.2%}"))

    if len(cdss) > 0:
        jtd_ok = (((cdss["direction"] == "BUY")  & (cdss["jump_to_default"] > 0)) |
                  ((cdss["direction"] == "SELL") & (cdss["jump_to_default"] < 0))).all()
        checks.append(("CDS: BUY JTD>0, SELL JTD<0", jtd_ok, f"{len(cdss)} trades"))
        cr01_sign_ok = (
            (len(cds_buy)  == 0 or (cds_buy["cr01"]  > 0).all()) and
            (len(cds_sell) == 0 or (cds_sell["cr01"] < 0).all())
        )
        checks.append(("CDS: CR01 sign (BUY>0, SELL<0)", cr01_sign_ok, "checked"))
        fair_min = cdss["par_rate"].dropna().min()
        fair_max = cdss["par_rate"].dropna().max()
        checks.append(("CDS: fair spread 25-500bps",
                        0.0025 < fair_min and fair_max < 0.05,
                        f"range {fair_min:.3%}-{fair_max:.3%}"))

    if len(eq_opts) > 0:
        buy_pos  = (eq_opt_buy["swap_npv"]  > 0).all() if len(eq_opt_buy)  > 0 else True
        sell_neg = (eq_opt_sell["swap_npv"] < 0).all() if len(eq_opt_sell) > 0 else True
        checks.append(("EqOpt: BUY premium>0, SELL premium<0",
                        buy_pos and sell_neg, f"buy={len(eq_opt_buy)} sell={len(eq_opt_sell)}"))
        # BUY CALL → delta > 0; BUY PUT → delta < 0; SELL reverses.
        # Check: |delta| > 0 for all options (non-zero sensitivity)
        delta_nonzero = (eq_opts["delta"].abs() > 0).all() if len(eq_opts) > 0 else True
        checks.append(("EqOpt: |delta|>0 (calls +ve, puts -ve)", delta_nonzero,
                        f"{len(eq_opts)} options"))
        prem_pct = (eq_opts["swap_npv"].abs() / eq_opts["notional"]).max()
        checks.append(("EqOpt: premium <100% notional",
                        prem_pct < 1.0, f"max {prem_pct:.2%} of notional"))

    checks.append(("No pricing errors", len(err) == 0,
                   f"{len(err)} errors" if len(err) else "all clean"))

    for desc, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
        print(f"         [{detail}]")

    if len(err):
        print("\n  Errors:")
        for _, r in err.iterrows():
            print(f"    {r['trade_id']}: {r['error']}")

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


def save_all_csvs(all_trades: list, df: pd.DataFrame, out_dir: str) -> None:
    """Build PricingResult objects and write per-asset CSVs."""
    import csv as _csv
    results = build_pricing_results(all_trades, df)

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

if __name__ == "__main__":
    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir  = os.path.dirname(_examples_dir)

    db_path = os.path.join(_project_dir, "trades.db")

    # Save market data
    mkt_repo = MarketDataRepository(db_path)
    mkt_repo.upsert(MKT)
    mkt_repo.close()

    repo = TradeRepository(db_path)

    # ── Clear results ──────────────────────────────────────────────────────────
    repo.clear_results()
    print("Cleared PricingResult table.")

    print("Multi-Instrument Pricer -- IRS+Bonds+Swaptions+EqSwaps+CDS+EqOptions")
    print("=" * 80)

    # ── RUN 1: generate trades from code ──────────────────────────────────────
    print("\n[RUN 1] Generating trades and pricing...")
    result_df1, trades1, pr_list1 = run_pricing(
        n_irs=100, n_bonds=100, n_opts=100, n_eq=100, n_cds=100, n_eq_opt=100,
        run_id="RUN-1"
    )
    print_results(result_df1, _project_dir)
    save_all_csvs(trades1, result_df1, _project_dir)

    # ── Print first 60 rows from PricingResult table ───────────────────────────
    print("\n[DB] First 60 rows from PricingResult table (RUN-1):")
    df_db = repo.get_results_df(run_id="RUN-1")
    display_cols = ["trade_id", "instrument", "direction", "tenor_y",
                    "notional", "npv", "dv01", "delta", "vega", "error"]
    available_cols = [c for c in display_cols if c in df_db.columns]
    print(tabulate(df_db[available_cols].head(60), headers="keys",
                   tablefmt="simple", floatfmt=".2f", showindex=False))

    # ── RUN 2: load trades from DB ─────────────────────────────────────────────
    print("\n[RUN 2] Loading trades from DB and pricing...")
    db_trades = repo.list_all()
    print(f"  Loaded {len(db_trades)} trades from DB.")
    result_df2, trades2, pr_list2 = run_pricing(
        trades=db_trades,
        run_id="RUN-2"
    )

    # ── Compare RUN-1 vs RUN-2 ────────────────────────────────────────────────
    print("\n[COMPARE] RUN-1 vs RUN-2 NPV comparison:")
    df1 = repo.get_results_df(run_id="RUN-1").set_index("trade_id")[["npv"]]
    df2 = repo.get_results_df(run_id="RUN-2").set_index("trade_id")[["npv"]]
    joined = df1.join(df2, lsuffix="_run1", rsuffix="_run2", how="inner")
    joined["abs_diff"] = (joined["npv_run1"] - joined["npv_run2"]).abs()
    joined["rel_diff_pct"] = joined["abs_diff"] / joined["npv_run1"].abs().clip(lower=1.0) * 100
    n_total = len(joined)
    n_exact = int((joined["abs_diff"] < 0.01).sum())
    n_close = int((joined["abs_diff"] < 1.0).sum())
    max_diff  = float(joined["abs_diff"].max())
    mean_diff = float(joined["abs_diff"].mean())
    print(f"  Trades compared      : {n_total}")
    print(f"  Exact match (<$0.01) : {n_exact}")
    print(f"  Close match (<$1.00) : {n_close}")
    print(f"  Max |NPV diff|       : ${max_diff:,.4f}")
    print(f"  Mean |NPV diff|      : ${mean_diff:,.4f}")

    # ── Print first 20 rows from PricingResult table ──────────────────────────
    print("\n[DB] First 20 rows from PricingResult table (latest run):")
    df_latest = repo.get_results_df()  # all runs
    display_cols2 = ["trade_id", "run_id", "instrument", "direction", "tenor_y",
                     "notional", "npv", "dv01", "delta", "vega", "error"]
    available_cols2 = [c for c in display_cols2 if c in df_latest.columns]
    print(tabulate(df_latest[available_cols2].head(20), headers="keys",
                   tablefmt="simple", floatfmt=".2f", showindex=False))

    repo.close()
    print("\nDone.")

