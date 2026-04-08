import os
import sys
import logging

# ── Silence all Spark / Java / py4j log noise before importing PySpark ────────
# Force workers to use the same Python executable as the driver (critical for PyCharm)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
# Force PySpark from conda, overriding any SPARK_HOME set in shell profile
os.environ["SPARK_HOME"] = "/opt/anaconda3/lib/python3.13/site-packages/pyspark"
os.environ.setdefault("JAVA_HOME", "/usr/local/opt/openjdk@17")
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--conf spark.driver.extraJavaOptions="
    "-Dlog4j2.rootLogger.level=OFF "
    "--conf spark.executor.extraJavaOptions="
    "-Dlog4j2.rootLogger.level=OFF "
    "pyspark-shell"
)
logging.getLogger("py4j").setLevel(logging.CRITICAL)
logging.getLogger("pyspark").setLevel(logging.CRITICAL)

import pandas as pd
import QuantLib as ql
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Dict, Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, DateType
)
from pyspark.sql.functions import pandas_udf


# ============================================================
# 1. Dataclasses
# ============================================================

@dataclass
class TradeData:
    trade_id: str
    book: str
    ccy: str
    counterparty: str
    valuation_date: date
    notional: float

@dataclass
class OptionData:
    option_type: str
    exercise_style: str
    expiry_date: date
    flat_vol: float

@dataclass
class SwapLeg:
    leg_id: str
    leg_type: str
    start_date: date
    maturity_date: date
    fixed_rate: Optional[float]
    spread: Optional[float]
    frequency: str
    day_count: str
    index_tenor_months: Optional[int]

@dataclass
class SwaptionTrade:
    trade: TradeData
    option: OptionData
    legs: List[SwapLeg]

    def to_flat_row(self) -> Dict[str, Any]:
        fixed_leg = next(l for l in self.legs if l.leg_type == "FIXED")
        float_leg = next(l for l in self.legs if l.leg_type == "FLOAT")

        return {
            "trade_id": self.trade.trade_id,
            "book": self.trade.book,
            "ccy": self.trade.ccy,
            "counterparty": self.trade.counterparty,
            "valuation_date": self.trade.valuation_date,
            "notional": self.trade.notional,
            "option_type": self.option.option_type,
            "exercise_style": self.option.exercise_style,
            "expiry_date": self.option.expiry_date,
            "flat_vol": self.option.flat_vol,
            "fixed_start_date": fixed_leg.start_date,
            "fixed_maturity_date": fixed_leg.maturity_date,
            "fixed_rate": fixed_leg.fixed_rate,
            "fixed_frequency": fixed_leg.frequency,
            "fixed_day_count": fixed_leg.day_count,
            "float_start_date": float_leg.start_date,
            "float_maturity_date": float_leg.maturity_date,
            "spread": float_leg.spread,
            "float_frequency": float_leg.frequency,
            "float_day_count": float_leg.day_count,
            "index_tenor_months": float_leg.index_tenor_months,
        }


# ============================================================
# 2. Synthetic Market Data (local dev / testing)
# ============================================================

def make_curve_data() -> pd.DataFrame:
    """Synthetic OIS and LIBOR3M zero curves for USD."""
    rows = []
    for curve_type, base in [("OIS", 0.045), ("LIBOR3M", 0.050)]:
        for tenor in [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]:
            rows.append({
                "ccy": "USD",
                "curve_type": curve_type,
                "tenor_years": tenor,
                "zero_rate": base + 0.001 * tenor,
            })
    return pd.DataFrame(rows)


def make_vol_surface() -> pd.DataFrame:
    """Synthetic swaption vol surface for USD covering all generated tenors."""
    rows = []
    for expiry in [1, 2, 3, 5, 7, 10]:
        for tenor in [2, 3, 5, 7, 10, 15, 20]:
            rows.append({
                "ccy": "USD",
                "expiry_years": float(expiry),
                "swap_tenor_years": float(tenor),
                "vol": max(0.10, 0.22 + 0.005 * expiry - 0.003 * tenor),
            })
    return pd.DataFrame(rows)


def make_trade_data(n: int = 100) -> List[Dict[str, Any]]:
    """Generate n synthetic swaption trades with varied parameters."""
    import random
    random.seed(42)

    base = date(2024, 1, 15)

    expiry_choices    = [1, 2, 3, 5, 7, 10]
    tenor_choices     = [2, 3, 5, 7, 10, 15, 20]
    notional_choices  = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
    rate_range        = (0.025, 0.060)
    vol_range         = (0.15, 0.30)
    option_types      = ["PAYER", "RECEIVER"]
    books             = ["IRD-NY", "IRD-LDN", "IRD-ASIA", "MACRO"]
    counterparties    = [f"CPTY-{c:02d}" for c in range(1, 11)]

    trades = []
    for i in range(n):
        expiry_y   = random.choice(expiry_choices)
        tenor_y    = random.choice(tenor_choices)
        notional   = float(random.choice(notional_choices))
        fixed_rate = round(random.uniform(*rate_range), 4)
        flat_vol   = round(random.uniform(*vol_range), 4)
        opt_type   = random.choice(option_types)

        expiry    = date(base.year + expiry_y, base.month, base.day)
        swap_end  = date(expiry.year + tenor_y, expiry.month, expiry.day)

        trades.append({
            "trade_id":            f"SWN-{i+1:04d}",
            "book":                random.choice(books),
            "ccy":                 "USD",
            "counterparty":        random.choice(counterparties),
            "valuation_date":      base,
            "notional":            notional,
            "option_type":         opt_type,
            "exercise_style":      "European",
            "expiry_date":         expiry,
            "flat_vol":            flat_vol,
            "fixed_start_date":    expiry,
            "fixed_maturity_date": swap_end,
            "fixed_rate":          fixed_rate,
            "fixed_frequency":     "Semiannual",
            "fixed_day_count":     "30/360",
            "float_start_date":    expiry,
            "float_maturity_date": swap_end,
            "spread":              0.0,
            "float_frequency":     "Quarterly",
            "float_day_count":     "ACT/360",
            "index_tenor_months":  3,
        })
    return trades


# ============================================================
# 3. Market Data Builders
# ============================================================

def build_zero_curve(valuation_date: ql.Date, rows: pd.DataFrame) -> ql.YieldTermStructureHandle:
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.Settlement)

    helpers = []
    for _, r in rows.iterrows():
        tenor = r["tenor_years"]
        rate = r["zero_rate"]
        helpers.append(
            ql.DepositRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(rate)),
                ql.Period(int(tenor * 12), ql.Months),
                2,
                calendar,
                ql.ModifiedFollowing,
                False,
                day_count,
            )
        )

    curve = ql.PiecewiseLogCubicDiscount(valuation_date, helpers, day_count)
    return ql.YieldTermStructureHandle(curve)


def build_swaption_vol_surface(valuation_date: ql.Date, rows: pd.DataFrame) -> ql.SwaptionVolatilityStructureHandle:
    calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
    day_count = ql.Actual365Fixed()

    expiries = sorted(rows["expiry_years"].unique())
    tenors   = sorted(rows["swap_tenor_years"].unique())

    vol_matrix = []
    for e in expiries:
        row_vols = []
        for t in tenors:
            subset = rows[(rows["expiry_years"] == e) & (rows["swap_tenor_years"] == t)]
            row_vols.append(float(subset["vol"].iloc[0]))
        vol_matrix.append(row_vols)

    ql_expiries = [ql.Period(int(e * 12), ql.Months) for e in expiries]
    ql_tenors   = [ql.Period(int(t), ql.Years) for t in tenors]

    surface = ql.SwaptionVolatilityMatrix(
        calendar,
        ql.ModifiedFollowing,
        ql_expiries,
        ql_tenors,
        vol_matrix,
        day_count,
    )
    return ql.SwaptionVolatilityStructureHandle(surface)


# ============================================================
# 4. QuantLib Helpers
# ============================================================

def ql_frequency(freq: str) -> int:
    return {
        "Annual":     ql.Annual,
        "Semiannual": ql.Semiannual,
        "Quarterly":  ql.Quarterly,
    }.get(freq, ql.Annual)


def ql_day_count(dc: str) -> ql.DayCounter:
    return {
        "30/360":  ql.Thirty360(ql.Thirty360.BondBasis),
        "ACT/365": ql.Actual365Fixed(),
        "ACT/360": ql.Actual360(),
    }.get(dc, ql.Actual365Fixed())


def ql_swap_type(t: str) -> int:
    return ql.VanillaSwap.Payer if t.upper() == "PAYER" else ql.VanillaSwap.Receiver


# ============================================================
# 5. Core Pricing Function (pure Python — called from UDF)
# ============================================================

def _price_one(row: Dict[str, Any], curve_pdf: pd.DataFrame, vol_pdf: pd.DataFrame) -> Dict[str, float]:
    nan_row = {
        "fixed_npv": float("nan"), "float_npv": float("nan"),
        "swap_npv":  float("nan"), "par_rate":  float("nan"),
        "dv01":      float("nan"), "duration":  float("nan"),
    }
    try:
        ccy = row["ccy"]
        val_dt = row["valuation_date"]
        ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
        ql.Settings.instance().evaluationDate = ql_val

        ois_rows   = curve_pdf[(curve_pdf["ccy"] == ccy) & (curve_pdf["curve_type"] == "OIS")]
        libor_rows = curve_pdf[(curve_pdf["ccy"] == ccy) & (curve_pdf["curve_type"] == "LIBOR3M")]

        ois_curve   = build_zero_curve(ql_val, ois_rows)
        libor_curve = build_zero_curve(ql_val, libor_rows)

        calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
        dc_fixed = ql_day_count(row["fixed_day_count"])
        dc_float = ql_day_count(row["float_day_count"])

        def to_ql(d):
            return ql.Date(d.day, d.month, d.year)

        fixed_schedule = ql.Schedule(
            to_ql(row["fixed_start_date"]), to_ql(row["fixed_maturity_date"]),
            ql.Period(ql_frequency(row["fixed_frequency"])),
            calendar, ql.ModifiedFollowing, ql.ModifiedFollowing,
            ql.DateGeneration.Forward, False,
        )
        float_schedule = ql.Schedule(
            to_ql(row["float_start_date"]), to_ql(row["float_maturity_date"]),
            ql.Period(int(row["index_tenor_months"]), ql.Months),
            calendar, ql.ModifiedFollowing, ql.ModifiedFollowing,
            ql.DateGeneration.Forward, False,
        )
        index = ql.IborIndex(
            "USDLibor3M",
            ql.Period(int(row["index_tenor_months"]), ql.Months),
            2, ql.USDCurrency(), calendar, ql.ModifiedFollowing, False,
            dc_float, libor_curve,
        )

        swap = ql.VanillaSwap(
            ql_swap_type(row["option_type"]),
            float(row["notional"]),
            fixed_schedule, float(row["fixed_rate"]), dc_fixed,
            float_schedule, index, float(row["spread"]), dc_float,
        )
        engine = ql.DiscountingSwapEngine(ois_curve)
        swap.setPricingEngine(engine)

        swap_npv  = swap.NPV()
        fixed_npv = swap.fixedLegNPV()
        float_npv = swap.floatingLegNPV()
        par_rate  = swap.fairRate()

        # DV01: bump OIS by 1bp
        bump = 0.0001
        bumped_ois = ois_rows.copy()
        bumped_ois["zero_rate"] += bump
        ois_bumped_curve = build_zero_curve(ql_val, bumped_ois)

        bumped_libor = libor_rows.copy()
        bumped_libor["zero_rate"] += bump
        libor_bumped_curve = build_zero_curve(ql_val, bumped_libor)

        index_b = ql.IborIndex(
            "USDLibor3M",
            ql.Period(int(row["index_tenor_months"]), ql.Months),
            2, ql.USDCurrency(), calendar, ql.ModifiedFollowing, False,
            dc_float, libor_bumped_curve,
        )
        swap_b = ql.VanillaSwap(
            ql_swap_type(row["option_type"]),
            float(row["notional"]),
            fixed_schedule, float(row["fixed_rate"]), dc_fixed,
            float_schedule, index_b, float(row["spread"]), dc_float,
        )
        swap_b.setPricingEngine(ql.DiscountingSwapEngine(ois_bumped_curve))
        dv01 = swap_b.NPV() - swap_npv

        # Modified duration = -dv01 / (NPV * bump) — guard zero NPV
        duration = abs(dv01 / (abs(swap_npv) * bump)) if abs(swap_npv) > 1e-6 else 0.0

        return {
            "fixed_npv": fixed_npv,
            "float_npv": float_npv,
            "swap_npv":  swap_npv,
            "par_rate":  par_rate,
            "dv01":      dv01,
            "duration":  duration,
        }

    except Exception as exc:
        print(f"[WARN] pricing failed: {exc}")
        return nan_row


# ============================================================
# 6. Pandas UDF (PySpark 4.0 compatible)
# ============================================================

# Market data is broadcast so workers can access it without hitting Spark.
_curve_broadcast = None
_vol_broadcast   = None

RESULT_SCHEMA = StructType([
    StructField("fixed_npv", DoubleType()),
    StructField("float_npv", DoubleType()),
    StructField("swap_npv",  DoubleType()),
    StructField("par_rate",  DoubleType()),
    StructField("dv01",      DoubleType()),
    StructField("duration",  DoubleType()),
])


def make_price_udf():
    """Build the pandas UDF after broadcast variables are set."""
    curve_bc = _curve_broadcast
    vol_bc   = _vol_broadcast

    @pandas_udf(RESULT_SCHEMA)
    def price_swaption_udf(
        trade_id: pd.Series, ccy: pd.Series, option_type: pd.Series,
        exercise_style: pd.Series, expiry_date: pd.Series, valuation_date: pd.Series,
        notional: pd.Series, fixed_start_date: pd.Series, fixed_maturity_date: pd.Series,
        fixed_rate: pd.Series, fixed_frequency: pd.Series, fixed_day_count: pd.Series,
        float_start_date: pd.Series, float_maturity_date: pd.Series, spread: pd.Series,
        float_frequency: pd.Series, float_day_count: pd.Series,
        index_tenor_months: pd.Series, flat_vol: pd.Series,
    ) -> pd.DataFrame:
        curve_pdf = curve_bc.value
        vol_pdf   = vol_bc.value

        records = []
        for i in range(len(trade_id)):
            row = {
                "ccy":                  ccy.iloc[i],
                "option_type":          option_type.iloc[i],
                "valuation_date":       valuation_date.iloc[i],
                "notional":             notional.iloc[i],
                "fixed_start_date":     fixed_start_date.iloc[i],
                "fixed_maturity_date":  fixed_maturity_date.iloc[i],
                "fixed_rate":           fixed_rate.iloc[i],
                "fixed_frequency":      fixed_frequency.iloc[i],
                "fixed_day_count":      fixed_day_count.iloc[i],
                "float_start_date":     float_start_date.iloc[i],
                "float_maturity_date":  float_maturity_date.iloc[i],
                "spread":               spread.iloc[i],
                "float_frequency":      float_frequency.iloc[i],
                "float_day_count":      float_day_count.iloc[i],
                "index_tenor_months":   index_tenor_months.iloc[i],
                "expiry_date":          expiry_date.iloc[i],
            }
            records.append(_price_one(row, curve_pdf, pd.DataFrame()))

        return pd.DataFrame(records)

    return price_swaption_udf


# ============================================================
# 7. Trade Schema
# ============================================================

TRADE_SCHEMA = StructType([
    StructField("trade_id",             StringType()),
    StructField("book",                 StringType()),
    StructField("ccy",                  StringType()),
    StructField("counterparty",         StringType()),
    StructField("valuation_date",       DateType()),
    StructField("notional",             DoubleType()),
    StructField("option_type",          StringType()),
    StructField("exercise_style",       StringType()),
    StructField("expiry_date",          DateType()),
    StructField("flat_vol",             DoubleType()),
    StructField("fixed_start_date",     DateType()),
    StructField("fixed_maturity_date",  DateType()),
    StructField("fixed_rate",           DoubleType()),
    StructField("fixed_frequency",      StringType()),
    StructField("fixed_day_count",      StringType()),
    StructField("float_start_date",     DateType()),
    StructField("float_maturity_date",  DateType()),
    StructField("spread",               DoubleType()),
    StructField("float_frequency",      StringType()),
    StructField("float_day_count",      StringType()),
    StructField("index_tenor_months",   IntegerType()),
])


# ============================================================
# 8. Spark Job
# ============================================================

def run_pricing(n_trades: int = 100):
    global _curve_broadcast, _vol_broadcast

    spark = SparkSession.builder \
        .appName("LocalSwaptionPricer") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.extraJavaOptions",
                "-Dlog4j.rootCategory=OFF,console "
                "-Dlog4j2.rootLogger.level=OFF") \
        .config("spark.executor.extraJavaOptions",
                "-Dlog4j.rootCategory=OFF,console "
                "-Dlog4j2.rootLogger.level=OFF") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")

    # Redirect any remaining stderr noise away from terminal
    _real_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    # Broadcast market data to workers
    curve_pdf = make_curve_data()
    vol_pdf   = make_vol_surface()
    _curve_broadcast = spark.sparkContext.broadcast(curve_pdf)
    _vol_broadcast   = spark.sparkContext.broadcast(vol_pdf)

    trade_rows = make_trade_data(n_trades)
    trades_df  = spark.createDataFrame(trade_rows, schema=TRADE_SCHEMA)

    price_udf = make_price_udf()

    priced = (
        trades_df
        .withColumn("risk", price_udf(
            "trade_id", "ccy", "option_type", "exercise_style",
            "expiry_date", "valuation_date", "notional",
            "fixed_start_date", "fixed_maturity_date", "fixed_rate",
            "fixed_frequency", "fixed_day_count",
            "float_start_date", "float_maturity_date", "spread",
            "float_frequency", "float_day_count", "index_tenor_months",
            "flat_vol",
        ))
        .select(
            "trade_id", "book", "ccy", "counterparty", "option_type",
            "notional", "fixed_rate",
            F.col("risk.fixed_npv").alias("fixed_npv"),
            F.col("risk.float_npv").alias("float_npv"),
            F.col("risk.swap_npv").alias("swap_npv"),
            F.col("risk.par_rate").alias("par_rate"),
            F.col("risk.dv01").alias("dv01"),
            F.col("risk.duration").alias("duration"),
        )
    )

    result_pdf = priced.toPandas()
    spark.stop()

    # Restore stderr so any post-run errors are visible
    sys.stderr.close()
    sys.stderr = _real_stderr
    sys.stdout.flush()

    from tabulate import tabulate

    # Save trade-level CSV to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path     = os.path.join(project_dir, "swap_trades.csv")
    portfolio_path = os.path.join(project_dir, "swap_portfolio.csv")
    result_pdf.to_csv(csv_path, index=False)

    # Confirm collection happened on driver
    print(f"\n✓ Collected {len(result_pdf)} rows on driver node\n", flush=True)

    # Split into two narrower tables so they fit any console width
    d = result_pdf.copy()

    table1 = pd.DataFrame({
        "Trade ID":   d["trade_id"],
        "Book":       d["book"],
        "Cpty":       d["counterparty"],
        "Type":       d["option_type"],
        "Notional $M": (d["notional"] / 1e6).map("{:>6.1f}".format),
        "Fix Rate":   d["fixed_rate"].map("{:.3%}".format),
        "Par Rate":   d["par_rate"].map(lambda x: f"{x:.3%}" if pd.notna(x) else "N/A"),
        "Swap NPV":   d["swap_npv"].map("${:>13,.0f}".format),
    })

    table2 = pd.DataFrame({
        "Trade ID":   d["trade_id"],
        "Fixed NPV":  d["fixed_npv"].map("${:>13,.0f}".format),
        "Float NPV":  d["float_npv"].map("${:>13,.0f}".format),
        "DV01 ($)":   d["dv01"].map("{:>9,.2f}".format),
        "Duration":   d["duration"].map("{:>7.2f}".format),
    })

    sep = "=" * 90
    print(sep, flush=True)
    print("  SWAP RESULTS (1/2) — Trade / Rates / NPV", flush=True)
    print(sep, flush=True)
    print(tabulate(table1, headers="keys", tablefmt="simple", showindex=False), flush=True)

    print(f"\n{sep}", flush=True)
    print("  SWAP RESULTS (2/2) — Leg NPVs / Risk", flush=True)
    print(sep, flush=True)
    print(tabulate(table2, headers="keys", tablefmt="simple", showindex=False), flush=True)

    # Summary block
    num = result_pdf
    payers   = num[num["option_type"] == "PAYER"]
    receivers = num[num["option_type"] == "RECEIVER"]
    print(f"""
{sep}
PORTFOLIO SUMMARY
  Trades                 : {len(num):>6}   (Payer: {len(payers)}  |  Receiver: {len(receivers)})
  Total Notional         : ${num['notional'].sum():>18,.0f}
  Total Fixed Leg NPV    : ${num['fixed_npv'].sum():>18,.2f}
  Total Float Leg NPV    : ${num['float_npv'].sum():>18,.2f}
  Total Swap NPV         : ${num['swap_npv'].sum():>18,.2f}
  Avg Par Rate           : {num['par_rate'].mean():>17.4%}
  Avg Fixed Rate         : {num['fixed_rate'].mean():>17.4%}
  Avg DV01               : ${num['dv01'].mean():>18,.2f}
  Avg Duration           : {num['duration'].mean():>17.2f} yrs
  Payer Total NPV        : ${payers['swap_npv'].sum():>18,.2f}
  Receiver Total NPV     : ${receivers['swap_npv'].sum():>18,.2f}
{sep}
""", flush=True)

    out_path = "/tmp/swap_results.parquet"
    result_pdf.to_parquet(out_path, index=False)

    # Save portfolio summary CSV to project directory
    summary = {
        "metric": [
            "trades", "payers", "receivers",
            "total_notional", "total_fixed_npv", "total_float_npv", "total_swap_npv",
            "avg_par_rate", "avg_fixed_rate", "avg_dv01", "avg_duration",
            "payer_total_npv", "receiver_total_npv"
        ],
        "value": [
            len(num), len(payers), len(receivers),
            num["notional"].sum(), num["fixed_npv"].sum(), num["float_npv"].sum(), num["swap_npv"].sum(),
            num["par_rate"].mean(), num["fixed_rate"].mean(), num["dv01"].mean(), num["duration"].mean(),
            payers["swap_npv"].sum(), receivers["swap_npv"].sum()
        ]
    }
    pd.DataFrame(summary).to_csv(portfolio_path, index=False)

    print(f"Trade CSV   saved to: {csv_path}", flush=True)
    print(f"Portfolio CSV saved : {portfolio_path}", flush=True)


# ============================================================
# 9. Main Entry Point
# ============================================================

def main():
    print("Running local swap pricing job — 100 trades...")
    run_pricing(n_trades=100)
    print("Done.")


if __name__ == "__main__":
    main()