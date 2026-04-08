"""
models/leg.py
=============
BaseLeg          — unified leg descriptor for all trade types.
FixedLeg         — fixed coupon leg subclass (rate known at inception).
FloatLeg         — floating rate leg subclass (reset from index + spread).
OptionLeg        — extends BaseLeg with option-specific fields.
EquityLeg        — extends BaseLeg with equity total-return fields.
CreditLeg        — extends BaseLeg with CDS / credit protection fields.
CDSPremiumLeg    — CDS fee/running-spread leg  (extends FixedLeg).
CDSProtectionLeg — CDS contingent/default leg  (extends CreditLeg).
EquityOptionLeg  — vanilla equity option        (extends BaseLeg).

leg_type values (LegType enum)
-------------------------------
"FIXED"          — fixed coupon leg  (swap fixed side, 30/360 SA)
"FLOAT"          — floating SOFR/IBOR leg  (swap floating side)
"BOND"           — coupon + redemption  (QuantLib FixedRateBond)
"OPTION"         — rate option / swaption  (OptionLeg subclass)
"EQUITY"         — equity total-return leg  (EquityLeg subclass)
"CREDIT"         — CDS protection leg       (CreditLeg subclass)
"CDS_PREMIUM"    — CDS running-spread payments  (CDSPremiumLeg subclass)
"CDS_PROTECTION" — CDS contingent default payment (CDSProtectionLeg subclass)
"EQUITY_OPTION"  — vanilla equity option    (EquityOptionLeg subclass)

All string convention keys map to QuantLib objects inside the pricer;
nothing QuantLib-specific lives here so this module has no QL dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class BaseLeg:
    """
    Unified leg descriptor for FIXED, FLOAT, and BOND cash-flow streams.

    Required fields (positional)
    ----------------------------
    leg_type    : "FIXED" | "FLOAT" | "BOND" | "OPTION"
    notional    : face / notional amount in trade currency
    start_date  : first accrual / effective date
    end_date    : final accrual / maturity date

    Common schedule conventions (all have defaults)
    ------------------------------------------------
    currency    : ISO 4217 code          (default "USD")
    calendar    : holiday calendar key   (default "US_GOVT")
    day_count   : accrual basis key      (default "30/360")
    bdc         : business-day convention (default "MOD_FOLLOWING")
    frequency   : coupon/reset frequency  (default "SEMIANNUAL")

    FIXED / BOND
    ------------
    coupon_rate : annual coupon (decimal)  e.g. 0.045 = 4.5 %

    FLOAT
    -----
    spread        : spread over index (decimal, default 0.0)
    index_name    : index key  e.g. "SOFR3M"
    index_tenor_m : reset tenor in months (default 3)
    fixing_lag    : fixing-to-start lag in business days (default 2)

    BOND
    ----
    redemption      : final redemption as % of face (default 100.0)
    settlement_days : T+N settlement lag (default 2)
    issue_date      : bond dated/issue date (Optional; if None → start_date)
    """

    # ── required (positional) ─────────────────────────────────────────────────
    leg_type:   str          # LegType value as string
    notional:   float
    start_date: date
    end_date:   date

    # ── schedule conventions ──────────────────────────────────────────────────
    currency:  str = "USD"
    calendar:  str = "US_GOVT"
    day_count: str = "30/360"
    bdc:       str = "MOD_FOLLOWING"
    frequency: str = "SEMIANNUAL"

    # ── FIXED / BOND ──────────────────────────────────────────────────────────
    coupon_rate: float = 0.0

    # ── FLOAT ─────────────────────────────────────────────────────────────────
    spread:        float = 0.0
    index_name:    str   = "SOFR3M"
    index_tenor_m: int   = 3
    fixing_lag:    int   = 2

    # ── BOND ──────────────────────────────────────────────────────────────────
    redemption:      float         = 100.0
    settlement_days: int           = 2
    issue_date:      Optional[date] = None


@dataclass
class FixedLeg(BaseLeg):
    """
    Fixed coupon leg — rate is known at trade inception.
    Inherits all BaseLeg fields; leg_type should be 'FIXED'.
    Brings coupon_rate explicitly to subclass level for clarity.
    """
    # coupon_rate already on BaseLeg; re-declare for clarity and type safety
    coupon_rate: float = 0.0
    day_count: str = "30/360"
    frequency: str = "SEMIANNUAL"


@dataclass
class FloatLeg(BaseLeg):
    """
    Floating rate leg — reset each period from an index + spread.
    Inherits all BaseLeg fields; leg_type should be 'FLOAT'.
    Brings floating-specific fields explicitly to subclass level.
    """
    spread: float = 0.0
    index_name: str = "SOFR3M"
    index_tenor_m: int = 3
    fixing_lag: int = 2
    day_count: str = "ACT/360"
    frequency: str = "QUARTERLY"


@dataclass
class OptionLeg(BaseLeg):
    """
    Option leg — inherits all BaseLeg fields and adds option-specific terms.

    The BaseLeg positional fields map as follows for options:
      leg_type   = "OPTION"
      notional   = option / swaption notional
      start_date = option expiry date (underlying swap starts here)
      end_date   = underlying maturity date

    Option-specific fields
    ----------------------
    strike           : fixed rate / price of underlying  (decimal)
    option_type      : OptionType value as string
                       "PAYER_SWAPTION" | "RECEIVER_SWAPTION" | "CAP" | "FLOOR"
    exercise_type    : ExerciseType value  "EUROPEAN" | "AMERICAN" | "BERMUDAN"
    vol              : implied volatility
                       lognormal: decimal (e.g. 0.40 = 40 %)
                       normal: decimal (e.g. 0.0100 = 100 bps)
    vol_type         : "LOGNORMAL" | "NORMAL"
    vol_shift        : shift for shifted-lognormal model (default 0.03 = 3 %)
    underlying_tenor_m : underlying swap/bond tenor in months (60 = 5 y)
    underlying_type  : "SWAP" | "BOND"

    Inherited BaseLeg fields used by OptionLeg
    -------------------------------------------
    day_count / frequency / calendar / bdc → underlying swap conventions
    coupon_rate → ignored (use strike instead)
    """

    # ── all have defaults so positional BaseLeg args still work ───────────────
    strike:            float         = 0.05
    option_type:       str           = "PAYER_SWAPTION"
    exercise_type:     str           = "EUROPEAN"
    vol:               float         = 0.40
    vol_type:          str           = "LOGNORMAL"
    vol_shift:         float         = 0.03
    underlying_tenor_m: int          = 60      # months
    underlying_type:   str           = "SWAP"


@dataclass
class EquityLeg(BaseLeg):
    """
    Equity total-return leg — receives/pays equity return on a notional.

    The BaseLeg positional fields map as follows:
      leg_type   = "EQUITY"
      notional   = notional amount in currency
      start_date = effective / reset date
      end_date   = maturity / final fixing date

    Equity-specific fields
    ----------------------
    underlying_ticker  : equity identifier  e.g. "SPY", "SX5E", "AAPL"
    initial_price      : S0 at trade inception (for P&L reference)
    dividend_yield     : continuous annual dividend yield  (e.g. 0.015 = 1.5%)
    equity_return_type : "TOTAL"  → price return + reinvested dividends
                         "PRICE" → price return only (no dividends)
                         "FUNDED"→ price return less funding cost
    reset_frequency    : periodic reset interval  "ANNUAL" | "QUARTERLY"
    participation_rate : leverage factor on equity return  (default 1.0 = 100%)
    funding_spread     : additional spread over SOFR on funding leg  (decimal)
    """

    underlying_ticker:  str   = "SPY"
    initial_price:      float = 100.0         # S0
    dividend_yield:     float = 0.015         # q
    equity_return_type: str   = "TOTAL"       # TOTAL | PRICE | FUNDED
    reset_frequency:    str   = "ANNUAL"
    participation_rate: float = 1.0
    funding_spread:     float = 0.0           # spread over SOFR on funding leg


@dataclass
class CreditLeg(BaseLeg):
    """
    Credit protection leg — the building block for CDS and credit-linked notes.

    The BaseLeg positional fields map as follows:
      leg_type   = "CREDIT"
      notional   = protection notional
      start_date = CDS effective / trade date
      end_date   = scheduled maturity date

    Credit-specific fields
    ----------------------
    reference_entity : reference obligor name  e.g. "FORD MOTOR CO"
    credit_spread    : running (par) CDS spread in decimal  (0.015 = 150 bps)
    recovery_rate    : assumed LGD recovery fraction  (0.40 = 40%)
    hazard_rate      : flat hazard rate; if 0.0 implied from spread
                       λ ≈ spread / (1 − recovery)
    seniority        : "SENIOR_UNSECURED" | "SUBORDINATED" | "SENIOR_SECURED"
    doc_clause       : ISDA documentation clause
                       "CR"  → complete restructuring
                       "MR"  → modified restructuring
                       "XR"  → no restructuring
                       "MM"  → mod-mod restructuring
    upfront_fee      : upfront payment as fraction of notional  (default 0.0)
    step_up_spread   : spread step-up after a credit event  (default 0.0)
    """

    reference_entity: str   = "CORP"
    credit_spread:    float = 0.015           # par CDS spread (decimal)
    recovery_rate:    float = 0.40
    hazard_rate:      float = 0.0             # 0 = imply from spread+recovery
    seniority:        str   = "SENIOR_UNSECURED"
    doc_clause:       str   = "CR"
    upfront_fee:      float = 0.0
    step_up_spread:   float = 0.0


@dataclass
class CDSPremiumLeg(FixedLeg):
    """
    CDS fee / running-spread leg  (leg_type = "CDS_PREMIUM").

    What it is
    ----------
    The protection BUYER pays this leg: a stream of fixed periodic payments
    calculated as  coupon_rate × notional × day_count_fraction  each period.
    It is the "price" the buyer pays for credit protection.

    How it is like a FixedLeg
    -------------------------
    CDSPremiumLeg IS-A FixedLeg in every structural sense:
      • coupon_rate  → the running CDS spread  (e.g. 0.015 = 150 bps)
      • notional     → protection notional (same as CDSProtectionLeg)
      • start/end    → CDS effective / scheduled maturity
    The only differences from a vanilla IRS FixedLeg are market conventions:
      • day_count  = ACT/360   (ISDA standard, not 30/360 like an IRS)
      • frequency  = QUARTERLY (ISDA standard, not SEMIANNUAL like an IRS)
    In QuantLib terms this maps to  ql.Schedule(quarterly) + ql.Actual360()
    exactly as used in  ql.CreditDefaultSwap(…, schedule, …, dc).

    What it is NOT
    --------------
    It does NOT model the contingent default payment; that is CDSProtectionLeg.
    It is NOT a FloatLeg — the spread payment amount is fixed at trade inception
    (the running spread does not reset from an index).

    Additional field
    ----------------
    accrued_on_default : bool  (default True)
        If True, the premium accrued since the last coupon date is also paid
        to the protection seller on the credit-event settlement date — standard
        ISDA (standard North-American CDS) behaviour.
    """

    # ISDA CDS premium-leg conventions (override FixedLeg defaults)
    day_count: str = "ACT/360"
    frequency: str = "QUARTERLY"

    # CDS-specific: accrued coupon is paid to seller on default settlement
    accrued_on_default: bool = True


@dataclass
class CDSProtectionLeg(CreditLeg):
    """
    CDS contingent / protection leg  (leg_type = "CDS_PROTECTION").

    What it is
    ----------
    The protection SELLER pays this leg upon a credit event:
        payment = (1 − recovery_rate) × notional
    This is a SINGLE contingent cash flow (not periodic) — it is triggered
    by a credit event (failure to pay, bankruptcy, restructuring under the
    doc_clause) and paid on the credit-event settlement date.

    How it is like a CreditLeg
    --------------------------
    CDSProtectionLeg IS-A CreditLeg; it inherits all reference-entity economics
    needed to value the contingent payment:
      • reference_entity  → the obligor whose default triggers the payment
      • credit_spread     → par running spread; used to imply hazard_rate
      • recovery_rate     → assumed recovery fraction (e.g. 0.40 = 40 %)
      • hazard_rate       → flat default intensity λ = spread / (1 − recovery)
      • seniority         → affects recovery assumption in practice
      • doc_clause        → defines qualifying credit events (CR / MR / XR / MM)
      • upfront_fee       → upfront points paid in addition to the running spread
    In QuantLib the protection leg NPV is obtained from
        ql.CreditDefaultSwap(…).protectionNPV()
    using a FlatHazardRate curve built from hazard_rate above.

    What it is NOT
    --------------
    It is NOT a periodic cash-flow leg — there is no schedule or coupon_rate.
    The BaseLeg.coupon_rate / frequency / day_count fields are inherited but
    unused for valuation; they are kept only for structural consistency with
    the rest of the leg hierarchy.
    The PREMIUM (periodic spread payments) is modelled by CDSPremiumLeg.

    How the two legs fit together
    -----------------------------
    For the protection BUYER:
        NPV = protectionLegNPV  −  premiumLegNPV
            = E[PV of (1−R)×N on default]  −  PV of quarterly spread payments
    QuantLib's MidPointCdsEngine computes both internally;
    we store the split so the risk system can show P&L attribution.
    """

    # Override leg_type to distinguish from bare CreditLeg.
    # Note: cannot be a dataclass field default (MRO constraint);
    # pass "CDS_PROTECTION" as the first positional argument when constructing.
    pass
@dataclass
class EquityOptionLeg(BaseLeg):
    """
    Equity option leg — vanilla call or put on an equity underlying.

    The BaseLeg positional fields map as follows:
      leg_type   = "EQUITY_OPTION"
      notional   = contract notional (number of contracts × lot size × price)
      start_date = trade date / premium payment date
      end_date   = option expiry date

    Equity option-specific fields
    -----------------------------
    underlying_ticker : equity identifier  e.g. "SPY", "AAPL", "SX5E"
    initial_price     : S0 — spot price at trade inception
    strike            : option strike price (absolute, same currency as S0)
    option_type       : "CALL" or "PUT"
    exercise_type     : "EUROPEAN" or "AMERICAN"
    vol               : implied volatility (e.g. 0.25 = 25%)
    dividend_yield    : continuous annual dividend yield (e.g. 0.013)
    risk_free_rate    : risk-free rate override; 0.0 = use SOFR curve
    pricing_model     : "BLACK_SCHOLES" (European) | "BINOMIAL" (American)
    n_steps           : binomial tree steps for AMERICAN pricing (default 200)
    lot_size          : contracts per lot (default 100 for equity options)
    """

    underlying_ticker: str   = "SPY"
    initial_price:     float = 100.0
    strike:            float = 100.0
    option_type:       str   = "CALL"        # CALL | PUT
    exercise_type:     str   = "EUROPEAN"    # EUROPEAN | AMERICAN
    vol:               float = 0.25
    dividend_yield:    float = 0.013
    risk_free_rate:    float = 0.0           # 0 = use SOFR curve
    pricing_model:     str   = "BLACK_SCHOLES"
    n_steps:           int   = 200
    lot_size:          int   = 100
