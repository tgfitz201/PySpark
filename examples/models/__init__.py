"""
models/__init__.py
==================
Public API for the trade model package.
"""

from models.enums import TradeDirection, LegType, OptionType, ExerciseType
from models.leg import (BaseLeg, FixedLeg, FloatLeg, OptionLeg, EquityLeg,
                        CreditLeg, CDSPremiumLeg, CDSProtectionLeg, EquityOptionLeg)
from models.trade_reference import TradeReference
from models.trade_base import TradeBase
from models.vanilla_swap import VanillaSwap
from models.interest_rate_swap import InterestRateSwap
from models.cross_currency_swap import CrossCurrencySwap
from models.bond import Bond
from models.option_trade import OptionTrade
from models.equity_swap import EquitySwap
from models.cds import CreditDefaultSwap
from models.equity_option import EquityOptionTrade
from models.interest_rate_swaption import InterestRateSwaption
from models.callable_bond import CallableBond
from models.pricing_result import PricingResult
from models.market_data import MarketDataSnapshot, MarketDataCache, make_default_snapshot

__all__ = [
    # enums
    "TradeDirection", "LegType", "OptionType", "ExerciseType",
    # legs
    "BaseLeg", "FixedLeg", "FloatLeg", "OptionLeg", "EquityLeg",
    "CreditLeg", "CDSPremiumLeg", "CDSProtectionLeg", "EquityOptionLeg",
    # reference
    "TradeReference",
    # trades
    "TradeBase", "VanillaSwap", "InterestRateSwap", "CrossCurrencySwap",
    "Bond", "CallableBond", "OptionTrade", "InterestRateSwaption", "EquitySwap", "CreditDefaultSwap",
    "EquityOptionTrade",
    # pricing result
    "PricingResult",
    # market data
    "MarketDataSnapshot", "MarketDataCache", "make_default_snapshot",
]
