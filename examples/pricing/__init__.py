"""pricing — per-instrument QuantLib pricers."""
from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW
from pricing.swap_pricer import price_swap, price_xccy, price_irs
from pricing.bond_pricer import price_bond
from pricing.optionable_bond_pricer import price_optionable_bond
from pricing.option_pricer import price_option, price_irs_swaption
from pricing.equity_pricer import price_equity_swap, price_equity_option
from pricing.cds_pricer import price_cds
from pricing.asset_swap_pricer import price_asset_swap


def price_trade(trade, curve_df) -> dict:
    """Convenience dispatcher — calls trade.price(curve_df)."""
    return trade.price(curve_df)
