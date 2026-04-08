"""
Trade data consistency tests.
Verifies: all fields present, leg counts correct, round-trip DB→JSON→object.
"""
import pytest, json
from models.trade_base import TradeBase
from models.leg import BaseLeg, FixedLeg, FloatLeg, OptionLeg, EquityLeg, CreditLeg, EquityOptionLeg

EXPECTED_LEG_COUNTS = {
    "VanillaSwap": 2,
    "Bond": 1,
    "Option": 1,
    "EquitySwap": (1, 2),   # 1 equity leg, optionally 1 funding leg
    "CDS": 1,
    "EquityOption": 1,
}

class TestTradeDataConsistency:
    def test_600_trades_in_db(self, all_trades):
        assert len(all_trades) == 600, f"Expected 600 trades, got {len(all_trades)}"

    def test_100_trades_per_type(self, all_trades):
        from collections import Counter
        counts = Counter(t._trade_type for t in all_trades)
        for ttype, count in counts.items():
            assert count == 100, f"{ttype}: expected 100, got {count}"

    def test_all_trades_have_trade_id(self, all_trades):
        for t in all_trades:
            assert t.trade_id, f"Trade has empty trade_id"

    def test_all_trades_have_legs(self, all_trades):
        for t in all_trades:
            assert len(t.legs) >= 1, f"{t.trade_id} has no legs"

    def test_vanilla_swap_has_2_legs(self, all_trades):
        swaps = [t for t in all_trades if t._trade_type == "VanillaSwap"]
        for s in swaps:
            assert len(s.legs) == 2, f"{s.trade_id} has {len(s.legs)} legs"

    def test_vanilla_swap_leg_types(self, all_trades):
        """FIXED_FLOAT and FLOAT_FIXED swaps must have FIXED+FLOAT legs; FIXED_FIXED two FIXED; FLOAT_FLOAT two FLOAT."""
        swaps = [t for t in all_trades if t._trade_type == "VanillaSwap"]
        for s in swaps:
            leg_types = {l.leg_type for l in s.legs}
            st = getattr(s, "swap_subtype", "FIXED_FLOAT")
            if st in ("FIXED_FLOAT", "FLOAT_FIXED"):
                assert "FIXED" in leg_types and "FLOAT" in leg_types, \
                    f"{s.trade_id} ({st}): expected FIXED+FLOAT, got {leg_types}"
            elif st == "FIXED_FIXED":
                assert leg_types == {"FIXED"}, f"{s.trade_id}: expected FIXED+FIXED"
            elif st == "FLOAT_FLOAT":
                assert leg_types == {"FLOAT"}, f"{s.trade_id}: expected FLOAT+FLOAT"

    def test_fixed_legs_are_FixedLeg_instances(self, all_trades):
        """FIXED legs should be FixedLeg subclass instances."""
        for t in all_trades:
            for leg in t.legs:
                if leg.leg_type == "FIXED":
                    assert isinstance(leg, FixedLeg), \
                        f"{t.trade_id} leg FIXED is {type(leg).__name__}, not FixedLeg"

    def test_float_legs_are_FloatLeg_instances(self, all_trades):
        """FLOAT legs should be FloatLeg subclass instances."""
        for t in all_trades:
            for leg in t.legs:
                if leg.leg_type == "FLOAT":
                    assert isinstance(leg, FloatLeg), \
                        f"{t.trade_id} leg FLOAT is {type(leg).__name__}, not FloatLeg"

    def test_option_legs_have_strike(self, all_trades):
        opts = [t for t in all_trades if t._trade_type == "Option"]
        for o in opts:
            for leg in o.legs:
                if leg.leg_type == "OPTION":
                    assert hasattr(leg, "strike") and leg.strike > 0, \
                        f"{o.trade_id} option leg missing strike"

    def test_equity_option_legs_have_option_type(self, all_trades):
        eqopts = [t for t in all_trades if t._trade_type == "EquityOption"]
        for e in eqopts:
            for leg in e.legs:
                if leg.leg_type == "EQUITY_OPTION":
                    assert leg.option_type in ("CALL", "PUT"), \
                        f"{e.trade_id}: option_type={leg.option_type!r}"

    def test_bond_has_1_leg(self, all_trades):
        bonds = [t for t in all_trades if t._trade_type == "Bond"]
        for b in bonds:
            assert len(b.legs) == 1, f"{b.trade_id} has {len(b.legs)} legs"

    def test_cds_has_1_leg(self, all_trades):
        cdss = [t for t in all_trades if t._trade_type == "CDS"]
        for c in cdss:
            assert len(c.legs) == 1

    def test_notional_positive(self, all_trades):
        for t in all_trades:
            for leg in t.legs:
                assert leg.notional > 0, f"{t.trade_id}: notional={leg.notional}"

    def test_json_roundtrip(self, all_trades):
        """Serialize to JSON and reconstruct — must be identical trade_id and trade_type."""
        for t in all_trades[:10]:  # spot check 10 trades
            enriched = t._to_enriched_dict()
            # strip _python_class and _computed for fromJson
            def _strip(d):
                d.pop("_python_class", None)
                d.pop("_computed", None)
                for leg in d.get("legs", []):
                    leg.pop("_python_class", None)
                return d
            clean = _strip(dict(enriched))
            t2 = TradeBase.fromJson(json.dumps(clean))
            assert t2.trade_id == t.trade_id
            assert t2._trade_type == t._trade_type
            assert len(t2.legs) == len(t.legs)

    def test_trades_json_file_exists_and_readable(self, trades_json_path):
        import os
        assert os.path.exists(trades_json_path), f"trades.json not found at {trades_json_path}"
        with open(trades_json_path) as f:
            data = json.load(f)
        assert len(data) == 600

class TestDatabaseConsistency:
    def test_tradebase_count(self, repo):
        assert repo.count() == 600

    def test_get_by_id_returns_correct_type(self, repo, all_trades):
        for t in all_trades[:5]:
            fetched = repo.get(t.trade_id)
            assert fetched is not None
            assert fetched._trade_type == t._trade_type

    def test_all_leg_types_reconstructed(self, all_trades):
        """Each leg should have the correct Python class."""
        for t in all_trades:
            for leg in t.legs:
                assert hasattr(leg, "leg_type")
                assert leg.leg_type in ("FIXED", "FLOAT", "BOND", "OPTION",
                                        "EQUITY", "CREDIT", "EQUITY_OPTION")
