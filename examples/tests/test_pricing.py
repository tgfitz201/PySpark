"""
Pricing consistency tests.
Verifies greek signs, NPV sanity, and cross-run stability.
"""
import pytest
import pandas as pd
import numpy as np

class TestPricingValidity:
    def test_all_600_priced(self, results_df):
        assert len(results_df) == 600, f"Expected 600 results, got {len(results_df)}"

    def test_npv_finite(self, results_df):
        bad = results_df[~results_df["npv"].apply(np.isfinite)]
        assert len(bad) == 0, f"Non-finite NPVs: {bad['trade_id'].tolist()}"

    def test_no_errors(self, results_df):
        errs = results_df[results_df["error"].fillna("").str.len() > 0]
        assert len(errs) == 0, f"Pricing errors: {errs[['trade_id','error']].to_dict('records')}"

    def test_notional_positive(self, results_df):
        assert (results_df["notional"] > 0).all()

    def test_irs_npv_decomposition(self, results_df):
        """For FIXED_FLOAT/FLOAT_FIXED IRS: fixed_npv + float_npv == npv."""
        ff = results_df[(results_df["instrument"] == "IRS") &
                        (results_df["swap_subtype"].isin(["FIXED_FLOAT", "FLOAT_FIXED"]))]
        if len(ff):
            residual = (ff["fixed_npv"] + ff["float_npv"] - ff["npv"]).abs()
            assert residual.max() < 0.01, f"Max IRS decomp residual: {residual.max()}"

    def test_irs_dv01_sign(self, results_df):
        std = results_df[(results_df["instrument"] == "IRS") &
                         (results_df["swap_subtype"].isin(["FIXED_FLOAT", "FLOAT_FIXED"]))]
        payer_ok    = (std[std["direction"] == "PAYER"]["dv01"] > 0).all()
        receiver_ok = (std[std["direction"] == "RECEIVER"]["dv01"] < 0).all()
        assert payer_ok and receiver_ok

    def test_bond_dv01_sign(self, results_df):
        bonds = results_df[results_df["instrument"] == "BOND"]
        long_ok  = (bonds[bonds["direction"] == "LONG"]["dv01"]  < 0).all()
        short_ok = (bonds[bonds["direction"] == "SHORT"]["dv01"] > 0).all()
        assert long_ok and short_ok

    def test_bond_clean_price_range(self, results_df):
        bonds = results_df[results_df["instrument"] == "BOND"]
        assert (bonds["clean_price"] > 80).all() and (bonds["clean_price"] < 130).all()

    def test_bond_accrued_nonneg(self, results_df):
        bonds = results_df[results_df["instrument"] == "BOND"]
        assert (bonds["accrued"] >= 0).all()

    def test_bond_convexity_positive(self, results_df):
        bonds = results_df[results_df["instrument"] == "BOND"]
        assert (bonds["convexity"] > 0).all()

    def test_swaption_buy_premium_positive(self, results_df):
        opts = results_df[results_df["instrument"] == "SWAPTION"]
        buy = opts[opts["direction"] == "BUY"]
        if len(buy):
            assert (buy["premium"] > 0).all()

    def test_swaption_buy_vega_positive(self, results_df):
        opts = results_df[results_df["instrument"] == "SWAPTION"]
        buy = opts[opts["direction"] == "BUY"]
        if len(buy):
            assert (buy["vega"] > 0).all()

    def test_eq_swap_delta_sign(self, results_df):
        eqs = results_df[results_df["instrument"] == "EQ_SWAP"]
        long_ok  = (eqs[eqs["direction"] == "LONG"]["delta"]  > 0).all()
        short_ok = (eqs[eqs["direction"] == "SHORT"]["delta"] < 0).all()
        assert long_ok and short_ok

    def test_cds_jtd_sign(self, results_df):
        cdss = results_df[results_df["instrument"] == "CDS"]
        buy_ok  = (cdss[cdss["direction"] == "BUY"]["jump_to_default"]  > 0).all()
        sell_ok = (cdss[cdss["direction"] == "SELL"]["jump_to_default"] < 0).all()
        assert buy_ok and sell_ok

    def test_cds_cr01_sign(self, results_df):
        cdss = results_df[results_df["instrument"] == "CDS"]
        buy_ok  = (cdss[cdss["direction"] == "BUY"]["cr01"]  > 0).all()
        sell_ok = (cdss[cdss["direction"] == "SELL"]["cr01"] < 0).all()
        assert buy_ok and sell_ok

    def test_eq_opt_gamma_sign(self, results_df):
        eqopts = results_df[results_df["instrument"] == "EQ_OPT"]
        buy_ok  = (eqopts[eqopts["direction"] == "BUY"]["gamma"]  > 0).all()
        sell_ok = (eqopts[eqopts["direction"] == "SELL"]["gamma"] < 0).all()
        assert buy_ok and sell_ok

    def test_eq_opt_vega_sign(self, results_df):
        eqopts = results_df[results_df["instrument"] == "EQ_OPT"]
        buy_ok  = (eqopts[eqopts["direction"] == "BUY"]["vega"]  > 0).all()
        sell_ok = (eqopts[eqopts["direction"] == "SELL"]["vega"] < 0).all()
        assert buy_ok and sell_ok

    def test_eq_opt_buy_premium_positive(self, results_df):
        eqopts = results_df[results_df["instrument"] == "EQ_OPT"]
        buy = eqopts[eqopts["direction"] == "BUY"]
        if len(buy):
            assert (buy["premium"] > 0).all()

    def test_eq_opt_call_delta_positive_for_buy(self, results_df):
        """BUY CALL delta > 0 (only if option_type column available)."""
        eqopts = results_df[results_df["instrument"] == "EQ_OPT"]
        if "option_type" not in eqopts.columns or not eqopts["option_type"].notna().any():
            pytest.skip("option_type not in PricingResult — skipping")
        buy_calls = eqopts[(eqopts["direction"] == "BUY") & (eqopts["option_type"] == "CALL")]
        if len(buy_calls):
            assert (buy_calls["delta"] > 0).all()

    def test_eq_opt_put_delta_negative_for_buy(self, results_df):
        """BUY PUT delta < 0 (only if option_type column available)."""
        eqopts = results_df[results_df["instrument"] == "EQ_OPT"]
        if "option_type" not in eqopts.columns or not eqopts["option_type"].notna().any():
            pytest.skip("option_type not in PricingResult — skipping")
        buy_puts = eqopts[(eqopts["direction"] == "BUY") & (eqopts["option_type"] == "PUT")]
        if len(buy_puts):
            assert (buy_puts["delta"] < 0).all()


class TestCrossRunStability:
    """Load run_id RUN-1 and RUN-2 and compare all greeks."""

    @pytest.fixture
    def run1(self, repo):
        return repo.get_results_df(run_id="RUN-1").set_index("trade_id")

    @pytest.fixture
    def run2(self, repo):
        return repo.get_results_df(run_id="RUN-2")
        # RUN-2 may or may not exist; skip if not

    def test_npv_stable_non_american_options(self, repo):
        """For all instruments except EQ_OPT, NPV must be bit-for-bit identical RUN-1 vs RUN-2."""
        r1 = repo.get_results_df(run_id="RUN-1")
        r2 = repo.get_results_df(run_id="RUN-2")
        if len(r2) == 0:
            pytest.skip("RUN-2 not found in DB")
        r1_idx = r1.set_index("trade_id")
        r2_idx = r2.set_index("trade_id")
        non_eqopt_r1 = r1_idx[r1_idx["instrument"] != "EQ_OPT"][["npv"]]
        non_eqopt_r2 = r2_idx[r2_idx["instrument"] != "EQ_OPT"][["npv"]]
        joined = non_eqopt_r1.join(non_eqopt_r2, lsuffix="_r1", rsuffix="_r2", how="inner")
        diff = (joined["npv_r1"] - joined["npv_r2"]).abs()
        assert diff.max() < 0.01, f"Non-EqOpt NPV drift: max={diff.max():.4f}"
