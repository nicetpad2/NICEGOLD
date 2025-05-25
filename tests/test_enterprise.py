import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import enterprise
import io

# Use predefined paths for all tests
TRADE_DIR_PATH = "/content/drive/MyDrive/NICEGOLD/logs"
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"
os.makedirs(TRADE_DIR_PATH, exist_ok=True)
enterprise.TRADE_DIR = TRADE_DIR_PATH
enterprise.M1_PATH = M1_PATH
enterprise.M15_PATH = M15_PATH

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if not os.path.exists(M1_PATH):
    os.makedirs(os.path.dirname(M1_PATH), exist_ok=True)
    import shutil
    shutil.copy(os.path.join(ROOT_DIR, "XAUUSD_M1.csv"), M1_PATH)
if not os.path.exists(M15_PATH):
    os.makedirs(os.path.dirname(M15_PATH), exist_ok=True)
    import shutil
    shutil.copy(os.path.join(ROOT_DIR, "XAUUSD_M15.csv"), M15_PATH)


class TestEnterprise(unittest.TestCase):
    def test_load_data_timestamp(self):
        csv = """Date,Timestamp,Open,High,Low,Close
25670501,0:00:00,1,1,1,1
25670501,0:01:00,1,1,1,1
"""
        df = enterprise.load_data(io.StringIO(csv))
        self.assertEqual(df["timestamp"].iloc[0], pd.Timestamp("2024-05-01 00:00:00"))

    def test_calc_indicators_columns(self):
        df = pd.DataFrame(
            {
                "close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "high": np.arange(14) + 1,
                "low": np.arange(14),
            }
        )
        res = enterprise.calc_indicators(df)
        for col in ["ema_fast", "ema_slow", "rsi", "atr", "adx", "ema_55", "rsi_14"]:
            self.assertIn(col, res.columns)

    def test_calc_indicators_rsi_34(self):
        df = pd.DataFrame(
            {
                "close": np.arange(40, dtype=float) + 1,
                "high": np.arange(40, dtype=float) + 2,
                "low": np.arange(40, dtype=float) + 1,
            }
        )
        res = enterprise.calc_indicators(df)
        self.assertIn("rsi_34", res.columns)


    def test_log_ram_usage_logs(self):
        logger = enterprise.logger
        with self.assertLogs(logger, level="INFO") as cm:
            enterprise.log_ram_usage("test")
        self.assertTrue(any("RAM usage" in msg for msg in cm.output))

    def test_rsi_function(self):
        ser = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        res = enterprise.rsi(ser, period=5)
        self.assertEqual(len(res), len(ser))
        self.assertGreater(res.iloc[-1], 50)

    def test_smart_entry_signal(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2] * 61,
                "ema_slow": [1] * 61,
                "rsi": [60] * 61,
                "atr": [4] * 61,
                "adx": [20] * 61,
                "close": [1] * 61,
                "high": [1] * 61,
                "low": [1] * 61,
            }
        )
        res = enterprise.smart_entry_signal(df)
        self.assertIn("buy", res["entry_signal"].values)

    def test_smart_entry_signal_counts(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2] * 61,
                "ema_slow": [1] * 61,
                "rsi": [60] * 61,
                "atr": [4] * 61,
                "adx": [20] * 61,
            }
        )
        res = enterprise.smart_entry_signal(df)
        self.assertEqual((res["entry_signal"] == "buy").sum(), 61 - 14)

    def test_is_strong_trend(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2] * 61,
                "ema_slow": [1] * 61,
                "adx": [20] * 61,
                "atr": [2] * 61,
            }
        )
        self.assertTrue(enterprise.is_strong_trend(df, 60))

    def test_oms_smart_lot(self):
        oms = enterprise.OMSManager(100, 0.5, 0.5)
        lot = oms.smart_lot(100, 3, 0.1)
        self.assertLessEqual(lot, enterprise.lot_cap_500)

    def test_oms_smart_lot_cap_limit(self):
        oms = enterprise.OMSManager(100, 0.5, 1.0)
        lot = oms.smart_lot(100, 100, 0.1)
        self.assertLessEqual(lot, enterprise.lot_cap_500)

    def test_run_backtest_outputs(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=30, freq="min"),
                "open": np.linspace(1, 1.3, 30),
                "high": np.linspace(1, 1.3, 30) + 0.1,
                "low": np.linspace(1, 1.3, 30) - 0.1,
                "close": np.linspace(1, 1.3, 30),
            }
        )
        df.to_csv(enterprise.M1_PATH, index=False)
        enterprise.run_backtest()
        self.assertTrue(os.path.exists(enterprise.M1_PATH))
        os.remove(enterprise.M1_PATH)
        # ensure logs saved
        logs = [f for f in os.listdir(enterprise.TRADE_DIR) if f.startswith("trade_log_")]
        for f in logs:
            os.remove(os.path.join(enterprise.TRADE_DIR, f))
        curves = [f for f in os.listdir(enterprise.TRADE_DIR) if f.startswith("equity_curve_")]
        for f in curves:
            os.remove(os.path.join(enterprise.TRADE_DIR, f))

    def test_add_m15_context_to_m1(self):
        df_m1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
                "close": [1, 2, 3],
            }
        )
        df_m15 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=1, freq="15min"),
                "ema_fast": [1.0],
                "ema_slow": [1.0],
                "rsi": [55],
            }
        )
        res = enterprise.add_m15_context_to_m1(df_m1, df_m15)
        self.assertIn("m15_ema_fast", res.columns)

    def test_smart_entry_signal_multi_tf(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2, 2],
                "ema_slow": [1, 1],
                "rsi": [60, 60],
                "m15_ema_fast": [2, 2],
                "m15_ema_slow": [1, 1],
                "m15_rsi": [60, 60],
            }
        )
        res = enterprise.smart_entry_signal_multi_tf(df)
        self.assertEqual(res["entry_signal"].iloc[0], "buy")

    def test_run_backtest_multi_tf_outputs(self):
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=30, freq="min"),
                "open": np.linspace(1, 1.3, 30),
                "high": np.linspace(1, 1.3, 30) + 0.1,
                "low": np.linspace(1, 1.3, 30) - 0.1,
                "close": np.linspace(1, 1.3, 30),
            }
        )
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="15min"),
                "open": [1, 1.1, 1.2],
                "high": [1.1, 1.2, 1.3],
                "low": [0.9, 1.0, 1.1],
                "close": [1, 1.1, 1.2],
            }
        )
        df1.to_csv(enterprise.M1_PATH, index=False)
        df2.to_csv(enterprise.M15_PATH, index=False)
        enterprise.run_backtest_multi_tf(enterprise.M1_PATH, enterprise.M15_PATH)
        os.remove(enterprise.M1_PATH)
        os.remove(enterprise.M15_PATH)
        logs = [f for f in os.listdir(enterprise.TRADE_DIR) if f.startswith("trade_log_")]
        for f in logs:
            os.remove(os.path.join(enterprise.TRADE_DIR, f))
        curves = [f for f in os.listdir(enterprise.TRADE_DIR) if f.startswith("equity_curve_")]
        for f in curves:
            os.remove(os.path.join(enterprise.TRADE_DIR, f))

    def test_smart_entry_signal_goldai2025_style(self):
        df = pd.DataFrame({"ema_fast": [2, 1], "ema_slow": [1, 2], "rsi": [60, 40]})
        res = enterprise.smart_entry_signal_goldai2025_style(df)
        self.assertEqual(res["entry_signal"].tolist(), ["buy", "sell"])

    def test_run_backtest_goldai2025(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=30, freq="min"),
                "open": np.linspace(1, 1.3, 30),
                "high": np.linspace(1, 1.3, 30) + 0.1,
                "low": np.linspace(1, 1.3, 30) - 0.1,
                "close": np.linspace(1, 1.3, 30),
            }
        )
        df.to_csv(enterprise.M1_PATH, index=False)
        enterprise.run_backtest()
        os.remove(enterprise.M1_PATH)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))

    def test_run_backtest_multi_tf_goldai2025(self):
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=30, freq="min"),
                "open": np.linspace(1, 1.3, 30),
                "high": np.linspace(1, 1.3, 30) + 0.1,
                "low": np.linspace(1, 1.3, 30) - 0.1,
                "close": np.linspace(1, 1.3, 30),
            }
        )
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="15min"),
                "open": [1, 1.1, 1.2],
                "high": [1.1, 1.2, 1.3],
                "low": [0.9, 1.0, 1.1],
                "close": [1, 1.1, 1.2],
            }
        )
        df1.to_csv(enterprise.M1_PATH, index=False)
        df2.to_csv(enterprise.M15_PATH, index=False)
        enterprise.run_backtest_multi_tf(enterprise.M1_PATH, enterprise.M15_PATH)
        os.remove(enterprise.M1_PATH)
        os.remove(enterprise.M15_PATH)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))

    def test_label_elliott_wave(self):
        df = pd.DataFrame({"high": [1, 2, 3, 2, 1], "low": [0, 1, 2, 1, 0]})
        res = enterprise.label_elliott_wave(df.copy())
        self.assertIn("wave_phase", res.columns)

    def test_detect_divergence_bullish(self):
        df = pd.DataFrame(
            {"low": [1, 0.9, 0.8], "high": [1, 1, 1], "rsi": [40, 41, 42]}
        )
        res = enterprise.detect_divergence(df.copy(), rsi_col="rsi")
        self.assertEqual(res["divergence"].iloc[2], "bullish")

    def test_detect_divergence_missing_rsi(self):
        df = pd.DataFrame(
            {"low": [1, 0.9, 0.8, 0.7], "high": [1, 1, 1, 1], "rsi": [40, 41, 42, 43]}
        )
        with self.assertLogs(enterprise.logger, level="WARNING") as cm:
            res = enterprise.detect_divergence(df.copy(), rsi_col="rsi")
        self.assertIn("divergence", res.columns)
        self.assertTrue(any("RSI_" in m for m in cm.output))

    def test_label_pattern(self):
        df = pd.DataFrame({"ema_fast": [2, 1], "ema_slow": [1, 2], "rsi": [60, 40]})
        res = enterprise.label_pattern(df.copy())
        self.assertEqual(res["pattern_label"].iloc[0], "first_pullback")

    def test_gain_zscore_and_meta_filter(self):
        df = pd.DataFrame(
            {
                "close": [1, 2, 3],
                "pattern_label": ["first_pullback", "first_pullback", "first_pullback"],
                "divergence": ["bullish", "bullish", "bullish"],
                "rsi": [60, 60, 60],
                "high": [1, 1, 1],
                "low": [0, 0, 0],
            }
        )
        df = enterprise.calc_gain_zscore(df, window=2)
        df = enterprise.label_elliott_wave(df)
        df = enterprise.calc_signal_score(df)
        df = enterprise.meta_classifier_filter(df)
        self.assertIn("gain_z", df.columns)
        self.assertTrue(df["meta_entry"].iloc[-1])


class TestDynamicTP2Session(unittest.TestCase):
    def test_calc_dynamic_tp2_column(self):
        df = pd.DataFrame({"atr": [1, 1, 5, 1]})
        res = enterprise.calc_dynamic_tp2(df.copy(), base_tp2_mult=2.0, atr_period=2)
        self.assertIn("tp2_dynamic", res.columns)
        self.assertAlmostEqual(res["tp2_dynamic"].iloc[2], 1.5, places=2)
        self.assertAlmostEqual(res["tp2_dynamic"].iloc[3], 2.5, places=2)

    def test_tag_session_and_bias(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2020-01-01 00:00",
                        "2020-01-01 08:00",
                        "2020-01-01 16:00",
                        "2020-01-01 23:30",
                    ]
                ),
                "entry_signal": ["buy"] * 4,
            }
        )
        df = enterprise.tag_session(df)
        self.assertEqual(df["session"].tolist(), ["Asia", "London", "NY", "Other"])
        df = enterprise.apply_session_bias(df)
        self.assertEqual(df["entry_signal"].tolist(), ["buy", "buy", "buy", None])

    def test_execute_backtest_dynamic_tp2(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
                "open": [1.0, 1.0, 1.0],
                "high": [2.5, 5.0, 5.0],
                "low": [-1.0, 0.8, 0.8],
                "close": [1.0, 1.1, 1.2],
                "ema_fast": [2, 2, 2],
                "ema_slow": [1, 1, 1],
                "rsi": [60, 60, 60],
                "adx": [20, 20, 20],
                "atr": [0.1, 0.1, 0.1],
                "entry_signal": ["buy", None, None],
                "tp2_dynamic": [1.5, 1.5, 1.5],
                "atr_long": [0.1, 0.1, 0.1],
            }
        )
        trades = enterprise._execute_backtest(df)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))
        self.assertTrue(trades.empty)

    def test_atr_filter_skip_trade(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "ema_fast": [2, 2, 2],
                "ema_slow": [1, 1, 1],
                "rsi": [60, 60, 60],
                "adx": [20, 20, 20],
                "atr": [enterprise.SPREAD_VALUE * 1.5] * 3,
                "entry_signal": ["buy", None, None],
                "tp2_dynamic": [1.5, 1.5, 1.5],
                "atr_long": [1.0, 1.0, 1.0],
            }
        )
        trades = enterprise._execute_backtest(df)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))
        self.assertTrue(trades.empty)

    def test_execute_backtest_debug_strings(self):
        import inspect

        src = inspect.getsource(enterprise._execute_backtest)
        self.assertIn("[Patch][Debug] Holding position", src)
        self.assertIn("Check TP/SL", src)


class TestSpikeNewsGuard(unittest.TestCase):
    def test_tag_spike_guard_flag(self):
        atr = [1] * 1000 + [5]
        high = [1] * 1000 + [10]
        low = [0] * 1000 + [5]
        df = pd.DataFrame({"atr": atr, "high": high, "low": low})
        res = enterprise.tag_spike_guard(df.copy())
        self.assertTrue(res["spike_guard"].iloc[-1])

    def test_tag_news_event_mask(self):
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2020-01-01", periods=3, freq="min")}
        )
        news_times = [(df["timestamp"].iloc[1], df["timestamp"].iloc[2])]
        res = enterprise.tag_news_event(df.copy(), news_times=news_times)
        self.assertTrue(res["news_guard"].iloc[2])

    def test_apply_spike_news_guard(self):
        df = pd.DataFrame(
            {
                "entry_signal": ["buy", "sell"],
                "spike_guard": [True, False],
                "news_guard": [False, True],
            }
        )
        res = enterprise.apply_spike_news_guard(df.copy())
        self.assertIsNone(res["entry_signal"].iloc[0])
        self.assertIsNone(res["entry_signal"].iloc[1])

    def test_split_folds_count(self):
        df = pd.DataFrame({"a": range(10)})
        folds = enterprise.split_folds(df, n_folds=3)
        self.assertEqual(len(folds), 3)

    def test_data_quality_check_basic(self):
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
                "open": [1.0, np.nan],
                "high": [1.0, 2.0],
                "low": [0.9, 0.8],
                "close": [1.0, 2.0],
            }
        )
        res = enterprise.data_quality_check(df)
        self.assertEqual(len(res), 1)

    def test_shap_feature_importance_placeholder(self):
        df = pd.DataFrame(
            {
                "ema_fast": [1.0],
                "ema_slow": [2.0],
                "rsi": [50.0],
                "atr": [1.0],
                "gain_z": [0.1],
                "signal_score": [1.0],
            }
        )
        res, imp = enterprise.shap_feature_importance_placeholder(df)
        for c in ["ema_fast", "ema_slow", "rsi", "atr", "gain_z", "signal_score"]:
            self.assertIn(f"shap_importance_{c}", res.columns)
        self.assertAlmostEqual(sum(imp.values()), 1.0, places=5)

    def test_apply_order_costs(self):
        entry, sl, tp1, tp2, com = enterprise.apply_order_costs(
            1.0, 0.9, 1.1, 1.2, 0.1, "buy", spread=0.1, commission=0.1, slippage=0.0
        )
        self.assertGreater(entry, 1.0)
        self.assertAlmostEqual(sl, 0.9)
        self.assertAlmostEqual(tp1, 1.1)
        self.assertAlmostEqual(tp2, 1.2)
        self.assertAlmostEqual(com, 2 * 0.1 * 0.1 * 100)

    def test_oms_audit_and_check(self):
        oms = enterprise.OMSManager(100, 0.5, 1.0)
        with self.assertLogs(enterprise.logger, level="INFO") as cm:
            oms.audit_log()
        self.assertTrue(any("OMS Audit" in m for m in cm.output))
        self.assertFalse(oms.check_max_orders([1], max_orders=1))
        self.assertTrue(oms.check_max_orders([], max_orders=1))

    def test_run_walkforward_backtest_returns(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=5, freq="min"),
                "open": [1] * 5,
                "high": [1] * 5,
                "low": [1] * 5,
                "close": [1] * 5,
                "atr": [0.1] * 5,
                "ema_fast": [1] * 5,
                "ema_slow": [1] * 5,
                "rsi": [50] * 5,
                "adx": [20] * 5,
                "entry_signal": ["buy"] * 5,
            }
        )
        results = enterprise.run_walkforward_backtest(df, n_folds=2)
        self.assertEqual(len(results), 2)

    def test_multi_session_trend_scalping_basic(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01 08:00", periods=60, freq="min"),
                "ema_fast": [2] * 60,
                "ema_slow": [1] * 60,
                "rsi": [60] * 60,
                "atr": np.linspace(0.5, 1.5, 60),
            }
        )
        res = enterprise.multi_session_trend_scalping(df)
        self.assertTrue((res["entry_signal"] == "buy").any())

    def test_entry_signal_always_on_pattern(self):
        df = pd.DataFrame({"ema_fast": [2, 2, 2]})
        res = enterprise.entry_signal_always_on(df)
        self.assertEqual(res["entry_signal"].tolist(), ["buy", "sell", "buy"])

    def test_entry_signal_trend_relax_basic(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="min"),
                "ema_fast": [1, 2, 1, 2],
                "ema_slow": [2, 1, 2, 1],
                "high": [1.1, 1.2, 1.1, 1.2],
                "low": [0.9, 0.8, 0.9, 0.8],
                "atr": [0.2, 0.3, 0.4, 0.5],
            }
        )
        res = enterprise.entry_signal_trend_relax(df, min_gap_minutes=0)
        self.assertIn("entry_signal", res.columns)
        self.assertTrue(res["entry_signal"].notna().any())

    def test_entry_signal_trend_scalp_force_gap(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2, 2, 2, 2, 2],
                "ema_slow": [1, 1, 1, 1, 1],
                "rsi": [60, 60, 40, 40, 60],
                "adx": [20] * 5,
                "atr": [1.2] * 5,
            }
        )
        res = enterprise.entry_signal_trend_scalp(df, force_gap=1)
        self.assertEqual(res["entry_signal"].iloc[2], "buy")
        self.assertTrue(res["entry_signal"].notna().any())

    def test_constants_values(self):
        self.assertEqual(enterprise.COMMISSION_PER_LOT, 0.10)
        self.assertEqual(enterprise.SLIPPAGE, 0.2)

    def test_smart_entry_signal_multi_tf_ema_adx_buy(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=60, freq="min"),
                "ema_fast": [2] * 60,
                "ema_slow": [1] * 60,
                "m15_ema_fast": [2] * 60,
                "m15_ema_slow": [1] * 60,
                "adx": [25] * 60,
                "rsi": [60] * 60,
                "high": [1.1] * 60,
                "low": [0.9] * 60,
                "close": [1.0] * 60,
            }
        )
        res = enterprise.smart_entry_signal_multi_tf_ema_adx(df)
        self.assertIn("buy", res["entry_signal"].values)

    def test_smart_entry_signal_multi_tf_ema_adx_optimized_buy(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=60, freq="min"),
                "ema_fast": [2] * 60,
                "ema_slow": [1] * 60,
                "m15_ema_fast": [2] * 60,
                "m15_ema_slow": [1] * 60,
                "adx": [25] * 60,
                "rsi": [60] * 60,
                "gain_z": [0.1] * 60,
                "atr": [1.0] * 60,
                "high": [1.1] * 60,
                "low": [0.9] * 60,
                "close": [1.0] * 60,
            }
        )
        res = enterprise.smart_entry_signal_multi_tf_ema_adx_optimized(df)
        self.assertIn("buy", res["entry_signal"].values)

    def test_calc_adaptive_lot(self):
        lot = enterprise.calc_adaptive_lot(1000, adx=30, recovery_mode=True, win_streak=2)
        self.assertGreaterEqual(lot, 0.01)

    def test_on_price_update_patch_partial(self):
        o = enterprise.Order(id=1, entry_price=1.0)
        enterprise.on_order_execute(o)
        enterprise.on_price_update_patch(o, 3.0, indicators={"ATR": 0.1})
        self.assertTrue(o.partial_taken)

    def test_on_price_update_patch_v2_partial(self):
        o = enterprise.Order(id=1, entry_price=1.0)
        enterprise.on_order_execute(o)
        enterprise.on_price_update_patch_v2(o, 3.0, indicators={"ATR": 0.1})
        self.assertTrue(o.partial_taken)

    def test_qa_validate_backtest(self):
        trades = pd.DataFrame({"pnl": [10] * 20})
        equity = pd.DataFrame({"equity": [100 + i for i in range(21)], "dd": [0.0] * 21})
        res = enterprise.qa_validate_backtest(trades, equity)
        self.assertGreaterEqual(res["trades"], 20)

    def test_qa_validate_backtest_winrate_warning(self):
        trades = pd.DataFrame({"pnl": [1, -1]})
        equity = pd.DataFrame({"equity": [100, 101, 100], "dd": [0.0, 0.0, 0.0]})
        with self.assertLogs(enterprise.logger, level="WARNING") as cm:
            enterprise.qa_validate_backtest(trades, equity, prev_winrate=1.0, min_trades=1, min_profit=0)
        self.assertTrue(any("Winrate dropped" in m for m in cm.output))

    def test_patch_confirm_on_lossy_indices(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
            "entry_signal": ["buy", "sell", None],
            "divergence": ["bearish", "bullish", "bullish"],
            "gain_z": [-1.0, 1.0, 0.5],
            "atr": [1.0, 1.0, 1.0],
        })
        res = enterprise.patch_confirm_on_lossy_indices(df.copy(), [0, 1])
        self.assertIsNone(res["entry_signal"].iloc[0])
        self.assertIsNone(res["entry_signal"].iloc[1])

    def test_analyze_tradelog_output_keys(self):
        trades = pd.DataFrame({"pnl": [1, -1, 2], "entry_idx": [0, 1, 2]})
        equity = pd.DataFrame({"dd": [0.0, 0.1, 0.05]})
        stats = enterprise.analyze_tradelog(trades, equity)
        self.assertIn("max_win_streak", stats)
        self.assertIn("max_loss_streak", stats)

    def test_calc_basic_indicators_columns(self):
        df = pd.DataFrame(
            {
                "close": [1, 2, 3, 4, 5],
                "high": [1, 2, 3, 4, 5],
                "low": [0.5, 1, 1.5, 2, 2.5],
            }
        )
        res = enterprise.calc_basic_indicators(df)
        for c in ["EMA_20", "EMA_50", "RSI_14", "ATR_14"]:
            self.assertIn(c, res.columns)

    def test_relaxed_entry_signal_force_gap(self):
        df = pd.DataFrame(
            {
                "open": [1, 1.1] * 10,
                "close": [1.2, 0.9] * 10,
                "high": [1.3, 1.2] * 10,
                "low": [0.8, 0.7] * 10,
            }
        )
        df = enterprise.calc_basic_indicators(df)
        res = enterprise.relaxed_entry_signal(df, force_gap=1)
        self.assertTrue(res["entry_signal"].notna().any())

    def test_smart_entry_signal_enterprise_v1_force_and_counts(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2, 2, 0, 0],
                "ema_slow": [1, 1, 1, 1],
                "rsi": [60, 60, 40, 40],
                "adx": [20, 20, 20, 20],
                "gain_z": [0.6, 0.6, -0.6, -0.6],
                "divergence": ["bullish", "bullish", "bearish", "bearish"],
                "wave_phase": ["trough", "trough", "peak", "peak"],
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="min"),
            }
        )
        res = enterprise.smart_entry_signal_enterprise_v1(df)
        self.assertEqual(res["entry_signal"].tolist(), ["buy", "buy", "sell", "sell"])

    def test_entry_count_tracking(self):
        enterprise._PREV_ENTRY_COUNT = None
        df1 = pd.DataFrame({
            "ema_fast": [2, 2],
            "ema_slow": [1, 1],
            "rsi": [60, 60],
            "adx": [20, 20],
            "gain_z": [0.6, 0.6],
            "divergence": ["bullish", "bullish"],
            "wave_phase": ["trough", "trough"],
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="min"),
        })
        enterprise.smart_entry_signal_enterprise_v1(df1)
        self.assertEqual(enterprise._PREV_ENTRY_COUNT, 2)
        df2 = df1.iloc[:1]
        enterprise.smart_entry_signal_enterprise_v1(df2)
        self.assertEqual(enterprise._PREV_ENTRY_COUNT, 1)

    def test_mid_wave_entry(self):
        df = pd.DataFrame(
            {
                "ema_fast": [2, 2, 0],
                "ema_slow": [1, 1, 1],
                "rsi": [60, 60, 40],
                "adx": [20, 20, 20],
                "gain_z": [0.6, 0.6, -0.6],
                "divergence": ["none", "none", "none"],
                "wave_phase": ["mid", "mid", "mid"],
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
            }
        )
        res = enterprise.smart_entry_signal_enterprise_v1(df)
        self.assertEqual(res["entry_signal"].tolist(), ["buy", "buy", "sell"])

    def test_strict_recovery_entry_conditions(self):
        class DummyOMS(enterprise.OMSManager):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.recovery_mode = True
                self.loss_streak = 4

        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
            "open": [1.0, 1.0, 1.0],
            "high": [1.2, 10.0, 10.0],
            "low": [0.8, 0.8, 0.8],
            "close": [1.0, 1.1, 1.2],
            "ema_fast": [2, 2, 2],
            "ema_slow": [1, 1, 1],
            "rsi": [60, 60, 60],
            "adx": [15, 15, 15],
            "atr": [2.0, 2.0, 2.0],
            "entry_signal": ["buy", None, None],
            "tp2_dynamic": [1.5, 1.5, 1.5],
            "atr_long": [0.1, 0.1, 0.1],
            "divergence": ["bullish", None, None],
            "gain_z": [0.6, 0.0, 0.0],
        })
        with patch.object(enterprise, "OMSManager", DummyOMS):
            trades = enterprise._execute_backtest(df)
        for f in os.listdir("."):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(f)
        self.assertTrue(trades.empty)

        df["adx"] = [25, 25, 25]
        df["gain_z"] = [0.8, 0.0, 0.0]
        with patch.object(enterprise, "OMSManager", DummyOMS):
            trades2 = enterprise._execute_backtest(df)
        for f in os.listdir("."):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(f)
        self.assertFalse(trades2.empty)

    def test_skip_entry_opposite_divergence(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=1, freq="min"),
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "ema_fast": [2],
                "ema_slow": [1],
                "rsi": [55],
                "adx": [20],
                "atr": [0.2],
                "entry_signal": ["buy"],
                "divergence": ["bearish"],
            }
        )
        trades = enterprise._execute_backtest(df)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))
        self.assertTrue(trades.empty)

    def test_walkforward_run_returns_list(self):
        df = pd.DataFrame(
            {
                "open": [1] * 10,
                "close": [1] * 10,
                "high": [1] * 10,
                "low": [1] * 10,
            }
        )
        folds = enterprise.walkforward_run(df, fold_size=3)
        self.assertEqual(len(folds), 4)

    def test_entry_signal_always_on_alternate(self):
        df = pd.DataFrame({"v": range(4)})
        res = enterprise.entry_signal_always_on(df)
        self.assertEqual(res["entry_signal"].tolist(), ["buy", "sell", "buy", "sell"])

    def test_calc_aggressive_lot(self):
        lot = enterprise.calc_aggressive_lot(100, 2)
        self.assertGreater(lot, 0)

    def test_run_backtest_aggressive_returns_df(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=10, freq="min"),
                "open": np.linspace(1, 1.1, 10),
                "high": np.linspace(1, 1.1, 10) + 0.05,
                "low": np.linspace(1, 1.1, 10) - 0.05,
                "close": np.linspace(1, 1.1, 10),
            }
        )
        df.to_csv(enterprise.M1_PATH, index=False)
        trades = enterprise.run_backtest_aggressive()
        os.remove(enterprise.M1_PATH)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))
        self.assertIsInstance(trades, pd.DataFrame)

    def test_config_defaults(self):
        assert enterprise.strategy_mode == "ib_commission_mode"
        assert enterprise.force_entry_gap == 200
        assert enterprise.partial_close_pct == 0.6
        assert enterprise.enable_micro_sl_exit

    def test_micro_sl_exit(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
            "open": [1.0, 0.8, 0.8],
            "high": [1.0, 0.8, 0.8],
            "low": [1.0, 0.7, 0.7],
            "close": [1.0, 0.8, 0.8],
            "ema_fast": [2, 2, 2],
            "ema_slow": [1, 1, 1],
            "rsi": [60, 60, 60],
            "adx": [20, 20, 20],
            "atr": [2.0, 2.0, 2.0],
            "entry_signal": ["buy", None, None],
        })
        trades = enterprise._execute_backtest(df)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))
        assert "MicroSL" in trades["exit"].values

    def test_calculate_auto_lot_basic(self):
        lot = enterprise.calculate_auto_lot(1000, 0.02, 10, 5)
        self.assertEqual(lot, 2.0)

    def test_equity_based_tp_sl_levels(self):
        tp, sl = enterprise.equity_based_tp_sl(400)
        self.assertEqual((tp, sl), (1.5, 0.8))
        tp2, sl2 = enterprise.equity_based_tp_sl(2000)
        self.assertEqual((tp2, sl2), (3.5, 1.2))

    def test_config_flags_true(self):
        self.assertTrue(enterprise.enable_auto_lot_scaling)
        self.assertTrue(enterprise.enable_equity_tp_sl_adjuster)

    def test_param_grid_length(self):
        self.assertGreaterEqual(len(enterprise.param_grid()), 1)

    def test_early_force_close_opposite_momentum(self):
        data = {
            "timestamp": pd.date_range("2020-01-01", periods=26, freq="min"),
            "open": [1.0] * 26,
            "high": [1.1] * 26,
            "low": [0.9] * 26,
            "close": [1.0] * 26,
            "ema_fast": [2] * 26,
            "ema_slow": [1] * 26,
            "rsi": [60] * 26,
            "adx": [20] * 26,
            "atr": [2.0] * 26,
            "entry_signal": ["buy"] + [None] * 25,
            "gain_z": [0.2] * 25 + [-0.5],
        }
        df = pd.DataFrame(data)
        trades = enterprise._execute_backtest(df)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))
        self.assertIn("EarlyForceClose", trades["exit"].values)

    def test_run_backtest_custom_keys(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=30, freq="min"),
                "open": np.linspace(1, 1.3, 30),
                "high": np.linspace(1, 1.3, 30) + 0.1,
                "low": np.linspace(1, 1.3, 30) - 0.1,
                "close": np.linspace(1, 1.3, 30),
            }
        )
        params = enterprise.param_grid()[0]
        res = enterprise.run_backtest_custom(df, params)
        self.assertIn("Final Equity", res)
        self.assertIn("Total Trades", res)
        for f in os.listdir(enterprise.TRADE_DIR):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(os.path.join(enterprise.TRADE_DIR, f))

    def test_walk_forward_run_outputs(self):
        # function is now deprecated and should return None
        res = enterprise.walk_forward_run("dummy.csv")
        self.assertIsNone(res)

    def test_main_runs_wfv(self):
        with patch.object(enterprise, "load_data", return_value=pd.DataFrame()), patch.object(
            enterprise, "data_quality_check", return_value=pd.DataFrame()
        ), patch.object(enterprise, "run_wfv_full_report") as mwfv:
            enterprise.main()
            mwfv.assert_called()

    def test_run_wfv_full_report_basic(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="min"),
                "open": [1] * 4,
                "high": [1] * 4,
                "low": [1] * 4,
                "close": [1] * 4,
            }
        )
        with patch.object(
            enterprise,
            "_execute_backtest",
            return_value=pd.DataFrame({"pnl": [1], "capital": [101]}),
        ):
            res = enterprise.run_wfv_full_report(df, n_folds=2)
        self.assertIsInstance(res, pd.DataFrame)
        summary_csv = os.path.join(enterprise.SUMMARY_DIR, "wfv_summary.csv")
        self.assertTrue(os.path.exists(summary_csv))
        os.remove(summary_csv)
        for metric in ["winrate", "profit", "max_dd"]:
            f = os.path.join(enterprise.SUMMARY_DIR, f"{metric}_plot.png")
            if os.path.exists(f):
                os.remove(f)

    def test_default_main_block_manual_wfv(self):
        import inspect

        src = inspect.getsource(enterprise)
        self.assertIn("Auto-detect mode: Run WFV", src)
        self.assertIn("run_wfv_full_report", src)



if __name__ == "__main__":
    unittest.main()
