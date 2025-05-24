import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import enterprise
import io


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
        for col in ["ema_fast", "ema_slow", "rsi", "atr", "adx"]:
            self.assertIn(col, res.columns)

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
                "atr": [3] * 61,
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
                "atr": [3] * 61,
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
        df.to_csv("tmp.csv", index=False)
        enterprise.M1_PATH = "tmp.csv"
        enterprise.TRADE_DIR = "."
        enterprise.run_backtest()
        self.assertTrue(os.path.exists("tmp.csv"))
        os.remove("tmp.csv")
        # ensure logs saved
        logs = [f for f in os.listdir(".") if f.startswith("trade_log_")]
        for f in logs:
            os.remove(f)
        curves = [f for f in os.listdir(".") if f.startswith("equity_curve_")]
        for f in curves:
            os.remove(f)

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
        df1.to_csv("tmp1.csv", index=False)
        df2.to_csv("tmp2.csv", index=False)
        enterprise.TRADE_DIR = "."
        enterprise.run_backtest_multi_tf("tmp1.csv", "tmp2.csv")
        os.remove("tmp1.csv")
        os.remove("tmp2.csv")
        logs = [f for f in os.listdir(".") if f.startswith("trade_log_")]
        for f in logs:
            os.remove(f)
        curves = [f for f in os.listdir(".") if f.startswith("equity_curve_")]
        for f in curves:
            os.remove(f)

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
        df.to_csv("tmp.csv", index=False)
        enterprise.M1_PATH = "tmp.csv"
        enterprise.TRADE_DIR = "."
        enterprise.run_backtest()
        os.remove("tmp.csv")
        for f in os.listdir("."):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(f)

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
        df1.to_csv("tmp1.csv", index=False)
        df2.to_csv("tmp2.csv", index=False)
        enterprise.TRADE_DIR = "."
        enterprise.run_backtest_multi_tf("tmp1.csv", "tmp2.csv")
        os.remove("tmp1.csv")
        os.remove("tmp2.csv")
        for f in os.listdir("."):
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(f)

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
        self.assertIsNone(df["entry_signal"].iloc[3])

    def test_execute_backtest_dynamic_tp2(self):
        enterprise.TRADE_DIR = "."
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
        for f in os.listdir("."):  # cleanup
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(f)
        for f in os.listdir("."):  # cleanup
            if f.startswith("trade_log_") or f.startswith("equity_curve_"):
                os.remove(f)
        self.assertAlmostEqual(trades["tp2"].iloc[0], 1 + 2.0 * 1.5)


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


if __name__ == "__main__":
    unittest.main()
