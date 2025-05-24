import importlib.util
import importlib
import unittest
import types

pandas_available = importlib.util.find_spec("pandas") is not None

if pandas_available:
    import pandas as pd
    import os
    import sys
    from unittest.mock import patch

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import walk_forward_engine as wfe


class TestWalkForwardEngine(unittest.TestCase):
    @unittest.skipUnless(pandas_available, "requires pandas")
    def test_split_data_into_folds(self):
        df = pd.DataFrame({
            "entry_time": pd.date_range("2020-01-01", periods=5, freq="D"),
            "v": range(5),
        })
        folds = wfe.split_data_into_folds(df, "entry_time", fold_size_days=2)
        self.assertEqual(len(folds), 2)

    @unittest.skipUnless(pandas_available, "requires pandas")
    def test_wfa_basic_flow(self):
        df = pd.DataFrame({"entry_time": ["2020-01-01"], "v": [1]})
        dummy = types.SimpleNamespace()
        with patch.dict("sys.modules", {"enterprise": dummy}):
            import importlib
            importlib.reload(wfe)
            dummy.data_quality_check = lambda x: x
            dummy.calc_indicators = lambda x: x
            dummy.calc_dynamic_tp2 = lambda x: x
            dummy.label_elliott_wave = lambda x: x
            dummy.detect_divergence = lambda x: x
            dummy.label_pattern = lambda x: x
            dummy.calc_gain_zscore = lambda x: x
            dummy.calc_signal_score = lambda x: x
            dummy.tag_session = lambda x: x
            dummy.tag_spike_guard = lambda x: x
            dummy.tag_news_event = lambda x: x
            dummy.smart_entry_signal_goldai2025_style = lambda x: x
            dummy.apply_session_bias = lambda x: x
            dummy.apply_spike_news_guard = lambda x: x
            dummy._execute_backtest = lambda x: pd.DataFrame({"pnl": [1]})
            dummy.analyze_tradelog = lambda trades, eq: {}
            dummy.initial_capital = 100
            res = wfe.wfa(df, {})
            self.assertIn("Final Equity", res)
            self.assertEqual(res["Total Trades"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
