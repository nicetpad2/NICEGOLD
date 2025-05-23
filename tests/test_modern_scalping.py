import unittest
import importlib.util

pandas_available = importlib.util.find_spec('pandas') is not None
numpy_available = importlib.util.find_spec('numpy') is not None
sklearn_available = importlib.util.find_spec('sklearn') is not None

if pandas_available and numpy_available and sklearn_available:
    import pandas as pd
    import numpy as np
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import nicegold

class TestModernScalping(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available, 'requires pandas, numpy, sklearn')
    def test_compute_features_columns(self):
        df = pd.DataFrame({
            'close': np.arange(1, 30, dtype=float),
            'high': np.arange(1, 30, dtype=float) + 0.1,
            'low': np.arange(1, 30, dtype=float) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=29, freq='T')
        })
        res = nicegold.compute_features(df)
        for col in ['rsi', 'atr', 'trend']:
            self.assertIn(col, res.columns)

    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available, 'requires pandas, numpy, sklearn')
    def test_train_signal_model(self):
        df = pd.DataFrame({
            'close': np.linspace(1, 2, 60),
            'high': np.linspace(1, 2, 60) + 0.1,
            'low': np.linspace(1, 2, 60) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=60, freq='T')
        })
        df = nicegold.compute_features(df)
        res = nicegold.train_signal_model(df)
        self.assertIn('signal_prob', res.columns)
        self.assertIn('entry_signal', res.columns)

    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available, 'requires pandas, numpy, sklearn')
    def test_run_backtest_returns_dataframe(self):
        df = pd.DataFrame({
            'close': np.linspace(1, 2, 80),
            'high': np.linspace(1, 2, 80) + 0.1,
            'low': np.linspace(1, 2, 80) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=80, freq='T')
        })
        df = nicegold.compute_features(df)
        df = nicegold.train_signal_model(df)
        trades = nicegold.run_backtest(df)
        self.assertIsInstance(trades, pd.DataFrame)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
