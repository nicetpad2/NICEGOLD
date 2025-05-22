import unittest
import importlib.util

pandas_available = importlib.util.find_spec('pandas') is not None
numpy_available = importlib.util.find_spec('numpy') is not None

if pandas_available and numpy_available:
    import pandas as pd
    import numpy as np
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import nicegold

class TestIndicators(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_calculate_macd_columns(self):
        df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        result = nicegold.calculate_macd(df.copy())
        self.assertIn('macd', result.columns)
        self.assertIn('signal', result.columns)
        self.assertIn('macd_hist', result.columns)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_generate_entry_signal_column(self):
        df = pd.DataFrame({'close': [1, 2, 3], 'high': [1, 2, 3], 'low': [1, 2, 3]})
        df = nicegold.calculate_macd(df.copy())
        df = nicegold.detect_macd_divergence(df)
        df = nicegold.macd_cross_signal(df)
        df = nicegold.apply_ema_trigger(df)
        df = nicegold.calculate_spike_guard(df)
        df = nicegold.validate_divergence(df)
        df = nicegold.generate_entry_signal(df)
        self.assertIn('entry_signal', df.columns)

if __name__ == '__main__':
    unittest.main()
