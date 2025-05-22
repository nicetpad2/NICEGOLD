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

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_trend_confirm_column(self):
        df = pd.DataFrame({'close': [1, 2, 3]})
        df = nicegold.calculate_trend_confirm(df.copy())
        self.assertIn('trend_confirm', df.columns)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_wave_phase_column(self):
        df = pd.DataFrame({
            'close': [1, 2, 3],
            'high': [1, 2, 3],
            'low': [1, 2, 3],
            'RSI': [60, 60, 60],
            'Pattern_Label': ['Breakout', 'Breakout', 'Breakout'],
        })
        df = nicegold.calculate_macd(df.copy())
        df = nicegold.detect_macd_divergence(df)
        df = nicegold.label_wave_phase(df)
        self.assertIn('Wave_Phase', df.columns)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_detect_elliott_wave_phase_column(self):
        df = pd.DataFrame({
            'close': [1, 2, 1, 3, 1, 4, 1],
            'RSI': [40, 60, 40, 60, 40, 60, 40],
            'divergence': ['bullish', 'bearish', 'bullish', 'bearish', 'bullish', 'bearish', 'bullish']
        })
        df = nicegold.detect_elliott_wave_phase(df)
        self.assertIn('Wave_Phase', df.columns)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_generate_entry_signal_wave_enhanced_column(self):
        df = pd.DataFrame({
            'close': [1, 2, 3, 4],
            'RSI': [60, 60, 60, 60],
            'ema35': [1, 2, 3, 4],
            'divergence': ['bullish', 'bullish', 'bullish', 'bullish'],
            'Wave_Phase': ['W.2', 'W.3', 'W.5', 'W.B']
        })
        df = nicegold.generate_entry_signal_wave_enhanced(df)
        self.assertIn('entry_signal', df.columns)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_generate_entry_score_signal_column(self):
        df = pd.DataFrame({
            'close': [1, 2, 3],
            'ema35': [1, 2, 3],
            'RSI': [55, 55, 55],
            'divergence': ['bullish', 'bullish', 'bullish'],
            'Wave_Phase': ['W.2', 'W.3', 'W.5']
        })
        df = nicegold.generate_entry_score_signal(df)
        self.assertIn('entry_score', df.columns)
        self.assertIn('entry_signal', df.columns)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
