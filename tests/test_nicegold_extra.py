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
    from datetime import datetime, timedelta

class TestNicegoldExtra(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_detect_macd_divergence_values(self):
        df = pd.DataFrame({
            'close': [1, 2, 0.5, 2, 3, 4],
            'macd_hist': [0, -1, 1, -2, 2, -3]
        })
        res = nicegold.detect_macd_divergence(df.copy())
        self.assertIn('bullish', res['divergence'].tolist())
        self.assertIn('bearish', res['divergence'].tolist())

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_generate_entry_signal_full(self):
        df = pd.DataFrame({
            'Gain_Z': [0.5, -0.5, 0.1, -0.1],
            'RSI': [60, 40, 55, 45],
            'Pattern_Label': ['Breakout', 'Reversal', 'Breakout', 'Reversal'],
            'divergence': ['bullish', 'bearish', 'bullish', 'bearish'],
            'ema_touch': [True, True, True, True],
            'trend_confirm': ['up', 'down', 'up', 'down']
        })
        res = nicegold.generate_entry_signal(df.copy())
        self.assertEqual(res['entry_signal'].tolist(), ['buy', 'sell', 'buy', 'sell'])

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_label_wave_phase_branches(self):
        df = pd.DataFrame({
            'divergence': ['bearish', 'bullish'],
            'RSI': [60, 40],
            'Pattern_Label': ['Breakout', 'Reversal']
        })
        result = nicegold.label_wave_phase(df.copy())
        self.assertIn('W.5', result['Wave_Phase'].tolist())
        self.assertIn('W.B', result['Wave_Phase'].tolist())

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_detect_elliott_wave_phase_reset(self):
        close = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        df = pd.DataFrame({
            'close': close,
            'RSI': [40, 60] * 6,
            'divergence': ['bullish', 'bearish'] * 6
        })
        result = nicegold.detect_elliott_wave_phase(df)
        self.assertGreaterEqual(list(result['Wave_Phase']).count('W.1'), 2)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_should_force_entry(self):
        row = {
            'entry_signal': None,
            'spike_score': 0.7,
            'Gain_Z': 0.6,
            'Pattern_Label': 'Breakout'
        }
        last_entry = datetime(2020, 1, 1, 0, 0, 0)
        current_time = last_entry + timedelta(minutes=300)
        self.assertTrue(nicegold.should_force_entry(row, last_entry, current_time))
        current_time_short = last_entry + timedelta(minutes=100)
        self.assertFalse(nicegold.should_force_entry(row, last_entry, current_time_short))

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_should_force_entry_branches(self):
        row = {'entry_signal': 'buy'}
        now = datetime.now()
        self.assertFalse(nicegold.should_force_entry(row, now - timedelta(minutes=500), now))

        row = {
            'entry_signal': None,
            'spike_score': 0.1,
            'Gain_Z': 0.1,
            'Pattern_Label': 'Breakout'
        }
        self.assertFalse(nicegold.should_force_entry(row, now - timedelta(minutes=500), now))

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_apply_wave_macd_cross_entry(self):
        df = pd.DataFrame({
            'entry_signal': [None]*5,
            'Wave_Phase': ['W.1', 'W.2', 'W.B', 'W.5', 'W.2'],
            'divergence': ['bullish', 'bullish', 'bullish', 'bearish', 'bullish'],
            'macd_cross_up': [False, True, True, False, True],
            'macd_cross_down': [False, False, False, True, False],
            'RSI': [50, 50, 50, 50, 50],
            'close': [1, 1, 1, 1, 1],
            'ema35': [1, 1, 1, 1, 1]
        })
        result = nicegold.apply_wave_macd_cross_entry(df.copy())
        self.assertEqual(result['entry_signal'].tolist()[2:], ['buy', 'sell', 'buy'])

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
