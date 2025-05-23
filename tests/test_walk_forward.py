import unittest
import importlib.util

pandas_available = importlib.util.find_spec('pandas') is not None
numpy_available = importlib.util.find_spec('numpy') is not None
sklearn_available = importlib.util.find_spec('sklearn') is not None
matplotlib_available = importlib.util.find_spec('matplotlib') is not None

if pandas_available and numpy_available and sklearn_available and matplotlib_available:
    import pandas as pd
    import numpy as np
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import nicegold

class TestWalkForward(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available and matplotlib_available,
                         'requires pandas, numpy, sklearn, matplotlib')
    def test_walk_forward_and_plot(self):
        df = pd.DataFrame({
            'close': np.linspace(1, 2, 1500),
            'high': np.linspace(1, 2, 1500) + 0.1,
            'low': np.linspace(1, 2, 1500) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=1500, freq='T')
        })
        cfg = {
            'initial_capital': 100.0,
            'risk_per_trade': 0.05,
            'tp1_mult': 0.8,
            'tp2_mult': 2.0,
            'trade_start_hour': 0,
            'trade_end_hour': 23,
            'kill_switch_min': 70,
        }
        result = nicegold.walk_forward_test(df, cfg, fold_days=1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(os.path.exists('trade_plot.png'))
        os.remove('trade_plot.png')

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
