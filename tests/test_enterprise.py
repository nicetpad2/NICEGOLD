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
        self.assertEqual(df['timestamp'].iloc[0], pd.Timestamp('2024-05-01 00:00:00'))

    def test_calc_indicators_columns(self):
        df = pd.DataFrame({
            'close': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
            'high': np.arange(14)+1,
            'low': np.arange(14)
        })
        res = enterprise.calc_indicators(df)
        for col in ['ema35','rsi','atr']:
            self.assertIn(col, res.columns)

    def test_smart_entry_signal(self):
        df = pd.DataFrame({
            'rsi':[60]*55 + [40]*5,
            'close':[2]*55 + [0.5]*5,
            'ema35':[1]*60,
            'atr':[1]*60
        })
        res = enterprise.smart_entry_signal(df)
        self.assertIn('buy', res['entry_signal'].values)
        self.assertIn('sell', res['entry_signal'].values)

    def test_oms_smart_lot(self):
        oms = enterprise.OMSManager(100,0.3,0.5,1)
        lot = oms.smart_lot(3,0.1)
        self.assertAlmostEqual(lot, min(3/0.1,1))

    def test_run_backtest_outputs(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=30, freq='min'),
            'open': np.linspace(1,1.3,30),
            'high': np.linspace(1,1.3,30)+0.1,
            'low': np.linspace(1,1.3,30)-0.1,
            'close': np.linspace(1,1.3,30)
        })
        df.to_csv('tmp.csv', index=False)
        enterprise.M1_PATH = 'tmp.csv'
        enterprise.TRADE_DIR = '.'
        enterprise.run_backtest()
        self.assertTrue(os.path.exists('tmp.csv'))
        os.remove('tmp.csv')
        # ensure logs saved
        logs = [f for f in os.listdir('.') if f.startswith('trade_log_')]
        for f in logs:
            os.remove(f)
        curves = [f for f in os.listdir('.') if f.startswith('equity_curve_')]
        for f in curves:
            os.remove(f)

if __name__ == '__main__':
    unittest.main()
