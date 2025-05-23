import unittest
import importlib.util

pandas_available = importlib.util.find_spec('pandas') is not None
numpy_available = importlib.util.find_spec('numpy') is not None
sklearn_available = importlib.util.find_spec('sklearn') is not None
matplotlib_available = importlib.util.find_spec('matplotlib') is not None

if pandas_available and numpy_available:
    import pandas as pd
    import numpy as np
    import io
    import os
    import sys
    from datetime import datetime, timedelta
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import nicegold
    from unittest.mock import patch


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
        self.assertIn('spike_score', df.columns)

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
            'divergence': ['bullish', 'bearish', 'bullish', 'bearish', 'bullish', 'bearish', 'bullish'],
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
    def test_should_force_entry_without_pattern(self):
        row = {
            'entry_signal': None,
            'spike_score': 0.7,
            'Gain_Z': 0.6,
        }
        last_entry = datetime(2020, 1, 1, 0, 0, 0)
        current_time = last_entry + timedelta(minutes=300)
        self.assertTrue(nicegold.should_force_entry(row, last_entry, current_time))

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_apply_wave_macd_cross_entry(self):
        df = pd.DataFrame({
            'entry_signal': [None] * 5,
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

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_load_data_timestamp_parsing(self):
        csv_data = """Date,Timestamp,Open,High,Low,Close,Volume
25630501,0:00:00,1,1,1,1,0
25630501,0:15:00,1,1,1,1,0
"""
        df = nicegold.load_data(io.StringIO(csv_data))
        self.assertEqual(df['timestamp'].iloc[0], pd.Timestamp('2020-05-01 00:00:00'))
        self.assertEqual(df['timestamp'].iloc[1], pd.Timestamp('2020-05-01 00:15:00'))

    def test_run_backtest_cli_path(self):
        import inspect
        source = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('M1_PATH', source)

    def test_run_backtest_cli_fillna_assignment(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('.fillna(50)', src)
        self.assertNotIn("fillna(50, inplace=True)", src)

    def test_run_backtest_cli_debug_tail_and_commission(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('generate_smart_signal', src)
        self.assertNotIn('capital * 0.05', src)

    def test_run_backtest_cli_updated_params(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('risk_per_trade = 0.1', src)

    def test_run_backtest_cli_atr_usage(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn("row['atr'] * sl_multiplier", src)

    def test_run_backtest_cli_drawdown_var(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('current_drawdown =', src)

    def test_run_backtest_cli_progress_logging(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('Backtest progress', src)


class TestModernScalping(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available, 'requires pandas, numpy, sklearn')
    def test_compute_features_columns(self):
        df = pd.DataFrame({
            'close': np.arange(1, 30, dtype=float),
            'high': np.arange(1, 30, dtype=float) + 0.1,
            'low': np.arange(1, 30, dtype=float) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=29, freq='min')
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
            'timestamp': pd.date_range('2020-01-01', periods=60, freq='min')
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
            'timestamp': pd.date_range('2020-01-01', periods=80, freq='min')
        })
        df = nicegold.compute_features(df)
        df = nicegold.train_signal_model(df)
        cfg = {
            'initial_capital': 100.0,
            'risk_per_trade': 0.05,
            'tp1_mult': 0.8,
            'tp2_mult': 2.0,
            'trade_start_hour': 0,
            'trade_end_hour': 23,
            'kill_switch_min': 70,
        }
        trades = nicegold.run_backtest(df, cfg)
        self.assertIsInstance(trades, pd.DataFrame)

    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available, 'requires pandas, numpy, sklearn')
    def test_run_backtest_creates_equity_curve(self):
        df = pd.DataFrame({
            'close': np.linspace(1, 2, 80),
            'high': np.linspace(1, 2, 80) + 0.1,
            'low': np.linspace(1, 2, 80) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=80, freq='min')
        })
        df = nicegold.compute_features(df)
        df = nicegold.train_signal_model(df)
        cfg = {
            'initial_capital': 100.0,
            'risk_per_trade': 0.05,
            'tp1_mult': 0.8,
            'tp2_mult': 2.0,
            'trade_start_hour': 0,
            'trade_end_hour': 23,
            'kill_switch_min': 70,
        }
        nicegold.run_backtest(df, cfg)
        self.assertTrue(os.path.exists('equity_curve.csv'))
        os.remove('equity_curve.csv')

    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available, 'requires pandas, numpy, sklearn')
    def test_load_config_and_time_filter(self):
        cfg = nicegold.load_config() if hasattr(nicegold, 'load_config') else {
            'trade_start_hour': 2,
            'trade_end_hour': 3,
            'initial_capital': 100.0,
            'risk_per_trade': 0.05,
            'tp1_mult': 0.8,
            'tp2_mult': 2.0,
            'kill_switch_min': 70,
        }

        df = pd.DataFrame({
            'close': np.linspace(1, 2, 80),
            'high': np.linspace(1, 2, 80) + 0.1,
            'low': np.linspace(1, 2, 80) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=80, freq='min')
        })
        df = nicegold.compute_features(df)
        df['entry_signal'] = 'buy'
        df['hour'] = df['timestamp'].dt.hour
        cfg['trade_start_hour'] = 2
        cfg['trade_end_hour'] = 3
        trades = nicegold.run_backtest(df, cfg)
        self.assertEqual(len(trades), 0)


class TestWalkForward(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available and sklearn_available and matplotlib_available,
                         'requires pandas, numpy, sklearn, matplotlib')
    def test_walk_forward_and_plot(self):
        df = pd.DataFrame({
            'close': np.linspace(1, 2, 1500),
            'high': np.linspace(1, 2, 1500) + 0.1,
            'low': np.linspace(1, 2, 1500) - 0.1,
            'timestamp': pd.date_range('2020-01-01', periods=1500, freq='min')
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


class TestNewFunctions(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_generate_smart_signal_column(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=3, freq='min'),
            'open': [0.9, 1.9, 2.9],
            'macd': [1, 2, 3],
            'signal': [0, 1, 2],
            'Wave_Phase': ['W.2', 'W.3', 'W.5'],
            'RSI': [55, 60, 65],
            'close': [1, 2, 3],
            'ema35': [1, 2, 3]
        })
        with patch.object(nicegold, 'load_csv_m15') as m15, \
             patch.object(nicegold, 'detect_ob_m15') as dob, \
             patch.object(nicegold, 'detect_fvg_m15') as dfvg, \
             patch.object(nicegold, 'detect_liquidity_grab_m15') as dlg:
            m15.return_value = pd.DataFrame({'timestamp':[pd.Timestamp('2020-01-01')],
                                            'open':[1.0],'high':[1.0],'low':[1.0],'close':[1.0]})
            dob.return_value = pd.DataFrame({'type':['bullish'],'zone':[1.0],'idx':[0],'time':[pd.Timestamp('2020-01-01')]})
            dfvg.return_value = pd.DataFrame({'type':['bullish'],'low':[0.9],'high':[1.1],'idx':[0],'time':[pd.Timestamp('2020-01-01')]})
            dlg.return_value = pd.DataFrame({'type':['grab_long'],'zone':[1.0],'idx':[0],'time':[pd.Timestamp('2020-01-01')]})
            res = nicegold.generate_smart_signal(df.copy())
        self.assertIn('entry_signal', res.columns)
        self.assertEqual(res['entry_signal'].iloc[0], 'buy')

    def test_check_drawdown_true(self):
        self.assertTrue(nicegold.check_drawdown(70, 100, limit=0.2))

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_backtest_with_partial_tp_returns(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, freq='min'),
            'open': [1, 1.1, 1.2, 1.3, 1.4],
            'close': [1, 1.1, 1.2, 1.3, 1.4],
            'high': [1, 1.2, 1.3, 1.4, 1.5],
            'low': [0.9, 1.0, 1.1, 1.2, 1.3],
            'atr': [0.1]*5,
            'ema35': [1]*5,
            'RSI': [60]*5,
            'Wave_Phase': ['W.2']*5,
            'macd': [1]*5,
            'signal': [0]*5
        })
        empty = pd.DataFrame({'type': pd.Series([],dtype=object),
                              'zone': pd.Series([],dtype=float),
                              'idx': pd.Series([],dtype=int),
                              'time': pd.Series([],dtype='datetime64[ns]')})
        with patch.object(nicegold, 'load_csv_m15', return_value=df[['timestamp','close','high','low','open']]), \
             patch.object(nicegold, 'detect_ob_m15', return_value=empty), \
             patch.object(nicegold, 'detect_fvg_m15', return_value=empty), \
             patch.object(nicegold, 'detect_liquidity_grab_m15', return_value=empty):
            df = nicegold.generate_smart_signal(df)
        trades, cap = nicegold.backtest_with_partial_tp(df)
        self.assertIsInstance(trades, pd.DataFrame)

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_backtest_with_partial_tp_dropna(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, freq='min'),
            'open': [1, 1.1, 1.2, 1.3, 1.4],
            'close': [1, 1.1, 1.2, 1.3, 1.4],
            'high': [1, 1.2, 1.3, 1.4, 1.5],
            'low': [0.9, 1.0, 1.1, 1.2, 1.3],
            'atr': [np.nan, 0.1, 0.1, 0.1, 0.1],
            'ema35': [np.nan, 1, 1, 1, 1],
            'RSI': [np.nan, 60, 60, 60, 60],
            'Wave_Phase': ['W.2'] * 5,
            'macd': [1] * 5,
            'signal': [0] * 5
        })
        empty = pd.DataFrame({'type': pd.Series([],dtype=object),
                              'zone': pd.Series([],dtype=float),
                              'idx': pd.Series([],dtype=int),
                              'time': pd.Series([],dtype='datetime64[ns]')})
        with patch.object(nicegold, 'load_csv_m15', return_value=df[['timestamp','close','high','low','open']]), \
             patch.object(nicegold, 'detect_ob_m15', return_value=empty), \
             patch.object(nicegold, 'detect_fvg_m15', return_value=empty), \
             patch.object(nicegold, 'detect_liquidity_grab_m15', return_value=empty):
            df = nicegold.generate_smart_signal(df)
        trades, cap = nicegold.backtest_with_partial_tp(df)
        self.assertIsInstance(trades, pd.DataFrame)


class TestConvertCSV(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_convert_csv_ad_to_be(self):
        csv_data = """Date,Timestamp,Open
20240501,0:00:00,1
20240502,0:15:00,1
"""
        df = nicegold.convert_csv_ad_to_be(io.StringIO(csv_data))
        self.assertEqual(df['date'].tolist(), ['25670501', '25670502'])


class TestLoadCSVFunctions(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_load_csv_m15_timestamp_parsing(self):
        csv_data = """Date,Timestamp,Open,High,Low,Close
25670501,0:00:00,1,1,1,1
25670501,0:15:00,1,1,1,1
"""
        df = nicegold.load_csv_m15(io.StringIO(csv_data))
        self.assertNotIn('date', df.columns)
        self.assertEqual(df['timestamp'].iloc[0], pd.Timestamp('2024-05-01 00:00:00'))

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_load_csv_m1_timestamp_parsing(self):
        csv_data = """Date,Timestamp,Open,High,Low,Close
25660501,0:00:00,1,1,1,1
25660501,0:01:00,1,1,1,1
"""
        df = nicegold.load_csv_m1(io.StringIO(csv_data))
        self.assertNotIn('date', df.columns)
        self.assertEqual(df['timestamp'].iloc[1], pd.Timestamp('2023-05-01 00:01:00'))

    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_load_csv_low_memory_flags(self):
        with patch('pandas.read_csv') as reader:
            reader.return_value = pd.DataFrame({
                'date': ['25670501'],
                'timestamp': ['0:00:00'],
                'open': [1], 'high': [1], 'low': [1], 'close': [1]
            })
            nicegold.load_csv_m15('dummy.csv')
            reader.assert_called_with('dummy.csv', low_memory=False)
        with patch('pandas.read_csv') as reader:
            reader.return_value = pd.DataFrame({
                'date': ['25660501'],
                'timestamp': ['0:00:00'],
                'open': [1], 'high': [1], 'low': [1], 'close': [1]
            })
            nicegold.load_csv_m1('dummy.csv')
            reader.assert_called_with('dummy.csv', low_memory=False)

    def test_default_paths_relative(self):
        self.assertFalse(nicegold.M1_PATH.startswith('/content'))


class TestOptimizeMemory(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_optimize_memory_downcast(self):
        df = pd.DataFrame({
            'a': np.arange(3, dtype='int64'),
            'b': np.arange(3, dtype='float64')
        })
        df_opt = nicegold.optimize_memory(df.copy())
        self.assertNotEqual(df_opt['a'].dtype, np.int64)
        self.assertNotEqual(df_opt['b'].dtype, np.float64)



class TestLogging(unittest.TestCase):
    def test_get_logger(self):
        logger = nicegold.get_logger() if hasattr(nicegold, 'get_logger') else None
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'nicegold')


class TestPatchGFix2(unittest.TestCase):
    def test_trade_dir_constant(self):
        self.assertTrue(hasattr(nicegold, 'TRADE_DIR'))

    def test_run_backtest_cli_paths(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('trade_log_', src)
        self.assertIn('equity_curve_', src)

class TestTrendConfirmFunctions(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_is_trending_and_confirm(self):
        df = pd.DataFrame({
            'ema_fast': [1]*200,
            'ema_slow': [0.5]*200,
            'atr': np.linspace(0.1, 1, 200),
            'high': np.linspace(1,2,200),
            'low': np.linspace(0.5,1.5,200),
            'open': np.linspace(1,2,200),
            'close': np.linspace(1,2,200)
        })
        self.assertTrue(nicegold.is_trending(df, 150))
        self.assertTrue(nicegold.is_confirm_bar(df.assign(high=df['high']*1.1), 150, 'buy'))

    def test_run_backtest_cli_uses_trend_confirm(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('is_trending', src)


class TestEnhancedRunBacktest(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_run_backtest_outputs_new_fields(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=30, freq='min'),
            'close': np.linspace(1, 1.5, 30),
            'high': np.linspace(1, 1.5, 30) + 0.2,
            'low': np.linspace(1, 1.5, 30) - 0.1,
            'atr': [0.1]*30,
            'hour': [h % 24 for h in range(30)],
            'volume': [1]*30,
            'entry_signal': [None]*30
        })
        df.at[25, 'entry_signal'] = 'buy'
        cfg = {
            'initial_capital': 100.0,
            'risk_per_trade': 0.05,
            'tp1_mult': 0.8,
            'tp2_mult': 2.0,
            'trade_start_hour': 0,
            'trade_end_hour': 23,
            'kill_switch_min': 70,
            'max_drawdown_pct': 30
        }
        trades = nicegold.run_backtest(df, cfg)
        for col in ['type', 'reason_entry', 'risk_price', 'risk_amount']:
            self.assertIn(col, trades.columns)

class TestSMCMultiTimeframe(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, "requires pandas and numpy")
    def test_align_mtf_zones_columns(self):
        df_m1 = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=1, freq="min"),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0]
        })
        ob = pd.DataFrame({"type":["bullish"],"zone":[1.0],"idx":[0],"time":[df_m1["timestamp"].iloc[0]]})
        fvg = pd.DataFrame({"type":["bullish"],"low":[0.9],"high":[1.1],"idx":[0],"time":[df_m1["timestamp"].iloc[0]]})
        lg = pd.DataFrame({"type":["grab_long"],"zone":[1.0],"idx":[0],"time":[df_m1["timestamp"].iloc[0]]})
        res = nicegold.align_mtf_zones(df_m1.copy(), ob, fvg, lg)
        self.assertIn("OB_Bull", res.columns)
        self.assertIn("FVG_Bull", res.columns)
        self.assertIn("LG_Bull", res.columns)

    @unittest.skipUnless(pandas_available and numpy_available, "requires pandas and numpy")
    def test_align_mtf_zones_flags(self):
        df_m1 = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.01, 1.0],
            "atr": [0.1, 0.1, 0.1]
        })
        ob = pd.DataFrame({"type": ["bullish"], "zone": [1.0], "idx": [0], "time": [df_m1["timestamp"].iloc[0]]})
        fvg = pd.DataFrame({"type": ["bullish"], "low": [0.99], "high": [1.01], "idx": [0], "time": [df_m1["timestamp"].iloc[0]]})
        lg = pd.DataFrame({"type": ["grab_long"], "zone": [1.0], "idx": [0], "time": [df_m1["timestamp"].iloc[0]]})
        res = nicegold.align_mtf_zones(df_m1.copy(), ob, fvg, lg)
        self.assertTrue(res['OB_Bull'].any())
        self.assertTrue(res['FVG_Bull'].any())
        self.assertTrue(res['LG_Bull'].any())

    @unittest.skipUnless(pandas_available and numpy_available, "requires pandas and numpy")
    def test_is_smc_entry_buy(self):
        df = pd.DataFrame({
            "open": [1.0, 1.2, 0.8],
            "close": [1.0, 1.0, 1.0],
            "high": [1.0, 1.3, 1.5],
            "low": [0.9, 0.9, 0.7],
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="min")
        })
        df["atr"] = 0.1
        ob_df = nicegold.detect_order_block(df)
        fvg_df = pd.DataFrame(columns=["idx","type","low","high","time"])
        lg_df = nicegold.detect_liquidity_grab(df, swing_window=2)
        from unittest.mock import patch
        with patch.object(nicegold, "is_confirm_bar", return_value=True):
            signal = nicegold.is_smc_entry(df, 2, ob_df, fvg_df, lg_df)
        self.assertEqual(signal, "buy")

class TestQAStepLogging(unittest.TestCase):
    def test_qa_log_step_logs_info(self):
        logger = nicegold.get_logger()
        with self.assertLogs(logger, level='INFO') as cm:
            nicegold.qa_log_step('test')
        self.assertTrue(any('STEP: test' in msg for msg in cm.output))


class TestMainBlockOrder(unittest.TestCase):
    def test_main_block_after_smc_funcs(self):
        import os
        with open(nicegold.__file__, 'r') as f:
            src = f.read()
        pos_main = src.rfind("if __name__ == \"__main__\"")
        pos_load = src.find("def load_csv_m15")
        self.assertGreater(pos_main, pos_load)


class TestFallbackSimpleSignal(unittest.TestCase):
    @unittest.skipUnless(pandas_available and numpy_available, 'requires pandas and numpy')
    def test_fallback_simple_signal_generates_entry(self):
        df = pd.DataFrame({
            'macd_hist': [0.1, 0.2, 0.3],
            'RSI': [60, 60, 60]
        })
        res = nicegold.fallback_simple_signal(df.copy())
        self.assertTrue((res['entry_signal'] == 'buy').all())

    def test_run_backtest_cli_calls_fallback(self):
        import inspect
        src = inspect.getsource(nicegold.run_backtest_cli)
        self.assertIn('fallback_simple_signal', src)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
