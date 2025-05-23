# NICEGOLD Enterprise QA Supergrowth v3 - Min SL, Lot Cap, Trend Only, RR Adaptive
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"

# [Patch] Parameters – MM & Growth
initial_capital = 100.0
risk_per_trade = 0.03
tp1_mult = 1.0
tp2_mult = 3.5
sl_mult = 1.0
min_sl_dist = 2.0    # [Patch] ป้องกัน SL ใกล้เกิน
lot_max = 0.25
lot_cap_500 = 0.25
lot_cap_2000 = 0.5
cooldown_bars = 2
oms_recovery_loss = 4
win_streak_boost = 1.1
recovery_multiplier = 0.3
trailing_atr_mult = 1.1
kill_switch_dd = 0.5
trend_lookback = 30
adx_period = 20
adx_thresh = 15


def load_data(path):
    """Load CSV and convert Buddhist year."""
    logger.info("[Patch] Loading data: %s", path)
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        year = df['date'].astype(str).str[:4].astype(int) - 543
        month = df['date'].astype(str).str[4:6]
        day = df['date'].astype(str).str[6:8]
        df['timestamp'] = pd.to_datetime(
            year.astype(str) + '-' + month + '-' + day + ' ' + df['timestamp'],
            format='%Y-%m-%d %H:%M:%S'
        )
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def calc_indicators(df):
    """Calculate EMA trend, RSI, ATR, and ADX."""
    logger.info("[Patch] Calculating indicators")
    df['ema_fast'] = df['close'].ewm(span=trend_lookback, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=trend_lookback*2, adjust=False).mean()
    df['rsi'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100 / (1 + (np.mean(np.clip(np.diff(x), 0, None)) / (np.mean(np.clip(-np.diff(x), 0, None)) + 1e-6))),
        raw=False
    )
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    # ADX for volatility filter
    up_move = df['high'].diff().clip(lower=0)
    down_move = -df['low'].diff().clip(upper=0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    plus_dm = up_move.where(up_move > down_move, 0.0)
    minus_dm = down_move.where(down_move > up_move, 0.0)
    atr = tr.rolling(adx_period).mean()
    plus_di = 100 * plus_dm.rolling(adx_period).sum() / atr
    minus_di = 100 * minus_dm.rolling(adx_period).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx'] = dx.rolling(adx_period).mean()
    return df


def is_strong_trend(df, i):
    """[Patch] Trend confirm ด้วย EMA cross และ Trend Strength rolling"""
    if i < trend_lookback * 2:
        return False
    trend = (
        (df['ema_fast'].iloc[i - trend_lookback + 1:i + 1] >
         df['ema_slow'].iloc[i - trend_lookback + 1:i + 1]).all() or
        (df['ema_fast'].iloc[i - trend_lookback + 1:i + 1] <
         df['ema_slow'].iloc[i - trend_lookback + 1:i + 1]).all()
    )
    adx = df['adx'].iloc[i] > adx_thresh
    atr_high = df['atr'].iloc[i] > df['atr'].rolling(1000).median().iloc[i] * 1.2
    return trend and (adx or atr_high)


def smart_entry_signal(df):
    logger.info("[Patch] Generating entry signals: trend-only, min SL guard")
    df = df.copy()
    df['entry_signal'] = None
    for i in range(len(df)):
        if not is_strong_trend(df, i):
            continue
        if df['atr'].iloc[i] < min_sl_dist:
            continue
        if (df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]) and (df['rsi'].iloc[i] > 53):
            df.at[df.index[i], 'entry_signal'] = 'buy'
        if (df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]) and (df['rsi'].iloc[i] < 47):
            df.at[df.index[i], 'entry_signal'] = 'sell'
    return df


class OMSManager:
    """Risk and drawdown management"""

    def __init__(self, capital, kill_switch_dd, lot_max):
        self.capital = capital
        self.peak = capital
        self.kill_switch_dd = kill_switch_dd
        self.lot_max = lot_max
        self.kill_switch = False
        self.win_streak = 0
        self.loss_streak = 0
        self.recovery_mode = False

    def update(self, capital, trade_win):
        self.capital = capital
        if capital > self.peak:
            self.peak = capital
        dd = (self.peak - capital) / self.peak
        if dd > self.kill_switch_dd:
            self.kill_switch = True
            logger.warning("[Patch] OMS Kill switch triggered: DD %.2f%%", dd*100)
        if trade_win is True:
            self.win_streak += 1
            self.loss_streak = 0
        elif trade_win is False:
            self.loss_streak += 1
            self.win_streak = 0
        if self.loss_streak >= oms_recovery_loss:
            self.recovery_mode = True
        if self.win_streak > 2:
            self.recovery_mode = False

    def smart_lot(self, capital, risk_amount, entry_sl_dist):
        # [Patch] Lot cap by capital tier
        lot_cap = self.lot_max
        if capital < 500:
            lot_cap = lot_cap_500
        elif capital < 2000:
            lot_cap = lot_cap_2000
        lot = risk_amount / max(entry_sl_dist, min_sl_dist)
        if self.recovery_mode:
            lot *= recovery_multiplier
        lot = max(0.01, min(lot, lot_cap))
        return lot


def run_backtest():
    """Run backtest with advanced OMS"""
    df = load_data(M1_PATH)
    df = calc_indicators(df)
    df = smart_entry_signal(df)
    df = df.dropna(subset=['atr', 'ema_fast', 'ema_slow', 'rsi']).reset_index(drop=True)

    capital = initial_capital
    oms = OMSManager(capital, kill_switch_dd, lot_max)
    trades = []
    equity_curve = []
    position = None
    last_entry_idx = -cooldown_bars

    for i, row in df.iterrows():
        equity_curve.append({'timestamp': row['timestamp'], 'equity': capital, 'dd': (oms.peak-capital)/oms.peak})
        if oms.kill_switch:
            logger.info("[Patch] OMS: Stop trading (kill switch)")
            break

        # Entry with min SL distance guard
        if position is None and row['entry_signal'] in ['buy', 'sell'] and (i - last_entry_idx) >= cooldown_bars:
            direction = row['entry_signal']
            atr = max(row['atr'], min_sl_dist)
            entry = row['close']
            sl = entry - atr*sl_mult if direction == 'buy' else entry + atr*sl_mult
            tp1 = entry + atr*tp1_mult if direction == 'buy' else entry - atr*tp1_mult
            tp2 = entry + atr*tp2_mult if direction == 'buy' else entry - atr*tp2_mult
            risk_amount = capital * risk_per_trade
            lot = oms.smart_lot(capital, risk_amount, abs(entry-sl))
            mode = "RECOVERY" if oms.recovery_mode else "NORMAL"
            position = {
                'entry_time': row['timestamp'],
                'type': direction,
                'entry': entry,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'lot': lot,
                'tp1_hit': False,
                'breakeven': False,
                'reason_entry': f"TrendOnly+SLGuard {direction}",
                'mode': mode,
                'risk': risk_amount,
                'dd_at_entry': (oms.peak-capital)/oms.peak,
                'peak_equity': oms.peak
            }
            last_entry_idx = i
            logger.info("[Patch] Entry: %s at %.2f, SL %.2f, TP1 %.2f, TP2 %.2f, Lot %.3f Mode %s", direction, entry, sl, tp1, tp2, lot, mode)

        if position:
            # Partial TP1
            if not position['tp1_hit']:
                hit_tp1 = (row['high'] >= position['tp1'] if position['type']=='buy' else row['low'] <= position['tp1'])
                if hit_tp1:
                    pnl = position['lot'] * abs(position['tp1']-position['entry']) * 0.5
                    capital += pnl
                    # Move stop to breakeven after TP1
                    position['sl'] = position['entry']
                    position['tp1_hit'] = True
                    position['breakeven'] = True
                    trades.append({**position, 'exit_time': row['timestamp'], 'exit': 'TP1', 'pnl': pnl, 'capital': capital, 'reason_exit': 'Partial TP1'})
                    logger.info("[Patch] Partial TP1 at %.2f (+%.2f$)", position['tp1'], pnl)
                    oms.update(capital, pnl>0)
                    continue

            # TP2
            hit_tp2 = (row['high'] >= position['tp2'] if position['type']=='buy' else row['low'] <= position['tp2'])
            if hit_tp2:
                pnl = position['lot'] * abs(position['tp2']-position['entry']) * (0.5 if position['tp1_hit'] else 1)
                capital += pnl
                trades.append({**position, 'exit_time': row['timestamp'], 'exit': 'TP2', 'pnl': pnl, 'capital': capital, 'reason_exit': 'TP2'})
                logger.info("[Patch] TP2 at %.2f (+%.2f$)", position['tp2'], pnl)
                oms.update(capital, pnl>0)
                position = None
                continue

            # Stop Loss / Breakeven
            hit_sl = (row['low'] <= position['sl'] if position['type']=='buy' else row['high'] >= position['sl'])
            if hit_sl:
                pnl = -position['lot'] * abs(position['entry']-position['sl'])
                capital += pnl
                trades.append({**position, 'exit_time': row['timestamp'], 'exit': 'SL', 'pnl': pnl, 'capital': capital, 'reason_exit': 'SL/BE'})
                logger.info("[Patch] SL/BE at %.2f (%.2f$)", position['sl'], pnl)
                oms.update(capital, pnl>0)
                position = None
                continue

            # Trailing SL after TP1
            if position['tp1_hit']:
                trailing_sl = row['close'] - row['atr']*trailing_atr_mult if position['type']=='buy' else row['close'] + row['atr']*trailing_atr_mult
                if position['type']=='buy' and trailing_sl > position['sl']:
                    position['sl'] = trailing_sl
                elif position['type']=='sell' and trailing_sl < position['sl']:
                    position['sl'] = trailing_sl

    # Save Trade Log & Equity Curve
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trade_log_path = os.path.join(TRADE_DIR, f"trade_log_{now_str}.csv")
    equity_curve_path = os.path.join(TRADE_DIR, f"equity_curve_{now_str}.csv")
    df_trades.to_csv(trade_log_path, index=False)
    df_equity.to_csv(equity_curve_path, index=False)
    logger.info("[Patch] Saved trade log: %s", trade_log_path)
    logger.info("[Patch] Saved equity curve: %s", equity_curve_path)

    # Summary
    print("Final Equity:", round(capital,2))
    print("Total Trades:", len(df_trades))
    print("Total Return: {:.2f}%".format((capital-initial_capital)/initial_capital*100))
    print("Winrate: {:.2f}%".format((df_trades['pnl']>0).mean()*100 if not df_trades.empty else 0))
    cols = ['entry_time','type','entry','exit','pnl','capital','mode','reason_entry','reason_exit']
    if not df_trades.empty:
        print(df_trades[cols].tail(12))
    max_equity = df_equity['equity'].max()
    min_equity = df_equity['equity'].min()
    print(f"[Patch] Max Equity: {max_equity:.2f} | Min Equity: {min_equity:.2f}")
    print(f"[Patch] OMS mode: recovery = {oms.recovery_mode}")


if __name__ == "__main__":
    run_backtest()
