# NICEGOLD Enterprise v1.1
# Backtest with OMS and relaxed entry signal
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"

# Strategy Parameters
initial_capital = 100.0
risk_per_trade = 0.03
max_drawdown_pct = 0.30
kill_switch_dd = 0.45
partial_tp_ratio = 0.5
tp1_mult = 0.8
tp2_mult = 2.2
sl_mult = 1.0
breakeven_buffer = 0.03
lot_max = 1.0
cooldown_bars = 3
oms_recovery_loss = 3
win_streak_boost = 1.25
recovery_multiplier = 0.5
trailing_atr_mult = 1.2

def load_data(path):
    """Load CSV and convert Buddhist year."""
    logger.debug("Loading data from %s", path)
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
    """Calculate EMA, RSI, and ATR."""
    logger.debug("Calculating indicators")
    df['ema35'] = df['close'].ewm(span=35, adjust=False).mean()
    df['rsi'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100 / (1 + (np.mean(np.clip(np.diff(x), 0, None)) / (np.mean(np.clip(-np.diff(x), 0, None)) + 1e-6))),
        raw=False
    )
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    return df

def smart_entry_signal(df):
    """Generate relaxed entry signal."""
    logger.debug("Generating relaxed entry signals")
    df = df.copy()
    df['entry_signal'] = None
    long_cond = (
        (df['rsi'] > 52) &
        (df['close'] > df['ema35']) &
        (df['atr'] > df['atr'].rolling(50).mean()*0.7)
    )
    short_cond = (
        (df['rsi'] < 48) &
        (df['close'] < df['ema35']) &
        (df['atr'] > df['atr'].rolling(50).mean()*0.7)
    )
    df.loc[long_cond, 'entry_signal'] = 'buy'
    df.loc[short_cond, 'entry_signal'] = 'sell'
    return df

class OMSManager:
    """Risk and drawdown management."""
    def __init__(self, capital, dd_pct, kill_switch_dd, lot_max):
        self.capital = capital
        self.peak = capital
        self.dd_pct = dd_pct
        self.kill_switch_dd = kill_switch_dd
        self.lot_max = lot_max
        self.kill_switch = False
        self.win_streak = 0
        self.loss_streak = 0

    def update(self, capital, trade_win):
        self.capital = capital
        if capital > self.peak:
            self.peak = capital
        dd = (self.peak - capital) / self.peak
        if dd > self.kill_switch_dd:
            self.kill_switch = True
            logger.warning("Kill switch triggered: %.2f%%", dd*100)
        if trade_win is True:
            self.win_streak += 1
            self.loss_streak = 0
        elif trade_win is False:
            self.loss_streak += 1
            self.win_streak = 0

    def smart_lot(self, risk_amount, entry_sl_dist):
        lot = risk_amount / max(entry_sl_dist, 1e-5)
        if self.win_streak >= 3:
            lot *= win_streak_boost
        if self.loss_streak >= oms_recovery_loss:
            lot *= recovery_multiplier
        return min(lot, self.lot_max)

def run_backtest():
    """Run simple backtest."""
    df = load_data(M1_PATH)
    df = calc_indicators(df)
    df = smart_entry_signal(df)
    df = df.dropna(subset=['atr', 'ema35', 'rsi']).reset_index(drop=True)

    capital = initial_capital
    oms = OMSManager(capital, max_drawdown_pct, kill_switch_dd, lot_max)
    trades = []
    equity_curve = []
    position = None
    last_entry_idx = -cooldown_bars

    for i, row in df.iterrows():
        equity_curve.append({'timestamp': row['timestamp'], 'equity': capital})
        if oms.kill_switch:
            break
        if position is None and row['entry_signal'] in ['buy', 'sell'] and (i - last_entry_idx) >= cooldown_bars:
            direction = row['entry_signal']
            atr = row['atr']
            entry = row['close']
            sl = entry - atr*sl_mult if direction == 'buy' else entry + atr*sl_mult
            tp1 = entry + atr*tp1_mult if direction == 'buy' else entry - atr*tp1_mult
            tp2 = entry + atr*tp2_mult if direction == 'buy' else entry - atr*tp2_mult
            risk_amount = capital * risk_per_trade
            lot = oms.smart_lot(risk_amount, abs(entry-sl))
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
                'reason_entry': f"Relaxed RSI/EMA35 entry [{direction}]"
            }
            last_entry_idx = i
            logger.info("Entry %s at %.2f lot %.3f", direction, entry, lot)

        if position:
            if not position['tp1_hit']:
                hit_tp1 = (row['high'] >= position['tp1'] if position['type']=='buy' else row['low'] <= position['tp1'])
                if hit_tp1:
                    pnl = position['lot'] * abs(position['tp1']-position['entry']) * partial_tp_ratio
                    capital += pnl
                    position['sl'] = position['entry'] + (breakeven_buffer if position['type']=='buy' else -breakeven_buffer)
                    position['tp1_hit'] = True
                    position['breakeven'] = True
                    trades.append({**position, 'exit_time': row['timestamp'], 'exit': 'TP1', 'pnl': pnl, 'capital': capital, 'reason_exit': 'Partial TP1'})
                    logger.info("Partial TP1 at %.2f (+%.2f$)", position['tp1'], pnl)
                    oms.update(capital, pnl>0)
                    continue
            hit_tp2 = (row['high'] >= position['tp2'] if position['type']=='buy' else row['low'] <= position['tp2'])
            if hit_tp2:
                pnl = position['lot'] * abs(position['tp2']-position['entry']) * (1-partial_tp_ratio if position['tp1_hit'] else 1)
                capital += pnl
                trades.append({**position, 'exit_time': row['timestamp'], 'exit': 'TP2', 'pnl': pnl, 'capital': capital, 'reason_exit': 'TP2'})
                logger.info("TP2 at %.2f (+%.2f$)", position['tp2'], pnl)
                oms.update(capital, pnl>0)
                position = None
                continue
            hit_sl = (row['low'] <= position['sl'] if position['type']=='buy' else row['high'] >= position['sl'])
            if hit_sl:
                pnl = -position['lot'] * abs(position['entry']-position['sl'])
                capital += pnl
                trades.append({**position, 'exit_time': row['timestamp'], 'exit': 'SL', 'pnl': pnl, 'capital': capital, 'reason_exit': 'SL/BE'})
                logger.info("SL at %.2f (%.2f$)", position['sl'], pnl)
                oms.update(capital, pnl>0)
                position = None
                continue
            if position['tp1_hit']:
                trailing_sl = row['close'] - row['atr']*trailing_atr_mult if position['type']=='buy' else row['close'] + row['atr']*trailing_atr_mult
                if position['type']=='buy' and trailing_sl > position['sl']:
                    position['sl'] = trailing_sl
                elif position['type']=='sell' and trailing_sl < position['sl']:
                    position['sl'] = trailing_sl

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trade_log_path = os.path.join(TRADE_DIR, f"trade_log_{now_str}.csv")
    equity_curve_path = os.path.join(TRADE_DIR, f"equity_curve_{now_str}.csv")
    df_trades.to_csv(trade_log_path, index=False)
    df_equity.to_csv(equity_curve_path, index=False)
    logger.info("Saved trade log: %s", trade_log_path)
    logger.info("Saved equity curve: %s", equity_curve_path)

    print("Final Equity:", round(capital,2))
    print("Total Trades:", len(df_trades))
    print("Total Return: {:.2f}%".format((capital-initial_capital)/initial_capital*100))
    print("Winrate: {:.2f}%".format((df_trades['pnl']>0).mean()*100 if not df_trades.empty else 0))

    max_equity = df_equity['equity'].max()
    min_equity = df_equity['equity'].min()
    print(f"Max Equity: {max_equity:.2f} | Min Equity: {min_equity:.2f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_backtest()
