# [Patch] NICEGOLD Enterprise Supergrowth v4 - Entry Boost, Adaptive Lot, OMS, Smart Exit, QA
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"

# [Patch] Parameters – MM & Growth
initial_capital = 100.0
risk_per_trade = 0.06      # [Patch] เพิ่มความเสี่ยง (6%)
tp1_mult = 0.9             # [Patch] ปรับ TP1 แคบขึ้นเพื่อเก็บกำไรบางส่วนเร็วขึ้น
tp2_mult = 3.0             # [Patch] TP2 ต่ำลง, เน้นยิงไวออกไว
sl_mult = 1.0
min_sl_dist = 2.0
lot_max = 5.0              # [Patch] ปลดลิมิตให้สามารถปั้นพอร์ตโต
lot_cap_500 = 0.5
lot_cap_2000 = 1.0
lot_cap_10000 = 2.5
lot_cap_max = 5.0
cooldown_bars = 0          # [Patch] ไม่มี cooldown
oms_recovery_loss = 3      # [Patch] Recovery เร็วขึ้น (จาก 4 → 3)
win_streak_boost = 1.3     # [Patch] Boost สูงขึ้น
recovery_multiplier = 3.0  # [Patch] Aggressive recovery
trailing_atr_mult = 1.1
kill_switch_dd = 0.35      # [Patch] Kill switch หาก DD > 35%
trend_lookback = 25        # [Patch] เร็วขึ้น (trend สั้นลง)
adx_period = 14
adx_thresh = 12            # [Patch] Relax adx guard
adx_strong = 23            # [Patch] ปรับ adx strong
force_entry_gap = 300      # [Patch] Force entry หากไม่มี order เกิน 300 แท่ง
trade_start_hour = 8
trade_end_hour = 23

def load_data(path):
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
    logger.info("[Patch] Calculating indicators")
    df['ema_fast'] = df['close'].ewm(span=trend_lookback, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=trend_lookback*2, adjust=False).mean()
    df['ema_fast_htf'] = df['close'].ewm(span=trend_lookback*4, adjust=False).mean()
    df['ema_slow_htf'] = df['close'].ewm(span=trend_lookback*8, adjust=False).mean()
    df['rsi'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100 / (1 + (np.mean(np.clip(np.diff(x), 0, None)) / (np.mean(np.clip(-np.diff(x), 0, None)) + 1e-6))),
        raw=False
    )
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
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

def smart_entry_signal(df):
    logger.info("[Patch] Vectorized entry signal (trend+min SL guard+relax+force entry)")
    df = df.copy()
    df['entry_signal'] = None

    # [Patch] 1. Relax ATR/ADX Guard (และกรองเฉพาะช่วงเวลาสำคัญ)
    mask_valid = (
        (df['atr'] > min_sl_dist * 0.7) &
        (df['adx'] > adx_thresh)
        & (df['timestamp'].dt.hour >= trade_start_hour)
        & (df['timestamp'].dt.hour < trade_end_hour)
    )

    # [Patch] 2. Rolling trend mask (trend สั้นลง 7 bar)
    n_trend = 7
    trend_up = (
        (df['ema_fast'] > df['ema_slow'])
        .rolling(n_trend, min_periods=n_trend)
        .apply(lambda x: x.all(), raw=True).fillna(0)
        .astype(bool)
    )
    trend_dn = (
        (df['ema_fast'] < df['ema_slow'])
        .rolling(n_trend, min_periods=n_trend)
        .apply(lambda x: x.all(), raw=True).fillna(0)
        .astype(bool)
    )

    # [Patch] 3. RSI + multi-timeframe + relax wick filter (ผ่านมากขึ้น)
    def relax_wick(row):
        price_range = max(row['high'] - row['low'], 1e-6)
        upper_wick = (row['high'] - row['close']) / price_range
        lower_wick = (row['close'] - row['low']) / price_range
        # [Patch] Relax ให้ Wick ถึง 80% ผ่านได้
        return upper_wick < 0.80 and lower_wick < 0.80

    entry_long = (
        mask_valid & trend_up & (df['rsi'] > 51) &
        (df['ema_fast_htf'] > df['ema_slow_htf']) &
        df.apply(relax_wick, axis=1)
    )
    entry_short = (
        mask_valid & trend_dn & (df['rsi'] < 49) &
        (df['ema_fast_htf'] < df['ema_slow_htf']) &
        df.apply(relax_wick, axis=1)
    )
    df.loc[entry_long, 'entry_signal'] = 'buy'
    df.loc[entry_short, 'entry_signal'] = 'sell'

    # [Patch] 4. Force Entry หากไม่มี signal เกิน force_entry_gap bar
    last_entry = -force_entry_gap
    for i in range(len(df)):
        if pd.notna(df['entry_signal'].iloc[i]):
            last_entry = i
        elif i - last_entry > force_entry_gap:
            if mask_valid.iloc[i]:
                # [Patch] เลือกทางเดียวกับ momentum/ema
                if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:
                    df.at[i, 'entry_signal'] = 'buy'
                else:
                    df.at[i, 'entry_signal'] = 'sell'
                last_entry = i
                logger.info("[Patch] Force Entry at %s", df['timestamp'].iloc[i])

    logger.info(
        "[Patch] Entry signal counts: buy=%d, sell=%d",
        (df['entry_signal'] == 'buy').sum(),
        (df['entry_signal'] == 'sell').sum(),
    )
    logger.debug(
        "Entry signals generated on indices: %s",
        df[df['entry_signal'].notna()].index.tolist(),
    )
    return df

class OMSManager:
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
            if not self.recovery_mode:
                self.recovery_mode = True
                logger.warning("[Patch] Recovery mode activated after %d consecutive losses", self.loss_streak)
        if self.recovery_mode and self.win_streak > 2:
            self.recovery_mode = False
            logger.info("[Patch] Recovery mode deactivated after win streak of %d", self.win_streak)

    def smart_lot(self, capital, risk_amount, entry_sl_dist):
        lot_cap = self.lot_max
        if capital < 500:
            lot_cap = lot_cap_500
        elif capital < 2000:
            lot_cap = lot_cap_2000
        elif capital < 10000:
            lot_cap = lot_cap_10000
        else:
            lot_cap = self.lot_max
        lot = risk_amount / max(entry_sl_dist, min_sl_dist)
        if self.recovery_mode:
            lot *= recovery_multiplier
        # [Patch] Boost lot ขณะ win streak
        if self.win_streak >= 2:
            lot *= win_streak_boost
        lot_cap = min(lot_cap, lot_cap_max)
        lot = max(0.01, min(lot, lot_cap))
        return lot

def run_backtest():
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

        # Entry
        if position is None and row['entry_signal'] in ['buy', 'sell'] and (i - last_entry_idx) >= cooldown_bars:
            direction = row['entry_signal']
            price_range = max(row['high'] - row['low'], 1e-6)
            upper_wick_ratio = (row['high'] - row['close']) / price_range
            lower_wick_ratio = (row['close'] - row['low']) / price_range
            # [Patch] Relax wick filter - ผ่าน 80%
            if direction == 'buy' and upper_wick_ratio > 0.80:
                continue
            if direction == 'sell' and lower_wick_ratio > 0.80:
                continue
            atr = max(row['atr'], min_sl_dist)
            entry = row['close']
            sl = entry - atr*sl_mult if direction == 'buy' else entry + atr*sl_mult
            tp1 = entry + atr*tp1_mult if direction == 'buy' else entry - atr*tp1_mult
            tp2 = entry + atr*tp2_mult if direction == 'buy' else entry - atr*tp2_mult
            risk_amount = capital * risk_per_trade
            # [Patch] Adaptive risk management + signal boost
            if not oms.recovery_mode:
                if oms.win_streak >= 2:
                    risk_amount *= win_streak_boost
                if row['adx'] > adx_strong:
                    risk_amount *= 1.5
                    logger.info("[Patch] Signal boost: ADX %.1f > %.0f, risk increased 50%%", row['adx'], adx_strong)
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
    max_dd = df_equity['dd'].max()
    print(f"[Patch] Max Drawdown: {max_dd*100:.2f}%")
    print(f"[Patch] OMS mode: recovery = {oms.recovery_mode}")

    # [Patch] Show equity curve visualization (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,5))
        plt.plot(df_equity['timestamp'], df_equity['equity'])
        plt.title('Equity Curve [Patch]')
        plt.xlabel('Timestamp')
        plt.ylabel('Equity')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning("[Patch] Matplotlib not available for equity plot: %s", e)

if __name__ == "__main__":
    run_backtest()
