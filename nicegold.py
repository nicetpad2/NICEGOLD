import pandas as pd
import numpy as np

# === คำนวณ Indicator ===
def calculate_macd(df, price_col='close', fast=12, slow=26, signal=9):
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df['macd'] = macd
    df['signal'] = signal_line
    df['macd_hist'] = macd - signal_line
    return df

def detect_macd_divergence(df, price_col='close'):
    price = df[price_col].values
    hist = df['macd_hist'].values
    divergence = [None, None]
    for i in range(2, len(df)):
        if price[i] < price[i - 2] and hist[i] > hist[i - 2]:
            divergence.append('bullish')
        elif price[i] > price[i - 2] and hist[i] < hist[i - 2]:
            divergence.append('bearish')
        else:
            divergence.append(None)
    df['divergence'] = divergence
    return df

def macd_cross_signal(df):
    df['macd_cross_up'] = (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))
    return df

def apply_ema_trigger(df, price_col='close'):
    df['ema35'] = df[price_col].ewm(span=35, adjust=False).mean()
    df['ema_touch'] = np.where(
        (df[price_col] <= df['ema35'] * 1.003) & (df[price_col] >= df['ema35'] * 0.997), True, False)
    return df

def calculate_spike_guard(df, window=20):
    df['hl_range'] = df['high'] - df['low']
    df['volatility'] = df['hl_range'].rolling(window=window).std()
    df['spike'] = df['hl_range'] > df['volatility'] * 3
    return df

def calculate_trend_confirm(df, price_col='close'):
    ema_fast = df[price_col].ewm(span=10, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=35, adjust=False).mean()
    df['ema_fast'] = ema_fast
    df['ema_slow'] = ema_slow
    df['trend_confirm'] = np.where(
        ema_fast > ema_slow, 'up',
        np.where(ema_fast < ema_slow, 'down', 'flat')
    )
    return df

def label_wave_phase(df):
    divergence = df.get('divergence', pd.Series(None, index=df.index))
    rsi = df.get('RSI', pd.Series(50, index=df.index))
    pattern = df.get('Pattern_Label', pd.Series('', index=df.index))

    phase = []
    for div, r, pat in zip(divergence, rsi, pattern):
        if div == 'bearish' and r > 55 and pat == 'Breakout':
            phase.append('W.5')
        elif div == 'bullish' and r < 45 and pat == 'Reversal':
            phase.append('W.B')
        else:
            phase.append(None)
    df['Wave_Phase'] = phase
    return df

def detect_elliott_wave_phase(df, price_col="close", rsi_col="RSI", divergence_col="divergence"):
    df = df.copy()
    df["Wave_Phase"] = None

    df["zz_high"] = (df[price_col] > df[price_col].shift(1)) & (df[price_col] > df[price_col].shift(-1))
    df["zz_low"] = (df[price_col] < df[price_col].shift(1)) & (df[price_col] < df[price_col].shift(-1))

    wave_counter = 1
    for i in range(2, len(df)-2):
        row = df.iloc[i]
        if row["zz_low"] and row[rsi_col] < 45 and row[divergence_col] == "bullish":
            df.at[df.index[i], "Wave_Phase"] = f"W.{wave_counter}"
            wave_counter += 1
        elif row["zz_high"] and row[rsi_col] > 55 and row[divergence_col] == "bearish":
            df.at[df.index[i], "Wave_Phase"] = f"W.{wave_counter}"
            wave_counter += 1

        if wave_counter > 5:
            wave_counter = 1

    return df

def validate_divergence(df, hist_threshold=0.03):
    df['hist_strength'] = df['macd_hist'].diff().abs()
    df['valid_divergence'] = np.where(
        ((df['divergence'] == 'bullish') & (df['macd_hist'] > hist_threshold)) |
        ((df['divergence'] == 'bearish') & (df['macd_hist'] < -hist_threshold)),
        df['divergence'], None)
    return df

def generate_entry_signal(df, gain_z_thresh=0.3, rsi_thresh=50):
    gain_z = df.get('Gain_Z', pd.Series(0, index=df.index))
    rsi = df.get('RSI', pd.Series(50, index=df.index))
    pattern = df.get('Pattern_Label', pd.Series('', index=df.index))

    df['Signal_Score'] = 0
    df['Signal_Score'] += np.where(gain_z > gain_z_thresh, 1, 0)
    df['Signal_Score'] -= np.where(gain_z < -gain_z_thresh, 1, 0)
    df['Signal_Score'] += np.where((rsi > rsi_thresh) & (gain_z > 0), 1, 0)
    df['Signal_Score'] -= np.where((rsi < rsi_thresh) & (gain_z < 0), 1, 0)
    df['Signal_Score'] += np.where(pattern.isin(['Breakout', 'StrongTrend']), 1, 0)

    df['entry_signal'] = np.where(
        df['Signal_Score'] >= 2, 'buy',
        np.where(df['Signal_Score'] <= -2, 'sell', None)
    )

    hybrid_buy = (
        (df.get('divergence') == 'bullish') &
        (rsi > 52) &
        pattern.isin(['Breakout', 'Reversal']) &
        df.get('ema_touch', False) &
        (df.get('trend_confirm') == 'up')
    )
    hybrid_sell = (
        (df.get('divergence') == 'bearish') &
        (rsi < 48) &
        pattern.isin(['Breakout', 'Reversal']) &
        df.get('ema_touch', False) &
        (df.get('trend_confirm') == 'down')
    )

    df['entry_signal'] = np.where(hybrid_buy, 'buy',
        np.where(hybrid_sell, 'sell', df['entry_signal']))
    return df

def generate_entry_signal_wave_enhanced(df, rsi_buy=52, rsi_sell=48, ema_col="ema35"):
    df = df.copy()
    df["entry_signal"] = None

    buy_cond = (
        (df["Wave_Phase"].isin(["W.2", "W.3", "W.5", "W.B"])) &
        (df["divergence"] == "bullish") &
        (df["RSI"] > rsi_buy) &
        (df["close"] >= df[ema_col] * 0.995) & (df["close"] <= df[ema_col] * 1.005)
    )

    sell_cond = (
        (df["Wave_Phase"].isin(["W.2", "W.3", "W.5", "W.B"])) &
        (df["divergence"] == "bearish") &
        (df["RSI"] < rsi_sell) &
        (df["close"] >= df[ema_col] * 0.995) & (df["close"] <= df[ema_col] * 1.005)
    )

    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"
    return df

def should_force_entry(row, last_entry_time, current_time, cooldown=240):
    if row['entry_signal'] is not None:
        return False
    if (current_time - last_entry_time).total_seconds() / 60 < cooldown:
        return False
    if row.get('spike_score', 0) > 0.6 and abs(row.get('Gain_Z', 0)) > 0.5 and row.get('Pattern_Label', '') in ['Breakout', 'StrongTrend']:
        return True
    return False

# === Apply Strategy ===
if __name__ == "__main__":
    # === โหลดและแปลงพ.ศ.เป็นค.ศ. ===
    df = pd.read_csv("XAUUSD_M1.csv")
    df.columns = [col.lower() for col in df.columns]

    year = df['date'].astype(str).str.slice(0, 4).astype(int) - 543
    month = df['date'].astype(str).str.slice(4, 6).astype(int)
    day = df['date'].astype(str).str.slice(6, 8).astype(int)
    time = df['timestamp']

    df['timestamp'] = pd.to_datetime(
        year.astype(str) + '-' + month.astype(str).str.zfill(2) + '-' + day.astype(str).str.zfill(2) + ' ' + time,
        format='%Y-%m-%d %H:%M:%S'
    )

    # === สร้าง close, high, low เป็น lowercase ===
    df.rename(columns={'close': 'close', 'high': 'high', 'low': 'low'}, inplace=True)

    df = calculate_macd(df)
    df = detect_macd_divergence(df)
    df = macd_cross_signal(df)
    df = apply_ema_trigger(df)
    df = calculate_spike_guard(df)
    df = validate_divergence(df)
    df = calculate_trend_confirm(df)
    df = label_wave_phase(df)

    fold_param = {
        0: {'gain_z_thresh': 0.3, 'rsi_thresh': 50},
        1: {'gain_z_thresh': 0.25, 'rsi_thresh': 48},
    }
    current_fold = 0
    param = fold_param.get(current_fold, {})
    gain_z_th = param.get('gain_z_thresh', 0.3)
    rsi_th = param.get('rsi_thresh', 50)
    df = generate_entry_signal(df, gain_z_thresh=gain_z_th, rsi_thresh=rsi_th)

# === Backtest สมจริง: ถือไม้เดียว, TP:SL = 2:1 ===
    initial_capital = 100.0
    capital = initial_capital
    risk_per_trade = 0.01
    tp_multiplier = 2.0
    sl_multiplier = 1.0
    pip_size = 0.1
    spread = 0.80  # 80 points = 0.80 USD (broker 3-digit)
    slippage = 0.05
    commission_per_lot = 0.10
    lot_unit = 0.01

    kill_switch_threshold = 50.0
    kill_switch_triggered = False
    cooldown_bars = 1
    last_entry_idx = -cooldown_bars

    consecutive_loss = 0
    recovery_multiplier = 0.5
    base_lot = 0.1
    prev_trade_result = None
    last_exit_idx = -cooldown_bars

    position = None
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        if i % 5000 == 0:
            position = None

        if row.get('spike_score', 0) > 0.75:
            continue

        if position is None:
            allow_reentry = True
            if prev_trade_result == 'SL' and (i - last_exit_idx) < cooldown_bars:
                allow_reentry = False
            if row['entry_signal'] in ['buy', 'sell'] and not row['spike'] and i - last_entry_idx > cooldown_bars and allow_reentry:
                if row['entry_signal'] == 'buy':
                    entry_price = row['close'] + spread + slippage
                    sl = entry_price - pip_size * sl_multiplier
                    tp = entry_price + pip_size * tp_multiplier
                    position_type = 'long'
                else:
                    entry_price = row['close'] - spread - slippage
                    sl = entry_price + pip_size * sl_multiplier
                    tp = entry_price - pip_size * tp_multiplier
                    position_type = 'short'

                risk_amount = capital * risk_per_trade
                distance = abs(entry_price - sl)
                if distance < 0.05:
                    continue
                lot_size = min(risk_amount / distance, 0.3)
                if consecutive_loss >= 2:
                    lot_size *= recovery_multiplier

                position = {
                    'type': position_type,
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'time': row['timestamp'],
                    'raw_entry': row['close'],
                    'lot_size': lot_size,
                    'tp1_hit': False,
                    'size': 1.0,
                }
                last_entry_idx = i
            elif should_force_entry(row, df.iloc[last_entry_idx]['timestamp'] if last_entry_idx >= 0 else row['timestamp'], row['timestamp']):
                entry_signal = 'buy' if row.get('Gain_Z', 0) > 0 else 'sell'
                if entry_signal == 'buy':
                    entry_price = row['close'] + spread + slippage
                    sl = entry_price - pip_size * sl_multiplier
                    tp = entry_price + pip_size * tp_multiplier
                    position_type = 'long'
                else:
                    entry_price = row['close'] - spread - slippage
                    sl = entry_price + pip_size * sl_multiplier
                    tp = entry_price - pip_size * tp_multiplier
                    position_type = 'short'

                risk_amount = capital * risk_per_trade
                distance = abs(entry_price - sl)
                if distance < 0.05:
                    continue
                lot_size = min(risk_amount / distance, 0.3)
                if consecutive_loss >= 2:
                    lot_size *= recovery_multiplier

                position = {
                    'type': position_type,
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'time': row['timestamp'],
                    'raw_entry': row['close'],
                    'lot_size': lot_size,
                    'tp1_hit': False,
                    'size': 1.0,
                }
                last_entry_idx = i
        else:
            if position['type'] == 'long':
                commission = min((position['lot_size'] / lot_unit) * commission_per_lot, capital * 0.05)
                if not position['tp1_hit'] and row['high'] >= position['entry'] + pip_size * 0.8:
                    pnl = capital * risk_per_trade * 0.5
                    pnl -= commission
                    capital += pnl
                    position['sl'] = position['entry']
                    position['tp1_hit'] = True
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['entry'] + pip_size * 0.8,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'TP1',
                    })
                if row['high'] >= position['tp'] - pip_size * 0.5:
                    position['sl'] = max(position['sl'], row['close'] - pip_size * 0.5)
                if row['low'] <= position['sl']:
                    pnl = -capital * risk_per_trade
                    pnl -= commission
                    capital += pnl
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['sl'],
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'SL',
                    })
                    consecutive_loss += 1
                    prev_trade_result = 'SL'
                    last_exit_idx = i
                    position = None
                elif row['high'] >= position['tp']:
                    pnl = capital * risk_per_trade * (0.5 if position['tp1_hit'] else 1.0)
                    pnl -= commission
                    capital += pnl
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['tp'],
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'TP2' if position['tp1_hit'] else 'TP',
                    })
                    consecutive_loss = 0
                    prev_trade_result = 'TP'
                    last_exit_idx = i
                    position = None
            elif position['type'] == 'short':
                commission = min((position['lot_size'] / lot_unit) * commission_per_lot, capital * 0.05)
                if not position['tp1_hit'] and row['low'] <= position['entry'] - pip_size * 0.8:
                    pnl = capital * risk_per_trade * 0.5
                    pnl -= commission
                    capital += pnl
                    position['sl'] = position['entry']
                    position['tp1_hit'] = True
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['entry'] - pip_size * 0.8,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'TP1',
                    })
                if row['low'] <= position['tp'] + pip_size * 0.5:
                    position['sl'] = min(position['sl'], row['close'] + pip_size * 0.5)
                if row['high'] >= position['sl']:
                    pnl = -capital * risk_per_trade
                    pnl -= commission
                    capital += pnl
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['sl'],
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'SL',
                    })
                    consecutive_loss += 1
                    prev_trade_result = 'SL'
                    last_exit_idx = i
                    position = None
                elif row['low'] <= position['tp']:
                    pnl = capital * risk_per_trade * (0.5 if position['tp1_hit'] else 1.0)
                    pnl -= commission
                    capital += pnl
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['tp'],
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'TP2' if position['tp1_hit'] else 'TP',
                    })
                    consecutive_loss = 0
                    prev_trade_result = 'TP'
                    last_exit_idx = i
                    position = None

        if capital < kill_switch_threshold:
            kill_switch_triggered = True
            break

# === สรุปผล ===
    df_trades = pd.DataFrame(trades)
    print("Final Equity:", round(capital, 2))
    print("Total Return: {:.2%}".format((capital - initial_capital) / initial_capital))
    print("Winrate: {:.2%}".format((df_trades['pnl'] > 0).mean()))
    print(df_trades.tail(10))
    if kill_switch_triggered:
        print("Kill switch activated: capital below threshold")
