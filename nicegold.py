import pandas as pd
import numpy as np

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
        (df[price_col] <= df['ema35'] * 1.002) & (df[price_col] >= df['ema35'] * 0.998), True, False)
    return df

def generate_entry_signal(df):
    df['entry_signal'] = np.where(
        (df['divergence'] == 'bullish') & df['macd_cross_up'] & df['ema_touch'], 'buy',
        np.where((df['divergence'] == 'bearish') & df['macd_cross_down'] & df['ema_touch'], 'sell', None)
    )
    return df

# === Apply Strategy ===
df = calculate_macd(df)
df = detect_macd_divergence(df)
df = macd_cross_signal(df)
df = apply_ema_trigger(df)
df = generate_entry_signal(df)

# === Backtest สมจริง: ถือไม้เดียว, TP:SL = 1.5:1 ===
initial_capital = 100.0
capital = initial_capital
risk_per_trade = 0.01
tp_multiplier = 1.5
sl_multiplier = 1.0
pip_size = 1.0
spread = 0.80  # 80 points = 0.80 USD (broker 3-digit)
slippage = 0.05
commission_per_lot = 0.10
lot_unit = 0.01

position = None
trades = []

for i in range(1, len(df)):
    row = df.iloc[i]
    if position is None:
        if row['entry_signal'] == 'buy':
            entry_price = row['close'] + spread + slippage
            sl = entry_price - pip_size * sl_multiplier
            tp = entry_price + pip_size * tp_multiplier
            position = {
                'type': 'long',
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'time': row['timestamp'],
                'raw_entry': row['close'],
            }
        elif row['entry_signal'] == 'sell':
            entry_price = row['close'] - spread - slippage
            sl = entry_price + pip_size * sl_multiplier
            tp = entry_price - pip_size * tp_multiplier
            position = {
                'type': 'short',
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'time': row['timestamp'],
                'raw_entry': row['close'],
            }
    else:
        if position['type'] == 'long':
            lot_size = (capital * risk_per_trade) / abs(position['entry'] - position['sl'])
            commission = (lot_size / lot_unit) * commission_per_lot
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
                position = None
            elif row['high'] >= position['tp']:
                pnl = capital * risk_per_trade * tp_multiplier
                pnl -= commission
                capital += pnl
                trades.append({
                    **position,
                    'exit_time': row['timestamp'],
                    'exit_price': position['tp'],
                    'pnl': pnl,
                    'commission': commission,
                    'capital': capital,
                    'exit': 'TP',
                })
                position = None
        elif position['type'] == 'short':
            lot_size = (capital * risk_per_trade) / abs(position['entry'] - position['sl'])
            commission = (lot_size / lot_unit) * commission_per_lot
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
                position = None
            elif row['low'] <= position['tp']:
                pnl = capital * risk_per_trade * tp_multiplier
                pnl -= commission
                capital += pnl
                trades.append({
                    **position,
                    'exit_time': row['timestamp'],
                    'exit_price': position['tp'],
                    'pnl': pnl,
                    'commission': commission,
                    'capital': capital,
                    'exit': 'TP',
                })
                position = None

# === สรุปผล ===
df_trades = pd.DataFrame(trades)
print("Final Equity:", round(capital, 2))
print("Total Return: {:.2%}".format((capital - initial_capital) / initial_capital))
print("Winrate: {:.2%}".format((df_trades['pnl'] > 0).mean()))
print(df_trades.tail(10))
