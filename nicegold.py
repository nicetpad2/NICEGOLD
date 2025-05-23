import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - optional dependency
    GradientBoostingClassifier = None
    StandardScaler = None
    train_test_split = None

CONFIG_PATH = "config.yaml"

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)

# === Default Parameters (Updated) ===
initial_capital = 100.0
risk_per_trade = 0.05  # ความเสี่ยงต่อไม้ 5% ของทุน (เพิ่มจาก 1% เดิมเพื่อเร่งการเติบโต) [Patch]
max_drawdown_pct = 0.30
partial_tp_ratio = 0.5
cooldown_minutes = 180
entry_signal_threshold = 2
def get_logger():
    return logger


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """ลดการใช้หน่วยความจำของ DataFrame"""
    logger.debug("Optimizing memory usage")
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
    return df



def load_config(path: str = CONFIG_PATH):
    """Load configuration from YAML file"""
    logger.debug("Loading config from %s", path)
    if yaml is None:
        raise ImportError("yaml is required for load_config")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug("Config keys: %s", list(config.keys()))
    return config

# === โหลดข้อมูล ===
def load_data(file_path=None):
    """อ่านไฟล์ CSV และแปลงคอลัมน์วันที่จากพ.ศ.เป็นค.ศ."""
    if file_path is None:
        file_path = "XAUUSD_M1.csv"
    logger.debug("Loading data from %s", file_path)
    df = pd.read_csv(file_path, dtype=str)
    df.columns = [c.lower() for c in df.columns]
    if 'date' not in df.columns or 'timestamp' not in df.columns:
        return optimize_memory(df)

    date_str = df['date'].str.zfill(8)
    year = date_str.str[:4].astype(int) - 543
    ts = year.astype(str) + '-' + date_str.str[4:6] + '-' + date_str.str[6:8]
    df['timestamp'] = pd.to_datetime(ts + ' ' + df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df.drop(columns=['date'], inplace=True)
    df = optimize_memory(df)
    logger.debug("Loaded %d rows", len(df))
    return df

# === แปลง CSV จาก ค.ศ. เป็น พ.ศ. ===
def convert_csv_ad_to_be(input_csv, output_csv=None):
    """แปลงคอลัมน์ Date ในไฟล์ CSV จาก ค.ศ. เป็น พ.ศ."""
    logger.debug("Converting AD dates to BE from %s", input_csv)
    df = pd.read_csv(input_csv)
    df.columns = [c.lower() for c in df.columns]
    if 'date' not in df.columns:
        logger.debug("Date column missing")
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df

    year = df['date'].astype(str).str.slice(0, 4).astype(int) + 543
    month = df['date'].astype(str).str.slice(4, 6).astype(int)
    day = df['date'].astype(str).str.slice(6, 8).astype(int)

    df['date'] = (
        year.astype(str)
        + month.astype(str).str.zfill(2)
        + day.astype(str).str.zfill(2)
    )
    logger.debug("Converted %d rows", len(df))
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.debug("Saved converted CSV to %s", output_csv)
    return df

# === คำนวณ Indicator ===
def calculate_macd(df, price_col='close', fast=12, slow=26, signal=9):
    logger.debug("Calculating MACD")
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df['macd'] = macd
    df['signal'] = signal_line
    df['macd_hist'] = macd - signal_line
    logger.debug("MACD columns added")
    return df

def detect_macd_divergence(df, price_col='close'):
    logger.debug("Detecting MACD divergence")
    prev_price = df[price_col].shift(2)
    prev_hist = df['macd_hist'].shift(2)
    bull = (df[price_col] < prev_price) & (df['macd_hist'] > prev_hist)
    bear = (df[price_col] > prev_price) & (df['macd_hist'] < prev_hist)
    df['divergence'] = np.where(bull, 'bullish', np.where(bear, 'bearish', None))
    logger.debug("Divergence column added")
    return df

def macd_cross_signal(df):
    logger.debug("Computing MACD cross signals")
    df['macd_cross_up'] = (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))
    return df

def apply_ema_trigger(df, price_col='close'):
    logger.debug("Applying EMA trigger")
    df['ema35'] = df[price_col].ewm(span=35, adjust=False).mean()
    df['ema_touch'] = np.where(
        (df[price_col] <= df['ema35'] * 1.003) & (df[price_col] >= df['ema35'] * 0.997), True, False)
    return df

def calculate_spike_guard(df, window=20):
    logger.debug("Calculating spike guard")
    df['hl_range'] = df['high'] - df['low']
    df['volatility'] = df['hl_range'].rolling(window=window).std()
    df['spike'] = df['hl_range'] > df['volatility'] * 3
    df['spike_score'] = df['hl_range'] / (df['volatility'] * 3)
    logger.debug("Spike guard columns added")
    return df

def calculate_trend_confirm(df, price_col='close'):
    logger.debug("Calculating trend confirmation")
    ema_fast = df[price_col].ewm(span=10, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=35, adjust=False).mean()
    df['ema_fast'] = ema_fast
    df['ema_slow'] = ema_slow
    df['trend_confirm'] = np.where(
        ema_fast > ema_slow, 'up',
        np.where(ema_fast < ema_slow, 'down', 'flat')
    )
    logger.debug("Trend confirm column added")
    return df

def label_wave_phase(df):
    logger.debug("Labeling wave phase")
    divergence = df.get('divergence', pd.Series(None, index=df.index, dtype=object))
    rsi = df.get('RSI', pd.Series(50, index=df.index, dtype=float))
    pattern = df.get('Pattern_Label', pd.Series('', index=df.index, dtype=object))

    phase = []
    for div, r, pat in zip(divergence, rsi, pattern):
        if div == 'bearish' and r > 55 and pat == 'Breakout':
            phase.append('W.5')
        elif div == 'bullish' and r < 45 and pat == 'Reversal':
            phase.append('W.B')
        else:
            phase.append(None)
    df['Wave_Phase'] = phase
    logger.debug("Wave phase column added")
    return df

def detect_elliott_wave_phase(df, price_col="close", rsi_col="RSI", divergence_col="divergence"):
    logger.debug("Detecting Elliott wave phase")
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

    logger.debug("Elliott wave phases labeled")
    return df

def validate_divergence(df, hist_threshold=0.03):
    logger.debug("Validating divergence")
    df['hist_strength'] = df['macd_hist'].diff().abs()
    df['valid_divergence'] = np.where(
        ((df['divergence'] == 'bullish') & (df['macd_hist'] > hist_threshold)) |
        ((df['divergence'] == 'bearish') & (df['macd_hist'] < -hist_threshold)),
        df['divergence'], None)
    logger.debug("Valid divergence column added")
    return df

def generate_entry_signal(df, gain_z_thresh=0.3, rsi_thresh=50):
    logger.debug("Generating entry signal")
    gain_z = df.get('Gain_Z', pd.Series(0, index=df.index))
    rsi = df.get('RSI', pd.Series(50, index=df.index, dtype=float))
    pattern = df.get('Pattern_Label', pd.Series('', index=df.index, dtype=object))

    df['Signal_Score'] = 0
    df['Signal_Score'] += np.where(gain_z > gain_z_thresh, 1, 0)
    df['Signal_Score'] -= np.where(gain_z < -gain_z_thresh, 1, 0)
    df['Signal_Score'] += np.where((rsi > rsi_thresh) & (gain_z > 0), 1, 0)
    df['Signal_Score'] -= np.where((rsi < rsi_thresh) & (gain_z < 0), 1, 0)
    df['Signal_Score'] += np.where(pattern.isin(['Breakout', 'StrongTrend']), 1, 0)
    df['Signal_Score'] += np.where(pattern == 'Reversal', 1, 0)

    df['entry_signal'] = np.where(
        df['Signal_Score'] >= 2, 'buy',
        np.where(df['Signal_Score'] <= -2, 'sell', None)
    )

    hybrid_buy = (
        (df.get('divergence') == 'bullish') &
        (rsi > 52) &
        df.get('ema_touch', False) &
        (df.get('trend_confirm') == 'up')
    )
    hybrid_sell = (
        (df.get('divergence') == 'bearish') &
        (rsi < 48) &
        df.get('ema_touch', False) &
        (df.get('trend_confirm') == 'down')
    )

    df['entry_signal'] = np.where(hybrid_buy, 'buy',
        np.where(hybrid_sell, 'sell', df['entry_signal']))
    logger.debug("Entry signal column added")
    return df

def generate_entry_signal_wave_enhanced(df, rsi_buy=52, rsi_sell=48, ema_col="ema35"):
    logger.debug("Generating entry signal (wave enhanced)")
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
    logger.debug("Entry signal (wave enhanced) column added")
    return df

def generate_entry_score_signal(df, ema_col="ema35", rsi_threshold=50):
    logger.debug("Generating entry score signal")
    df = df.copy()

    df["entry_score"] = 0
    df["entry_score"] += df["Wave_Phase"].isin(["W.2", "W.3", "W.5", "W.B"]).astype(int)
    df["entry_score"] += df["divergence"].isin(["bullish", "bearish"]).astype(int)

    rsi_buy_cond = (df["divergence"] == "bullish") & (df["RSI"] > rsi_threshold)
    rsi_sell_cond = (df["divergence"] == "bearish") & (df["RSI"] < rsi_threshold)
    df["entry_score"] += (rsi_buy_cond | rsi_sell_cond).astype(int)

    ema_prox = (df["close"] >= df[ema_col] * 0.995) & (df["close"] <= df[ema_col] * 1.005)
    df["entry_score"] += ema_prox.astype(int)

    df["entry_signal"] = None
    buy_cond = (df["entry_score"] >= 2) & (df["divergence"] == "bullish") & (df["RSI"] > rsi_threshold)
    sell_cond = (df["entry_score"] >= 2) & (df["divergence"] == "bearish") & (df["RSI"] < rsi_threshold)

    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"
    logger.debug("Entry score and signal columns added")
    return df

def apply_wave_macd_cross_entry(df, ema_col="ema35"):
    logger.debug("Applying wave MACD cross entry")
    df = df.copy()
    for i in range(2, len(df)):
        if pd.isna(df.at[df.index[i], 'entry_signal']):
            if (
                df.at[df.index[i], 'Wave_Phase'] in ['W.2', 'W.3', 'W.5', 'W.B'] and
                df.at[df.index[i], 'divergence'] == 'bullish' and
                df.at[df.index[i], 'macd_cross_up'] and
                df.at[df.index[i], 'RSI'] > 45 and
                df.at[df.index[i], 'close'] >= df.at[df.index[i], ema_col] * 0.995 and
                df.at[df.index[i], 'close'] <= df.at[df.index[i], ema_col] * 1.005
            ):
                df.at[df.index[i], 'entry_signal'] = 'buy'
            elif (
                df.at[df.index[i], 'Wave_Phase'] in ['W.2', 'W.3', 'W.5', 'W.B'] and
                df.at[df.index[i], 'divergence'] == 'bearish' and
                df.at[df.index[i], 'macd_cross_down'] and
                df.at[df.index[i], 'RSI'] < 55 and
                df.at[df.index[i], 'close'] >= df.at[df.index[i], ema_col] * 0.995 and
                df.at[df.index[i], 'close'] <= df.at[df.index[i], ema_col] * 1.005
            ):
                df.at[df.index[i], 'entry_signal'] = 'sell'
    return df

def should_force_entry(row, last_entry_time, current_time, cooldown=180):
    logger.debug("Checking force entry conditions")
    if row['entry_signal'] is not None:
        return False
    if (current_time - last_entry_time).total_seconds() / 60 < cooldown:
        return False
    if row.get('spike_score', 0) > 0.6 and abs(row.get('Gain_Z', 0)) > 0.5:
        return True
    return False

# === Modern Scalping Strategy ===
def compute_features(df):
    """คำนวณตัวชี้วัดสำหรับกลยุทธ์ ModernScalping"""
    logger.debug("Computing features for ML model")
    df = df.copy()
    df['ret1'] = df['close'].pct_change(fill_method=None)
    df['ret5'] = df['close'].pct_change(5, fill_method=None)
    df['ret10'] = df['close'].pct_change(10, fill_method=None)
    df['rsi'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100 / (1 + np.mean(np.clip(np.diff(x), 0, None)) / (1e-6 + np.mean(np.clip(-np.diff(x), 0, None)))),
        raw=False,
    )
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['trend'] = (df['ma5'] > df['ma20']).astype(int)
    df.dropna(inplace=True)
    logger.debug("Feature columns added")
    return df


def train_signal_model(df):
    """ฝึกโมเดล Machine Learning เพื่อสร้างสัญญาณ พร้อม debug log"""
    logger.debug("Training ML signal model")
    if GradientBoostingClassifier is None:
        raise ImportError('scikit-learn is required for train_signal_model')
    df = df.copy()
    df['future_ret'] = df['close'].shift(-5).pct_change(periods=5, fill_method=None)
    df['target'] = (df['future_ret'] > 0.0015).astype(int)
    features = ['ret1', 'ret5', 'ret10', 'rsi', 'atr', 'trend']
    X = df[features]
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.2)
    scaler = StandardScaler().fit(X_train)
    clf = GradientBoostingClassifier(n_estimators=50).fit(scaler.transform(X_train), y_train)
    df['signal_prob'] = clf.predict_proba(scaler.transform(df[features]))[:, 1]

    df['entry_signal'] = np.where(
        (df['signal_prob'] > 0.55) &
        (df['atr'] > df['atr'].rolling(50).mean()),
        'buy', None
    )

    logger.debug("ENTRY_SIGNAL COUNTS:\n%s", df['entry_signal'].value_counts(dropna=False))
    logger.debug("Mean Signal Prob: %s", round(df['signal_prob'].mean(), 4))
    logger.debug("Mean ATR: %s", round(df['atr'].mean(), 5))

    logger.debug("Signal probabilities and entry signals generated")
    return df


def run_backtest(df, cfg):
    """รันแบ็กเทสต์แบบง่ายสำหรับกลยุทธ์ ModernScalping"""
    logger.debug("Starting backtest on %d rows", len(df))
    capital = cfg.get('initial_capital', 100.0)
    peak_equity = capital
    max_drawdown = 0.0
    trades = []
    equity_curve = []
    position = None
    for i in range(20, len(df)):
        row = df.iloc[i]
        equity_curve.append({'timestamp': row['timestamp'], 'equity': capital})
        if not (cfg.get('trade_start_hour', 0) <= row.get('hour', 0) <= cfg.get('trade_end_hour', 23)):
            continue
        if position is None and row['entry_signal'] == 'buy':
            logger.debug("Open long position at %s", row['timestamp'])
            entry_price = row['close'] + 0.10
            sl = entry_price - row['atr']
            tp1 = entry_price + row['atr'] * cfg.get('tp1_mult', 0.8)
            tp2 = entry_price + row['atr'] * cfg.get('tp2_mult', 2.0)
            lot = min(0.01, capital * cfg.get('risk_per_trade', 0.05) / max(entry_price - sl, 1e-6))
            position = {
                'entry_time': row['timestamp'],
                'entry': entry_price,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'lot': lot,
                'tp1_hit': False,
                'capital_before': capital,
            }
        elif position:
            if not position['tp1_hit'] and row['high'] >= position['tp1']:
                pnl = position['lot'] * (position['tp1'] - position['entry']) * 100
                capital += pnl
                position['tp1_hit'] = True
                position['sl'] = position['entry']
                trades.append({**position, 'exit': 'TP1', 'pnl': pnl, 'capital_after': capital, 'exit_time': row['timestamp']})
            elif row['low'] <= position['sl']:
                pnl = -position['lot'] * (position['entry'] - position['sl']) * 100
                capital += pnl
                trades.append({**position, 'exit': 'SL', 'pnl': pnl, 'capital_after': capital, 'exit_time': row['timestamp']})
                position = None
            elif row['high'] >= position['tp2']:
                pnl = position['lot'] * (position['tp2'] - position['entry']) * 100
                capital += pnl
                trades.append({**position, 'exit': 'TP2', 'pnl': pnl, 'capital_after': capital, 'exit_time': row['timestamp']})
                logger.debug("Exit position at %s with TP2", row['timestamp'])
                position = None

            drawdown = (peak_equity - capital) / peak_equity
            if capital > peak_equity:
                peak_equity = capital
            else:
                max_drawdown = max(max_drawdown, drawdown)

            if capital < cfg.get('kill_switch_min', 70):
                logger.debug("Kill switch triggered")
                break

    df_trades = pd.DataFrame(trades)
    df_trades.to_csv('trade_log.csv', index=False)
    df_equity = pd.DataFrame(equity_curve)
    df_equity.to_csv('equity_curve.csv', index=False)
    logger.debug('Final Equity: %s', round(capital, 2))
    logger.debug('Total Return: %s', capital - cfg.get('initial_capital', 100.0))
    logger.debug('Total Trades: %s', len(df_trades))
    if 'pnl' in df_trades.columns and not df_trades.empty:
        logger.debug('Win Rate: %s', (df_trades['pnl'] > 0).mean())
    else:
        logger.debug('Win Rate: N/A (no trades)')
    logger.debug('Max Drawdown: %s', round(max_drawdown * 100, 2))
    plot_trades(df, df_trades)
    logger.debug("Backtest finished")
    return df_trades


def plot_trades(df, trades):
    """Save a trade visualization plot if matplotlib is available."""
    if plt is None:
        return
    logger.debug("Plotting trades")
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], df['close'], label='Price', alpha=0.6)
    for _, t in trades.iterrows():
        if t['exit'] == 'SL':
            plt.axvline(t['entry_time'], color='red', alpha=0.2)
        elif t['exit'] == 'TP2':
            plt.axvline(t['entry_time'], color='green', alpha=0.2)
        else:
            plt.axvline(t['entry_time'], color='orange', alpha=0.2)
    plt.title('Entry Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trade_plot.png')
    plt.close()
    logger.debug("Trade plot saved")


def walk_forward_test(df, cfg, fold_days=60):
    """Execute walk-forward validation across multiple folds."""
    logger.debug("Starting walk-forward test")
    step = int(fold_days * 24 * 60)
    total = len(df)
    fold = 0
    results = []
    for start in range(0, total - step, step):
        end = start + step
        fold_df = df.iloc[start:end].copy()
        fold_df = compute_features(fold_df)
        fold_df = train_signal_model(fold_df)
        logger.debug("=== Fold %d ===", fold + 1)
        df_trades = run_backtest(fold_df, cfg)
        results.append(df_trades)
        fold += 1
    logger.debug("Walk-forward test completed")
    return pd.concat(results) if results else pd.DataFrame()

# === Apply Strategy ===
def run_backtest_cli():  # pragma: no cover
    """Execute the realistic backtest when run as a script."""
    logger.debug("Running backtest CLI")
    # === โหลดและแปลงพ.ศ.เป็นค.ศ. ===
    df = pd.read_csv("/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv")
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

    # Calculate technical indicators and signals
    df = calculate_macd(df)
    df = detect_macd_divergence(df)
    df = macd_cross_signal(df)
    df = apply_ema_trigger(df)
    df = calculate_spike_guard(df)
    df = validate_divergence(df)
    df = calculate_trend_confirm(df)
    # Compute RSI (14-period) for momentum indication
    df['RSI'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100 / (1 + (np.mean(np.clip(np.diff(x), 0, None)) /
                                    ((np.mean(np.clip(-np.diff(x), 0, None))) + 1e-6))),
        raw=False
    )
    df['RSI'] = df['RSI'].fillna(50)  # [Patch G-Fix1] fix chained assignment warning
    logger.debug("RSI calculated")
    # Compute short-term momentum Z-score (Gain_Z) over 10-bar returns
    ret10 = df['close'].pct_change(10, fill_method=None)
    df['Gain_Z'] = ((ret10 - ret10.rolling(60).mean()) / (ret10.rolling(60).std() + 1e-6)).fillna(0)
    logger.debug("Gain_Z calculated")
    df['atr'] = (df['high'] - df['low']).rolling(14).mean().bfill()
    logger.debug("ATR calculated")
    # Label pattern signals: Breakout, Reversal, StrongTrend
    df['Pattern_Label'] = ''
    df.loc[(df['divergence'] == 'bearish') & (df['RSI'] > 55), 'Pattern_Label'] = 'Breakout'
    df.loc[(df['divergence'] == 'bullish') & (df['RSI'] < 45), 'Pattern_Label'] = 'Reversal'
    df.loc[(df['Pattern_Label'] == '') & (df['trend_confirm'] == 'up') & (df['RSI'] > 60), 'Pattern_Label'] = 'StrongTrend'
    df.loc[(df['Pattern_Label'] == '') & (df['trend_confirm'] == 'down') & (df['RSI'] < 40), 'Pattern_Label'] = 'StrongTrend'
    logger.debug("Pattern labels assigned")
    df = label_wave_phase(df)

    fold_param = {
        0: {'gain_z_thresh': 0.3, 'rsi_thresh': 50},
        1: {'gain_z_thresh': 0.25, 'rsi_thresh': 48},
    }
    current_fold = 1
    param = fold_param.get(current_fold, {})
    gain_z_th = param.get('gain_z_thresh', 0.3)
    rsi_th = param.get('rsi_thresh', 50)
    df = generate_entry_signal(df, gain_z_thresh=gain_z_th, rsi_thresh=rsi_th)
    df = apply_wave_macd_cross_entry(df)
    logger.debug(df[['timestamp', 'entry_signal', 'Wave_Phase', 'RSI', 'divergence']].tail(30))  # [Patch G-Fix1] signal debug tail

# === Backtest ปรับปรุง: ถือไม้เดียว, TP:SL = 2:1 (ใช้ ATR) ===
    initial_capital = 100.0
    capital = initial_capital
    risk_per_trade = 0.05  # เพิ่มความเสี่ยงต่อไม้เป็น 5% [Patch]
    tp_multiplier = 2.0    # TP สุดท้าย 2 ATR จากจุดเข้า (2R) [Patch]
    sl_multiplier = 1.0    # SL ที่ 1 ATR จากจุดเข้า (1R) [Patch]
    pip_size = 0.1         # 1 pip = $0.1 (ใช้สำหรับ trailing เล็กน้อย)
    spread = 0.50          # ลดสเปรดสมมุติลงให้สมเหตุสมผลกับตลาดจริง (~$0.5) [Patch]
    slippage = 0.05        # ค่า slippage คงที่
    commission_per_lot = 0.10
    lot_unit = 0.01
    trailing_atr_multiplier = 1.0   # ระยะ trailing SL ตาม ATR [Patch]
    drawdown_threshold = 0.20      # หยุดเข้าออเดอร์ใหม่หาก drawdown สูงเกิน 20% [Patch]
    extreme_vol_factor = 2.0       # ไม่เข้าออเดอร์หาก ATR > 2 เท่าของ ATR เฉลี่ย [Patch]

    kill_switch_threshold = 50.0  # หยุดเทรดทั้งหมดหากทุนลดต่ำกว่า $50 (50% ของเริ่มต้น) [Patch]
    kill_switch_triggered = False
    cooldown_bars = 1  # รออย่างน้อย 1 แท่ง (1 นาที) หลัง SL ก่อนเข้าใหม่ (Cool-down) [Patch]
    last_entry_idx = -cooldown_bars

    consecutive_loss = 0
    consecutive_win = 0
    recovery_multiplier = 0.5
    win_streak_boost = 1.2
    base_lot = 0.1
    prev_trade_result = None
    last_exit_idx = -cooldown_bars

    position = None
    trades = []
    equity_curve = []
    peak_capital = capital  # ติดตามจุดสูงสุดของพอร์ตเพื่อคำนวณ drawdown [Patch]
    atr_rolling_mean = df['atr'].rolling(50).mean().bfill().values  # ATR เฉลี่ย 50 แท่ง สำหรับตรวจสอบ volatility [Patch]

    for i in range(1, len(df)):
        row = df.iloc[i]
        equity_curve.append({'timestamp': row['timestamp'], 'equity': capital})

        if i % 5000 == 0:
            position = None

        if row.get('spike_score', 0) > 0.75:
            continue  # ข้ามแท่งที่ความผันผวนสูงผิดปกติ (anti-spike) [Patch]

        if position is None:
            allow_reentry = True
            if prev_trade_result == 'SL' and (i - last_exit_idx) < cooldown_bars:
                allow_reentry = False
            current_drawdown = (peak_capital - capital) / peak_capital
            if current_drawdown > drawdown_threshold:
                logger.debug(f"Skip entry due to high drawdown: {current_drawdown:.2%}")
                allow_reentry = False
            if df['atr'].iat[i] > atr_rolling_mean[i] * extreme_vol_factor:
                logger.debug(f"Skip entry due to extreme volatility: ATR {df['atr'].iat[i]:.4f}")
                allow_reentry = False
            if row['entry_signal'] in ['buy', 'sell'] and not row['spike'] and (i - last_entry_idx) > cooldown_bars and allow_reentry:
                # เข้าใหม่เฉพาะช่วงเวลาตลาดหลักที่มีสภาพคล่องสูง (13:00-22:00) [Patch]
                hour = row['timestamp'].hour
                if hour < 13 or hour > 22:
                    continue
                if row['entry_signal'] == 'buy':
                    entry_price = row['close'] + spread + slippage
                    sl = entry_price - row['atr'] * sl_multiplier
                    position_type = 'long'
                else:
                    entry_price = row['close'] - spread - slippage
                    sl = entry_price + row['atr'] * sl_multiplier
                    position_type = 'short'

                risk_amount = capital * risk_per_trade
                # คำนวณระยะ SL ตาม ATR (volatility) แทนค่าคงที่
                distance = abs(entry_price - sl)
                if distance < 1e-6:
                    continue
                lot_size = min(risk_amount / distance, 0.3)
                if consecutive_loss >= 2:
                    lot_size *= recovery_multiplier
                if consecutive_win >= 2:
                    lot_size *= win_streak_boost

                factor_by_streak = 1.0
                if consecutive_loss >= 3:
                    factor_by_streak *= 0.75
                if consecutive_win >= 3:
                    factor_by_streak *= 1.25
                tp = entry_price + (row['atr'] * tp_multiplier * factor_by_streak * (1 if position_type == 'long' else -1))

                reason_components = []
                wave_phase = row.get('Wave_Phase', '')
                divergence = row.get('divergence', '')
                if position_type == 'long':
                    if str(wave_phase).startswith('W.') and str(divergence) == 'bullish' and row.get('macd_cross_up', False):
                        reason_components.append(f"WavePhase {wave_phase} bullish divergence")
                        reason_components.append("MACD cross up")
                    else:
                        reason_components.append("Bullish signal triggered")
                else:
                    if str(wave_phase).startswith('W.') and str(divergence) == 'bearish' and row.get('macd_cross_down', False):
                        reason_components.append(f"WavePhase {wave_phase} bearish divergence")
                        reason_components.append("MACD cross down")
                    else:
                        reason_components.append("Bearish signal triggered")
                reason_entry = "; ".join(reason_components)

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
                    'risk_amount': risk_amount,
                    'risk_price': distance,
                    'reason_entry': reason_entry,
                    'risk': round(risk_amount, 2),
                    'reward': round(risk_amount * tp_multiplier * factor_by_streak, 2)
                }
                last_entry_idx = i
                logger.info(f"Opened {position_type} position at {position['time']} - Entry: {position['entry']:.2f}, SL: {position['sl']:.2f}, TP: {position['tp']:.2f}, Lot: {position['lot_size']:.2f} | Reason: {reason_entry}")
            elif should_force_entry(row, df.iloc[last_entry_idx]['timestamp'] if last_entry_idx >= 0 else row['timestamp'], row['timestamp']):
                entry_signal = 'buy' if row.get('Gain_Z', 0) > 0 else 'sell'
                if entry_signal == 'buy':
                    entry_price = row['close'] + spread + slippage
                    sl = entry_price - row['atr'] * sl_multiplier
                    position_type = 'long'
                else:
                    entry_price = row['close'] - spread - slippage
                    sl = entry_price + row['atr'] * sl_multiplier
                    position_type = 'short'

                risk_amount = capital * risk_per_trade
                distance = abs(entry_price - sl)
                if distance < 1e-6:
                    continue
                lot_size = min(risk_amount / distance, 0.3)
                if consecutive_loss >= 2:
                    lot_size *= recovery_multiplier
                if consecutive_win >= 2:
                    lot_size *= win_streak_boost

                factor_by_streak = 1.0
                if consecutive_loss >= 3:
                    factor_by_streak *= 0.75
                if consecutive_win >= 3:
                    factor_by_streak *= 1.25
                tp = entry_price + (row['atr'] * tp_multiplier * factor_by_streak * (1 if position_type == 'long' else -1))

                reason_components = []
                wave_phase = row.get('Wave_Phase', '')
                divergence = row.get('divergence', '')
                if position_type == 'long':
                    if str(wave_phase).startswith('W.') and str(divergence) == 'bullish' and row.get('macd_cross_up', False):
                        reason_components.append(f"WavePhase {wave_phase} bullish divergence")
                        reason_components.append("MACD cross up")
                    else:
                        reason_components.append("Bullish signal triggered")
                else:
                    if str(wave_phase).startswith('W.') and str(divergence) == 'bearish' and row.get('macd_cross_down', False):
                        reason_components.append(f"WavePhase {wave_phase} bearish divergence")
                        reason_components.append("MACD cross down")
                    else:
                        reason_components.append("Bearish signal triggered")
                reason_entry = "; ".join(reason_components)

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
                    'risk_amount': risk_amount,
                    'risk_price': distance,
                    'reason_entry': reason_entry,
                    'risk': round(risk_amount, 2),
                    'reward': round(risk_amount * tp_multiplier * factor_by_streak, 2)
                }
                last_entry_idx = i
                logger.info(f"Opened {position_type} position at {position['time']} - Entry: {position['entry']:.2f}, SL: {position['sl']:.2f}, TP: {position['tp']:.2f}, Lot: {position['lot_size']:.2f} | Reason: {reason_entry}")
        else:
            if position['type'] == 'long':
                commission = (position['lot_size'] / lot_unit) * commission_per_lot  # [Patch G-Fix1] charged once, realistic
                if not position['tp1_hit'] and row['high'] >= position['entry'] + (position['entry'] - position['sl']):
                    pnl = capital * risk_per_trade * 0.5
                    capital += pnl
                    peak_capital = max(peak_capital, capital)
                    position['sl'] = position['entry']
                    position['tp1_hit'] = True
                    logger.info(f"TP1 reached for {position['type']} at {position['tp']:.2f} - Capital: {capital:.2f}")
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['entry'] + pip_size * sl_multiplier,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'TP1',
                    })
                if row['high'] >= position['tp'] - pip_size * 0.5:
                    position['sl'] = max(position['sl'], row['close'] - pip_size * 0.5)
                # Trailing SL หลังจากย้าย SL มาที่ทุน (breakeven)
                if position['tp1_hit']:
                    new_sl = position['entry'] + (row['close'] - position['entry'] - row['atr'] * trailing_atr_multiplier if position['type']=='long' else position['entry'] - (position['entry'] - row['close']) + row['atr'] * trailing_atr_multiplier)
                    if position['type'] == 'long':
                        position['sl'] = max(position['sl'], new_sl)
                    else:
                        position['sl'] = min(position['sl'], new_sl)
                # ตรวจสอบการออกออเดอร์ (SL/TP)
                if row['low'] <= position['sl']:
                    exit_price = position['sl']
                    direction = 1 if position['type'] == 'long' else -1
                    effective_risk_amount = position['risk_amount'] * (0.5 if position.get('tp1_hit') else 1.0)
                    pnl_move = direction * (exit_price - position['entry']) / position['risk_price'] * effective_risk_amount
                    pnl = pnl_move
                    pnl -= commission
                    capital += pnl
                    peak_capital = max(peak_capital, capital)
                    reason_exit = "Stopped out at breakeven" if abs(position['entry'] - exit_price) < 1e-9 else "Stop Loss hit"
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'SL',
                        'reason_exit': reason_exit
                    })
                    consecutive_loss += 1
                    consecutive_win = 0
                    prev_trade_result = 'SL'
                    last_exit_idx = i
                    position = None
                elif row['high'] >= position['tp']:
                    exit_price = position['tp']
                    direction = 1 if position['type'] == 'long' else -1
                    effective_risk_amount = position['risk_amount'] * (0.5 if position.get('tp1_hit') else 1.0)
                    pnl_move = direction * (exit_price - position['entry']) / position['risk_price'] * effective_risk_amount
                    pnl = pnl_move
                    pnl -= commission
                    capital += pnl
                    exit_label = 'TP2' if position['tp1_hit'] else 'TP'
                    reason_exit = "Final Take Profit hit"
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': exit_label,
                        'reason_exit': reason_exit
                    })
                    consecutive_loss = 0
                    consecutive_win += 1
                    prev_trade_result = 'TP'
                    last_exit_idx = i
                    position = None
            elif position['type'] == 'short':
                commission = (position['lot_size'] / lot_unit) * commission_per_lot  # [Patch G-Fix1] charged once at exit only
                if not position['tp1_hit'] and row['low'] <= position['entry'] - (position['sl'] - position['entry']):
                    pnl = capital * risk_per_trade * 0.5
                    capital += pnl
                    peak_capital = max(peak_capital, capital)
                    position['sl'] = position['entry']
                    position['tp1_hit'] = True
                    logger.info(f"TP1 reached for {position['type']} at {position['tp']:.2f} - Capital: {capital:.2f}")
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': position['entry'] - pip_size * sl_multiplier,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'TP1',
                    })
                if row['low'] <= position['tp'] + pip_size * 0.5:
                    position['sl'] = min(position['sl'], row['close'] + pip_size * 0.5)
                # Trailing SL หลังจากย้าย SL มาที่ทุน (breakeven)
                if position['tp1_hit']:
                    new_sl = position['entry'] + (row['close'] - position['entry'] - row['atr'] * trailing_atr_multiplier if position['type']=='long' else position['entry'] - (position['entry'] - row['close']) + row['atr'] * trailing_atr_multiplier)
                    if position['type'] == 'long':
                        position['sl'] = max(position['sl'], new_sl)
                    else:
                        position['sl'] = min(position['sl'], new_sl)
                if row['high'] >= position['sl']:
                    exit_price = position['sl']
                    direction = 1 if position['type'] == 'long' else -1
                    effective_risk_amount = position['risk_amount'] * (0.5 if position.get('tp1_hit') else 1.0)
                    pnl_move = direction * (exit_price - position['entry']) / position['risk_price'] * effective_risk_amount
                    pnl = pnl_move
                    pnl -= commission
                    capital += pnl
                    reason_exit = "Stopped out at breakeven" if abs(position['entry'] - exit_price) < 1e-9 else "Stop Loss hit"
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': 'SL',
                        'reason_exit': reason_exit
                    })
                    consecutive_loss += 1
                    consecutive_win = 0
                    prev_trade_result = 'SL'
                    last_exit_idx = i
                    position = None
                elif row['low'] <= position['tp']:
                    exit_price = position['tp']
                    direction = 1 if position['type'] == 'long' else -1
                    effective_risk_amount = position['risk_amount'] * (0.5 if position.get('tp1_hit') else 1.0)
                    pnl_move = direction * (exit_price - position['entry']) / position['risk_price'] * effective_risk_amount
                    pnl = pnl_move
                    pnl -= commission
                    capital += pnl
                    exit_label = 'TP2' if position['tp1_hit'] else 'TP'
                    reason_exit = "Final Take Profit hit"
                    trades.append({
                        **position,
                        'exit_time': row['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'commission': commission,
                        'capital': capital,
                        'exit': exit_label,
                        'reason_exit': reason_exit
                    })
                    consecutive_loss = 0
                    consecutive_win += 1
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
    if 'pnl' in df_trades.columns and not df_trades.empty:
        print("Winrate: {:.2%}".format((df_trades['pnl'] > 0).mean()))
    else:
        print("Winrate: N/A (no trades)")
    print(df_trades.tail(10))
    if kill_switch_triggered:
        print("Kill switch activated: capital below threshold")

    df_equity = pd.DataFrame(equity_curve)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trade_log_path = os.path.join(TRADE_DIR, f"trade_log_{now_str}.csv")
    equity_curve_path = os.path.join(TRADE_DIR, f"equity_curve_{now_str}.csv")

    try:
        df_trades.to_csv(trade_log_path, index=False)
        print(f"✅ บันทึก trade log แล้ว: {trade_log_path}")
    except Exception as e:
        print(f"❌ ไม่สามารถบันทึก trade_log.csv ได้: {e}")

    try:
        df_equity.to_csv(equity_curve_path, index=False)
        print(f"✅ บันทึก equity curve แล้ว: {equity_curve_path}")
    except Exception as e:
        print(f"❌ ไม่สามารถบันทึก equity_curve.csv ได้: {e}")

    if plt is not None:
        try:
            plt.figure(figsize=(12, 4))
            df_equity.plot(x='timestamp', y='equity', legend=False)
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity')
            plt.grid(True)
            curve_path = os.path.join(TRADE_DIR, f"equity_plot_{now_str}.png")
            plt.savefig(curve_path)
            plt.close()
            print(f"✅ บันทึกกราฟ equity curve แล้ว: {curve_path}")
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึกกราฟ equity curve ได้: {e}")


def generate_smart_signal(df):
    """สร้างสัญญาณเข้าซื้อแบบให้คะแนนหลายเงื่อนไข"""
    logger.debug("Generating smart signal")
    df = df.copy()
    df['signal_score'] = 0
    df['signal_score'] += (df['macd'] > df['signal']).astype(int)
    df['signal_score'] += df['Wave_Phase'].isin(['W.2', 'W.3', 'W.5', 'W.B']).astype(int)
    df['signal_score'] += (df['RSI'] > 50).astype(int)
    df['signal_score'] += ((df['close'] >= df['ema35'] * 0.995) & (df['close'] <= df['ema35'] * 1.005)).astype(int)
    df['entry_signal'] = np.where(df['signal_score'] >= entry_signal_threshold, 'buy', None)
    logger.debug("Smart signal column added")
    return df


def check_drawdown(capital: float, peak_capital: float, limit: float = 0.30) -> bool:
    """ตรวจสอบว่าเกินขีดจำกัด drawdown หรือไม่"""
    drawdown = (peak_capital - capital) / peak_capital
    logger.debug("Current drawdown: %s", drawdown)
    return drawdown > limit


def backtest_with_partial_tp(df):
    """ทดสอบกลยุทธ์พร้อม TP1 และเบรกอีเวน"""
    logger.debug("Starting simple backtest with partial TP")
    # Remove rows with NaN indicators to avoid NaN-related issues
    before = len(df)
    df = df.dropna(subset=["atr", "RSI", "ema35"])
    after = len(df)
    logger.debug("Dropped %d rows with NaN indicators", before - after)

    capital = initial_capital
    peak = capital
    trades = []
    position = None
    last_entry_time = pd.Timestamp('2000-01-01')

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = row['timestamp']
        if position is None:
            if row['entry_signal'] == 'buy' and (ts - last_entry_time).total_seconds() > cooldown_minutes * 60:
                atr = row['atr']
                entry = row['close'] + 0.10
                sl = entry - atr
                tp1 = entry + atr * 0.8
                tp2 = entry + atr * 2.0
                risk = capital * risk_per_trade
                lot = risk / max(entry - sl, 1e-6)
                position = {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'lot': lot, 'tp1_hit': False, 'entry_time': ts}
                last_entry_time = ts
                logger.debug("Open position at %s", ts)
        else:
            high, low = row['high'], row['low']
            if not position['tp1_hit'] and high >= position['tp1']:
                gain = position['lot'] * (position['tp1'] - position['entry']) * partial_tp_ratio
                capital += gain
                position['sl'] = position['entry']
                position['tp1_hit'] = True
                trades.append({**position, 'exit': 'TP1', 'pnl': gain, 'time': ts})
                logger.debug("TP1 hit at %s", ts)
            elif high >= position['tp2']:
                gain = position['lot'] * (position['tp2'] - position['entry']) * (1 - partial_tp_ratio)
                capital += gain
                trades.append({**position, 'exit': 'TP2', 'pnl': gain, 'time': ts})
                position = None
                logger.debug("TP2 hit at %s", ts)
            elif low <= position['sl']:
                loss = -position['lot'] * (position['entry'] - position['sl'])
                capital += loss
                trades.append({**position, 'exit': 'SL', 'pnl': loss, 'time': ts})
                position = None
                logger.debug("Stop loss hit at %s", ts)

            if capital > peak:
                peak = capital
            if check_drawdown(capital, peak):
                logger.debug("Drawdown limit reached. stop trading")
                break

    return pd.DataFrame(trades), capital


def run():
    """โหลดข้อมูลและรัน backtest แบบย่อ"""
    logger.debug("Running simple integration")
    df = pd.read_csv("XAUUSD_M1.csv")
    df.columns = [c.lower() for c in df.columns]
    year = df['date'].astype(str).str[:4].astype(int) - 543
    month = df['date'].astype(str).str[4:6].astype(int)
    day = df['date'].astype(str).str[6:8].astype(int)
    df['timestamp'] = pd.to_datetime(year.astype(str) + '-' + month.astype(str).str.zfill(2) + '-' + day.astype(str).str.zfill(2) + ' ' + df['timestamp'])
    df['ema35'] = df['close'].ewm(span=35).mean()
    df['RSI'] = df['close'].rolling(14).apply(lambda x: 100 - 100 / (1 + (np.mean(np.clip(np.diff(x), 0, None)) / (np.mean(np.clip(-np.diff(x), 0, None)) + 1e-6))))
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    # Drop rows with NaN indicators so backtest starts with valid values
    before = len(df)
    df = df.dropna(subset=["atr", "RSI", "ema35"])
    logger.debug("Dropped %d rows with NaN indicators before backtest", before - len(df))
    df = generate_smart_signal(df)
    trades, final_capital = backtest_with_partial_tp(df)
    print(f"Final Equity: {final_capital:.2f}, Total Trades: {len(trades)}")
    print(trades.tail())

if __name__ == "__main__":  # pragma: no cover
    run_backtest_cli()
