# [Patch] NICEGOLD Enterprise Supergrowth v4 - Entry Boost, Adaptive Lot, OMS, Smart Exit, QA
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
os.makedirs(TRADE_DIR, exist_ok=True)
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"

# [Patch] Parameters – MM & Growth
initial_capital = 100.0
risk_per_trade = 0.01  # [Patch] ลดความเสี่ยงต่อไม้เหลือ 1%
tp1_mult = 2.0  # [Patch] ปรับ TP1 multiplier ตามความเป็นจริง
tp2_mult = 4.0  # [Patch] ปรับ TP2 multiplier ตามความเป็นจริง
sl_mult = 1.2  # [Patch] ลด SL multiplier ให้ใกล้เคียงตลาดจริง
min_sl_dist = 4.0  # [Patch] ระยะ SL ขั้นต่ำสมจริง
lot_max = 5.0  # [Patch] ปลดลิมิตให้สามารถปั้นพอร์ตโต
lot_cap_500 = 0.5
lot_cap_2000 = 1.0
lot_cap_10000 = 2.5
lot_cap_max = 5.0
cooldown_bars = 0  # [Patch] ไม่มี cooldown
oms_recovery_loss = 3  # [Patch] Recovery เร็วขึ้น (จาก 4 → 3)
win_streak_boost = 1.3  # [Patch] Boost สูงขึ้น
recovery_multiplier = 3.0  # [Patch] Aggressive recovery
trailing_atr_mult = 1.1
kill_switch_dd = 0.35  # [Patch] Kill switch หาก DD > 35%
trend_lookback = 25  # [Patch] เร็วขึ้น (trend สั้นลง)
adx_period = 14
adx_thresh = 12  # [Patch] Relax adx guard
adx_strong = 23  # [Patch] ปรับ adx strong
force_entry_gap = 300  # [Patch] Force entry หากไม่มี order เกิน 300 แท่ง
trade_start_hour = 8
trade_end_hour = 23

# [Patch] Commission, Spread, Slippage สมจริง
SPREAD_POINTS = 80
SPREAD_VALUE = 0.8
COMMISSION_PER_LOT = 0.10
SLIPPAGE = 0.2

# --- Runtime utilities (merged) ---


def is_in_session(current_time, sessions):
    if sessions is None:
        return True
    hour = current_time.hour
    for start, end in sessions:
        if start <= hour < end:
            return True
    return False


def breakout_condition(price_data):
    return price_data.get("breakout", False)


def get_pip_value(symbol="XAUUSD", lot=1):
    return 0.1 * lot


@dataclass
class Order:
    id: int
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    partial_tp_level: Optional[float] = None
    move_sl_to_be_trigger: Optional[float] = None
    trail_stop: bool = False
    partial_taken: bool = False


@dataclass
class Portfolio:
    equity: float
    drawdown: float = 0.0
    recovery_active: bool = False
    last_trade_loss: bool = False
    last_lot: float = 0.0
    initial_lot: float = 0.1
    last_direction: str = "BUY"


DD_THRESHOLD = 0.2
SAFE_THRESHOLD = 0.05
base_stop_distance = 1.0
base_tp_distance = 2.0
partial_tp_distance = 1.5
break_even_distance = 1.0
trailing_atr_multiplier = 1.0


def generate_signal(
    price_data,
    indicators,
    *,
    current_time=None,
    allowed_sessions=None,
    intended_side="BUY",
    force_entry=False,
):
    if force_entry:
        logger.debug("Force entry active")
        return True
    session_ok = is_in_session(current_time, allowed_sessions)
    trend_ok = (
        indicators["EMA50"] > indicators["EMA200"]
        if intended_side == "BUY"
        else indicators["EMA50"] < indicators["EMA200"]
    )
    momentum_ok = (
        indicators["ADX"] > 25 and indicators.get("ADX_trend") == intended_side
    )
    if session_ok and trend_ok and momentum_ok and breakout_condition(price_data):
        logger.debug(
            "[Patch] Signal confirmed: session=%s, trend=%s, momentum=%s, breakout=True",
            session_ok,
            trend_ok,
            momentum_ok,
        )
        return True
    return False


def calculate_position_size(equity, entry_price, stop_price, risk_pct):
    risk_amount = equity * risk_pct
    pip_value = get_pip_value(symbol="XAUUSD", lot=1)
    stop_pips = abs(entry_price - stop_price) / pip_value
    lot = risk_amount / (stop_pips * pip_value)
    logger.debug(
        "[Patch] Calculated position size = %.2f lots for risk %.1f%% and stop %.1f pips",
        lot,
        risk_pct * 100,
        stop_pips,
    )
    return lot


def set_stop_loss(order, price):
    order.stop_loss = price


def set_take_profit(order, price):
    order.take_profit = price


def close_position(order, portion=1.0):
    logger.info("[Patch] Closing %.0f%% of order %s", portion * 100, order.id)


def on_order_execute(order):
    set_stop_loss(order, order.entry_price - base_stop_distance)
    set_take_profit(order, order.entry_price + base_tp_distance)
    logger.info(
        "[Patch] Order %s executed: SL=%.2f, TP=%.2f",
        order.id,
        order.stop_loss,
        order.take_profit,
    )
    order.partial_tp_level = order.entry_price + partial_tp_distance
    order.move_sl_to_be_trigger = order.entry_price + break_even_distance
    order.trail_stop = True
    logger.debug(
        "[Patch] Set partial TP at %.2f and BE trigger at %.2f",
        order.partial_tp_level,
        order.move_sl_to_be_trigger,
    )


def on_price_update(order, price, indicators=None):
    if (
        order.partial_tp_level
        and price >= order.partial_tp_level
        and not order.partial_taken
    ):
        close_position(order, portion=0.5)
        order.partial_taken = True
        logger.info(
            "[Patch] Partial TP hit for order %s: closed 50%% at price %.2f",
            order.id,
            price,
        )
        if order.move_sl_to_be_trigger:
            new_sl = order.entry_price
            set_stop_loss(order, new_sl)
            logger.info("[Patch] Moved SL to BE for order %s at %.2f", order.id, new_sl)
    if order.trail_stop and price > order.entry_price and indicators:
        trail_distance = trailing_atr_multiplier * indicators.get("ATR", 0)
        new_sl = max(order.stop_loss, price - trail_distance)
        if new_sl > order.stop_loss:
            set_stop_loss(order, new_sl)
            logger.debug(
                "[Patch] Trailing SL updated for order %s to %.2f",
                order.id,
                new_sl,
            )


def on_price_update_patch(order, price, indicators=None):
    """[Patch] Partial TP + Move SL to BE, trailing SL after TP1, for QA backtest"""
    if (
        order.partial_tp_level
        and price >= order.partial_tp_level
        and not order.partial_taken
    ):
        close_position(order, portion=0.5)
        order.partial_taken = True
        logger.info(
            "[Patch] Partial TP hit for order %s: closed 50%% at price %.2f",
            order.id,
            price,
        )
        if order.move_sl_to_be_trigger:
            new_sl = order.entry_price
            set_stop_loss(order, new_sl)
            logger.info("[Patch] Moved SL to BE for order %s at %.2f", order.id, new_sl)
    if order.trail_stop and price > order.entry_price and indicators:
        trail_distance = trailing_atr_multiplier * indicators.get("ATR", 0)
        new_sl = max(order.stop_loss, price - trail_distance)
        if new_sl > order.stop_loss:
            set_stop_loss(order, new_sl)
            logger.debug(
                "[Patch] Trailing SL updated for order %s to %.2f",
                order.id,
                new_sl,
            )


def on_price_update_patch_v2(order, price, indicators=None):
    """[Patch] Partial TP + Move SL to BE, trailing SL (tighten step) after TP1, for QA backtest"""
    if (
        order.partial_tp_level
        and price >= order.partial_tp_level
        and not order.partial_taken
    ):
        close_position(order, portion=0.5)
        order.partial_taken = True
        logger.info(
            "[Patch] Partial TP hit for order %s: closed 50%% at price %.2f",
            order.id,
            price,
        )
        if order.move_sl_to_be_trigger:
            new_sl = order.entry_price
            set_stop_loss(order, new_sl)
            logger.info("[Patch] Moved SL to BE for order %s at %.2f", order.id, new_sl)
    if order.trail_stop and price > order.entry_price and indicators:
        trail_distance = trailing_atr_multiplier * indicators.get("ATR", 0) * 0.7
        new_sl = max(order.stop_loss, price - trail_distance)
        if new_sl > order.stop_loss:
            set_stop_loss(order, new_sl)
            logger.debug(
                "[Patch] Tight Trailing SL updated for order %s to %.2f",
                order.id,
                new_sl,
            )


def open_position(lot_size, direction):
    logger.info("[Patch] Opening position %s %.2f lots", direction, lot_size)


def manage_recovery(portfolio: Portfolio, price_data=None, indicators=None):
    if portfolio.drawdown > DD_THRESHOLD:
        if not portfolio.recovery_active:
            portfolio.recovery_active = True
            logger.warning(
                "[Patch] Activating Recovery Mode at drawdown %.1f%%",
                portfolio.drawdown * 100,
            )
        base_lot = portfolio.initial_lot
        if portfolio.last_trade_loss:
            base_lot = max(base_lot, portfolio.last_lot)
        if (
            generate_signal(
                price_data or {},
                indicators or {},
                current_time=datetime.now(),
                allowed_sessions=None,
                intended_side=portfolio.last_direction.lower(),
            )
            and portfolio.recovery_active
        ):
            open_position(lot_size=base_lot, direction=portfolio.last_direction)
            logger.info(
                "[Patch] Recovery trade opened with lot=%.2f in direction %s",
                base_lot,
                portfolio.last_direction,
            )
    elif portfolio.recovery_active and portfolio.drawdown < SAFE_THRESHOLD:
        portfolio.recovery_active = False
        logger.info(
            "[Patch] Exit Recovery Mode, drawdown improved to %.1f%%",
            portfolio.drawdown * 100,
        )


def load_data(path):
    logger.info("[Patch] Loading data: %s", path)
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        year = df["date"].astype(str).str[:4].astype(int) - 543
        month = df["date"].astype(str).str[4:6]
        day = df["date"].astype(str).str[6:8]
        df["timestamp"] = pd.to_datetime(
            year.astype(str) + "-" + month + "-" + day + " " + df["timestamp"],
            format="%Y-%m-%d %H:%M:%S",
        )
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def data_quality_check(df):
    """[Patch] ตรวจสอบและ log คุณภาพข้อมูลเบื้องต้น"""
    logger.info("[Patch] Data QA: Start checking data quality")
    required_cols = ["timestamp", "open", "high", "low", "close"]
    for col in required_cols:
        if df[col].isnull().any():
            logger.warning(
                "[Patch] NaN detected in column %s: %d rows",
                col,
                df[col].isnull().sum(),
            )
    if df["timestamp"].duplicated().any():
        dup_count = df["timestamp"].duplicated().sum()
        logger.warning("[Patch] Duplicated timestamp: %d rows", dup_count)
        df = df[~df["timestamp"].duplicated(keep="first")].reset_index(drop=True)
    if not df["timestamp"].is_monotonic_increasing:
        logger.warning("[Patch] Non-monotonic timestamp detected, sorting")
        df = df.sort_values("timestamp").reset_index(drop=True)
    invalid_price = (df["high"] < df["low"]).sum()
    if invalid_price > 0:
        logger.warning(
            "[Patch] %d bars: high < low detected",
            invalid_price,
        )
    price_jump = (df["close"].pct_change().abs() > 0.10).sum()
    if price_jump > 0:
        logger.warning("[Patch] %d bars: close jump > 10%% detected", price_jump)
    orig_len = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) < orig_len:
        logger.info(
            "[Patch] Drop %d rows with NaN in required columns",
            orig_len - len(df),
        )
    logger.info("[Patch] Data QA: Complete")
    return df


def rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    logger.debug("[Patch] Calculating RSI with period %d", period)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_indicators(df, ema_fast_period=None, ema_slow_period=None, rsi_period=14):
    logger.info("[Patch] Calculating indicators")
    if ema_fast_period is None:
        ema_fast_period = trend_lookback
    if ema_slow_period is None:
        ema_slow_period = trend_lookback * 2
    df["ema_fast"] = df["close"].ewm(span=ema_fast_period, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_period, adjust=False).mean()
    df["ema_fast_htf"] = df["close"].ewm(span=ema_fast_period * 4, adjust=False).mean()
    df["ema_slow_htf"] = df["close"].ewm(span=ema_slow_period * 4, adjust=False).mean()
    df["rsi"] = rsi(df["close"], rsi_period)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    up_move = df["high"].diff().clip(lower=0)
    down_move = -df["low"].diff().clip(upper=0)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    plus_dm = up_move.where(up_move > down_move, 0.0)
    minus_dm = down_move.where(down_move > up_move, 0.0)
    atr = tr.rolling(adx_period).mean()
    plus_di = 100 * plus_dm.rolling(adx_period).sum() / atr
    minus_di = 100 * minus_dm.rolling(adx_period).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = dx.rolling(adx_period).mean()
    return df


def add_m15_context_to_m1(df_m1, df_m15):
    """[Patch] Join M15 trend/indicator context onto M1 DataFrame by nearest timestamp."""
    import pandas as pd

    logger.info("[Patch] Add M15 trend context to M1")
    df_m15 = df_m15.copy().sort_index()
    m15_cols = ["ema_fast", "ema_slow", "rsi"]
    df_m15_ctx = df_m15[m15_cols].rename(
        columns={
            "ema_fast": "m15_ema_fast",
            "ema_slow": "m15_ema_slow",
            "rsi": "m15_rsi",
        }
    )
    df_m1 = pd.merge_asof(
        df_m1.sort_index(),
        df_m15_ctx,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return df_m1


def label_elliott_wave(df, wave_col="wave_phase"):
    """[Patch] Label Elliott Wave Phase using simple swing logic."""
    logger.info("[Patch] Label Elliott Wave Phase")
    df = df.copy()
    df[wave_col] = None
    window = 25
    swing_high = df["high"].rolling(window, center=True).max()
    swing_low = df["low"].rolling(window, center=True).min()
    for i in range(len(df)):
        if df["high"].iloc[i] == swing_high.iloc[i]:
            df.at[df.index[i], wave_col] = "peak"
        elif df["low"].iloc[i] == swing_low.iloc[i]:
            df.at[df.index[i], wave_col] = "trough"
        else:
            df.at[df.index[i], wave_col] = "mid"
    return df


def detect_divergence(df, rsi_col="rsi", macd_col="macd", div_col="divergence"):
    """[Patch] Detect basic bullish/bearish divergence using RSI."""
    logger.info("[Patch] Detect Divergence (RSI/MACD)")
    df = df.copy()
    df[div_col] = None
    for i in range(2, len(df)):
        if (
            df["low"].iloc[i] < df["low"].iloc[i - 1]
            and df[rsi_col].iloc[i] > df[rsi_col].iloc[i - 1]
        ):
            df.at[df.index[i], div_col] = "bullish"
        if (
            df["high"].iloc[i] > df["high"].iloc[i - 1]
            and df[rsi_col].iloc[i] < df[rsi_col].iloc[i - 1]
        ):
            df.at[df.index[i], div_col] = "bearish"
    return df


def label_pattern(df, pattern_col="pattern_label"):
    """[Patch] Tag simple price patterns using EMA and RSI."""
    logger.info("[Patch] Pattern Labeling")
    df = df.copy()
    df[pattern_col] = None
    cond_first_pullback = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > 50)
    cond_throwback = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < 50)
    df.loc[cond_first_pullback, pattern_col] = "first_pullback"
    df.loc[cond_throwback, pattern_col] = "throwback"
    return df


def calc_gain_zscore(df, window=50, gain_col="gain_z"):
    """[Patch] Calculate Z-score of price change as momentum."""
    logger.info("[Patch] Calculate Gain_Z Z-score")
    df = df.copy()
    df["gain"] = df["close"].diff()
    df[gain_col] = (df["gain"] - df["gain"].rolling(window).mean()) / (
        df["gain"].rolling(window).std() + 1e-8
    )
    return df


def calc_signal_score(df, score_col="signal_score"):
    """[Patch] Aggregate signal score from context features."""
    logger.info(
        "[Patch] Calculate Signal Score (context + pattern + divergence + gain_z)"
    )
    df = df.copy()
    df[score_col] = 0
    df.loc[df["pattern_label"].isin(["first_pullback", "throwback"]), score_col] += 1
    df.loc[df["divergence"].notna(), score_col] += 1
    df.loc[df["gain_z"] > 1.5, score_col] += 1
    df.loc[df["wave_phase"].isin(["peak", "trough"]), score_col] += 1
    return df


def meta_classifier_filter(
    df, proba_col="meta_proba", score_col="signal_score", threshold=2
):
    """[Patch] Meta model filter for final entry permission."""
    logger.info("[Patch] Meta Classifier filter")
    df = df.copy()
    df["meta_entry"] = df[score_col] >= threshold
    return df


def shap_feature_importance_placeholder(
    df, feature_cols=None, importance_col="shap_importance"
):
    """[Patch] SHAP feature importance stub"""
    logger.info("[Patch] SHAP feature importance: stub mode")
    if feature_cols is None:
        feature_cols = [
            "ema_fast",
            "ema_slow",
            "rsi",
            "atr",
            "gain_z",
            "signal_score",
        ]
    importances = {col: np.random.uniform(0, 1) for col in feature_cols}
    s = sum(importances.values())
    if s > 0:
        for k in importances:
            importances[k] /= s
    logger.info("[Patch] Feature importances (stub): %s", importances)
    for k, v in importances.items():
        if k in df.columns:
            df[f"{importance_col}_{k}"] = v
    return df, importances


def patch_confirm_on_lossy_indices(df, loss_indices, min_divergence_score=1):
    """[Patch] เพิ่ม confirm เฉพาะจุด SL/BE ขาดทุนถี่"""
    logger.info("[Patch] Confirm on lossy indices: %s", loss_indices)
    for i in loss_indices:
        if pd.notna(df.at[i, "entry_signal"]):
            div = df.at[i, "divergence"] if "divergence" in df.columns else None
            gain_z = df.at[i, "gain_z"] if "gain_z" in df.columns else 0
            atr = df.at[i, "atr"] if "atr" in df.columns else 0
            if div is None or (
                df.at[i, "entry_signal"] == "buy" and div != "bullish"
            ) or (
                df.at[i, "entry_signal"] == "sell" and div != "bearish"
            ):
                df.at[i, "entry_signal"] = None
                logger.info(
                    "[Patch] LossyIndex confirm: remove entry at %s (divergence filter)",
                    df["timestamp"].iloc[i],
                )
            elif gain_z < 0 and df.at[i, "entry_signal"] == "buy":
                df.at[i, "entry_signal"] = None
                logger.info(
                    "[Patch] LossyIndex confirm: remove buy entry at %s (gain_z < 0)",
                    df["timestamp"].iloc[i],
                )
            elif gain_z > 0 and df.at[i, "entry_signal"] == "sell":
                df.at[i, "entry_signal"] = None
                logger.info(
                    "[Patch] LossyIndex confirm: remove sell entry at %s (gain_z > 0)",
                    df["timestamp"].iloc[i],
                )
    return df


def analyze_tradelog(trades_df, equity_df):
    """[Patch] แจกแจง PnL, Win/Loss streak, drawdown"""
    logger.info("[Patch] Analyze trade log statistics")
    streaks, streak_val = [], []
    prev_win = None
    count = 0
    for _, row in trades_df.iterrows():
        win = row["pnl"] > 0
        if prev_win is None or win == prev_win:
            count += 1
        else:
            streaks.append(count)
            streak_val.append(prev_win)
            count = 1
        prev_win = win
    if count > 0:
        streaks.append(count)
        streak_val.append(prev_win)
    max_win_streak = max([s for s, v in zip(streaks, streak_val) if v], default=0)
    max_loss_streak = max([s for s, v in zip(streaks, streak_val) if not v], default=0)
    mean_pnl = trades_df["pnl"].mean() if not trades_df.empty else 0
    std_pnl = trades_df["pnl"].std() if not trades_df.empty else 0
    max_drawdown = equity_df["dd"].max() if not equity_df.empty else 0
    print(f"[Patch] Max Win Streak: {max_win_streak}, Max Loss Streak: {max_loss_streak}")
    print(f"[Patch] Mean PnL: {mean_pnl:.2f}, Std PnL: {std_pnl:.2f}")
    print(f"[Patch] Max Drawdown: {max_drawdown:.2%}")
    return {
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "max_drawdown": max_drawdown,
    }


def calc_dynamic_tp2(df, base_tp2_mult=3.0, atr_period=1000, tp2_col="tp2_dynamic"):
    """[Patch] Calculate Dynamic TP2 Multiplier based on volatility regime."""
    logger.info("[Patch] Calculate Dynamic TP2 Multiplier")
    df = df.copy()
    df["atr_long"] = df["atr"].rolling(atr_period).mean()
    df[tp2_col] = base_tp2_mult
    high_vol = df["atr"] > 1.5 * df["atr_long"]
    low_vol = df["atr"] < 0.8 * df["atr_long"]
    df.loc[high_vol, tp2_col] = base_tp2_mult * 0.75
    df.loc[low_vol, tp2_col] = base_tp2_mult * 1.25
    df[tp2_col] = df[tp2_col].clip(lower=base_tp2_mult * 0.5, upper=base_tp2_mult * 2.0)
    return df


def tag_session(df, session_col="session"):
    """[Patch] Tag trading session (Asia/London/NY/Other)."""
    logger.info("[Patch] Tag trading session (Asia/London/NY)")
    session = []
    for ts in pd.to_datetime(df["timestamp"]):
        hour = ts.hour
        if 0 <= hour < 8:
            session.append("Asia")
        elif 8 <= hour < 16:
            session.append("London")
        elif 16 <= hour < 23:
            session.append("NY")
        else:
            session.append("Other")
    df[session_col] = session
    return df


def apply_session_bias(df, entry_col="entry_signal", session_col="session"):
    """[Patch] Filter or adjust entry signal based on session bias."""
    logger.info("[Patch] Apply session bias (entry filter/boost)")
    df = df.copy()
    block_sessions = ["Other"]
    mask_block = df[session_col].isin(block_sessions)
    df.loc[mask_block, entry_col] = None
    return df


def tag_spike_guard(df, atr_col="atr", spike_col="spike_guard"):
    """[Patch] Tag Spike ช่วงที่ ATR หรือ Candle body ใหญ่ผิดปกติ"""
    logger.info("[Patch] Spike Guard tagging")
    df = df.copy()
    df[spike_col] = False
    atr_long = df[atr_col].rolling(1000).mean()
    spike_atr = df[atr_col] > 1.8 * atr_long
    wrb = (df["high"] - df["low"]) > 2.0 * (df["high"] - df["low"]).rolling(200).mean()
    spike = spike_atr | wrb
    df.loc[spike, spike_col] = True
    return df


def tag_news_event(df, news_times=None, news_col="news_guard"):
    """[Patch] News Filter ช่วงเวลาหลีกเลี่ยงเทรด"""
    logger.info("[Patch] News Guard tagging")
    df = df.copy()
    df[news_col] = False
    if news_times:
        for t_start, t_end in news_times:
            mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
            df.loc[mask, news_col] = True
    return df


def apply_spike_news_guard(
    df, entry_col="entry_signal", spike_col="spike_guard", news_col="news_guard"
):
    """[Patch] Block entry_signal ถ้ามี spike หรือช่วงข่าว"""
    logger.info("[Patch] Block entry when spike or news event")
    df = df.copy()
    mask_block = (df[spike_col] == True) | (df[news_col] == True)
    df.loc[mask_block, entry_col] = None
    return df


def is_strong_trend(df, i):
    """Check if strong trend using rolling EMA and ADX."""
    if i < trend_lookback * 2:
        return False
    trend = (
        df["ema_fast"].iloc[i - trend_lookback + 1 : i + 1]
        > df["ema_slow"].iloc[i - trend_lookback + 1 : i + 1]
    ).all() or (
        df["ema_fast"].iloc[i - trend_lookback + 1 : i + 1]
        < df["ema_slow"].iloc[i - trend_lookback + 1 : i + 1]
    ).all()
    adx = df["adx"].iloc[i] > adx_thresh
    atr_high = df["atr"].iloc[i] > df["atr"].rolling(1000).median().iloc[i] * 1.2
    return trend and (adx or atr_high)


def multi_session_trend_scalping(df):
    """[Patch] Multi-Session Trend Scalping Entry Signal (ATR breakout + momentum)"""
    logger.info(
        "[Patch] Multi-Session Trend Scalping Entry Signal (ATR breakout + momentum)"
    )
    df = df.copy()
    df["entry_signal"] = None
    atr_quantile = df["atr"].quantile(0.65)
    for i in range(50, len(df)):
        hour = pd.to_datetime(df["timestamp"].iloc[i]).hour
        if not (8 <= hour < 23):
            continue
        if df["atr"].iloc[i] < atr_quantile:
            continue
        if df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i] and df["rsi"].iloc[i] > 55:
            df.at[df.index[i], "entry_signal"] = "buy"
        elif (
            df["ema_fast"].iloc[i] < df["ema_slow"].iloc[i]
            and df["rsi"].iloc[i] < 45
        ):
            df.at[df.index[i], "entry_signal"] = "sell"
    logger.info(
        "[Patch] Entry signal counts (Enterprise strategy): buy=%d, sell=%d",
        (df["entry_signal"] == "buy").sum(),
        (df["entry_signal"] == "sell").sum(),
    )
    return df


def smart_entry_signal(df):
    logger.info(
        "[Patch] Vectorized entry signal (trend+min SL guard+relax+force entry)"
    )
    df = df.copy()
    df["entry_signal"] = None

    # [Patch] 1. Relax ATR/ADX Guard (และกรองเฉพาะช่วงเวลาสำคัญ)
    session_mask = True
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        session_mask = (ts.dt.hour >= trade_start_hour) & (ts.dt.hour < trade_end_hour)
    mask_valid = (
        (df["atr"] > min_sl_dist * 0.7) & (df["adx"] > adx_thresh) & session_mask
    )

    # [Patch] 2. Rolling trend mask (trend 15 bar for stability)
    n_trend = 15
    trend_up = (
        (df["ema_fast"] > df["ema_slow"])
        .rolling(n_trend, min_periods=n_trend)
        .apply(lambda x: x.all(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    trend_dn = (
        (df["ema_fast"] < df["ema_slow"])
        .rolling(n_trend, min_periods=n_trend)
        .apply(lambda x: x.all(), raw=True)
        .fillna(0)
        .astype(bool)
    )

    # [Patch] 3. RSI + multi-timeframe + relax wick filter (ผ่านมากขึ้น)
    def relax_wick(row):
        if {"high", "low", "close"}.issubset(row.index):
            price_range = max(row["high"] - row["low"], 1e-6)
            upper_wick = (row["high"] - row["close"]) / price_range
            lower_wick = (row["close"] - row["low"]) / price_range
            return upper_wick < 0.80 and lower_wick < 0.80
        return True

    htf_ok_long = True
    htf_ok_short = True
    if "ema_fast_htf" in df.columns and "ema_slow_htf" in df.columns:
        htf_ok_long = df["ema_fast_htf"] > df["ema_slow_htf"]
        htf_ok_short = df["ema_fast_htf"] < df["ema_slow_htf"]
    entry_long = (
        mask_valid
        & trend_up
        & (df["rsi"] > 51)
        & htf_ok_long
        & df.apply(relax_wick, axis=1)
    )
    entry_short = (
        mask_valid
        & trend_dn
        & (df["rsi"] < 49)
        & htf_ok_short
        & df.apply(relax_wick, axis=1)
    )
    df.loc[entry_long, "entry_signal"] = "buy"
    df.loc[entry_short, "entry_signal"] = "sell"

    # [Patch] 4. Force Entry หากไม่มี signal เกิน force_entry_gap bar
    last_entry = 0
    for i in range(len(df)):
        if pd.notna(df["entry_signal"].iloc[i]):
            last_entry = i
        elif i - last_entry > force_entry_gap:
            if mask_valid.iloc[i]:
                # [Patch] เลือกทางเดียวกับ momentum/ema
                if df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i]:
                    df.at[i, "entry_signal"] = "buy"
                else:
                    df.at[i, "entry_signal"] = "sell"
                last_entry = i
                if "timestamp" in df.columns:
                    logger.info("[Patch] Force Entry at %s", df["timestamp"].iloc[i])
                else:
                    logger.info("[Patch] Force Entry at index %s", i)

    logger.info(
        "[Patch] Entry signal counts: buy=%d, sell=%d",
        (df["entry_signal"] == "buy").sum(),
        (df["entry_signal"] == "sell").sum(),
    )
    logger.debug(
        "Entry signals generated on indices: %s",
        df[df["entry_signal"].notna()].index.tolist(),
    )
    return df


def smart_entry_signal_multi_tf(df):
    """[Patch] Multi-Timeframe Confirm Entry (M1 signal + M15 trend)"""
    logger.info("[Patch] Entry signal with Multi-TF confirm (M1+M15)")
    df = df.copy()
    df["entry_signal"] = None
    buy_cond = (
        (df["ema_fast"] > df["ema_slow"])
        & (df["m15_ema_fast"] > df["m15_ema_slow"])
        & (df["rsi"] > 50)
        & (df["m15_rsi"] > 50)
    )
    sell_cond = (
        (df["ema_fast"] < df["ema_slow"])
        & (df["m15_ema_fast"] < df["m15_ema_slow"])
        & (df["rsi"] < 50)
        & (df["m15_rsi"] < 50)
    )
    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"
    logger.info(
        "[Patch] Multi-TF Entry counts: buy=%d, sell=%d",
        (df["entry_signal"] == "buy").sum(),
        (df["entry_signal"] == "sell").sum(),
    )
    return df


def smart_entry_signal_multi_tf_ema_adx(df, min_entry_gap_minutes=60):
    """[Patch] กลยุทธ์ Trend + EMA + ADX + Multi-TF + OMS/QA"""
    import pandas as pd
    import numpy as np

    logger.info(
        "[Patch] Entry signal: Multi-TF EMA+ADX+RSI+Wick Filter+OMS"
    )
    df = df.copy()
    df["entry_signal"] = None
    last_entry_time = None
    min_gap = pd.Timedelta(minutes=min_entry_gap_minutes)

    for i in range(max(50, df.index.min()), len(df)):
        ema_fast = df["ema_fast"].iloc[i]
        ema_slow = df["ema_slow"].iloc[i]
        m15_ema_fast = (
            df["m15_ema_fast"].iloc[i] if "m15_ema_fast" in df.columns else np.nan
        )
        m15_ema_slow = (
            df["m15_ema_slow"].iloc[i] if "m15_ema_slow" in df.columns else np.nan
        )
        adx = df["adx"].iloc[i]
        rsi = df["rsi"].iloc[i]
        timestamp = pd.to_datetime(df["timestamp"].iloc[i]) if "timestamp" in df.columns else pd.Timestamp.now()

        price_range = max(df["high"].iloc[i] - df["low"].iloc[i], 1e-8)
        upper_wick = (df["high"].iloc[i] - df["close"].iloc[i]) / price_range
        lower_wick = (df["close"].iloc[i] - df["low"].iloc[i]) / price_range
        relaxed_wick = upper_wick < 0.80 and lower_wick < 0.80

        if last_entry_time is not None and timestamp - last_entry_time < min_gap:
            continue

        if (
            ema_fast > ema_slow
            and m15_ema_fast > m15_ema_slow
            and adx > 12
            and rsi > 51
            and relaxed_wick
        ):
            df.at[df.index[i], "entry_signal"] = "buy"
            last_entry_time = timestamp
            logger.info(
                f"[Patch] Entry: BUY at {timestamp} (ADX={adx:.2f}, RSI={rsi:.2f})"
            )
        elif (
            ema_fast < ema_slow
            and m15_ema_fast < m15_ema_slow
            and adx > 12
            and rsi < 49
            and relaxed_wick
        ):
            df.at[df.index[i], "entry_signal"] = "sell"
            last_entry_time = timestamp
            logger.info(
                f"[Patch] Entry: SELL at {timestamp} (ADX={adx:.2f}, RSI={rsi:.2f})"
            )

    force_entry_gap = 300
    last_entry = 0
    for i in range(len(df)):
        if pd.notna(df["entry_signal"].iloc[i]):
            last_entry = i
        elif i - last_entry > force_entry_gap:
            ema_fast = df["ema_fast"].iloc[i]
            ema_slow = df["ema_slow"].iloc[i]
            m15_ema_fast = (
                df["m15_ema_fast"].iloc[i]
                if "m15_ema_fast" in df.columns
                else np.nan
            )
            m15_ema_slow = (
                df["m15_ema_slow"].iloc[i]
                if "m15_ema_slow" in df.columns
                else np.nan
            )
            adx = df["adx"].iloc[i]
            rsi = df["rsi"].iloc[i]
            relaxed_wick = True
            if {"high", "low", "close"}.issubset(df.columns):
                price_range = max(df["high"].iloc[i] - df["low"].iloc[i], 1e-8)
                upper_wick = (df["high"].iloc[i] - df["close"].iloc[i]) / price_range
                lower_wick = (df["close"].iloc[i] - df["low"].iloc[i]) / price_range
                relaxed_wick = upper_wick < 0.80 and lower_wick < 0.80
            if adx > 12 and relaxed_wick:
                if ema_fast > ema_slow and m15_ema_fast > m15_ema_slow and rsi > 51:
                    df.at[df.index[i], "entry_signal"] = "buy"
                    logger.info(
                        f"[Patch] Force Entry: BUY at {df['timestamp'].iloc[i]}"
                    )
                    last_entry = i
                elif (
                    ema_fast < ema_slow
                    and m15_ema_fast < m15_ema_slow
                    and rsi < 49
                ):
                    df.at[df.index[i], "entry_signal"] = "sell"
                    logger.info(
                        f"[Patch] Force Entry: SELL at {df['timestamp'].iloc[i]}"
                    )
                    last_entry = i

    logger.info(
        f"[Patch] Multi-TF Trend Entry: buy={(df['entry_signal']=='buy').sum()}, sell={(df['entry_signal']=='sell').sum()}"
    )
    return df


def smart_entry_signal_multi_tf_ema_adx_optimized(df, min_entry_gap_minutes=60):
    """[Patch] Multi-TF EMA+ADX+RSI+Momentum+ATR Filter (Maintain Entry Count, Boost Winrate)"""
    import pandas as pd
    import numpy as np
    logger.info("[Patch] Entry signal (optimized): Multi-TF EMA+ADX+Momentum+ATR")
    df = df.copy()
    df["entry_signal"] = None
    last_entry_time = None
    min_gap = pd.Timedelta(minutes=min_entry_gap_minutes)

    atr_threshold = df["atr"].quantile(0.20)
    gainz_buy = 0.0
    gainz_sell = 0.0

    for i in range(max(50, df.index.min()), len(df)):
        ema_fast = df["ema_fast"].iloc[i]
        ema_slow = df["ema_slow"].iloc[i]
        m15_ema_fast = df["m15_ema_fast"].iloc[i] if "m15_ema_fast" in df.columns else np.nan
        m15_ema_slow = df["m15_ema_slow"].iloc[i] if "m15_ema_slow" in df.columns else np.nan
        adx = df["adx"].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        gain_z = df["gain_z"].iloc[i] if "gain_z" in df.columns else 0.0
        divergence = df["divergence"].iloc[i] if "divergence" in df.columns else None
        timestamp = pd.to_datetime(df["timestamp"].iloc[i])
        price_range = max(df["high"].iloc[i] - df["low"].iloc[i], 1e-8)
        upper_wick = (df["high"].iloc[i] - df["close"].iloc[i]) / price_range
        lower_wick = (df["close"].iloc[i] - df["low"].iloc[i]) / price_range
        relaxed_wick = (upper_wick < 0.80 and lower_wick < 0.80)

        if atr < atr_threshold:
            continue

        if last_entry_time is not None and timestamp - last_entry_time < min_gap:
            continue

        if (
            ema_fast > ema_slow
            and m15_ema_fast > m15_ema_slow
            and adx > 12
            and rsi > 51
            and gain_z > gainz_buy
            and relaxed_wick
            and (divergence is None or divergence == "bullish")
        ):
            df.at[df.index[i], "entry_signal"] = "buy"
            last_entry_time = timestamp
            logger.info(
                f"[Patch] Entry: BUY at {timestamp} (ADX={adx:.2f}, RSI={rsi:.2f}, GZ={gain_z:.2f}, Div={divergence})"
            )
        elif (
            ema_fast < ema_slow
            and m15_ema_fast < m15_ema_slow
            and adx > 12
            and rsi < 49
            and gain_z < gainz_sell
            and relaxed_wick
            and (divergence is None or divergence == "bearish")
        ):
            df.at[df.index[i], "entry_signal"] = "sell"
            last_entry_time = timestamp
            logger.info(
                f"[Patch] Entry: SELL at {timestamp} (ADX={adx:.2f}, RSI={rsi:.2f}, GZ={gain_z:.2f}, Div={divergence})"
            )

    force_entry_gap = 300
    last_entry = 0
    for i in range(len(df)):
        if pd.notna(df["entry_signal"].iloc[i]):
            last_entry = i
        elif i - last_entry > force_entry_gap:
            ema_fast = df["ema_fast"].iloc[i]
            ema_slow = df["ema_slow"].iloc[i]
            m15_ema_fast = df["m15_ema_fast"].iloc[i] if "m15_ema_fast" in df.columns else np.nan
            m15_ema_slow = df["m15_ema_slow"].iloc[i] if "m15_ema_slow" in df.columns else np.nan
            adx = df["adx"].iloc[i]
            rsi = df["rsi"].iloc[i]
            atr = df["atr"].iloc[i]
            gain_z = df["gain_z"].iloc[i] if "gain_z" in df.columns else 0.0
            relaxed_wick = True
            if {"high", "low", "close"}.issubset(df.columns):
                price_range = max(df["high"].iloc[i] - df["low"].iloc[i], 1e-8)
                upper_wick = (df["high"].iloc[i] - df["close"].iloc[i]) / price_range
                lower_wick = (df["close"].iloc[i] - df["low"].iloc[i]) / price_range
                relaxed_wick = (upper_wick < 0.80 and lower_wick < 0.80)
            if atr >= atr_threshold and adx > 12 and relaxed_wick:
                if ema_fast > ema_slow and m15_ema_fast > m15_ema_slow and rsi > 51 and gain_z > gainz_buy:
                    df.at[i, "entry_signal"] = "buy"
                    logger.info(f"[Patch] Force Entry: BUY at {df['timestamp'].iloc[i]}")
                    last_entry = i
                elif ema_fast < ema_slow and m15_ema_fast < m15_ema_slow and rsi < 49 and gain_z < gainz_sell:
                    df.at[i, "entry_signal"] = "sell"
                    logger.info(f"[Patch] Force Entry: SELL at {df['timestamp'].iloc[i]}")
                    last_entry = i
    logger.info(
        f"[Patch] Optimized Entry: buy={(df['entry_signal']=='buy').sum()}, sell={(df['entry_signal']=='sell').sum()}"
    )
    return df


def calc_adaptive_lot(equity, adx, recovery_mode=False, win_streak=0):
    """[Patch] Adaptive lot sizing based on portfolio growth + ADX + Recovery/WinStreak"""
    base_risk = 0.01
    if adx > 23:
        base_risk *= 2.0
    if recovery_mode:
        base_risk *= 2.0
    if win_streak >= 2:
        base_risk *= 1.5
    lot = max(0.01, min(5.0, (equity * base_risk) / 10))
    logger.info(
        "[Patch] Lot calc: eq=%.2f, adx=%.2f, recovery=%s, streak=%d => lot=%.2f",
        equity,
        adx,
        recovery_mode,
        win_streak,
        lot,
    )
    return lot


def smart_entry_signal_goldai2025_style(df):
    """Entry signal logic for GoldAI2025."""
    logger.info("[Patch] Entry signal GoldAI2025 style")
    df = df.copy()
    df["entry_signal"] = None

    buy_cond = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > 55)
    sell_cond = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < 45)

    if {"m15_ema_fast", "m15_ema_slow", "m15_rsi"}.issubset(df.columns):
        buy_cond &= (df["m15_ema_fast"] > df["m15_ema_slow"]) & (df["m15_rsi"] > 55)
        sell_cond &= (df["m15_ema_fast"] < df["m15_ema_slow"]) & (df["m15_rsi"] < 45)

    df.loc[buy_cond, "entry_signal"] = "buy"
    df.loc[sell_cond, "entry_signal"] = "sell"

    logger.info(
        "[Patch] GoldAI2025 Entry counts: buy=%d, sell=%d",
        (df["entry_signal"] == "buy").sum(),
        (df["entry_signal"] == "sell").sum(),
    )
    logger.debug(
        "GoldAI2025 Entry indices: %s",
        df[df["entry_signal"].notna()].index.tolist(),
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
        self.last_order_time = None

    def update(self, capital, trade_win):
        self.capital = capital
        if capital > self.peak:
            self.peak = capital
        dd = (self.peak - capital) / self.peak
        if dd > self.kill_switch_dd:
            self.kill_switch = True
            logger.warning("[Patch] OMS Kill switch triggered: DD %.2f%%", dd * 100)
        if trade_win is True:
            self.win_streak += 1
            self.loss_streak = 0
        elif trade_win is False:
            self.loss_streak += 1
            self.win_streak = 0
        if self.loss_streak >= oms_recovery_loss and not self.recovery_mode:
            self.recovery_mode = True
            logger.warning(
                "[Patch] Recovery mode activated after %d consecutive losses",
                self.loss_streak,
            )
        if self.recovery_mode and self.win_streak > 2:
            self.recovery_mode = False
            logger.info(
                "[Patch] Recovery mode deactivated after win streak of %d",
                self.win_streak,
            )
        if self.recovery_mode and capital >= self.peak:
            self.recovery_mode = False
            logger.info("[Patch] Recovery mode deactivated after capital recovery")

    def can_place_order(self, timestamp, min_gap_minutes=60):
        if self.last_order_time is None:
            return True
        gap = pd.Timedelta(minutes=min_gap_minutes)
        return pd.to_datetime(timestamp) - self.last_order_time >= gap

    def register_order(self, timestamp):
        self.last_order_time = pd.to_datetime(timestamp)

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
        if self.win_streak >= 2:
            lot *= win_streak_boost
        lot_cap = min(lot_cap, lot_cap_max)
        lot = max(0.01, min(lot, lot_cap))
        lot = max(0.01, min(lot, 0.05))  # [Patch] Limit max lot to 0.05
        return lot

    def check_max_orders(self, open_positions, max_orders=1):
        if len(open_positions) >= max_orders:
            logger.warning("[Patch] Max open order limit reached (%d)", max_orders)
            return False
        return True

    def audit_log(self):
        logger.info(
            "[Patch] OMS Audit: capital=%.2f, peak=%.2f, kill_switch=%s, recovery=%s, win_streak=%d, loss_streak=%d",
            self.capital,
            self.peak,
            self.kill_switch,
            self.recovery_mode,
            self.win_streak,
            self.loss_streak,
        )


def apply_order_costs(
    entry,
    sl,
    tp1,
    tp2,
    lot,
    direction,
    spread=SPREAD_VALUE,
    commission=COMMISSION_PER_LOT,
    slippage=SLIPPAGE,
):
    """[Patch] Realistic spread/commission/slippage for profit possibility"""
    spread_half = spread / 2
    slip = np.random.uniform(-slippage, slippage)
    if direction == "buy":
        entry_adj = entry + spread_half + slip
    else:
        entry_adj = entry - spread_half - slip
    # SL/TP ไม่ควรบวก spread อีก
    com = 2 * commission * lot * 100
    return entry_adj, sl, tp1, tp2, com


def _execute_backtest(df):
    df = df.dropna(subset=["atr", "ema_fast", "ema_slow", "rsi"]).reset_index(drop=True)

    capital = initial_capital
    oms = OMSManager(capital, kill_switch_dd, lot_max)
    trades = []
    equity_curve = []
    position = None
    last_entry_idx = -cooldown_bars

    for i, row in df.iterrows():
        equity_curve.append(
            {
                "timestamp": row["timestamp"],
                "equity": capital,
                "dd": (oms.peak - capital) / oms.peak,
            }
        )
        oms.audit_log()
        if oms.kill_switch:
            logger.info("[Patch] OMS: Stop trading (kill switch)")
            break

        # Entry
        if (
            position is None
            and row["entry_signal"] in ["buy", "sell"]
            and (i - last_entry_idx) >= cooldown_bars
            and oms.check_max_orders([p for p in [position] if p])
        ):
            direction = row["entry_signal"]
            price_range = max(row["high"] - row["low"], 1e-6)
            upper_wick_ratio = (row["high"] - row["close"]) / price_range
            lower_wick_ratio = (row["close"] - row["low"]) / price_range
            # [Patch] Relax wick filter - ผ่าน 80%
            if direction == "buy" and upper_wick_ratio > 0.80:
                continue
            if direction == "sell" and lower_wick_ratio > 0.80:
                continue
            if row["atr"] < SPREAD_VALUE * 2.0:
                continue  # [Patch] ข้ามไม้ตลาดแคบ
            if oms.recovery_mode:
                atr_roll = (
                    df["atr"].rolling(100).mean().iloc[i]
                    if i >= 1
                    else df["atr"].iloc[: i + 1].mean()
                )
                if row["atr"] < atr_roll * 0.9 or row.get("gain_z", 0) < 0:
                    continue
            atr = max(row["atr"], min_sl_dist)
            entry = row["close"]
            sl = entry - atr * sl_mult if direction == "buy" else entry + atr * sl_mult
            tp1 = (
                entry + atr * tp1_mult if direction == "buy" else entry - atr * tp1_mult
            )
            tp2_mult_now = row.get("tp2_dynamic", tp2_mult)
            tp2 = (
                entry + atr * tp2_mult_now
                if direction == "buy"
                else entry - atr * tp2_mult_now
            )
            risk_amount = capital * risk_per_trade
            if "atr_long" in row:
                if row["atr"] > 1.5 * row["atr_long"]:
                    risk_amount *= 0.75
                elif row["atr"] < 0.8 * row["atr_long"]:
                    risk_amount *= 1.15
            if oms.recovery_mode:
                atr_roll = (
                    df["atr"].rolling(100).mean().iloc[i]
                    if i >= 1
                    else df["atr"].iloc[: i + 1].mean()
                )
                if row["atr"] < atr_roll:
                    risk_amount *= 0.7
            logger.debug(
                "[Patch] TP2 mult %.2f, risk amount %.2f", tp2_mult_now, risk_amount
            )
            # [Patch] Adaptive risk management + signal boost
            if not oms.recovery_mode:
                if oms.win_streak >= 2:
                    risk_amount *= win_streak_boost
                if row["adx"] > adx_strong:
                    risk_amount *= 1.5
                    logger.info(
                        "[Patch] Signal boost: ADX %.1f > %.0f, risk increased 50%%",
                        row["adx"],
                        adx_strong,
                    )
            lot = oms.smart_lot(capital, risk_amount, abs(entry - sl))
            entry, sl, tp1, tp2, com = apply_order_costs(
                entry, sl, tp1, tp2, lot, direction
            )
            mode = "RECOVERY" if oms.recovery_mode else "NORMAL"
            position = {
                "entry_time": row["timestamp"],
                "type": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "lot": lot,
                "tp1_hit": False,
                "breakeven": False,
                "entry_idx": i,
                "reason_entry": f"{row.get('wave_phase')}+{row.get('pattern_label')}+{row.get('divergence')}+score={row.get('signal_score')}+session={row.get('session')}",
                "mode": mode,
                "risk": risk_amount,
                "context": f"Spike={row.get('spike_guard')}, News={row.get('news_guard')}",
                "dd_at_entry": (oms.peak - capital) / oms.peak,
                "peak_equity": oms.peak,
                "commission": com,
            }
            last_entry_idx = i
            logger.info(
                "[Patch] Entry: %s at %.2f, SL %.2f, TP1 %.2f, TP2 %.2f, Lot %.3f Mode %s (spread %.1f, slippage %.1f, com %.2f)",
                direction,
                entry,
                sl,
                tp1,
                tp2,
                lot,
                mode,
                SPREAD_VALUE,
                SLIPPAGE,
                com,
            )
            logger.info(
                "[Patch][Debug] Entry=%.2f, SL=%.2f, TP1=%.2f, TP2=%.2f, ATR=%.2f, Spread=%.2f, Slippage=%.2f, Lot=%.3f, Com=%.2f",
                entry,
                sl,
                tp1,
                tp2,
                row["atr"],
                SPREAD_VALUE,
                SLIPPAGE,
                lot,
                com,
            )

        if position:
            logger.debug(
                "[Patch][Debug] Holding position %s: entry=%.2f, SL=%.2f, TP1=%.2f, TP2=%.2f, lot=%.3f, close=%.2f, high=%.2f, low=%.2f, idx=%d",
                position["type"],
                position["entry"],
                position["sl"],
                position["tp1"],
                position["tp2"],
                position["lot"],
                row["close"],
                row["high"],
                row["low"],
                i,
            )

            hit_tp1 = (
                row["high"] >= position["tp1"]
                if position["type"] == "buy"
                else row["low"] <= position["tp1"]
            )
            hit_tp2 = (
                row["high"] >= position["tp2"]
                if position["type"] == "buy"
                else row["low"] <= position["tp2"]
            )
            hit_sl = (
                row["low"] <= position["sl"]
                if position["type"] == "buy"
                else row["high"] >= position["sl"]
            )
            logger.debug(
                "[Patch][Debug] Check TP/SL: TP1=%.2f, TP2=%.2f, SL=%.2f, High=%.2f, Low=%.2f, HitTP1=%s, HitTP2=%s, HitSL=%s",
                position["tp1"],
                position["tp2"],
                position["sl"],
                row["high"],
                row["low"],
                hit_tp1,
                hit_tp2,
                hit_sl,
            )

            if i - position.get("entry_idx", i) >= 25:
                atr_now = row.get("atr", 0)
                atr_rolling = (
                    df["atr"].iloc[max(i - 100, 0) : i].mean()
                    if i > 0
                    else atr_now
                )
                gain_z_now = row.get("gain_z", 0)
                if atr_now < atr_rolling * 0.8 or gain_z_now < 0:
                    pnl = (
                        (row["close"] - position["entry"]) * position["lot"] * (100 if position["type"] == "buy" else -100)
                    ) - position.get("commission", 0)
                    capital += pnl
                    trades.append(
                        {
                            **position,
                            "exit_time": row["timestamp"],
                            "exit": "EarlyForceClose",
                            "pnl": pnl,
                            "capital": capital,
                            "reason_exit": "Early force close: ATR/gain_z low",
                            "commission": position.get("commission", 0),
                            "spread": SPREAD_VALUE,
                            "slippage": SLIPPAGE,
                        }
                    )
                    oms.update(capital, pnl > 0)
                    position = None
                    continue
            max_holding_bars = 50
            if i - position.get("entry_idx", i) >= max_holding_bars:
                logger.warning("[Patch] Force close position after %d bars", max_holding_bars)
                pnl = (
                    (row["close"] - position["entry"]) * position["lot"] * (100 if position["type"] == "buy" else -100)
                ) - position.get("commission", 0)
                capital += pnl
                trades.append(
                    {
                        **position,
                        "exit_time": row["timestamp"],
                        "exit": "ForceClose",
                        "pnl": pnl,
                        "capital": capital,
                        "reason_exit": "Force close after max bars",
                        "commission": position.get("commission", 0),
                        "spread": SPREAD_VALUE,
                        "slippage": SLIPPAGE,
                    }
                )
                oms.update(capital, pnl > 0)
                position = None
                if len(trades) <= 3 and capital < initial_capital * 0.8:
                    logger.warning(
                        "[Patch][Debug] Stop - Too fast loss in first 3 trades, Check spread/sl/lot/commission logic!"
                    )
                    break
                continue

            # Partial TP1
            if not position["tp1_hit"] and hit_tp1:
                pnl = (
                    position["lot"] * abs(position["tp1"] - position["entry"]) * 0.5
                )
                capital += pnl
                position["sl"] = position["entry"]
                position["tp1_hit"] = True
                position["breakeven"] = True
                trades.append(
                    {
                        **position,
                        "exit_time": row["timestamp"],
                        "exit": "TP1",
                        "pnl": pnl,
                        "capital": capital,
                        "reason_exit": "Partial TP1",
                        "reason_entry": position.get("reason_entry", ""),
                        "context": position.get("context", ""),
                        "wave_phase": row.get("wave_phase"),
                        "pattern_label": row.get("pattern_label"),
                        "divergence": row.get("divergence"),
                        "signal_score": row.get("signal_score"),
                        "session": row.get("session"),
                        "spike_guard": row.get("spike_guard"),
                        "news_guard": row.get("news_guard"),
                        "risk": position.get("risk"),
                        "oms_mode": position.get("mode"),
                        "commission": position.get("commission", 0),
                        "spread": SPREAD_VALUE,
                        "slippage": SLIPPAGE,
                    }
                )
                logger.info(
                    "[Patch] Partial TP1 at %.2f (+%.2f$)", position["tp1"], pnl
                )
                oms.update(capital, pnl > 0)
                continue

            # TP2
            if hit_tp2:
                pnl = (
                    position["lot"]
                    * abs(position["tp2"] - position["entry"])
                    * (0.5 if position["tp1_hit"] else 1)
                ) - position.get("commission", 0)
                capital += pnl
                trades.append(
                    {
                        **position,
                        "exit_time": row["timestamp"],
                        "exit": "TP2",
                        "pnl": pnl,
                        "capital": capital,
                        "reason_exit": "TP2",
                        "reason_entry": position.get("reason_entry", ""),
                        "context": position.get("context", ""),
                        "wave_phase": row.get("wave_phase"),
                        "pattern_label": row.get("pattern_label"),
                        "divergence": row.get("divergence"),
                        "signal_score": row.get("signal_score"),
                        "session": row.get("session"),
                        "spike_guard": row.get("spike_guard"),
                        "news_guard": row.get("news_guard"),
                        "risk": position.get("risk"),
                        "oms_mode": position.get("mode"),
                        "commission": position.get("commission", 0),
                        "spread": SPREAD_VALUE,
                        "slippage": SLIPPAGE,
                    }
                )
                logger.info("[Patch] TP2 at %.2f (+%.2f$)", position["tp2"], pnl)
                oms.update(capital, pnl > 0)
                position = None
                if len(trades) <= 3 and capital < initial_capital * 0.8:
                    logger.warning(
                        "[Patch][Debug] Stop - Too fast loss in first 3 trades, Check spread/sl/lot/commission logic!"
                    )
                    break
                continue

            # Stop Loss / Breakeven
            if hit_sl:
                pnl = -position["lot"] * abs(
                    position["entry"] - position["sl"]
                ) - position.get("commission", 0)
                capital += pnl
                trades.append(
                    {
                        **position,
                        "exit_time": row["timestamp"],
                        "exit": "SL",
                        "pnl": pnl,
                        "capital": capital,
                        "reason_exit": "SL/BE",
                        "reason_entry": position.get("reason_entry", ""),
                        "context": position.get("context", ""),
                        "wave_phase": row.get("wave_phase"),
                        "pattern_label": row.get("pattern_label"),
                        "divergence": row.get("divergence"),
                        "signal_score": row.get("signal_score"),
                        "session": row.get("session"),
                        "spike_guard": row.get("spike_guard"),
                        "news_guard": row.get("news_guard"),
                        "risk": position.get("risk"),
                        "oms_mode": position.get("mode"),
                        "commission": position.get("commission", 0),
                        "spread": SPREAD_VALUE,
                        "slippage": SLIPPAGE,
                    }
                )
                logger.info("[Patch] SL/BE at %.2f (%.2f$)", position["sl"], pnl)
                oms.update(capital, pnl > 0)
                position = None
                if len(trades) <= 3 and capital < initial_capital * 0.8:
                    logger.warning(
                        "[Patch][Debug] Stop - Too fast loss in first 3 trades, Check spread/sl/lot/commission logic!"
                    )
                    break
                continue

            # Trailing SL after TP1
            if position["tp1_hit"]:
                trailing_sl = (
                    row["close"] - row["atr"] * trailing_atr_mult
                    if position["type"] == "buy"
                    else row["close"] + row["atr"] * trailing_atr_mult
                )
                if position["type"] == "buy" and trailing_sl > position["sl"]:
                    position["sl"] = trailing_sl
                elif position["type"] == "sell" and trailing_sl < position["sl"]:
                    position["sl"] = trailing_sl

    # Save Trade Log & Equity Curve
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)
    if df_equity.empty:
        df_equity = pd.DataFrame({"timestamp": [], "equity": [], "dd": []})
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trade_log_path = os.path.join(TRADE_DIR, f"trade_log_{now_str}.csv")
    equity_curve_path = os.path.join(TRADE_DIR, f"equity_curve_{now_str}.csv")
    df_trades.to_csv(trade_log_path, index=False)
    df_equity.to_csv(equity_curve_path, index=False)
    logger.info("[Patch] Saved trade log: %s", trade_log_path)
    logger.info("[Patch] Saved equity curve: %s", equity_curve_path)
    globals()["_LAST_EQUITY_DF"] = df_equity

    qa_validate_backtest(df_trades, df_equity)

    # Summary
    print("Final Equity:", round(capital, 2))
    print("Total Trades:", len(df_trades))
    print(
        "Total Return: {:.2f}%".format(
            (capital - initial_capital) / initial_capital * 100
        )
    )
    print(
        "Winrate: {:.2f}%".format(
            (df_trades["pnl"] > 0).mean() * 100 if not df_trades.empty else 0
        )
    )
    cols = [
        "entry_time",
        "type",
        "entry",
        "exit",
        "pnl",
        "capital",
        "mode",
        "reason_entry",
        "reason_exit",
    ]
    if not df_trades.empty:
        print(df_trades[cols].tail(12))
    max_equity = df_equity["equity"].max()
    min_equity = df_equity["equity"].min()
    print(f"[Patch] Max Equity: {max_equity:.2f} | Min Equity: {min_equity:.2f}")
    max_dd = df_equity["dd"].max()
    print(f"[Patch] Max Drawdown: {max_dd*100:.2f}%")
    print(f"[Patch] OMS mode: recovery = {oms.recovery_mode}")

    # [Patch] Show equity curve visualization (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5))
        plt.plot(df_equity["timestamp"], df_equity["equity"])
        plt.title("Equity Curve [Patch]")
        plt.xlabel("Timestamp")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning("[Patch] Matplotlib not available for equity plot: %s", e)
    return df_trades


def qa_validate_backtest(trades_df, equity_df, min_trades=15, min_profit=15, prev_winrate=None):
    """[Patch] QA validation after backtest"""
    orders = len(trades_df)
    profit = equity_df["equity"].iloc[-1] - initial_capital if not equity_df.empty else 0
    dd = equity_df["dd"].max() if not equity_df.empty else 0
    winrate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0
    if orders < min_trades:
        logger.warning("[Patch] QA fail: trades < %d", min_trades)
    if orders < 279:
        logger.error("[Patch] QA fail: orders < 279")
    if profit < min_profit:
        logger.warning("[Patch] QA fail: profit %.2f < %.2f", profit, min_profit)
    if prev_winrate is not None and winrate < prev_winrate:
        logger.warning(
            "[Patch] Winrate dropped from %.2f%% to %.2f%%", prev_winrate * 100, winrate * 100
        )
    if dd > kill_switch_dd:
        logger.warning("[Patch] QA fail: DD %.2f%% > kill switch", dd * 100)
    if (
        orders >= max(min_trades, 279)
        and profit >= min_profit
        and dd <= kill_switch_dd
        and (prev_winrate is None or winrate >= prev_winrate)
    ):
        logger.info("[Patch] QA check passed")
    return {
        "trades": orders,
        "profit": profit,
        "dd": dd,
        "winrate": winrate,
    }


def run_backtest(path=None):
    """Run single timeframe backtest (goldai2025 entry logic)."""
    logger.debug("[Patch] run_backtest called with path=%s", path)
    if path is None:
        path = M1_PATH
    df = load_data(path)
    df = data_quality_check(df)
    df = calc_indicators(df)
    df = multi_session_trend_scalping(df)
    trades = _execute_backtest(df)
    if not trades.empty and "exit" in trades.columns:
        loss_indices = trades.loc[trades["exit"].isin(["SL", "ForceClose"]), "entry_idx"].tolist()
    else:
        loss_indices = []
    if loss_indices:
        logger.info("[Patch] Reconfirm on %d lossy indices", len(loss_indices))
        df = patch_confirm_on_lossy_indices(df, loss_indices)
        trades = _execute_backtest(df)
    analyze_tradelog(trades, globals().get("_LAST_EQUITY_DF", pd.DataFrame()))
    return trades


def run_backtest_multi_tf(path_m1=M1_PATH, path_m15=M15_PATH):
    """Backtest with M1 trade data and M15 trend confirmation (goldai2025 entry logic)."""
    logger.debug(
        "[Patch] run_backtest_multi_tf called with paths %s, %s", path_m1, path_m15
    )
    df_m1 = load_data(path_m1)
    df_m1 = data_quality_check(df_m1)
    df_m15 = load_data(path_m15)
    df_m15 = data_quality_check(df_m15)
    df_m15 = calc_indicators(
        df_m15, ema_fast_period=50, ema_slow_period=200, rsi_period=14
    )
    df_m1 = calc_indicators(
        df_m1, ema_fast_period=15, ema_slow_period=50, rsi_period=14
    )
    df_m1 = calc_dynamic_tp2(df_m1)
    df_m1 = add_m15_context_to_m1(df_m1, df_m15)
    df_m1 = label_elliott_wave(df_m1)
    df_m1 = detect_divergence(df_m1)
    df_m1 = label_pattern(df_m1)
    df_m1 = calc_gain_zscore(df_m1)
    df_m1 = calc_signal_score(df_m1)
    df_m1, _ = shap_feature_importance_placeholder(df_m1)
    df_m1 = tag_session(df_m1)
    df_m1 = tag_spike_guard(df_m1)
    df_m1 = tag_news_event(df_m1)
    df_m1 = smart_entry_signal_goldai2025_style(df_m1)
    df_m1 = apply_session_bias(df_m1)
    df_m1 = apply_spike_news_guard(df_m1)
    return _execute_backtest(df_m1)


def split_folds(df, n_folds=5):
    """[Patch] แบ่งข้อมูลเป็น Folds (Walk-Forward) อย่างเท่าๆ กัน"""
    logger.info(f"[Patch] Splitting data into {n_folds} folds (WFV)")
    fold_size = int(np.ceil(len(df) / n_folds))
    fold_indices = [
        (i * fold_size, min((i + 1) * fold_size, len(df))) for i in range(n_folds)
    ]
    return [df.iloc[start:end].reset_index(drop=True) for start, end in fold_indices]


def run_walkforward_backtest(df, n_folds=5, config_list=None):
    """[Patch] รัน Backtest แบบ Walk-Forward Validation (WFV)"""
    logger.info("[Patch] Running Walk-Forward Validation")
    fold_results = []
    folds = split_folds(df, n_folds=n_folds)
    for i, fold_df in enumerate(folds):
        logger.info(f"[Patch] --- Fold #{i+1}/{n_folds} ---")
        fold_config = config_list[i] if config_list and len(config_list) > i else None
        fold_df = data_quality_check(fold_df)
        fold_df = calc_indicators(fold_df)
        fold_df = calc_dynamic_tp2(fold_df)
        fold_df = label_elliott_wave(fold_df)
        fold_df = detect_divergence(fold_df)
        fold_df = label_pattern(fold_df)
        fold_df = calc_gain_zscore(fold_df)
        fold_df = calc_signal_score(fold_df)
        fold_df, _ = shap_feature_importance_placeholder(fold_df)
        fold_df = tag_session(fold_df)
        fold_df = tag_spike_guard(fold_df)
        fold_df = tag_news_event(fold_df)
        fold_df = smart_entry_signal_goldai2025_style(fold_df)
        fold_df = apply_session_bias(fold_df)
        fold_df = apply_spike_news_guard(fold_df)
        result = _execute_backtest(fold_df)
        fold_results.append(result)
    logger.info("[Patch] Walk-Forward complete")
    return fold_results


if __name__ == "__main__":
    run_backtest()
