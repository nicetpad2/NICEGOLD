import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Helper functions

def is_in_session(current_time, sessions):
    if sessions is None:
        return True
    hour = current_time.hour
    for start, end in sessions:
        if start <= hour < end:
            return True
    return False

def breakout_condition(price_data):
    return price_data.get('breakout', False)

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
    last_direction: str = 'BUY'

DD_THRESHOLD = 0.2
SAFE_THRESHOLD = 0.05
base_stop_distance = 1.0
base_tp_distance = 2.0
partial_tp_distance = 1.5
break_even_distance = 1.0
trailing_atr_multiplier = 1.0


def generate_signal(price_data, indicators, *, current_time=None,
                    allowed_sessions=None, intended_side='BUY', force_entry=False):
    """Generate trade signal with trend, momentum and session checks."""
    if force_entry:
        logger.debug("Force entry active")
        return True
    session_ok = is_in_session(current_time, allowed_sessions)
    trend_ok = (
        indicators['EMA50'] > indicators['EMA200']
        if intended_side == 'BUY'
        else indicators['EMA50'] < indicators['EMA200']
    )
    momentum_ok = indicators['ADX'] > 25 and indicators.get('ADX_trend') == intended_side
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
    """Calculate lot size based on risk percentage."""
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
    if order.partial_tp_level and price >= order.partial_tp_level and not order.partial_taken:
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
        trail_distance = trailing_atr_multiplier * indicators.get('ATR', 0)
        new_sl = max(order.stop_loss, price - trail_distance)
        if new_sl > order.stop_loss:
            set_stop_loss(order, new_sl)
            logger.debug(
                "[Patch] Trailing SL updated for order %s to %.2f",
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
        if generate_signal(price_data or {}, indicators or {}, current_time=datetime.now(), allowed_sessions=None, intended_side=portfolio.last_direction.lower()) and portfolio.recovery_active:
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
