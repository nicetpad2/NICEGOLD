import unittest
from datetime import datetime
import runtime

class TestRuntime(unittest.TestCase):
    def test_generate_signal_true(self):
        price = {'breakout': True}
        ind = {'EMA50': 2, 'EMA200': 1, 'ADX': 30, 'ADX_trend': 'BUY'}
        now = datetime(2020,1,1,10)
        self.assertTrue(runtime.generate_signal(price, ind, current_time=now, allowed_sessions=[(8,12)], intended_side='BUY'))

    def test_calculate_position_size(self):
        lot = runtime.calculate_position_size(1000, 10, 9.5, 0.01)
        self.assertGreater(lot, 0)

    def test_on_order_execute_and_update(self):
        o = runtime.Order(id=1, entry_price=10)
        runtime.on_order_execute(o)
        self.assertIsNotNone(o.stop_loss)
        runtime.on_price_update(o, 11.5, indicators={'ATR':1})
        self.assertTrue(o.partial_taken)

    def test_manage_recovery(self):
        p = runtime.Portfolio(equity=1000, drawdown=0.3, last_trade_loss=True, last_lot=0.2)
        runtime.manage_recovery(p, {'breakout': True}, {'EMA50':2,'EMA200':1,'ADX':30,'ADX_trend':'buy'})
        self.assertTrue(p.recovery_active)
        p.drawdown = 0.01
        runtime.manage_recovery(p)
        self.assertFalse(p.recovery_active)

if __name__ == '__main__':
    unittest.main()
