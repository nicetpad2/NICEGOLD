# 📘 changelog.md — Gold AI Strategy System

## [v1.0.0] — 2025-05-22
### Added
- [Patch] Introduced realistic backtest system
  - ✅ Spread 80 point (0.80 USD)
  - ✅ Slippage 0.05
  - ✅ Commission: 0.10 USD per 0.01 lot (applied on exit only)
- [Patch] Integrated `EMA35 + MACD Divergence` signal system (multi-trigger)
- [Patch] Created agent.md specification for Codex/Agent integration

## [v1.1.0] — 2025-05-23
### Added
- Spike Guard and volatility filter
- Histogram divergence threshold with valid signals
- Cooldown and walk-forward fold simulation
- Trailing stop and break-even logic
- Kill switch on capital below $50

## [v1.1.1] — 2025-05-24
### Changed
- Expanded EMA touch range to ±0.3%
- Reduced cooldown bars to 1 and adjusted entry logic
- Added overlot and commission limits
- Improved BE-SL and trailing stop handling

## [v1.2.0] — 2025-05-25
### Added
- Entry signal scoring system with RSI and Gain Z
- Forced entry strategy with spike filter
- Re-entry logic and recovery lot adjustment
- Partial take profit with two targets
- Walk-forward fold parameters

## [v1.3.0] — 2025-05-26
### Added
- Trend confirmation with EMA10 vs EMA35
- Wave phase labeling system
- Elliott-DivMeta Hybrid signal
- TP1 at 0.8R with break-even stop

## [v1.3.1] — 2025-05-27
### Added
- Elliott Wave Dynamic Labeller v2.1 with zigzag + RSI divergence
- Entry Signal enhanced by wave phase filtering

## [v1.4.0] — 2025-05-28
### Added
- Entry Score Based System (`generate_entry_score_signal`)

## [v1.5.0] — 2025-05-28
### Added
- Unit tests for full coverage (100%)

## [v1.6.0] — 2025-05-29
### Added
- Patch G33: Fold #1 parameter optimization, MACD cross divergence entry, spike score for force entry, and reversal scoring

## [v1.7.0] — 2025-05-30
### Added
- Patch G34: Wave phase divergence logic with pd.isna fix, spike_score calculation, and force entry cooldown tweak

## [v1.8.0] — 2025-05-31
### Added
- Patch G35: `load_data` now parses Buddhist `Date` and non zero-padded `Timestamp`

## [v1.9.0] — 2025-06-01
### Added
- Patch G36: ModernScalpingXAUUSD v1.0 algorithm with ML signal and dynamic stop

## [v1.9.1] — 2025-06-02
### Changed
- Integrate ModernScalpingXAUUSD logic into `nicegold.py`

## [v1.9.2] — 2025-06-03
### Changed
- Updated CSV path to `/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv`

## [v1.9.3] — 2025-06-04
### Added
- Configurable parameters via `config.yaml`
- Trade time filter (13:00–22:00 UTC)
- Trade log export with drawdown tracking

## [v1.9.4] — 2025-06-05
### Added
- Patch G33.3: Trade visualization plot and walk-forward validation helper


## [v1.9.5] — 2025-06-06
### Changed
- Combined all unit tests into a single `test_all.py` for easier maintenance
=======

## [v1.9.6] — 2025-06-07
### Added
- Patch G33.4: Equity curve tracking saved to `equity_curve.csv`
- Patch G33.5: Reduced ML thresholds and added debug logging


## [v1.9.7] — 2025-06-08
### Added
- Extensive debug logging across all functions


## [v1.9.8] — 2025-06-09
### Changed
- Documented debug logging requirement in agent.md
=======

## [v1.9.9] — 2025-06-10
### Added
- Patch G-RESTOREENTRYV1: Smart entry signals, OMS drawdown guard and partial TP
- Added `generate_smart_signal`, `check_drawdown`, `backtest_with_partial_tp`, and `run`
- Extra unit tests for new features

