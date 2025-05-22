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
