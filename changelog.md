# 📘 changelog.md — Gold AI Strategy System

## [v1.0.0] — 2025-05-22
### Added
- [Patch] Introduced realistic backtest system
  - ✅ Spread 80 point (0.80 USD)
  - ✅ Slippage 0.05
  - ✅ Commission: 0.10 USD per 0.01 lot (applied on exit only)
- [Patch] Integrated `EMA35 + MACD Divergence` signal system (multi-trigger)
- [Patch] Created agent.md specification for Codex/Agent integration

## [v1.0.1] — 2025-05-24
### Changed
- Updated risk management parameters (recovery multiplier 2.0, trailing ATR 1.3, kill switch 30%)
- Reduced force entry gap to 200 and simplified entry signal logic

## [v1.0.2] — 2025-05-25
### Changed
- Replaced WFA CLI with Walk-Forward Validation approach.

