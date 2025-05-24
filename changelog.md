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

## [v1.9.10] — 2025-06-11
### Fixed
- Patch G-FixNoTrade-NaNATR: Drop rows with NaN indicators before backtest
- `run()` now drops NaN ATR/RSI/EMA35 rows before generating signals

## [v1.9.11] — 2025-06-12
### Added
- Script `convert_csv_ad_to_be` for converting AD dates to Buddhist Era
- Additional unit test for the new function

## [v1.9.12] — 2025-06-13
### Changed
- Expanded indicator preprocessing with RSI, Gain_Z, and Pattern_Label
- Updated hybrid entry logic and force entry conditions
- Added win streak boost for position sizing
- Unit tests updated for new behavior

## [v1.9.13] — 2025-06-14
### Changed
- Added `optimize_memory` for efficient DataFrame handling on Colab L4
- Refactored `load_data` and `detect_macd_divergence` for vectorized operations
- Extra unit test for the new function

## [v1.9.14] — 2025-06-15
### Fixed
- Patch G-Fix1: removed chained assignment warning for RSI fillna
- Updated commission calculation to $0.10 per 0.01 lot without capital limit
- Added debug log tail to inspect entry signals


## [v1.9.15] — 2025-06-15
### Fixed
- Removed deprecated 'T' frequency in tests
- Specified `fill_method=None` in `pct_change` to silence FutureWarning

=======

## [v1.9.16] — 2025-06-16
### Changed
- ปรับ `run_backtest_cli` ใช้ความเสี่ยงต่อไม้ 5% และลดสเปรดสมจริง
- เพิ่มตัวกรองเวลาซื้อขาย 13:00-22:00 และคำนวณ SL/TP ตาม ATR


## [v1.9.17] — 2025-06-17
### Changed
- เพิ่ม ATR คำนวณและบันทึกใน run_backtest_cli
- ปรับ SL/TP ให้ใช้ ATR
- เพิ่ม logger.info ติดตามสถานะเปิดสถานะและ TP1
- ไม่หักค่าคอมมิชชั่นเมื่อ TP1

## [v1.9.18] — 2025-06-18
### Added
- Patch G-Fix2: ระบบบันทึก trade log และ equity curve พร้อมพาธเต็มและ timestamp
- บันทึกกราฟ Equity Curve หลังจบการทดสอบ
## [v1.9.19] — 2025-06-19
### Added
- เพิ่มระบบตรวจสอบ drawdown และความผันผวนก่อนเข้าออเดอร์
- ปรับขนาด TP และขนาด lot ตาม win/loss streak พร้อมเหตุผลการเข้าออเดอร์
- Trailing SL ตาม ATR หลัง TP1
## [v1.9.20] — 2025-06-20
### Fixed
- แก้ FutureWarning `fillna(method='bfill')`
- แก้ NameError `current_drawdown` ใน `run_backtest_cli`
## [v1.9.21] — 2025-06-21
### Changed
- ปรับปรุง `run_backtest_cli` ให้มี Partial TP และ Trailing Stop ตาม ATR
- พิมพ์ผลสรุปรายการเทรดพร้อมจำนวนและเหตุผลเข้าออกออเดอร์

## [v1.9.22] — 2025-06-22
### Added
- ระบบ Trend + Confirm Bar ก่อนเข้าออเดอร์
- แจ้งเตือนหาก Winrate 50 ไม้ล่าสุดต่ำกว่า 35%
- สรุปสาเหตุเข้าออเดอร์ที่ทำกำไรและขาดทุน Top 3

## [v1.9.23] — 2025-06-23
### Changed
- เพิ่ม kill switch หยุดเทรดเมื่อ Equity ต่ำกว่า $50
- ข้ามแท่งที่มี volatility spike
- เข้าตลาดเฉพาะช่วง 13:00-22:00
- ปรับ lot ตาม streak แพ้/ชนะ และเพิ่ม TP ตาม ADX

## [v1.9.24] — 2025-06-24
### Changed
- ปรับ `run_backtest` รองรับการเข้าออเดอร์ทั้งขา Long/Short
- ตรวจจับความผันผวนสูงเพื่อข้ามการเข้าใหม่
- เพิ่ม Partial TP 50% และปรับ Trailing SL
- เพิ่ม dynamic risk scaling และบันทึกเหตุผลเข้าออเดอร์ใน trade log
## [v1.9.25] — 2025-06-25
### Added
- ฟังก์ชัน `qa_log_step` สำหรับแสดงข้อความทุกขั้นตอน
- เรียกใช้ใน `run_backtest_cli` และ `run`
## [v1.9.26] — 2025-06-26
### Added
- เพิ่มชุดฟังก์ชัน SMC Multi-Timeframe และตัวช่วยตรวจจับโซน
- ฟังก์ชันใหม่ `detect_order_block`, `detect_fvg`, `detect_liquidity_grab`, `is_smc_entry`

## [v1.9.27] — 2025-06-27
### Changed
- ปรับ `run_backtest_cli` ใช้สัญญาณแบบ SMC Multi-Timeframe
- ปรับ `generate_smart_signal` ให้โหลดข้อมูล M15 และแมพโซน OB/FVG/Liquidity Grab
- เพิ่มเหตุผลการเข้าออเดอร์แบบ SMC ในบันทึกเทรด

## [v1.9.28] — 2025-06-28
### Fixed
- แก้ NameError `load_csv_m15` ใน `run_backtest_cli` โดยย้ายบล็อก `__main__` ไปท้ายไฟล์

## [v1.9.29] — 2025-06-29
### Fixed
- กำหนด `format='%Y-%m-%d %H:%M:%S'` ในการแปลงค่า timestamp ทุกจุด
- ป้องกันคำเตือนจาก `pd.to_datetime`

## [v1.9.30] — 2025-06-30
### Fixed
- แก้การแปลง timestamp ใน `load_csv_m15` และ `load_csv_m1` ให้รองรับวันที่ พ.ศ.

## [v1.9.31] — 2025-07-01
### Changed
- ปรับปรุง `align_mtf_zones` ให้ใช้การแม็พแบบเวกเตอร์
### Fixed
- ยืนยันการแปลง timestamp ใน `load_csv_m15` และ `load_csv_m1`

## [v1.9.32] — 2025-07-02

### Fixed
- แก้การอ่าน CSV ด้วย `low_memory=False` ใน `load_csv_m15` และ `load_csv_m1`
### Changed
- ปรับ `align_mtf_zones` ให้ใช้เงื่อนไขเวกเตอร์และตัวแปรใหม่


## [v1.9.33] — 2025-07-03
### Changed
- `run_backtest_cli` โหลดไฟล์ด้วย `low_memory=False` และ optimize หน่วยความจำ
- เพิ่ม `calculate_trend_confirm` ก่อนสร้างสัญญาณ SMC
- แสดง RAM ที่ใช้ผ่าน logger เพื่อการตรวจสอบ QA


## [v1.9.34] — 2025-07-04
### Added
- NICEGOLD Enterprise v1.1: OMS manager, relaxed entry, smart sizing, logging improvements


## [v1.9.35] — 2025-07-05
### Added
- NICEGOLD Enterprise v2.0: ระบบ Risk Sizing ตามเทรนด์และ RR, OMS สมบูรณ์, บันทึกเทรดและ equity curve แบบละเอียด

## [v1.9.36] — 2025-07-06
### Added
- NICEGOLD Enterprise QA Supergrowth v3: ป้องกัน SL ใกล้เกิน, จำกัดล็อตตามทุน, ตรวจเทรนด์อย่างเดียว, RR Adaptive
## [v1.9.37] — 2025-07-07
### Changed
- ปรับ `smart_entry_signal` เป็นเวกเตอร์ พร้อม Mask Trend และ Min SL Guard

## [v1.9.38] — 2025-07-08
### Added
- ระบบ signal และ recovery ปรับปรุงตาม QA patch
- คำนวณ position size ตามความเสี่ยง
- Partial TP, BE, trailing stop และ Smart Recovery

## [v1.9.39] — 2025-07-09
### Changed
- รวมโมดูล runtime เข้ากับ `enterprise.py` ลดไฟล์และใช้งานสะดวกขึ้น
## [v1.9.40] — 2025-07-10
### Fixed
- รองรับการอ่านเวลารูปแบบ AM/PM ใน `load_data`, `load_csv_m1`, `load_csv_m15` และ `run_backtest_cli`

## [v1.9.41] — 2025-07-11
### Added
- เพิ่ม `fallback_simple_signal` ใช้สร้างสัญญาณ MACD/RSI หากไม่มีสัญญาณ SMC
- `run_backtest_cli` เรียกใช้ฟังก์ชันใหม่นี้ เพื่อเพิ่มจำนวนรายการเทรด

## [v1.9.42] — 2025-07-12
### Changed
- ปรับเส้นทางไฟล์ข้อมูลและ log ให้เป็นแบบ relative ภายใน repo
- เพิ่มความเสี่ยงต่อไม้เป็น 10% และยกเลิกข้อจำกัด drawdown เพื่อเน้นทดสอบจำนวนเทรด

## [v1.9.43] — 2025-07-13
### Added
- ระบบเทรด Multi-TF (M1 ยืนยันด้วย M15)
- ฟังก์ชัน `run_backtest_multi_tf` และ `smart_entry_signal_multi_tf`
- `calc_indicators` รองรับกำหนดค่า period

## [v1.9.44] — 2025-07-14
### Added
- ฟังก์ชัน `rsi` สำหรับคำนวณค่า Relative Strength Index
- `calc_indicators` ปรับให้เรียกใช้ `rsi`
- `add_m15_context_to_m1` ใช้ `merge_asof` บนดัชนี

## [v1.9.45] — 2025-07-15
### Added
- ฟังก์ชัน `smart_entry_signal_goldai2025_style` สำหรับสร้างสัญญาณเทรดแบบ GoldAI2025
- ปรับ `run_backtest` และ `run_backtest_multi_tf` ให้ใช้ตรรกะสัญญาณใหม่นี้

