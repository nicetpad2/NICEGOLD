# NICEGOLD Enterprise Agent

เวอร์ชัน v3.1 เพิ่มระบบสัญญาณเข้าแบบเวกเตอร์ ตรวจเทรนด์และ Min SL Guard ได้เร็วขึ้น
ใช้สำหรับทดสอบกลยุทธ์บนข้อมูล XAUUSD M1


## v3.2 QA Patch
- เพิ่มฟังก์ชัน runtime สำหรับสัญญาณและ Recovery อัตโนมัติ

## v3.3 QA Patch
- รวมฟังก์ชัน runtime เข้าไฟล์ enterprise.py เพื่อใช้งานสะดวกขึ้น

## v3.4 QA Patch
- รองรับไฟล์เวลารูปแบบ AM/PM ในฟังก์ชันโหลดข้อมูล

## v3.5 QA Patch
- เพิ่มฟังก์ชัน `fallback_simple_signal` เพื่อสร้างสัญญาณเมื่อไม่มีสัญญาณ SMC
- `run_backtest_cli` เรียกใช้ฟังก์ชันใหม่นี้เพื่อให้มีการเทรดมากขึ้น

## v3.6 QA Patch
- ปรับเส้นทางไฟล์ข้อมูลและ log ให้ใช้งานภายใน repo โดยตรง
- เพิ่มความเสี่ยงต่อไม้เป็น 10% และยกเลิกข้อจำกัด drawdown เพื่อการทดสอบ

## v3.7 QA Patch
- รองรับระบบ Multi-TF (M1 ยืนยันด้วย M15)
- ปรับ `calc_indicators` ให้กำหนด period ได้
- เพิ่ม `run_backtest_multi_tf` สำหรับทดสอบหลายกรอบเวลา

## v3.8 QA Patch
- เพิ่มฟังก์ชัน `rsi` และปรับ `calc_indicators` ใช้งาน
- `add_m15_context_to_m1` ปรับเป็น `merge_asof` ตามดัชนี

## v3.9 QA Patch
- เพิ่มฟังก์ชัน `smart_entry_signal_goldai2025_style` สำหรับสร้างสัญญาณแบบ GoldAI2025
- ปรับ `run_backtest` และ `run_backtest_multi_tf` ให้เรียกใช้ฟังก์ชันใหม่

## v3.10 QA Patch
- เพิ่มฟังก์ชัน label_elliott_wave, detect_divergence, label_pattern
- เพิ่ม calc_gain_zscore, calc_signal_score และ meta_classifier_filter
- ปรับ pipeline ใน run_backtest และ run_backtest_multi_tf ให้เรียกใช้โมดูลใหม่

## v3.11 QA Patch
- เพิ่มฟังก์ชัน `calc_dynamic_tp2`, `tag_session` และ `apply_session_bias`
- ปรับ backtest pipeline ใช้ Dynamic TP2 และ Session Bias

## v3.12 QA Patch
- เพิ่มระบบ Spike Guard และ News Filter
- บันทึก Reason/Context ลง trade log อย่างละเอียด
- ฟังก์ชัน Walk-Forward Validation และ split folds อัตโนมัติ

## v3.13 QA Patch
- Reformatted code with Black and adjusted unit tests for quoting changes.

## v3.14 QA Patch
- เพิ่มฟังก์ชัน `data_quality_check` ตรวจสอบและทำความสะอาดข้อมูลเบื้องต้น
- เพิ่ม `shap_feature_importance_placeholder` สำหรับแสดงความสำคัญของฟีเจอร์แบบ stub
- ระบบคำนวณค่าใช้จ่ายคำสั่งซื้อจริง (Spread, Commission, Slippage) และบันทึกลง trade log

## v3.15 QA Patch
- เพิ่ม debug log แสดงสถานะขณะถือ position ใน `_execute_backtest`
- บันทึกค่า hit TP/SL เพื่อช่วยตรวจสอบ logic
- ปิดสถานะอัตโนมัติเมื่อถือครบ 50 แท่ง

## v3.16 QA Patch
- ปรับ `apply_order_costs` ให้บวกสเปรดเฉพาะราคาเข้า ไม่ปรับ SL/TP
- ลด `risk_per_trade` และขยาย `sl_mult`/`min_sl_dist`
- กรอง ATR หากต่ำกว่า `spread x2` ไม่เปิดสถานะ
- เพิ่ม debug log รายไม้และ fail-safe หยุดแบ็กเทสต์เมื่อขาดทุน 20% ใน 3 ไม้แรก

## v3.17 QA Patch
- ลดค่าคอมมิชชันเหลือ 0.02 และ slippage 0.05
- ขยาย TP multiplier (2.5/6.0) และ SL multiplier 2.0
- ระยะ SL ขั้นต่ำ 5 จุด และกรอง ATR < spread x2.5
- จำกัด lot ไม่เกิน 0.05 ต่อไม้
- เพิ่ม debug log รายงาน ATR และค่าใช้จ่ายทุกไม้

## v3.18 QA Patch
- ปรับค่าคอมมิชชันเป็น 0.10 และ slippage 0.2
- ปรับ risk management (tp1_mult=2.0, tp2_mult=4.0, sl_mult=1.2, min_sl_dist=4)
- เพิ่มฟังก์ชัน `multi_session_trend_scalping` และใช้ใน pipeline `run_backtest`
- บันทึกจำนวนสัญญาณเข้าออเดอร์แบบละเอียด

## v3.19 QA Patch
- เพิ่มตัวแปร `MAX_RAM_MODE` และฟังก์ชัน `log_ram_usage`
- คำนวณ EMA/RSI/ATR หลายช่วงเมื่อเปิดโหมดใช้แรมสูง
- บันทึกการใช้ RAM ก่อนและหลังรัน backtest


## v3.20 QA Patch
- เพิ่มฟังก์ชัน `smart_entry_signal_enterprise_v1` ใช้สัญญาณแบบ Multi-confirm
- ปรับ `_execute_backtest` ตรวจ strict confirm ใน Recovery และ QA monitor loss streak
- `run_backtest` เรียกใช้สัญญาณใหม่

## v3.21 QA Patch
- แก้ `calc_indicators` เพิ่มสร้าง `rsi_34` สำหรับ MAX_RAM_MODE
- เพิ่มการตรวจสอบคอลัมน์ RSI ใน `detect_divergence` และแจ้งเตือนผ่าน logger

## v3.22 QA Patch
- `smart_entry_signal_enterprise_v1` เพิ่ม QA Guard ตรวจจำนวนสัญญาณลดลง
- `_execute_backtest` ปรับ Adaptive Entry เฉพาะจุดเมื่ออยู่ใน Recovery Mode
- ปรับ Trailing SL หลัง TP1 ให้ทำงานเมื่อ ADX > 20

## v3.23 QA Patch
- เพิ่มโหมด Aggressive Entry ใช้สัญญาณสลับทุกแท่งและ lot sizing ใหม่

## v3.24 QA Patch
- เพิ่มฟังก์ชัน `entry_signal_trend_scalp` ใช้ Trend Scalping พร้อม Force Entry

## v3.25 QA Patch
- ปรับ risk_per_trade เป็น 3% และขยาย TP1/TP2 เป็น 3.0/6.0
- เพิ่มตรวจ signal_score และเทรนด์ M15 ใน generate_signal
- calculate_position_size ปรับความเสี่ยงตาม ADX และจำกัด lot ตาม equity

## v3.26 QA Patch
- ลด risk_per_trade เป็น 2%
- เพิ่ม kill_switch_dd เป็น 50%
- adx_thresh ปรับเป็น 15
- force_entry_gap เป็น 300
- ยกเลิกลิมิต lot สูงสุด 0.05
- apply_session_bias ไม่กรอง session ใด ๆ
- `_execute_backtest` ข้ามสัญญาณที่ divergence ตรงข้าม


## v3.27 QA Patch
- เปิดโหมด ib_commission_mode เพิ่ม force_entry_gap เป็น 100
- Partial TP 60% และ trail_stop_mult 0.4
- เพิ่ม micro SL exit และระบบ re-entry สูงสุด 2 ไม้

## v3.28 QA Patch
- เพิ่มฟังก์ชันคำนวณล็อตอัตโนมัติ `calculate_auto_lot`
- เพิ่มฟังก์ชัน `equity_based_tp_sl` ปรับ TP/SL ตามทุน
- ปรับ `_execute_backtest` ใช้ค่าจากฟังก์ชันใหม่เมื่อเปิดใช้งาน


## v3.29 QA Patch
- เพิ่มโมดูล `walk_forward_engine.py` สำหรับ Walk Forward Analysis และบันทึก log แบบละเอียด

## v3.30 QA Patch
- รวมระบบ Walk Forward Analysis เข้า enterprise.py และเพิ่มเมนู CLI เลือกโหมด

## v3.31 QA Patch
- [Patch G] ปรับสัญญาณ smart_entry_signal_enterprise_v1 ลด gain_z threshold, รองรับ wave_phase="mid" และ force_entry_gap=50

## v3.32 QA Patch
- ปรับเมนู CLI ตัด Basic Backtest และเหลือเฉพาะ Walk Forward กับ Multi-TF
- โหมดเริ่มต้นเรียก `walk_forward_run` พร้อม debug log

