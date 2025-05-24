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
