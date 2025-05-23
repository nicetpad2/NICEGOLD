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

