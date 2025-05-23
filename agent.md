# 🧠 agent.md — Gold AI: Elliott-MACD Realistic QA Agent



**Version:** v1.9.18




**Maintainer:** AI Studio QA / Dev Agent System  

## 📌 Agent Role: `elliott_macd_backtest_agent`

### 🧭 Objective:
-This agent runs **realistic backtests** on XAUUSD M1 historical data using:
- **Elliott Wave-based Trend Context**
- **MACD Divergence + EMA35 Trigger**
- **ModernScalpingXAUUSD v1.0 (ML-based Scalping, integrated in main module)**
- Real-world trading constraints:
  - Spread (80 points)
  - Slippage
  - Commission per execution (0.10 USD per 0.01 lot)

### 🔨 Responsibilities:
| Task | Description |
|------|-------------|
| `load_data()` | Parse and convert Pali/Buddhist date to AD. Handle `timestamp` as datetime index. |
| `calculate_indicators()` | Calculate MACD line, signal line, histogram, EMA35 |
| `detect_signal()` | Generate entry signal using Divergence + MACD Cross + EMA Touch |
| `detect_elliott_wave_phase()` | Dynamic W1-W5/A-B-C labeling using RSI and divergence |
| `generate_entry_signal_wave_enhanced()` | Entry only during W.2/W.3/W.5/B with EMA filter |
| `generate_entry_score_signal()` | Score-based entry ranking system |
| `run_backtest()` | Execute realistic backtest on last N rows (e.g., 200,000) |
| `apply_constraints()` | Apply spread, slippage, and commission before evaluating PnL |
| `generate_log()` | Export `trade_log.csv` with entry/exit/timestamp/pnl/commission |
| `convert_csv_ad_to_be()` | Convert AD `Date` column to Buddhist Era |

### ⚙️ Environment Assumptions:
- Input data is from `/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv` with Buddhist `Date` + `Timestamp`
- 3-digit Gold broker logic: `1 pip = 0.01`, `80 point = 0.80`
- Commission model is **per execution only**
- Single position (no scaling / pyramiding)
- TP:SL = 2:1 | Risk per trade = 1% of capital | Starting Capital: $100

### 🧩 Integration Format:
```python
agent = ElliottMACDBacktestAgent()
df = agent.load_data("/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv")
df = agent.calculate_indicators(df)
df = agent.detect_elliott_wave_phase(df)
signal_df = agent.generate_entry_score_signal(df)
result_df = agent.run_backtest(signal_df, last_n_rows=200000)
agent.export_log(result_df, path="realistic_trade_log.csv")
```

### ✅ Required After Patch
- [x] `agent.md` updated
- [x] `changelog.md` updated with patch note
- [x] เพิ่ม unit tests เพื่อให้ครอบคลุม 100%
- [x] ใส่ debug logging ทุกครั้งเมื่ออัปเดตหรือเพิ่มฟังก์ชันใหม่

### 📝 Patch v1.9.13
- เพิ่มฟังก์ชัน `optimize_memory` เพื่อประหยัดหน่วยความจำบน Colab L4
- ปรับ `load_data` และ `detect_macd_divergence` ให้เป็นเวกเตอร์และมีประสิทธิภาพมากขึ้น

### 📝 Patch v1.9.14
- แก้คำเตือน chained assignment ในการใช้ `fillna`
- ปรับค่าคอมมิชชันเป็น 0.10 USD ต่อ 0.01 lot คิดครั้งเดียวเมื่อปิดสถานะ
- เพิ่ม debug log แสดงข้อมูลสัญญาณท้ายตาราง


### 📝 Patch v1.9.15
- แก้ FutureWarning ใน unit test (`freq='T'` → `freq='min'`)
- ปรับ `.pct_change()` ใส่ `fill_method=None` เพื่อลด warning
=======

### 📝 Patch v1.9.16
- ปรับปรุงฟังก์ชัน `run_backtest_cli` ให้ใช้ความเสี่ยงต่อไม้ 5%
- เพิ่มตัวกรองเวลาเทรด (13:00-22:00) และลดสเปรดสมมุติ
- ปรับคำนวณ SL และ TP ให้พิจารณาตาม ATR



### 📝 Patch v1.9.17
- เพิ่ม ATR ใน `run_backtest_cli` และปรับ SL/TP ตาม ATR
- เพิ่ม logger.info เมื่อเปิดสถานะและเมื่อ TP1
- ยกเลิกหักค่าคอมมิชชั่นในขั้น TP1

### 📝 Patch v1.9.18
- [Patch G-Fix2] เพิ่มระบบบันทึก trade log และ equity curve พร้อม timestamp
- แก้ปัญหา `to_csv` ใน Colab โดยระบุพาธแบบเต็ม
- บันทึกกราฟ Equity Curve อัตโนมัติ
\n### 📝 Patch v1.9.19\n- ปรับปรุงระบบจัดการความเสี่ยง เพิ่มการตรวจสอบ drawdown และ volatility\n- เพิ่ม dynamic lot sizing และ reward scaling ตาม win/loss streak\n- เพิ่ม contextual reasoning ในเหตุผลการเข้าออเดอร์และ trailing SL แบบ ATR\n
