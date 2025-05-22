# 🧠 agent.md — Gold AI: Elliott-MACD Realistic QA Agent
**Version:** v1.1.1
**Last updated:** 2025-05-24
**Maintainer:** AI Studio QA / Dev Agent System  

## 📌 Agent Role: `elliott_macd_backtest_agent`

### 🧭 Objective:
This agent runs **realistic backtests** on XAUUSD M1 historical data using:
- **Elliott Wave-based Trend Context**
- **MACD Divergence + EMA35 Trigger**
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
| `run_backtest()` | Execute realistic backtest on last N rows (e.g., 200,000) |
| `apply_constraints()` | Apply spread, slippage, and commission before evaluating PnL |
| `generate_log()` | Export `trade_log.csv` with entry/exit/timestamp/pnl/commission |

### ⚙️ Environment Assumptions:
- Input data is from `XAUUSD_M1.csv` with Buddhist `Date` + `Timestamp`
- 3-digit Gold broker logic: `1 pip = 0.01`, `80 point = 0.80`
- Commission model is **per execution only**
- Single position (no scaling / pyramiding)
- TP:SL = 2:1 | Risk per trade = 1% of capital | Starting Capital: $100

### 🧩 Integration Format:
```python
agent = ElliottMACDBacktestAgent()
df = agent.load_data("XAUUSD_M1.csv")
df = agent.calculate_indicators(df)
signal_df = agent.detect_signal(df)
result_df = agent.run_backtest(signal_df, last_n_rows=200000)
agent.export_log(result_df, path="realistic_trade_log.csv")
```

### ✅ Required After Patch
- [x] `agent.md` updated
- [x] `changelog.md` updated with patch note