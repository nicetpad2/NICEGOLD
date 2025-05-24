import pandas as pd
from datetime import timedelta
import logging
import os
from itertools import product

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

import numpy as np
import json

logger = logging.getLogger("WFA")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("wfa_engine.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def split_data_into_folds(df, date_col, fold_size_days=30):
    """Split dataframe into time-based folds."""
    logger.debug("Splitting data into folds")
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = df[date_col].min()
    end_date = df[date_col].max()
    folds = []
    while start_date < end_date:
        fold_end = start_date + timedelta(days=fold_size_days)
        fold = df[(df[date_col] >= start_date) & (df[date_col] < fold_end)]
        if not fold.empty:
            folds.append((start_date, fold_end, fold.copy()))
        start_date = fold_end
    return folds


def param_grid():
    """Generate parameter grid for optimization."""
    logger.debug("Generating parameter grid")
    grid = {
        "risk_per_trade": [0.02, 0.03],
        "force_entry_gap": [100, 150],
        "partial_close_pct": [0.5, 0.6],
        "trail_stop_mult": [0.4, 0.6],
        "lot_max": [5.0, 10.0],
    }
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in product(*values)]


def optimize_params(df):
    """Search best parameters using heuristic score."""
    logger.debug("Optimizing parameters")
    best_score = -np.inf
    best_params = None
    objectives = []
    for params in param_grid():
        try:
            result = wfa(df.copy(), params)
            return_score = result.get("Total Return", 0)
            winrate_score = result.get("Winrate", 0)
            dd_score = result.get("max_drawdown", 0)
            trade_count = result.get("Total Trades", 0)

            score = return_score + 0.5 * winrate_score - 1000 * dd_score + 0.1 * trade_count
            objectives.append((score, params))

            if score > best_score:
                best_score = score
                best_params = params
        except Exception as e:  # pragma: no cover - logging
            logger.warning(f"[WFA] Param combo failed: {params} => {e}")

    objectives.sort(reverse=True)
    logger.info("[WFA] Top 3 parameter sets:")
    for rank, (score, p) in enumerate(objectives[:3]):
        logger.info(f"  {rank+1}: Score={score:.2f}, Params={p}")

    return best_params


def validate_result(result):
    """Validate wfa result dictionary."""
    required_keys = ["Final Equity", "Total Trades", "Total Return", "Winrate"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"[WFA QA] Missing key in result: {key}")
    return True


def save_shap_summary(df, fold_id):
    """Save SHAP summary per fold if columns available."""
    try:
        shap_cols = [col for col in df.columns if col.startswith("shap_importance_")]
        summary = {col.replace("shap_importance_", ""): df[col].mean() for col in shap_cols}
        with open(f"shap_summary_fold_{fold_id}.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[WFA] SHAP summary saved for Fold {fold_id}")
    except Exception as e:  # pragma: no cover - optional logging
        logger.warning(f"[WFA] Failed to save SHAP summary for Fold {fold_id}: {e}")


def wfa(df, params):
    """Run backtest with given params and return statistics."""
    logger.debug("Running WFA backtest")
    import enterprise

    enterprise.risk_per_trade = params.get("risk_per_trade", 0.03)
    enterprise.force_entry_gap = params.get("force_entry_gap", 100)
    enterprise.partial_close_pct = params.get("partial_close_pct", 0.6)
    enterprise.trail_stop_mult = params.get("trail_stop_mult", 0.4)
    enterprise.lot_max = params.get("lot_max", 10.0)

    df = enterprise.data_quality_check(df)
    df = enterprise.calc_indicators(df)
    df = enterprise.calc_dynamic_tp2(df)
    df = enterprise.label_elliott_wave(df)
    df = enterprise.detect_divergence(df)
    df = enterprise.label_pattern(df)
    df = enterprise.calc_gain_zscore(df)
    df = enterprise.calc_signal_score(df)
    df = enterprise.tag_session(df)
    df = enterprise.tag_spike_guard(df)
    df = enterprise.tag_news_event(df)
    df = enterprise.smart_entry_signal_goldai2025_style(df)
    df = enterprise.apply_session_bias(df)
    df = enterprise.apply_spike_news_guard(df)

    trades = enterprise._execute_backtest(df)
    equity = globals().get("_LAST_EQUITY_DF", pd.DataFrame())

    stats = enterprise.analyze_tradelog(trades, equity)
    final_eq = equity['equity'].iloc[-1] if not equity.empty else enterprise.initial_capital
    total_trades = len(trades)
    total_return = final_eq - enterprise.initial_capital
    winrate = (trades['pnl'] > 0).mean() * 100 if not trades.empty else 0

    logger.debug("WFA backtest complete: equity=%.2f, trades=%d", final_eq, total_trades)

    return {
        "Final Equity": final_eq,
        "Total Trades": total_trades,
        "Total Return": total_return,
        "Winrate": winrate,
        **stats,
    }


def walk_forward_run(trade_data_path, date_col='entry_time', fold_days=30, output_csv="wfa_summary_results.csv"):
    """Execute walk forward analysis across folds."""
    if not os.path.exists(trade_data_path):
        logger.error(f"[WFA] Trade file not found: {trade_data_path}")
        raise FileNotFoundError("Trade file not found")

    df = pd.read_csv(trade_data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    folds = split_data_into_folds(df, date_col, fold_days)
    all_results = []

    for i, (start, end, fold_df) in enumerate(folds):
        logger.info(f"[WFA] Running Fold {i+1}: {start.date()} → {end.date()} | Trades: {len(fold_df)}")
        try:
            best_params = optimize_params(fold_df)
            logger.info(f"[WFA] Fold {i+1} Best Params: {best_params}")
            result = wfa(fold_df.copy(), best_params)
            validate_result(result)
            result['Fold'] = f"{start.date()} → {end.date()}"
            result['Trades'] = len(fold_df)
            result.update(best_params)
            all_results.append(result)
            save_shap_summary(fold_df, i+1)
            logger.info(
                f"[WFA] Fold {i+1} completed: Final Equity = {result['Final Equity']}, Trades = {result['Total Trades']}, Winrate = {result['Winrate']:.2f}%"
            )
        except Exception as e:  # pragma: no cover - logging
            logger.error(f"[WFA] Error in Fold {i+1}: {str(e)}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_csv, index=False)

    try:
        if plt is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(summary_df['Fold'], summary_df['Final Equity'], marker='o', label='Final Equity')
            plt.plot(summary_df['Fold'], summary_df['Total Return'], marker='x', label='Total Return')
            plt.xticks(rotation=45)
            plt.title("Walk Forward Fold Performance")
            plt.ylabel("USD")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("wfa_equity_plot.png")
            logger.info("[WFA] Plot saved to wfa_equity_plot.png")
    except Exception as e:  # pragma: no cover - optional logging
        logger.warning(f"[WFA] Failed to plot equity chart: {e}")

    logger.info("[WFA] Walk Forward Analysis completed. Summary saved.")
    print("\n✅ Walk Forward Analysis Completed. Summary saved to", output_csv)


if __name__ == "__main__":  # pragma: no cover - CLI
    print("\n🔢 Select Mode:")
    print("  [1] Walk Forward Analysis (WFA)")
    print("  [2] Basic Backtest")
    mode_input = input("Enter mode number [1 or 2]: ").strip()
    file_input = input(
        "Enter CSV file path (default: trade_log_20250525_000842.csv): "
    ).strip()
    if not file_input:
        file_input = "trade_log_20250525_000842.csv"

    if mode_input == "1":
        print("[MODE 1] Running Walk Forward Pipeline...")
        walk_forward_run(file_input)
    elif mode_input == "2":
        from enterprise import run_backtest

        print("[MODE 2] Running Basic Backtest...")
        run_backtest(file_input)
    else:
        print("❌ Invalid mode. Please enter 1 or 2.")
