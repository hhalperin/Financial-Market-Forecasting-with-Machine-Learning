"""
trade_analysis.py

This script analyzes the results of the trading simulation by loading the trade log CSV,
computing metrics such as the equity curve, drawdowns, trade profit/loss distribution, and win rate,
and generating visualizations to facilitate analysis.

Usage:
    python trade_analysis.py --log_file path/to/trade_log.csv --output_dir path/to/output_dir
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set seaborn style for nicer plots
sns.set(style="whitegrid")

def load_trade_log(csv_path: str) -> pd.DataFrame:
    """
    Loads the trade log CSV into a DataFrame.
    
    :param csv_path: Path to the trade log CSV file.
    :return: DataFrame with trade log data.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Compute trade PnL if not explicitly provided
    if "profit" not in df.columns and "loss" not in df.columns:
        df["trade_pnl"] = df["capital_after"] - df["capital_before"]
    else:
        # For BUY, profit is recorded; for SELL, loss is recorded as a positive number here
        df["trade_pnl"] = df.apply(
            lambda row: row.get("profit", 0) if row["decision"] == "BUY" 
                        else -row.get("loss", 0) if row["decision"] == "SELL" 
                        else 0, axis=1)
    return df

def compute_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the equity curve (capital over time) from the trade log.
    
    :param df: Trade log DataFrame.
    :return: DataFrame with timestamp and equity (capital_after) columns, sorted by timestamp.
    """
    df_sorted = df.sort_values("timestamp")
    equity_df = df_sorted[["timestamp", "capital_after"]].copy()
    equity_df.rename(columns={"capital_after": "equity"}, inplace=True)
    return equity_df

def calculate_drawdowns(equity_series: pd.Series) -> pd.Series:
    """
    Calculates the drawdown (percentage drop from the running maximum) for the equity curve.
    
    :param equity_series: Pandas Series representing equity values over time.
    :return: Pandas Series with drawdown percentages.
    """
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max * 100  # in percent
    return drawdowns

def plot_equity_curve(equity_df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots the equity curve over time and saves the figure.
    
    :param equity_df: DataFrame containing 'timestamp' and 'equity' columns.
    :param output_dir: Directory where the figure will be saved.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["timestamp"], equity_df["equity"], marker="o", linestyle="-", color="blue")
    plt.title("Equity Curve Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity (Capital)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
def plot_drawdown(equity_df: pd.DataFrame, output_dir: str) -> None:
    """
    Computes drawdowns from the equity curve and plots them.
    
    :param equity_df: DataFrame containing the equity curve.
    :param output_dir: Directory where the figure will be saved.
    """
    drawdowns = calculate_drawdowns(equity_df["equity"])
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["timestamp"], drawdowns, marker="o", linestyle="-", color="red")
    plt.title("Drawdowns Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Drawdown (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "drawdowns.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_trade_pnl_histogram(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots a histogram of trade profit/loss values.
    
    :param df: Trade log DataFrame.
    :param output_dir: Directory where the figure will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df["trade_pnl"], bins=20, kde=True, color="purple")
    plt.title("Histogram of Trade Profit/Loss")
    plt.xlabel("Trade PnL")
    plt.ylabel("Frequency")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "trade_pnl_histogram.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_trade_pnl_over_time(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plots individual trade PnL over time as a scatter plot.
    
    :param df: Trade log DataFrame.
    :param output_dir: Directory where the figure will be saved.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(df["timestamp"], df["trade_pnl"], color="green")
    plt.title("Trade Profit/Loss Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Trade PnL")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "trade_pnl_over_time.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """
    Computes summary metrics from the trade log.
    
    :param df: Trade log DataFrame.
    :return: Dictionary with summary statistics.
    """
    total_trades = len(df)
    total_profit = df["trade_pnl"].sum()
    average_profit = df["trade_pnl"].mean() if total_trades > 0 else 0
    win_trades = df[df["trade_pnl"] > 0]
    win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0
    max_drawdown = df["capital_after"].cummax().subtract(df["capital_after"]).max()
    summary = {
        "total_trades": total_trades,
        "total_profit": total_profit,
        "average_profit": average_profit,
        "win_rate_percent": win_rate,
        "max_drawdown": max_drawdown
    }
    return summary

def save_summary_table(summary: dict, output_dir: str) -> None:
    """
    Saves the summary metrics to a CSV file.
    
    :param summary: Dictionary with summary metrics.
    :param output_dir: Directory where the file will be saved.
    """
    df = pd.DataFrame([summary])
    output_path = os.path.join(output_dir, "trading_summary.csv")
    df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Analyze trading simulation results and generate visualizations.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the trade log CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations and summary metrics.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading trade log from: {args.log_file}")
    df = load_trade_log(args.log_file)
    
    if df.empty:
        print("Trade log is empty. Exiting.")
        return

    # Compute equity curve and summary statistics
    equity_df = compute_equity_curve(df)
    summary_metrics = compute_summary_metrics(df)
    
    print("Summary Metrics:")
    for k, v in summary_metrics.items():
        print(f"  {k}: {v}")
    
    # Generate and save visualizations
    plot_equity_curve(equity_df, args.output_dir)
    plot_drawdown(equity_df, args.output_dir)
    plot_trade_pnl_histogram(df, args.output_dir)
    plot_trade_pnl_over_time(df, args.output_dir)
    save_summary_table(summary_metrics, args.output_dir)
    
    print(f"Visualizations and summary saved to {args.output_dir}")

if __name__ == "__main__":
    main()
