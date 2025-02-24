# main.py
"""
Main Module for Trading

This script serves as the entry point for the trading directory.
It initializes the TradingEngine and starts a trading session.
"""

from .trading_engine import TradingEngine

def main():
    engine = TradingEngine()
    engine.run()

if __name__ == "__main__":
    main()
