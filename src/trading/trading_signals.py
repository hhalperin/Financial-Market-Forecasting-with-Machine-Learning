# trading_signals.py
"""
Trading Signals Module

This module defines functions to generate trading signals from model outputs.
It uses configurable thresholds to decide whether to buy, sell, or hold.
In a more advanced implementation, it can combine multiple model predictions (weighted)
to generate a robust signal.
"""

import logging

def generate_signal(model_output, config):
    """
    Generates a trading signal based on the aggregated model output and configuration thresholds.
    
    :param model_output: The aggregated output from multiple ML models (a weighted prediction).
    :param config: The trading configuration containing buy/sell thresholds.
    :return: A string signal: "buy", "sell", or "hold".
    """
    logger = logging.getLogger("TradingSignals")
    logger.info("Generating trading signal based on aggregated model output.")

    if model_output >= config.buy_threshold:
        signal = "buy"
    elif model_output <= config.sell_threshold:
        signal = "sell"
    else:
        signal = "hold"

    logger.info(f"Signal determined: {signal}")
    return signal
