# trading_rules.py
"""
Trading Rules Module

This module defines functions to apply trading rules and risk management to generate
trade orders from signals. Adjust the logic to suit your risk management strategy.
"""

import logging

def apply_trading_rules(signal, config):
    """
    Applies trading rules to a given signal and returns an order dictionary.
    
    :param signal: The trading signal (e.g., "buy", "sell", "hold").
    :param config: The trading configuration containing thresholds and trade parameters.
    :return: A dictionary representing the trade order.
    """
    logger = logging.getLogger("TradingRules")
    logger.info("Applying trading rules.")

    order = {}
    if signal == "buy":
        order = {
            "action": "buy",
            "trade_size": config.trade_size,
            "stop_loss": config.stop_loss,       # Default value; may be updated later.
            "take_profit": config.take_profit    # Default value; may be updated later.
        }
    elif signal == "sell":
        order = {
            "action": "sell",
            "trade_size": config.trade_size,
            "stop_loss": config.stop_loss,
            "take_profit": config.take_profit
        }
    else:
        order = {"action": "hold"}

    logger.info(f"Order generated: {order}")
    return order
