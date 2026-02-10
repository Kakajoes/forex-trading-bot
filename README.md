# Forex Trading Bot

Personal Python-based forex bot (EUR/USD, GBP/USD, USD/JPY focus)

## Features
- EMA 10/30 crossover + RSI(14) + MACD histogram signals
- ATR-based dynamic ranges for entry/TP/SL
- Simple PyTorch ML model to refine signals
- SQLite trade history logging
- Streamlit dashboard with equity curve, win rate, monthly P/L, weekday performance
- OANDA live trading support (use practice account!)
- News/high-volatility filters

## How to Run (on computer/VPS)
1. `pip install pandas polygon-api-client v20 torch streamlit plotly`
2. Add your Polygon & OANDA keys in `forex_bot.py`
3. Backtest: `python forex_bot.py`
4. View dashboard: `streamlit run dashboard.py`

**Warning**: This is for educational/demo use. Trading involves risk — test thoroughly on demo account.

Built with help from Grok – February 2026
