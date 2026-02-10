import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

DB_FILE = "trades_history.db"

st.set_page_config(page_title="Joachim Forex Dashboard", layout="wide")

@st.cache_data(ttl=300)
def load_trades():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['cum_profit'] = df['profit'].cumsum().fillna(0)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['weekday'] = df['timestamp'].dt.day_name()
    return df

def get_suggestion(pair="EUR_USD"):
    from forex_bot import fetch_data, compute_indicators, generate_rule_signals, train_ml_model, get_ml_signal
    df = fetch_data(pair, days_back=30)
    if df.empty:
        return "No data"
    df = compute_indicators(df)
    df = generate_rule_signals(df)
    model = train_ml_model(df)
    last = df.iloc[-1]
    sig = get_ml_signal(model, last)
    atr = last['atr']
    price = last['close']
    if sig == 1:
        return f"BUY | Entry: {price - 0.5*atr:.5f}-{price + 0.5*atr:.5f} | TP: {price + 2*atr:.5f}-{price + 3*atr:.5f} | SL: {price - 1.5*atr:.5f}"
    elif sig == -1:
        return f"SELL | Entry: {price - 0.5*atr:.5f}-{price + 0.5*atr:.5f} | TP: {price - 2*atr:.5f}-{price - 3*atr:.5f} | SL: {price + 1.5*atr:.5f}"
    return "NEUTRAL"

def main():
    st.title("Joachim's Complete Forex Bot Dashboard")
    trades_df = load_trades()
    if trades_df.empty:
        st.warning("Run forex_bot.py first!")
        return

    # Metrics & Suggestions
    cols = st.columns(5)
    # ... (same as previous metrics)
    st.subheader("Future Trade Suggestions")
    for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
        st.markdown(f"**{pair}**: {get_suggestion(pair)}")

    # Tabs & Charts (same as previous)

if __name__ == "__main__":
    main()
