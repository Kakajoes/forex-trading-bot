import pandas as pd
from v20 import Context as V20Context  # OANDA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from polygon import RESTClient
import sqlite3
from datetime import datetime, timedelta
import time
import sys
import requests  # For news filter

# ================= CONFIG =================
POLYGON_KEY = "your_polygon_api_key_here"  # Polygon for data
OANDA_TOKEN = "your_oanda_access_token_here"  # OANDA for live
OANDA_ACCOUNT_ID = "your_oanda_account_id_here"  # e.g., 101-001-1234567-001
OANDA_HOST = "api-fxpractice.oanda.com"  # Practice; use api-fxtrade for live
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]  # Multi-pair (OANDA format)
TIMEFRAME = "H1"  # OANDA timeframe
DAYS_BACK = 365
DB_FILE = "trades_history.db"

INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.01
MAX_DRAWDOWN = 0.20  # 20% halt
HIGH_ATR_THRESHOLD = 0.002  # Skip if volatility too high

polygon_client = RESTClient(POLYGON_KEY)
oanda = V20Context(OANDA_HOST, 443, token=OANDA_TOKEN, poll_timeout=5) if OANDA_TOKEN != "your_oanda_access_token_here" else None

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pair TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            profit REAL,
            reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(pair, direction, entry_price, exit_price=None, profit=None, reason=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    ts = datetime.now().isoformat()
    c.execute('''
        INSERT INTO trades (timestamp, pair, direction, entry_price, exit_price, profit, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ts, pair, direction, entry_price, exit_price, profit, reason))
    conn.commit()
    conn.close()
    print(f"[{ts}] LOGGED: {direction} {pair} @ {entry_price:.5f} | {reason}")

# ================= NEWS FILTER =================
def check_news_filter():
    # Simple: Search for today's high-impact events
    today = datetime.now().strftime("%Y-%m-%d")
    response = requests.get(f"https://www.forexfactory.com/calendar?day={today}")
    if "high impact" in response.text.lower() or "fomc" in response.text.lower() or "nfp" in response.text.lower():
        print("High-impact news today – skipping trades.")
        return True  # Skip
    return False

# ================= DATA & INDICATORS =================
def fetch_data(pair, days_back=DAYS_BACK):
    polygon_pair = f"C:{pair.replace('_', '')}"
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back + 10)).strftime("%Y-%m-%d")
    print(f"Fetching {polygon_pair} from {from_date} to {to_date}...")
    try:
        aggs = polygon_client.get_aggs(polygon_pair, 1, "hour", from_date, to_date, limit=50000)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    if not aggs:
        return pd.DataFrame()
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').set_index('timestamp')
    print(f"→ {len(df)} candles.")
    return df

def compute_indicators(df):
    df = df.copy()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    df['tr'] = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    return df

def generate_rule_signals(df):
    df = df.copy()
    df['signal'] = 0
    df['ema10_prev'] = df['ema10'].shift(1)
    df['ema30_prev'] = df['ema30'].shift(1)
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    buy = ((df['ema10'] > df['ema30']) & (df['ema10_prev'] <= df['ema30_prev']) & (df['rsi'] > 50) & (df['macd_hist'] > 0) & (df['macd_hist_prev'] <= 0))
    sell = ((df['ema10'] < df['ema30']) & (df['ema10_prev'] >= df['ema30_prev']) & (df['rsi'] < 50) & (df['macd_hist'] < 0) & (df['macd_hist_prev'] >= 0))
    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = -1
    return df

# ================= ML MODEL =================
class SignalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)  # Inputs: EMA diff, RSI, MACD hist, ATR, prev close
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 3)  # 0: hold, 1: buy, 2: sell

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.out(x), dim=1)

def train_ml_model(df):
    df = df.dropna()
    features = df[['ema10', 'ema30', 'rsi', 'macd_hist', 'atr']].values
    labels = df['signal'].shift(-1).fillna(0).astype(int).values + 1  # -1->0, 0->1, 1->2
    X = torch.tensor(features[:-1], dtype=torch.float32)
    y = torch.tensor(labels[:-1], dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SignalNet()
    optimizer = optim.Adam(model.params(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    return model

def get_ml_signal(model, row):
    features = torch.tensor([[row['ema10'], row['ema30'], row['rsi'], row['macd_hist'], row['atr']]], dtype=torch.float32)
    pred = model(features)
    conf, signal = pred.max(1)
    if conf > 0.7:
        return signal.item() - 1  # 0->-1 sell, 1->0 hold, 2->1 buy
    return 0

# ================= BACKTEST / LIVE =================
def process_pair(pair, df, mode="backtest", model=None):
    if check_news_filter() or df.iloc[-1]['atr'] > HIGH_ATR_THRESHOLD:
        return  # Skip
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    lot_size = 0
    trades = []
    peak_balance = balance
    for idx, row in df.iterrows():
        rule_sig = row['signal']
        ml_sig = get_ml_signal(model, row) if model else 0
        sig = ml_sig if ml_sig != 0 else rule_sig  # ML overrides
        if sig == 1 and position <= 0:
            if position == -1:
                profit = (entry_price - row['close']) * lot_size
                log_trade(pair, "SELL CLOSE", entry_price, row['close'], profit, "Opposite")
                balance += profit
                trades.append(profit)
            risk = balance * RISK_PER_TRADE
            lot_size = int(risk / (row['atr'] * 10000))  # Dynamic based on ATR (pips \~ ATR*10000 for sizing)
            entry_price = row['close']
            position = 1
            if mode == "live":
                place_order(pair, lot_size, "BUY", row['atr'])
            log_trade(pair, "BUY", entry_price, reason="Signal")
        elif sig == -1 and position >= 0:
            if position == 1:
                profit = (row['close'] - entry_price) * lot_size
                log_trade(pair, "BUY CLOSE", entry_price, row['close'], profit, "Opposite")
                balance += profit
                trades.append(profit)
            risk = balance * RISK_PER_TRADE
            lot_size = int(risk / (row['atr'] * 10000))
            entry_price = row['close']
            position = -1
            if mode == "live":
                place_order(pair, -lot_size, "SELL", row['atr'])
            log_trade(pair, "SELL", entry_price, reason="Signal")
        # Drawdown check
        if (peak_balance - balance) / peak_balance > MAX_DRAWDOWN:
            print("Max drawdown hit – halting.")
            break
        peak_balance = max(peak_balance, balance)
    # Close at end
    if position != 0:
        exit_price = df.iloc[-1]['close']
        profit = (exit_price - entry_price) * lot_size if position == 1 else (entry_price - exit_price) * lot_size
        log_trade(pair, "CLOSE (end)", entry_price, exit_price, profit, "End")
    if trades:
        win_rate = sum(1 for p in trades if p > 0) / len(trades) * 100
        print(f"{pair} {mode}: {len(trades)} trades | Win: {win_rate:.1f}%")

# ================= OANDA LIVE =================
def place_order(pair, units, side, atr):
    if not oanda:
        print("OANDA not configured – simulation only.")
        return
    order = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": str(entry_price + 2*atr if units > 0 else entry_price - 2*atr)},
            "stopLossOnFill": {"price": str(entry_price - 1.5*atr if units > 0 else entry_price + 1.5*atr)}
        }
    }
    response = oanda.order.create(OANDA_ACCOUNT_ID, order)
    print(f"Order placed: {response.get('orderCreateTransaction', {}).get('id')}")

def close_positions(pair):
    response = oanda.position.close(OANDA_ACCOUNT_ID, pair, longUnits="ALL" if position > 0 else shortUnits="ALL")
    print("Positions closed.")

# ================= MAIN =================
if __name__ == "__main__":
    init_db()
    for pair in PAIRS:
        df = fetch_data(pair)
        if not df.empty:
            df = compute_indicators(df)
            df = generate_rule_signals(df)
            model = train_ml_model(df)
            process_pair(pair, df, mode="backtest", model=model)
    # For live: process_pair(pair, recent_df, "live", model)
    print("Done. For cloud: streamlit run dashboard.py --server.port 80")
