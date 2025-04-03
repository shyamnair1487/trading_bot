import ccxt
import pandas as pd
from config.settings import settings

def fetch_historical_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data from exchange"""
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def save_data(data: pd.DataFrame, filename: str):
    """Save historical data to parquet"""
    data.to_parquet(f"data/historical/{filename}")

def load_data(filename: str) -> pd.DataFrame:
    """Load historical data from parquet"""
    return pd.read_parquet(f"data/historical/{filename}")