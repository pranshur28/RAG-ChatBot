import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

class MarketDataHandler:
    def __init__(self):
        self.cache = {}

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock market data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
                
            # Calculate basic technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            
            # Store in cache
            self.cache[symbol] = {
                'last_updated': datetime.now(),
                'data': data
            }
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_market_summary(self, symbol: str) -> Dict:
        """Generate a market summary for the given symbol"""
        data = self.fetch_stock_data(symbol)
        if data is None:
            return {"error": "Unable to fetch market data"}

        latest = data.iloc[-1]
        prev_day = data.iloc[-2]

        return {
            "symbol": symbol,
            "current_price": round(latest['Close'], 2),
            "daily_change": round(((latest['Close'] - prev_day['Close']) / prev_day['Close']) * 100, 2),
            "volume": latest['Volume'],
            "sma_20": round(latest['SMA_20'], 2) if not pd.isna(latest['SMA_20']) else None,
            "sma_50": round(latest['SMA_50'], 2) if not pd.isna(latest['SMA_50']) else None,
            "timestamp": latest.name.strftime("%Y-%m-%d %H:%M:%S")
        }
