import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AISwingAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model_cache = {}
        self.data_cache = {}

    def fetch_data(self, ticker, period="1y", force_update=True):
        cache_file = f"cache/swing_{ticker}_{period}.pkl"
        if not force_update and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data, timestamp = pickle.load(f)
            if datetime.now() - timestamp < timedelta(minutes=5):
                logging.info(f"Using cached data for {ticker}")
                return data
        
        logging.info(f"Fetching fresh data for {ticker}")
        stock = yf.Ticker(ticker)
        try:
            data = stock.history(period=period)
            if len(data) == 0:
                raise ValueError(f"No data returned for {ticker}")
            if not os.path.exists("cache"):
                os.makedirs("cache")
            with open(cache_file, 'wb') as f:
                pickle.dump((data, datetime.now()), f)
            return data
        except Exception as e:
            logging.error(f"Failed to fetch data for {ticker}: {str(e)}")
            return pd.DataFrame()  # Return empty if fetch fails

    def calculate_features(self, df):
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['BB_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
        df['BB_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
        df['ATR'] = ((df['High'] - df['Low']).rolling(14).mean())
        df['Target'] = np.where(df['Close'].shift(-3) > df['Close'], 1, 0)
        df = df.dropna()
        if len(df) < 60:
            logging.warning(f"Insufficient data for {ticker} after dropping NaN: {len(df)} rows")
        return df

    def calculate_rsi(self, prices, period):
        if len(prices) < period + 1:
            return pd.Series(50, index=prices.index)  # Default to neutral RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_model(self, ticker):
        if ticker in self.model_cache:
            return self.model_cache[ticker]
        
        df = self.fetch_data(ticker, force_update=True)
        if df.empty:
            logging.error(f"No data available for {ticker}—returning default model")
            return xgb.XGBClassifier()  # Default model if data fails
        df = self.calculate_features(df)
        if len(df) < 60:
            logging.error(f"Insufficient data for {ticker}—cannot train model")
            return xgb.XGBClassifier()  # Default model if data insufficient
        features = ['Returns', 'EMA20', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']
        X = self.scaler.fit_transform(df[features])
        y = df['Target']
        
        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                 n_jobs=-1,  # Use all available cores
                                 use_label_encoder=False,  # Modern XGBoost
                                 eval_metric='logloss')
        model.fit(X, y)
        self.model_cache[ticker] = model
        logging.info(f"Trained model for {ticker} with {len(X)} samples")
        return model

    def get_signal(self, ticker):
        df = self.fetch_data(ticker, force_update=True)
        if df.empty:
            logging.error(f"No data for {ticker}—defaulting to 'Hold'")
            return "Hold", 0.5  # Default to neutral
        df = self.calculate_features(df)
        if len(df) < 60:
            logging.warning(f"Insufficient data for {ticker}—defaulting to 'Hold'")
            return "Hold", 0.5
        model = self.train_model(ticker)
        X = self.scaler.transform(df[['Returns', 'EMA20', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']].iloc[-1:])
        pred = model.predict_proba(X)[0]
        logging.info(f"{ticker} Signal Probabilities - Buy: {pred[1]:.3f}, Sell: {pred[0]:.3f}")
        signal = "Strong Buy" if pred[1] > 0.8 else "Buy" if pred[1] > 0.7 else "Sell" if pred[0] > 0.7 else "Hold"
        return signal, max(pred)

    def analyze_ticker(self, ticker):
        df = self.fetch_data(ticker, force_update=True)
        if df.empty:
            logging.error(f"No data for {ticker}—returning default analysis")
            return {
                'signal': "Hold",
                'confidence': 0.5,
                'volatility': 0,
                'visualization': "",
                'summary': f"Analysis for {ticker}: No data available—recommend 'Hold'.\nTip: Ensure ticker is valid.",
                'report': "AI Report: Unable to analyse due to missing data.",
                'currency': "$",
                'current_price': 0,
                'day_range': "N/A",
                'week_52_range': "N/A",
                'company_name': ticker
            }
        df = self.calculate_features(df)
        if len(df) < 60:
            logging.warning(f"Insufficient data for {ticker}—defaulting to 'Hold'")
            return {
                'signal': "Hold",
                'confidence': 0.5,
                'volatility': 0,
                'visualization': "",
                'summary': f"Analysis for {ticker}: Insufficient data—recommend 'Hold'.\nTip: Add more historical data.",
                'report': "AI Report: Limited data prevents accurate analysis.",
                'currency': "$",
                'current_price': df['Close'].iloc[-1] if not df.empty else 0,
                'day_range': "N/A",
                'week_52_range': "N/A",
                'company_name': ticker
            }
        model = self.train_model(ticker)
        arima = ARIMA(df['Close'], order=(5,1,0)).fit()
        forecast = arima.forecast(steps=7)
        X = self.scaler.transform(df[['Returns', 'EMA20', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']].iloc[-1:])
        pred = model.predict_proba(X)[0]
        stock = yf.Ticker(ticker)
        info = stock.info

        # Enhanced currency detection
        currency = info.get('currency', 'USD')
        if currency in ['GBp', 'GBP']:
            currency = 'GBP'  # Normalize GBP variants
        elif currency not in ['USD', 'GBP', 'EUR', 'CAD', 'JPY']:
            currency = 'USD'  # Default to USD for other currencies
        currency_symbol = "£" if currency == "GBP" else "$" if currency == "USD" else "€" if currency == "EUR" else "C$" if currency == "CAD" else "¥" if currency == "JPY" else "$"
        logging.info(f"{ticker} Currency detected: {currency} ({currency_symbol})")

        current_price = df['Close'].iloc[-1]
        day_range = f"{df['Low'].iloc[-1]:,.2f} - {df['High'].iloc[-1]:,.2f}"
        week_52_range = f"{info.get('fiftyTwoWeekLow', 'N/A'):.2f} - {info.get('fiftyTwoWeekHigh', 'N/A'):.2f}"

        fig = make_subplots(rows=2, cols=1, subplot_titles=('7-Day Price Forecast', 'RSI & MACD'))
        fig.add_trace(go.Scatter(x=df.index[-30:], y=df['Close'][-30:], name='Price'), row=1, col=1)
        future_dates = pd.date_range(start=df.index[-1], periods=8)[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='7-Day Forecast', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[-30:], y=df['RSI_14'][-30:], name='RSI'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index[-30:], y=df['MACD'][-30:], name='MACD'), row=2, col=1)
        fig.update_layout(title=f"{ticker} Swing Analysis", height=800, width=900, showlegend=True)
        fig.write_html(f"swing_{ticker}.html")

        signal = "Strong Buy" if pred[1] > 0.8 else "Buy" if pred[1] > 0.7 else "Sell" if pred[0] > 0.7 else "Hold"
        summary = (
            f"Analysis for {ticker} ({info.get('longName', ticker)}):\n"
            f"- Recommendation: {signal} (Confidence: {max(pred):.2%})\n"
            f"- Current Price: {currency_symbol}{current_price:,.2f}\n"
            f"- Day Range: {currency_symbol}{day_range}\n"
            f"- 52-Week Range: {currency_symbol}{week_52_range}\n"
            f"- Volatility (ATR): {df['ATR'].iloc[-1]:.2f}\n"
            f"Tip: 'Hold' suggests waiting for a clearer trend."
        )
        report = (
            f"AI Report for {ticker}:\n"
            f"The model predicts a {signal.lower()} signal based on recent price action.\n"
            f"- RSI ({df['RSI_14'].iloc[-1]:.2f}): {'Overbought—possible peak' if df['RSI_14'].iloc[-1] > 70 else 'Oversold—potential bounce' if df['RSI_14'].iloc[-1] < 30 else 'Neutral'}.\n"
            f"- MACD Trend: {'Bullish—upward momentum' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'Bearish—downward pressure'}.\n"
            f"- ATR ({df['ATR'].iloc[-1]:.2f}): {'High—expect volatility' if df['ATR'].iloc[-1] > 1 else 'Low—stable moves'}.\n"
            f"Consider timing your trade based on these signals."
        )

        logging.info(f"{ticker} Final Signal: {signal} (Buy Prob: {pred[1]:.3f}, Sell Prob: {pred[0]:.3f})")
        return {
            'signal': signal,
            'confidence': max(pred),
            'volatility': df['ATR'].iloc[-1],
            'visualization': f"swing_{ticker}.html",
            'summary': summary,
            'report': report,
            'currency': currency_symbol,
            'current_price': current_price,
            'day_range': day_range,
            'week_52_range': week_52_range,
            'company_name': info.get('longName', ticker)
        }

if __name__ == "__main__":
    analyzer = AISwingAnalysis()
    result = analyzer.analyze_ticker("TSLA")
    print(result)