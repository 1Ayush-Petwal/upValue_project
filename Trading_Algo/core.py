import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import json
import random
from functools import wraps


def get_markets():
    return {
        "India National Stock Exchange": "NS",
        "US Stock Market": "",
        "Bombay Stock Exchange": "BO",
        "Malaysian Stock Exchange": "KL"
    }

def get_historical_price(df, days_ago):
    try:
        if df.empty:
            return None
        if len(df) > days_ago:
            return df['Close'].iloc[-days_ago]
        return df['Close'].iloc[0]
    except Exception:
        return None

def get_currency_symbol(market_index):
    currency_map = {
        "": "$",  # US Market
        "KL": "RM",  # Malaysian Market
        "NS": "₹",  # Indian NSE
        "BO": "₹"  # Indian BSE
    }
    return currency_map.get(market_index, "$")

def get_ticker_symbol(symbol, market_index):
    # Malaysian stock mapping
    malaysian_tickers = {
        "MAYBANK": "1155.KL",
        "CIMB": "1023.KL",
        "PBBANK": "1295.KL",
        "TENAGA": "5347.KL",
        "PCHEM": "5183.KL",
        "IOICORP": "1961.KL",
        "KLK": "2445.KL",
        "SIME": "4197.KL",
        "GENTING": "3182.KL",
        "HAPSENG": "3034.KL"
    }
    
    # US stock mapping
    us_tickers = {
        "APPLE": "AAPL",
        "MICROSOFT": "MSFT",
        "GOOGLE": "GOOGL",
        "AMAZON": "AMZN",
        "META": "META",
        "TESLA": "TSLA",
        "NVIDIA": "NVDA",
        "JPMORGAN": "JPM",
        "VISA": "V",
        "WALMART": "WMT"
    }
    
    # Indian stock mapping
    indian_tickers = {
        "SBI": "SBIN.NS",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC": "HDFCBANK.NS",
        "INFOSYS": "INFY.NS",
        "ICICI": "ICICIBANK.NS",
        "ITC": "ITC.NS",
        "KOTAK": "KOTAKBANK.NS",
        "AXIS": "AXISBANK.NS",
        "L&T": "LT.NS",
        "BHARTI": "BHARTIARTL.NS"
    }
    
    symbol = symbol.upper()
    
    if market_index == "KL":
        return malaysian_tickers.get(symbol, f"{symbol}.KL")
    elif market_index == "":
        return us_tickers.get(symbol, symbol)
    elif market_index == "NS":
        return indian_tickers.get(symbol, f"{symbol}.NS")
    else:  # BO
        return indian_tickers.get(symbol, f"{symbol}.BO")

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
            return None
        return wrapper
    return decorator



def get_yahoo_finance_history(symbol, market_index, range_period="6mo", interval="1d"):
    """
    Fetch historical data directly from Yahoo Finance API (API-compatible version)
    """
    # Check if symbol already has a suffix
    if "." in symbol:
        # Symbol already has suffix, use it as is
        full_symbol = symbol
    else:
        # Symbol doesn't have suffix, add appropriate one
        suffixes = {
            "NS": ".NS",
            "": "",
            "KL": ".KL",
            "BO": ".BO"
        }
        suffix = suffixes.get(market_index, "")
        full_symbol = symbol + suffix
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{full_symbol}?interval={interval}&range={range_period}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"Failed to fetch data for {full_symbol} (Status: {r.status_code})")
            return None
        
        data = r.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        indicators = result['indicators']['quote'][0]
        adjclose = result['indicators'].get('adjclose', [{}])[0].get('adjclose', [])

        df = pd.DataFrame({
            "Date": [datetime.fromtimestamp(ts) for ts in timestamps],
            "Open": indicators['open'],
            "High": indicators['high'],
            "Low": indicators['low'],
            "Close": indicators['close'],
            "Adj Close": adjclose if adjclose else indicators['close'],
            "Volume": indicators['volume']
        })
        
        # Set Date as index and remove any NaN values
        df.set_index('Date', inplace=True)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching/parsing data for {full_symbol}: {str(e)}")
        return None

@retry_on_failure(max_retries=3, delay=2)
def get_mean_reversion_signals_api(symbol, index="NS", lookback_days=60, investment_amount=100000):
    """
    API-compatible version of mean reversion analysis using Yahoo Finance direct API
    """
    try:
        # Get the correct ticker symbol
        ticker_symbol = get_ticker_symbol(symbol, index)
        currency_symbol = get_currency_symbol(index)
        
        # Calculate range period based on lookback days
        buffer_days = max(15, int(lookback_days * 0.25))
        total_days = lookback_days + buffer_days
        
        # Convert days to Yahoo Finance range format
        if total_days <= 7:
            range_period = "7d"
        elif total_days <= 30:
            range_period = "1mo"
        elif total_days <= 90:
            range_period = "3mo"
        elif total_days <= 180:
            range_period = "6mo"
        else:
            range_period = "1y"
        
        # Fetch data using our integrated function
        df = get_yahoo_finance_history(ticker_symbol, index, range_period=range_period)
        
        if df is None or df.empty:
            print(f"No data available for {symbol} (Ticker: {ticker_symbol})")
            return None

        # Trim data to the requested lookback period
        if len(df) > lookback_days:
            df = df.tail(lookback_days)

        # Dynamic SMA periods based on lookback days
        if lookback_days <= 20:
            short_sma_period = 3
            long_sma_period = 8
        elif lookback_days <= 40:
            short_sma_period = 5
            long_sma_period = 10
        else:
            short_sma_period = min(5, max(3, int(lookback_days * 0.08)))
            long_sma_period = min(15, max(8, int(lookback_days * 0.15)))

        # Calculate current and historical prices
        current_price = df['Close'].iloc[-1]
        price_1d_ago = get_historical_price(df, 1) or current_price
        price_5d_ago = get_historical_price(df, min(5, lookback_days)) or current_price
        price_10d_ago = get_historical_price(df, min(10, lookback_days)) or current_price
        price_30d_ago = get_historical_price(df, min(30, lookback_days)) or current_price
        price_60d_ago = get_historical_price(df, min(60, lookback_days)) or current_price

        # Calculate technical indicators
        df['SMA_Short'] = df['Close'].rolling(window=short_sma_period, min_periods=1).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_sma_period, min_periods=1).mean()
        df['STD_Long'] = df['Close'].rolling(window=long_sma_period, min_periods=1).std()

        # Bollinger Bands
        df['Upper_Band'] = df['SMA_Long'] + (df['STD_Long'] * 2)
        df['Lower_Band'] = df['SMA_Long'] - (df['STD_Long'] * 2)

        # Z-Score calculation
        df['Z_Score'] = np.where(
            df['STD_Long'] != 0,
            (df['Close'] - df['SMA_Long']) / df['STD_Long'],
            0
        )
        current_zscore = df['Z_Score'].iloc[-1]

        # Volatility calculation
        volatility_window = min(10, max(5, int(lookback_days * 0.1)))
        recent_volatility = df['Close'].pct_change().tail(volatility_window).std() * np.sqrt(252) * 100

        # Dynamic Z-Score threshold
        z_score_threshold = 0.8
        if lookback_days > 30:
            z_score_threshold = min(1.2, 0.8 + (lookback_days - 30) / 100)

        # Determine action
        action = "HOLD"
        if current_zscore <= -z_score_threshold:
            action = "BUY"
        elif current_zscore >= z_score_threshold:
            action = "SELL"

        # Calculate expected reversion and potential return
        expected_reversion = df['SMA_Long'].iloc[-1]
        potential_return = abs((expected_reversion - current_price) / current_price * 100) if current_price > 0 else 0

        # Risk management calculations
        stop_loss_pct = 0.7 + (0.3 * recent_volatility / 15)
        take_profit_pct = potential_return * 0.6

        stop_loss = round(current_price * (1 - stop_loss_pct/100), 2) if action == "BUY" else round(current_price * (1 + stop_loss_pct/100), 2)
        take_profit = round(current_price * (1 + take_profit_pct/100), 2) if action == "BUY" else round(current_price * (1 - take_profit_pct/100), 2)

        # VWAP calculation
        vwap_period = min(lookback_days, 20)
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=vwap_period).sum() / df['Volume'].rolling(window=vwap_period).sum()
        current_vwap = df['VWAP'].iloc[-1]

        # Prepare result with Python native types (not numpy types)
        # Note: Excluding 'Data' DataFrame for API compatibility - too large for JSON response
        result = {
            'Symbol': symbol,
            'Current Price': round(float(current_price), 2),
            '1D Ago Price': round(float(price_1d_ago), 2),
            '5D Ago Price': round(float(price_5d_ago), 2),
            '10D Ago Price': round(float(price_10d_ago), 2),
            '30D Ago Price': round(float(price_30d_ago), 2),
            '60D Ago Price': round(float(price_60d_ago), 2),
            'Volatility %': round(float(recent_volatility), 2),
            f'SMA_{short_sma_period}': round(float(df['SMA_Short'].iloc[-1]), 2),
            f'SMA_{long_sma_period}': round(float(df['SMA_Long'].iloc[-1]), 2),
            'VWAP': round(float(current_vwap), 2) if not np.isnan(current_vwap) else None,
            'Z-Score': round(float(current_zscore), 2),
            'Z-Score Threshold': round(float(z_score_threshold), 2),
            'Action': action,
            'Expected Price': round(float(expected_reversion), 2),
            'Potential Return %': round(float(potential_return), 2),
            'Stop Loss': float(stop_loss),
            'Take Profit': float(take_profit),
            'Lookback Days': int(lookback_days),
            'Short SMA Period': int(short_sma_period),
            'Long SMA Period': int(long_sma_period),
            'Currency': currency_symbol,
            # Include some basic chart data points for the last few days
            'Recent_Prices': {
                'dates': [d.strftime('%Y-%m-%d') for d in df.index[-5:].tolist()],
                'prices': [round(float(p), 2) for p in df['Close'].tail(5).tolist()],
                'sma_short': [round(float(p), 2) for p in df['SMA_Short'].tail(5).tolist()],
                'sma_long': [round(float(p), 2) for p in df['SMA_Long'].tail(5).tolist()]
            }
        }
        return result
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def run_analysis_api(symbols, lookback_days=60, investment_per_stock=100000, market_index="NS"):
    """API-compatible version of run analysis"""
    analysis_results = []

    print(f"Running analysis with {lookback_days} days lookback period...")
    for symbol in symbols:
        result = get_mean_reversion_signals_api(
            symbol, 
            index=market_index, 
            lookback_days=lookback_days, 
            investment_amount=investment_per_stock
        )
        if result is not None:
            analysis_results.append(result)

    return analysis_results

def get_yahoo_finance_suggestions_api(query, market_index):
    """API-compatible version of Yahoo Finance suggestions"""
    print(f"DEBUG: get_yahoo_finance_suggestions_api called with query='{query}', market_index='{market_index}'")
    try:
        market_suffixes = {
            "NS": ".NS",
            "": "",
            "KL": ".KL",
            "BO": ".BO"
        }
        suffix = market_suffixes.get(market_index, "")
        
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"DEBUG: Making request to URL: {url}")
        response = requests.get(url, headers=headers)
        print(f"DEBUG: Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            suggestions = []
            print(f"DEBUG: Response data keys: {data.keys() if data else 'None'}")
            
            if 'quotes' in data:
                print(f"DEBUG: Found {len(data['quotes'])} quotes in response")
                for quote in data['quotes']:
                    if 'symbol' in quote and 'longname' in quote:
                        if suffix == "" and "." not in quote['symbol']:
                            suggestions.append({
                                'symbol': quote['symbol'],
                                'name': quote['longname']
                            })
                        elif suffix in quote['symbol']:
                            suggestions.append({
                                'symbol': quote['symbol'].replace(suffix, ""),
                                'name': quote['longname']
                            })
            else:
                print("DEBUG: No 'quotes' key found in response")
            
            print(f"DEBUG: Returning {len(suggestions)} suggestions from Yahoo Finance")
            return suggestions[:5]
        else:
            print(f"DEBUG: Yahoo Finance request failed with status {response.status_code}")
    except Exception as e:
        print(f"Error fetching suggestions: {str(e)}")
    return []

def get_stock_recommendations_api(search_term, market_index):
    """API-compatible version of stock recommendations"""
    print(f"DEBUG: get_stock_recommendations_api called with search_term='{search_term}', market_index='{market_index}'")
    
    if not search_term:
        print("DEBUG: Empty search term, returning empty list")
        return []
    
    # First try Yahoo Finance API
    suggestions = get_yahoo_finance_suggestions_api(search_term, market_index)
    print(f"DEBUG: Yahoo Finance returned {len(suggestions)} suggestions")
    
    # If no suggestions from Yahoo, fall back to local database
    if not suggestions:
        print("DEBUG: No Yahoo Finance suggestions, falling back to local database")
        all_stocks = {
            "NS": {
                "SBI": "State Bank of India",
                "RELIANCE": "Reliance Industries",
                "TCS": "Tata Consultancy Services",
                "HDFC": "HDFC Bank",
                "INFOSYS": "Infosys Limited",
                "ICICI": "ICICI Bank",
                "ITC": "ITC Limited",
                "KOTAK": "Kotak Mahindra Bank",
                "AXIS": "Axis Bank",
                "L&T": "Larsen & Toubro",
                "BHARTI": "Bharti Airtel",
            },
            "": {
                "APPLE": "Apple Inc.",
                "MICROSOFT": "Microsoft Corporation",
                "GOOGLE": "Alphabet Inc.",
                "AMAZON": "Amazon.com Inc.",
                "META": "Meta Platforms Inc.",
                "TESLA": "Tesla Inc.",
                "NVIDIA": "NVIDIA Corporation",
                "JPMORGAN": "JPMorgan Chase & Co.",
                "VISA": "Visa Inc.",
                "WALMART": "Walmart Inc.",
            },
            "KL": {
                "MAYBANK": "Malayan Banking Berhad",
                "CIMB": "CIMB Group Holdings",
                "PBBANK": "Public Bank Berhad",
                "TENAGA": "Tenaga Nasional",
                "PCHEM": "Petronas Chemicals",
                "IOICORP": "IOI Corporation",
                "KLK": "Kuala Lumpur Kepong",
                "SIME": "Sime Darby",
                "GENTING": "Genting Berhad",
                "HAPSENG": "Hap Seng Consolidated",
            },
            "BO": {
                "RELIANCE": "Reliance Industries",
                "TCS": "Tata Consultancy Services",
                "HDFC": "HDFC Bank",
                "INFOSYS": "Infosys Limited",
                "ICICI": "ICICI Bank",
                "ITC": "ITC Limited",
                "KOTAK": "Kotak Mahindra Bank",
                "AXIS": "Axis Bank",
                "L&T": "Larsen & Toubro",
                "BHARTI": "Bharti Airtel"
            }
        }
        
        market_stocks = all_stocks.get(market_index, {})
        print(f"DEBUG: Market '{market_index}' has {len(market_stocks)} stocks")
        search_term_upper = search_term.upper()
        print(f"DEBUG: Searching for '{search_term_upper}' in local database")
        
        suggestions = []
        for symbol, company in market_stocks.items():
            if (search_term_upper in symbol.upper() or 
                search_term_upper in company.upper()):
                suggestions.append({
                    'symbol': symbol,
                    'name': company
                })
                print(f"DEBUG: Found match: {symbol} - {company}")
        
        print(f"DEBUG: Local database search found {len(suggestions)} matches")
        
        suggestions.sort(key=lambda x: (
            not x['symbol'].startswith(search_term_upper),
            not x['name'].upper().startswith(search_term_upper),
            len(x['symbol'])
        ))
    
    final_suggestions = suggestions[:5]
    print(f"DEBUG: Returning {len(final_suggestions)} final suggestions")
    return final_suggestions

def get_default_stocks(market: str):
    default_stocks = {
        "NS": ["SBI", "RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI", "WIPRO","HCLTECH","SUNPHARMA",
"TATAMOTORS"],
        "US": ["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META", "TESLA", "NVIDIA", "JPMORGAN", "VISA", "WALMART", "NETFLIX",
                "INTEL",
                "AMD",
                "COCA-COLA",

                "DISNEY"],
        "BO": ["RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
        "KL": ["MAYBANK", "CIMB", "PBBANK", "TENAGA", "PCHEM", "IOICORP", "KLK", "SIME", "GENTING", "HAPSENG", "AXIATA",
                "DIGI",
                "MAXIS",
                "PETGAS",
                "MISC"]
    }
    return default_stocks.get(market, [])