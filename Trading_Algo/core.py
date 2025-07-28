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
    print("The Symbol got on calling get_ticker_symbol")
    
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

@retry_on_failure(max_retries=3, delay=2)
def _get_mean_reversion_signals_(symbol, index="NS", lookback_days=60, investment_amount=100000):
        # Get the correct ticker symbol
            ticker_symbol = get_ticker_symbol(symbol, index)
            ticker = yf.Ticker(ticker_symbol)    #### 1st Call
            currency_symbol = get_currency_symbol(index)
            
            end_date = datetime.now()
            buffer_days = max(15, int(lookback_days * 0.25))
            start_date = end_date - timedelta(days=lookback_days + buffer_days)
            
            try:
                df = ticker.history(start=start_date, end=end_date, interval="1d")
                print(f"DEBUG: Successfully fetched data, shape: {df.shape}")

                if df is None or df.empty:
                    # st.error(f"No data available for {symbol} (Ticker: {ticker_symbol})")
                    return None
            except Exception as e:
                # st.error(f"Error fetching data for {symbol} (Ticker: {ticker_symbol}): {str(e)}")
                return None

            if lookback_days <= 20:
                short_sma_period = 3
                long_sma_period = 8
            elif lookback_days <= 40:
                short_sma_period = 5
                long_sma_period = 10
            else:
                short_sma_period = min(5, max(3, int(lookback_days * 0.08)))
                long_sma_period = min(15, max(8, int(lookback_days * 0.15)))

            current_price = df['Close'].iloc[-1]
            price_1d_ago = get_historical_price(df, 1) or current_price
            price_5d_ago = get_historical_price(df, min(5, lookback_days)) or current_price
            price_10d_ago = get_historical_price(df, min(10, lookback_days)) or current_price
            price_30d_ago = get_historical_price(df, min(30, lookback_days)) or current_price
            price_60d_ago = get_historical_price(df, min(60, lookback_days)) or current_price

            df['SMA_Short'] = df['Close'].rolling(window=short_sma_period, min_periods=1).mean()
            df['SMA_Long'] = df['Close'].rolling(window=long_sma_period, min_periods=1).mean()
            df['STD_Long'] = df['Close'].rolling(window=long_sma_period, min_periods=1).std()

            df['Upper_Band'] = df['SMA_Long'] + (df['STD_Long'] * 2)
            df['Lower_Band'] = df['SMA_Long'] - (df['STD_Long'] * 2)

            df['Z_Score'] = np.where(
                df['STD_Long'] != 0,
                (df['Close'] - df['SMA_Long']) / df['STD_Long'],
                0
            )
            current_zscore = df['Z_Score'].iloc[-1]

            volatility_window = min(10, max(5, int(lookback_days * 0.1)))
            recent_volatility = df['Close'].pct_change().tail(volatility_window).std() * np.sqrt(252) * 100

            z_score_threshold = 0.8
            if lookback_days > 30:
                z_score_threshold = min(1.2, 0.8 + (lookback_days - 30) / 100)

            action = "HOLD"

            if current_zscore <= -z_score_threshold:
                action = "BUY"
            elif current_zscore >= z_score_threshold:
                action = "SELL"

            expected_reversion = df['SMA_Long'].iloc[-1]
            potential_return = abs((expected_reversion - current_price) / current_price * 100) if current_price > 0 else 0

            stop_loss_pct = 0.7 + (0.3 * recent_volatility / 15)
            take_profit_pct = potential_return * 0.6

            stop_loss = round(current_price * (1 - stop_loss_pct/100), 2) if action == "BUY" else round(current_price * (1 + stop_loss_pct/100), 2)
            take_profit = round(current_price * (1 + take_profit_pct/100), 2) if action == "BUY" else round(current_price * (1 - take_profit_pct/100), 2)

            vwap_period = min(lookback_days, 20)
            df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=vwap_period).sum() / df['Volume'].rolling(window=vwap_period).sum()
            current_vwap = df['VWAP'].iloc[-1]

            result = {
                'Symbol': symbol,
                'Current Price': round(current_price, 2),
                '1D Ago Price': round(price_1d_ago, 2),
                '5D Ago Price': round(price_5d_ago, 2),
                '10D Ago Price': round(price_10d_ago, 2),
                '30D Ago Price': round(price_30d_ago, 2),
                '60D Ago Price': round(price_60d_ago, 2),
                'Volatility %': round(recent_volatility, 2),
                f'SMA_{short_sma_period}': round(df['SMA_Short'].iloc[-1], 2),
                f'SMA_{long_sma_period}': round(df['SMA_Long'].iloc[-1], 2),
                'VWAP': round(current_vwap, 2) if not np.isnan(current_vwap) else None,
                'Z-Score': round(current_zscore, 2),
                'Z-Score Threshold': round(z_score_threshold, 2),
                'Action': action,
                'Expected Price': round(expected_reversion, 2),
                'Potential Return %': round(potential_return, 2),
                'Stop Loss': stop_loss,
                'Take Profit': take_profit,
                'Lookback Days': lookback_days,
                'Short SMA Period': short_sma_period,
                'Long SMA Period': long_sma_period,
                'Data': df,
                'Currency': currency_symbol
            }
            return result

def _run_analysis_(symbols, lookback_days=60, investment_per_stock=100000, market_index="NS"):
    analysis_results = []
    for symbol in symbols:
        result = _get_mean_reversion_signals_(symbol, index=market_index, lookback_days=lookback_days, investment_amount=investment_per_stock)
        if result is not None:
            analysis_results.append(result)

    return analysis_results

def get_yahoo_finance_suggestions(query, market_index):
    try:
        # Map market indices to Yahoo Finance suffixes
        market_suffixes = {
            "NS": ".NS",  # NSE
            "": "",       # US
            "KL": ".KL",  # Malaysia
            "BO": ".BO"   # BSE
        }
        suffix = market_suffixes.get(market_index, "")
        
        # Construct the Yahoo Finance search URL
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            suggestions = []
            
            if 'quotes' in data:
                for quote in data['quotes']:
                    if 'symbol' in quote and 'longname' in quote:
                        # Check if the symbol matches our market
                        if suffix == "" and "." not in quote['symbol']:  # US market
                            suggestions.append({
                                'symbol': quote['symbol'],
                                'name': quote['longname']
                            })
                        elif suffix in quote['symbol']:  # Other markets
                            suggestions.append({
                                'symbol': quote['symbol'].replace(suffix, ""),
                                'name': quote['longname']
                            })
            
            return suggestions[:5]  # Return top 5 suggestions
    except Exception as e:
        # st.error(f"Error fetching suggestions: {str(e)}")
        print(e)
    return []

def get_stock_recommendations(search_term, market_index):
    if not search_term:
        return []
    
    # First try Yahoo Finance API
    suggestions = get_yahoo_finance_suggestions(search_term, market_index)
    
    # If no suggestions from Yahoo, fall back to our local database
    if not suggestions:
        all_stocks = {
            "NS": {  # Indian NSE
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
                "WIPRO": "Wipro Limited",
                "HCLTECH": "HCL Technologies",
                "SUNPHARMA": "Sun Pharmaceutical",
                "TATAMOTORS": "Tata Motors"
            },
            "": {  # US Market
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
                "NETFLIX": "Netflix Inc.",
                "INTEL": "Intel Corporation",
                "AMD": "Advanced Micro Devices",
                "COCA-COLA": "The Coca-Cola Company",
                "DISNEY": "The Walt Disney Company"
            },
            "KL": {  # Malaysian Market
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
                "AXIATA": "Axiata Group",
                "DIGI": "DiGi.Com",
                "MAXIS": "Maxis Berhad",
                "PETGAS": "Petronas Gas",
                "MISC": "MISC Berhad"
            },
            "BO": {  # Bombay Stock Exchange
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
        search_term = search_term.upper()
        
        suggestions = []
        for symbol, company in market_stocks.items():
            if (search_term in symbol.upper() or 
                search_term in company.upper()):
                suggestions.append({
                    'symbol': symbol,
                    'name': company
                })
        
        suggestions.sort(key=lambda x: (
            not x['symbol'].startswith(search_term),
            not x['name'].upper().startswith(search_term),
            len(x['symbol'])
        ))
    
    return suggestions[:5]

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