# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from datetime import datetime, timedelta
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import requests
# from bs4 import BeautifulSoup
# import time
# import json

# from core import _get_mean_reversion_signals_, _run_analysis_

# st.set_page_config(
#     page_title="Mean Reversion Strategy Analyzer",
#     page_icon="📈",
#     layout="wide",
# ) 

# def get_mean_reversion_signals(symbol, index="NS", lookback_days=60, investment_amount=100000):
#     try:
#         with st.spinner(f"Fetching data for {symbol}..."):
#             return _get_mean_reversion_signals_(symbol, index=index, lookback_days=lookback_days, investment_amount=investment_amount);
#     except Exception as e:
#         st.error(f"Error processing {symbol}: {str(e)}")
#     return None

# def run_analysis(symbols, lookback_days=60, investment_per_stock=100000, market_index="NS"):
#     with st.spinner(f"Running analysis with {lookback_days} days lookback period..."):
#         return _run_analysis_(symbols=symbols, lookback_days=lookback_days, investment_per_stock=investment_per_stock, market_index=market_index)



# def create_plotly_chart(result):
#     symbol = result['Symbol']
#     df = result['Data'].copy()
#     short_sma_period = result['Short SMA Period']
#     long_sma_period = result['Long SMA Period']
#     currency_symbol = result['Currency']

#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
#                         vertical_spacing=0.08, 
#                         subplot_titles=(f'{symbol} Price Analysis', 'Z-Score'),
#                         row_heights=[0.7, 0.3])

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='royalblue', width=2)),
#         row=1, col=1
#     )

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['SMA_Short'], name=f'SMA {short_sma_period}',
#                   line=dict(color='orange', width=1.5, dash='dash')),
#         row=1, col=1
#     )

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['SMA_Long'], name=f'SMA {long_sma_period}',
#                   line=dict(color='red', width=1.5, dash='dash')),
#         row=1, col=1
#     )

#     if 'VWAP' in df.columns:
#         fig.add_trace(
#             go.Scatter(x=df.index, y=df['VWAP'], name='VWAP',
#                       line=dict(color='purple', width=1.5, dash='dot')),
#             row=1, col=1
#         )

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['Upper_Band'], name='Upper Band',
#                   line=dict(color='green', width=1, dash='dot')),
#         row=1, col=1
#     )

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['Lower_Band'], name='Lower Band',
#                   line=dict(color='green', width=1, dash='dot')),
#         row=1, col=1
#     )

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['Upper_Band'], 
#                   fill='tonexty', fillcolor='rgba(0,128,0,0.1)',
#                   line=dict(color='rgba(0,0,0,0)'), showlegend=False),
#         row=1, col=1
#     )

#     current_price = result['Current Price']
#     stop_loss = result['Stop Loss']
#     take_profit = result['Take Profit']

#     fig.add_hline(y=current_price, line_width=1, line_color="black", line_dash="solid",
#                  annotation_text="Current", row=1, col=1)

#     fig.add_hline(y=stop_loss, line_width=1.5, line_color="red", line_dash="dash",
#                  annotation_text="Stop Loss", row=1, col=1)

#     fig.add_hline(y=take_profit, line_width=1.5, line_color="green", line_dash="dash",
#                  annotation_text="Take Profit", row=1, col=1)

#     if result['Action'] != "HOLD":
#         marker_color = 'green' if result['Action'] == 'BUY' else 'red'
#         marker_symbol = 'triangle-up' if result['Action'] == 'BUY' else 'triangle-down'

#         fig.add_trace(
#             go.Scatter(x=[df.index[-1]], y=[current_price], 
#                       mode='markers', marker=dict(color=marker_color, size=15, symbol=marker_symbol),
#                       name=result['Action']),
#             row=1, col=1
#         )

#     fig.add_trace(
#         go.Scatter(x=df.index, y=df['Z_Score'], name='Z-Score', line=dict(color='purple', width=2)),
#         row=2, col=1
#     )

#     z_threshold = result['Z-Score Threshold']
#     fig.add_hline(y=0, line_width=1, line_color="black", line_dash="solid", row=2, col=1)
#     fig.add_hline(y=z_threshold, line_width=1, line_color="red", line_dash="dash", row=2, col=1)
#     fig.add_hline(y=-z_threshold, line_width=1, line_color="green", line_dash="dash", row=2, col=1)

#     df_threshold = pd.DataFrame({
#         'x': df.index,
#         'y1': [z_threshold] * len(df),
#         'y2': [3] * len(df)
#     })

#     fig.add_trace(
#         go.Scatter(x=df_threshold['x'], y=df_threshold['y1'], 
#                   fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
#                   line=dict(color='rgba(0,0,0,0)'), showlegend=False,
#                   name='Sell Zone'),
#         row=2, col=1
#     )

#     df_threshold = pd.DataFrame({
#         'x': df.index,
#         'y1': [-z_threshold] * len(df),
#         'y2': [-3] * len(df)
#     })

#     fig.add_trace(
#         go.Scatter(x=df_threshold['x'], y=df_threshold['y1'], 
#                   fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
#                   line=dict(color='rgba(0,0,0,0)'), showlegend=False,
#                   name='Buy Zone'),
#         row=2, col=1
#     )

#     fig.update_layout(
#         title=f'{symbol} - Mean Reversion Analysis ({result["Lookback Days"]} Days)',
#         height=700,
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         template="plotly_white"
#     )

#     fig.update_yaxes(title_text=f"Price ({currency_symbol})", row=1, col=1)
#     fig.update_yaxes(title_text="Z-Score", range=[-3, 3], row=2, col=1)
#     fig.update_xaxes(title_text="Date", row=2, col=1)

#     return fig


# ## Frontend Streamlit
# def main():
#     st.title("📈 Day Trading Mean Reversion Strategy Analyzer")
#     st.markdown("This tool analyzes stocks using a mean reversion strategy optimized for day trading. It provides buy/sell signals based on price deviations from moving averages.")

#     st.sidebar.header("Configuration")

#     market_options = {
#         "India National Stock Exchange": "NS",
#         "US Stock Market": "",
#         "Bombay Stock Exchange": "BO",
#         "Malaysian Stock Exchange": "KL"
#     }
#     selected_market = st.sidebar.selectbox(
#         "Select Market", 
#         list(market_options.keys()),
#         index=0
#     )
#     market_index = market_options[selected_market]

#     default_stocks = {
#         "India National Stock Exchange": ["SBI", "RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
#         "US Stock Market": ["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META", "TESLA", "NVIDIA", "JPMORGAN", "VISA", "WALMART"],
#         "Bombay Stock Exchange": ["RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
#         "Malaysian Stock Exchange": ["MAYBANK", "CIMB", "PBBANK", "TENAGA", "PCHEM", "IOICORP", "KLK", "SIME", "GENTING", "HAPSENG"]
#     }

#     # Initialize session state for stock list
#     if "stock_list" not in st.session_state:
#         st.session_state.stock_list = default_stocks[selected_market].copy()

#     # Create a container for the stock input area
#     with st.sidebar:
#         st.subheader("Selected Stocks")
        
#         # Text area for the stock list
#         stock_input = st.text_area(
#             "Enter stock symbols (one per line)", 
#             value="\n".join(st.session_state.stock_list),
#             height=150,
#             key="stock_input"
#         )

#     # Update session state with current input
#     st.session_state.stock_list = [symbol.strip().upper() for symbol in stock_input.split("\n") if symbol.strip()]
#     entered_symbols = st.session_state.stock_list

#     lookback_days = st.sidebar.slider(
#         "Lookback Period (days)", 
#         min_value=10, 
#         max_value=60, 
#         value=25,
#         help="Number of days to look back for analysis (10-60 recommended for day trading)"
#     )

#     investment_amount = st.sidebar.number_input(
#         "Investment Amount per Stock", 
#         min_value=10000, 
#         max_value=1000000, 
#         value=100000,
#         step=10000,
#         format="%d"
#     )

#     col1, col2 = st.sidebar.columns(2)
#     with col1:
#         run_button = st.button("Run Analysis", type="primary")
#     with col2:
#         clear_button = st.button("Clear Results")

#     if clear_button:
#         st.session_state.results = None
#         st.rerun()  # Updated from experimental_rerun

#     if "results" not in st.session_state:
#         st.session_state.results = None


#     ####### Run Analysis: ############
#     if run_button:
#         if not entered_symbols:
#             st.error("Please enter at least one stock symbol")
#         else:
#             st.session_state.results = run_analysis(
#                 entered_symbols, 
#                 lookback_days=lookback_days,
#                 investment_per_stock=investment_amount,
#                 market_index=market_index
#             )
    
    
#     ### Displaying the results of the call 
#     if st.session_state.results:
#         results = st.session_state.results

#         df_display = pd.DataFrame([{k: v for k, v in result.items() if k != 'Data'} for result in results])
#         df_display = df_display.set_index('Symbol')

#         df_display['Abs_Z_Score'] = abs(df_display['Z-Score'])
#         df_display = df_display.sort_values('Abs_Z_Score', ascending=False)
#         df_display = df_display.drop('Abs_Z_Score', axis=1)

#         tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💹 Trade Signals", "📈 Charts", "📉 Price History"])

#         with tab1:
#             st.header("Analysis Overview")

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 buy_signals = len([r for r in results if r['Action'] == 'BUY'])
#                 st.metric("Buy Signals", buy_signals)
#             with col2:
#                 sell_signals = len([r for r in results if r['Action'] == 'SELL'])
#                 st.metric("Sell Signals", sell_signals)
#             with col3:
#                 hold_signals = len([r for r in results if r['Action'] == 'HOLD'])
#                 st.metric("Hold Signals", hold_signals)

#             st.subheader("Moving Averages")
#             sma_cols = []
#             if len(results) > 0:
#                 short_sma_period = results[0]["Short SMA Period"]
#                 long_sma_period = results[0]["Long SMA Period"]
#                 sma_cols = [f'SMA_{short_sma_period}', f'SMA_{long_sma_period}']

#             st.dataframe(df_display[['Current Price'] + sma_cols], use_container_width=True)

#         with tab2:
#             st.header("Trading Signals")

#             signal_stocks = df_display[df_display['Action'] != 'HOLD']

#             if not signal_stocks.empty:
#                 cols = st.columns(2)
#                 with cols[0]:
#                     st.metric("Total Signals", len(signal_stocks))
#                 with cols[1]:
#                     avg_potential = signal_stocks['Potential Return %'].mean()
#                     st.metric("Avg. Potential Return", f"{avg_potential:.2f}%")

#                 st.subheader("Trading Recommendations")
#                 signal_cols = ['Action', 'Current Price', 'Expected Price', 'Potential Return %', 'Stop Loss', 'Take Profit', 'Z-Score']
#                 st.dataframe(signal_stocks[signal_cols], use_container_width=True)

#                 st.subheader("Detailed Recommendations")
#                 for idx, symbol in enumerate(signal_stocks.index):
#                     row = signal_stocks.loc[symbol]

#                     with st.expander(f"{symbol} - {row['Action']}", expanded=True):
#                         cols = st.columns([1, 2])

#                         with cols[0]:
#                             color = "green" if row['Action'] == "BUY" else "red"
#                             st.markdown(f"<h3 style='color:{color}'>{row['Action']}</h3>", unsafe_allow_html=True)
#                             currency = row.get('Currency', '₹')  # Get currency symbol from result
#                             st.metric("Current Price", f"{currency}{row['Current Price']}")
#                             st.metric("Expected Price", f"{currency}{row['Expected Price']}", 
#                                     delta=f"{row['Potential Return %']}%")

#                             st.markdown("#### Risk Management")
#                             st.metric("Stop Loss", f"{currency}{row['Stop Loss']}")
#                             st.metric("Take Profit", f"{currency}{row['Take Profit']}")

#                         with cols[1]:
#                             result_obj = next((r for r in results if r['Symbol'] == symbol), None)
#                             if result_obj:
#                                 mini_fig = create_plotly_chart(result_obj)
#                                 st.plotly_chart(mini_fig, use_container_width=True, key=f"signal_chart_{symbol}_{idx}")
#             else:
#                 st.info("No trading signals (BUY/SELL) found for the current stocks and parameters.")

#         with tab3:
#             st.header("Stock Charts")

#             selected_stock = st.selectbox(
#                 "Select Stock to View",
#                 options=[r['Symbol'] for r in results],
#                 index=0,
#                 key="chart_stock_selector"
#             )

#             selected_result = next((r for r in results if r['Symbol'] == selected_stock), None)
#             if selected_result:
#                 fig = create_plotly_chart(selected_result)
#                 st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{selected_stock}")

#                 with st.expander("Stock Details"):
#                     cols = st.columns(3)
#                     with cols[0]:
#                         currency = selected_result.get('Currency', '₹')  # Get currency symbol from result
#                         st.metric("Current Price", f"{currency}{selected_result['Current Price']}")
#                         st.metric("Z-Score", selected_result['Z-Score'])
#                     with cols[1]:
#                         st.metric("Volatility", f"{selected_result['Volatility %']}%")
#                         st.metric("Signal", selected_result['Action'])
#                     with cols[2]:
#                         short_period = selected_result['Short SMA Period']
#                         long_period = selected_result['Long SMA Period']
#                         st.metric("SMA (Short)", f"{currency}{selected_result[f'SMA_{short_period}']}")
#                         st.metric("SMA (Long)", f"{currency}{selected_result[f'SMA_{long_period}']}")

#         with tab4:
#             st.header("Price History")

#             price_cols = ['Current Price', '1D Ago Price', '5D Ago Price', '10D Ago Price', '30D Ago Price']
#             if '60D Ago Price' in df_display.columns:
#                 price_cols.append('60D Ago Price')

#             st.dataframe(df_display[price_cols], use_container_width=True)

#             st.subheader("Price Changes")
#             changes_df = pd.DataFrame(index=df_display.index)

#             changes_df['1D Change %'] = ((df_display['Current Price'] / df_display['1D Ago Price']) - 1) * 100
#             changes_df['5D Change %'] = ((df_display['Current Price'] / df_display['5D Ago Price']) - 1) * 100
#             changes_df['10D Change %'] = ((df_display['Current Price'] / df_display['10D Ago Price']) - 1) * 100
#             changes_df['30D Change %'] = ((df_display['Current Price'] / df_display['30D Ago Price']) - 1) * 100

#             changes_df = changes_df.round(2)

#             def color_negative_red(val):
#                 color = 'red' if val < 0 else 'green'
#                 return f'color: {color}'

#             styled_changes = changes_df.style.applymap(color_negative_red)
#             st.dataframe(styled_changes, use_container_width=True)
#     else:
#         st.info("Welcome! Please enter stock symbols and click 'Run Analysis' to start.")
#         st.write("""
#         ### How to use this tool:
#         1. Enter stock symbols in the sidebar (one per line)
#         2. Adjust the lookback period (10-60 days recommended for day trading)
#         3. Set your investment amount per stock
#         4. Click 'Run Analysis' to get results
#         """)

#         st.write("""
#         ### Mean Reversion Strategy:
#         This strategy identifies stocks that have deviated significantly from their historical average prices and are likely to revert back. The analysis uses:
#         - Dynamic moving averages
#         - Z-Scores to measure deviation
#         - Bollinger Bands
#         - Volume-weighted average price (VWAP)
#         """)

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json

st.set_page_config(
    page_title="Mean Reversion Strategy Analyzer",
    page_icon="📈",
    layout="wide",
)

def get_yahoo_finance_history(symbol, market_index, range_period="6mo", interval="1d"):
    """
    Fetch historical data directly from Yahoo Finance API
    """
    suffixes = {
        "NS": ".NS",
        "": "",
        "KL": ".KL",
        "BO": ".BO"
    }
    full_symbol = symbol + suffixes.get(market_index, "")
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{full_symbol}?interval={interval}&range={range_period}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            st.warning(f"Failed to fetch data for {full_symbol} (Status: {r.status_code})")
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
        st.error(f"Error fetching/parsing data for {full_symbol}: {str(e)}")
        return None

def get_historical_price(df, days_ago):
    """Get historical price from days ago"""
    try:
        if df.empty:
            return None
        if len(df) > days_ago:
            return df['Close'].iloc[-days_ago]
        return df['Close'].iloc[0]
    except Exception:
        return None

def get_currency_symbol(market_index):
    """Get currency symbol based on market"""
    currency_map = {
        "": "$",  # US Market
        "KL": "RM",  # Malaysian Market
        "NS": "₹",  # Indian NSE
        "BO": "₹"  # Indian BSE
    }
    return currency_map.get(market_index, "$")

def get_ticker_symbol(symbol, market_index):
    """Map common stock names to their ticker symbols"""
    # Malaysian stock mapping
    malaysian_tickers = {
        "MAYBANK": "1155",
        "CIMB": "1023",
        "PBBANK": "1295",
        "TENAGA": "5347",
        "PCHEM": "5183",
        "IOICORP": "1961",
        "KLK": "2445",
        "SIME": "4197",
        "GENTING": "3182",
        "HAPSENG": "3034"
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
        "SBI": "SBIN",
        "RELIANCE": "RELIANCE",
        "TCS": "TCS",
        "HDFC": "HDFCBANK",
        "INFOSYS": "INFY",
        "ICICI": "ICICIBANK",
        "ITC": "ITC",
        "KOTAK": "KOTAKBANK",
        "AXIS": "AXISBANK",
        "L&T": "LT",
        "BHARTI": "BHARTIARTL"
    }
    
    symbol = symbol.upper()
    
    if market_index == "KL":
        return malaysian_tickers.get(symbol, symbol)
    elif market_index == "":
        return us_tickers.get(symbol, symbol)
    elif market_index in ["NS", "BO"]:
        return indian_tickers.get(symbol, symbol)
    else:
        return symbol

def get_mean_reversion_signals(symbol, index="NS", lookback_days=60, investment_amount=100000):
    """
    Analyze stock using mean reversion strategy with integrated Yahoo Finance data
    """
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
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
                st.error(f"No data available for {symbol} (Ticker: {ticker_symbol})")
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

            # Prepare result
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
            
    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
        return None

def run_analysis(symbols, lookback_days=60, investment_per_stock=100000, market_index="NS"):
    """Run analysis for multiple symbols"""
    analysis_results = []

    with st.spinner(f"Running analysis with {lookback_days} days lookback period..."):
        for symbol in symbols:
            result = get_mean_reversion_signals(
                symbol, 
                index=market_index, 
                lookback_days=lookback_days, 
                investment_amount=investment_per_stock
            )
            if result is not None:
                analysis_results.append(result)

    return analysis_results

def create_plotly_chart(result):
    """Create interactive Plotly chart for stock analysis"""
    symbol = result['Symbol']
    df = result['Data'].copy()
    short_sma_period = result['Short SMA Period']
    long_sma_period = result['Long SMA Period']
    currency_symbol = result['Currency']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        subplot_titles=(f'{symbol} Price Analysis', 'Z-Score'),
                        row_heights=[0.7, 0.3])

    # Price chart
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='royalblue', width=2)),
        row=1, col=1
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_Short'], name=f'SMA {short_sma_period}',
                  line=dict(color='orange', width=1.5, dash='dash')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_Long'], name=f'SMA {long_sma_period}',
                  line=dict(color='red', width=1.5, dash='dash')),
        row=1, col=1
    )

    # VWAP
    if 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VWAP'], name='VWAP',
                      line=dict(color='purple', width=1.5, dash='dot')),
            row=1, col=1
        )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper_Band'], name='Upper Band',
                  line=dict(color='green', width=1, dash='dot')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['Lower_Band'], name='Lower Band',
                  line=dict(color='green', width=1, dash='dot')),
        row=1, col=1
    )

    # Fill between bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper_Band'], 
                  fill='tonexty', fillcolor='rgba(0,128,0,0.1)',
                  line=dict(color='rgba(0,0,0,0)'), showlegend=False),
        row=1, col=1
    )

    # Price levels
    current_price = result['Current Price']
    stop_loss = result['Stop Loss']
    take_profit = result['Take Profit']

    fig.add_hline(y=current_price, line_width=1, line_color="black", line_dash="solid",
                 annotation_text="Current", row=1, col=1)
    fig.add_hline(y=stop_loss, line_width=1.5, line_color="red", line_dash="dash",
                 annotation_text="Stop Loss", row=1, col=1)
    fig.add_hline(y=take_profit, line_width=1.5, line_color="green", line_dash="dash",
                 annotation_text="Take Profit", row=1, col=1)

    # Signal marker
    if result['Action'] != "HOLD":
        marker_color = 'green' if result['Action'] == 'BUY' else 'red'
        marker_symbol = 'triangle-up' if result['Action'] == 'BUY' else 'triangle-down'

        fig.add_trace(
            go.Scatter(x=[df.index[-1]], y=[current_price], 
                      mode='markers', marker=dict(color=marker_color, size=15, symbol=marker_symbol),
                      name=result['Action']),
            row=1, col=1
        )

    # Z-Score chart
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Z_Score'], name='Z-Score', line=dict(color='purple', width=2)),
        row=2, col=1
    )

    # Z-Score threshold lines
    z_threshold = result['Z-Score Threshold']
    fig.add_hline(y=0, line_width=1, line_color="black", line_dash="solid", row=2, col=1)
    fig.add_hline(y=z_threshold, line_width=1, line_color="red", line_dash="dash", row=2, col=1)
    fig.add_hline(y=-z_threshold, line_width=1, line_color="green", line_dash="dash", row=2, col=1)

    # Z-Score zones
    df_threshold = pd.DataFrame({
        'x': df.index,
        'y1': [z_threshold] * len(df),
        'y2': [3] * len(df)
    })

    fig.add_trace(
        go.Scatter(x=df_threshold['x'], y=df_threshold['y1'], 
                  fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
                  line=dict(color='rgba(0,0,0,0)'), showlegend=False,
                  name='Sell Zone'),
        row=2, col=1
    )

    df_threshold = pd.DataFrame({
        'x': df.index,
        'y1': [-z_threshold] * len(df),
        'y2': [-3] * len(df)
    })

    fig.add_trace(
        go.Scatter(x=df_threshold['x'], y=df_threshold['y1'], 
                  fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
                  line=dict(color='rgba(0,0,0,0)'), showlegend=False,
                  name='Buy Zone'),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title=f'{symbol} - Mean Reversion Analysis ({result["Lookback Days"]} Days)',
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    fig.update_yaxes(title_text=f"Price ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", range=[-3, 3], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig

def get_yahoo_finance_suggestions(query, market_index):
    """Get stock suggestions from Yahoo Finance API"""
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
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            suggestions = []
            
            if 'quotes' in data:
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
            
            return suggestions[:5]
    except Exception as e:
        st.error(f"Error fetching suggestions: {str(e)}")
    return []

def get_stock_recommendations(search_term, market_index):
    """Get stock recommendations with fallback to local database"""
    if not search_term:
        return []
    
    # First try Yahoo Finance API
    suggestions = get_yahoo_finance_suggestions(search_term, market_index)
    
    # If no suggestions from Yahoo, fall back to local database
    if not suggestions:
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

def main():
    st.title("📈 Enhanced Mean Reversion Strategy Analyzer")
    st.markdown("""
    This tool analyzes stocks using a mean reversion strategy with **direct Yahoo Finance integration**. 
    It provides buy/sell signals based on price deviations from moving averages with improved data reliability.
    """)

    st.sidebar.header("Configuration")

    # Market selection
    market_options = {
        "India National Stock Exchange": "NS",
        "US Stock Market": "",
        "Bombay Stock Exchange": "BO",
        "Malaysian Stock Exchange": "KL"
    }
    selected_market = st.sidebar.selectbox(
        "Select Market", 
        list(market_options.keys()),
        index=0
    )
    market_index = market_options[selected_market]

    # Default stocks for each market
    default_stocks = {
        "India National Stock Exchange": ["SBI", "RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
        "US Stock Market": ["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META", "TESLA", "NVIDIA", "JPMORGAN", "VISA", "WALMART"],
        "Bombay Stock Exchange": ["RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
        "Malaysian Stock Exchange": ["MAYBANK", "CIMB", "PBBANK", "TENAGA", "PCHEM", "IOICORP", "KLK", "SIME", "GENTING", "HAPSENG"]
    }

    # Initialize session state for stock list
    if "stock_list" not in st.session_state:
        st.session_state.stock_list = default_stocks[selected_market].copy()

    # Stock input
    with st.sidebar:
        st.subheader("Selected Stocks")
        
        stock_input = st.text_area(
            "Enter stock symbols (one per line)", 
            value="\n".join(st.session_state.stock_list),
            height=150,
            key="stock_input"
        )

    # Update session state with current input
    st.session_state.stock_list = [symbol.strip().upper() for symbol in stock_input.split("\n") if symbol.strip()]
    entered_symbols = st.session_state.stock_list

    # Analysis parameters
    lookback_days = st.sidebar.slider(
        "Lookback Period (days)", 
        min_value=10, 
        max_value=60, 
        value=25,
        help="Number of days to look back for analysis (10-60 recommended for day trading)"
    )

    investment_amount = st.sidebar.number_input(
        "Investment Amount per Stock", 
        min_value=10000, 
        max_value=1000000, 
        value=100000,
        step=10000,
        format="%d"
    )

    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_button = st.button("Run Analysis", type="primary")
    with col2:
        clear_button = st.button("Clear Results")

    if clear_button:
        st.session_state.results = None
        st.rerun()

    if "results" not in st.session_state:
        st.session_state.results = None

    # Run analysis
    if run_button:
        if not entered_symbols:
            st.error("Please enter at least one stock symbol")
        else:
            st.session_state.results = run_analysis(
                entered_symbols, 
                lookback_days=lookback_days,
                investment_per_stock=investment_amount,
                market_index=market_index
            )

    # Display results
    if st.session_state.results:
        results = st.session_state.results

        # Prepare display dataframe
        df_display = pd.DataFrame([{k: v for k, v in result.items() if k != 'Data'} for result in results])
        df_display = df_display.set_index('Symbol')

        # Sort by absolute Z-Score for most interesting signals first
        df_display['Abs_Z_Score'] = abs(df_display['Z-Score'])
        df_display = df_display.sort_values('Abs_Z_Score', ascending=False)
        df_display = df_display.drop('Abs_Z_Score', axis=1)

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💹 Trade Signals", "📈 Charts", "📉 Price History"])

        with tab1:
            st.header("Analysis Overview")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                buy_signals = len([r for r in results if r['Action'] == 'BUY'])
                st.metric("Buy Signals", buy_signals)
            with col2:
                sell_signals = len([r for r in results if r['Action'] == 'SELL'])
                st.metric("Sell Signals", sell_signals)
            with col3:
                hold_signals = len([r for r in results if r['Action'] == 'HOLD'])
                st.metric("Hold Signals", hold_signals)

            # Show data source info
            st.info("✅ **Enhanced Data Source**: Using direct Yahoo Finance API integration for improved reliability and faster data fetching.")

            # Moving averages table
            st.subheader("Moving Averages")
            sma_cols = []
            if len(results) > 0:
                short_sma_period = results[0]["Short SMA Period"]
                long_sma_period = results[0]["Long SMA Period"]
                sma_cols = [f'SMA_{short_sma_period}', f'SMA_{long_sma_period}']

            st.dataframe(df_display[['Current Price'] + sma_cols], use_container_width=True)

        with tab2:
            st.header("Trading Signals")

            signal_stocks = df_display[df_display['Action'] != 'HOLD']

            if not signal_stocks.empty:
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Total Signals", len(signal_stocks))
                with cols[1]:
                    avg_potential = signal_stocks['Potential Return %'].mean()
                    st.metric("Avg. Potential Return", f"{avg_potential:.2f}%")

                st.subheader("Trading Recommendations")
                signal_cols = ['Action', 'Current Price', 'Expected Price', 'Potential Return %', 'Stop Loss', 'Take Profit', 'Z-Score']
                st.dataframe(signal_stocks[signal_cols], use_container_width=True)

                st.subheader("Detailed Recommendations")
                for idx, symbol in enumerate(signal_stocks.index):
                    row = signal_stocks.loc[symbol]

                    with st.expander(f"{symbol} - {row['Action']}", expanded=True):
                        cols = st.columns([1, 2])

                        with cols[0]:
                            color = "green" if row['Action'] == "BUY" else "red"
                            st.markdown(f"<h3 style='color:{color}'>{row['Action']}</h3>", unsafe_allow_html=True)
                            currency = row.get('Currency', '₹')
                            st.metric("Current Price", f"{currency}{row['Current Price']}")
                            st.metric("Expected Price", f"{currency}{row['Expected Price']}", 
                                     delta=f"{row['Potential Return %']}%")

                            st.markdown("#### Risk Management")
                            st.metric("Stop Loss", f"{currency}{row['Stop Loss']}")
                            st.metric("Take Profit", f"{currency}{row['Take Profit']}")

                        with cols[1]:
                            result_obj = next((r for r in results if r['Symbol'] == symbol), None)
                            if result_obj:
                                mini_fig = create_plotly_chart(result_obj)
                                st.plotly_chart(mini_fig, use_container_width=True, key=f"signal_chart_{symbol}_{idx}")
            else:
                st.info("No trading signals (BUY/SELL) found for the current stocks and parameters.")

        with tab3:
            st.header("Stock Charts")

            selected_stock = st.selectbox(
                "Select Stock to View",
                options=[r['Symbol'] for r in results],
                index=0,
                key="chart_stock_selector"
            )

            selected_result = next((r for r in results if r['Symbol'] == selected_stock), None)
            if selected_result:
                fig = create_plotly_chart(selected_result)
                st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{selected_stock}")

                with st.expander("Stock Details"):
                    cols = st.columns(3)
                    with cols[0]:
                        currency = selected_result.get('Currency', '₹')
                        st.metric("Current Price", f"{currency}{selected_result['Current Price']}")
                        st.metric("Z-Score", selected_result['Z-Score'])
                    with cols[1]:
                        st.metric("Volatility", f"{selected_result['Volatility %']}%")
                        st.metric("Signal", selected_result['Action'])
                    with cols[2]:
                        short_period = selected_result['Short SMA Period']
                        long_period = selected_result['Long SMA Period']
                        st.metric("SMA (Short)", f"{currency}{selected_result[f'SMA_{short_period}']}")
                        st.metric("SMA (Long)", f"{currency}{selected_result[f'SMA_{long_period}']}")

        with tab4:
            st.header("Price History")

            price_cols = ['Current Price', '1D Ago Price', '5D Ago Price', '10D Ago Price', '30D Ago Price']
            if '60D Ago Price' in df_display.columns:
                price_cols.append('60D Ago Price')

            st.dataframe(df_display[price_cols], use_container_width=True)

            st.subheader("Price Changes")
            changes_df = pd.DataFrame(index=df_display.index)

            changes_df['1D Change %'] = ((df_display['Current Price'] / df_display['1D Ago Price']) - 1) * 100
            changes_df['5D Change %'] = ((df_display['Current Price'] / df_display['5D Ago Price']) - 1) * 100
            changes_df['10D Change %'] = ((df_display['Current Price'] / df_display['10D Ago Price']) - 1) * 100
            changes_df['30D Change %'] = ((df_display['Current Price'] / df_display['30D Ago Price']) - 1) * 100

            changes_df = changes_df.round(2)

            def color_negative_red(val):
                color = 'red' if val < 0 else 'green'
                return f'color: {color}'

            styled_changes = changes_df.style.applymap(color_negative_red)
            st.dataframe(styled_changes, use_container_width=True)

    else:
        st.info("Welcome! Please enter stock symbols and click 'Run Analysis' to start.")
        
        # Information panels
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            ### 🚀 New Features:
            - **Direct Yahoo Finance Integration**: Faster, more reliable data fetching
            - **Enhanced Error Handling**: Better error messages and data validation
            - **Improved Market Support**: Consistent handling across all markets
            - **Optimized Performance**: Reduced API calls and faster processing
            """)
            
        with col2:
            st.write("""
            ### How to use this tool:
            1. **Select Market**: Choose your preferred stock market
            2. **Enter Symbols**: Add stock symbols in the sidebar (one per line)
            3. **Adjust Parameters**: Set lookback period (10-60 days for day trading)
            4. **Set Investment**: Define your investment amount per stock
            5. **Run Analysis**: Click 'Run Analysis' to get results
            """)

        st.write("""
        ### 📊 Mean Reversion Strategy Overview:
        
        This enhanced strategy identifies stocks that have deviated significantly from their historical average prices 
        and are likely to revert back to the mean. The analysis includes:
        
        **Technical Indicators:**
        - **Dynamic Moving Averages**: Short and long-term SMAs based on lookback period
        - **Z-Scores**: Statistical measure of price deviation from the mean
        - **Bollinger Bands**: Volatility-based trading bands
        - **VWAP**: Volume-weighted average price for better entry/exit points
        
        **Risk Management:**
        - **Dynamic Stop Loss**: Calculated based on current volatility
        - **Take Profit Levels**: Based on expected mean reversion potential
        - **Volatility Analysis**: Helps determine position sizing and risk
        
        **Signal Generation:**
        - **BUY Signal**: When Z-Score drops below negative threshold (oversold)
        - **SELL Signal**: When Z-Score rises above positive threshold (overbought)
        - **HOLD**: When price is within normal deviation range
        """)
        
        # Market-specific information
        market_info = {
            "India National Stock Exchange": {
                "currency": "₹ (INR)",
                "suffix": ".NS",
                "examples": "SBI, RELIANCE, TCS, HDFC, INFOSYS"
            },
            "US Stock Market": {
                "currency": "$ (USD)", 
                "suffix": "No suffix",
                "examples": "AAPL, MSFT, GOOGL, AMZN, TSLA"
            },
            "Bombay Stock Exchange": {
                "currency": "₹ (INR)",
                "suffix": ".BO", 
                "examples": "RELIANCE, TCS, HDFC, INFOSYS, ICICI"
            },
            "Malaysian Stock Exchange": {
                "currency": "RM (MYR)",
                "suffix": ".KL",
                "examples": "1155 (Maybank), 1023 (CIMB), 1295 (Public Bank)"
            }
        }
        
        if selected_market in market_info:
            info = market_info[selected_market]
            st.info(f"""
            **Selected Market**: {selected_market}
            - **Currency**: {info['currency']}
            - **Yahoo Finance Suffix**: {info['suffix']}
            - **Example Symbols**: {info['examples']}
            """)

if __name__ == "__main__":
    main()