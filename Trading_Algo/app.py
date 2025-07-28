import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import time
import json

from core import _get_mean_reversion_signals_, _run_analysis_

st.set_page_config(
    page_title="Mean Reversion Strategy Analyzer",
    page_icon="📈",
    layout="wide",
) 

def get_mean_reversion_signals(symbol, index="NS", lookback_days=60, investment_amount=100000):
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            return _get_mean_reversion_signals_(symbol, index=index, lookback_days=lookback_days, investment_amount=investment_amount);
    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
    return None

def run_analysis(symbols, lookback_days=60, investment_per_stock=100000, market_index="NS"):
    with st.spinner(f"Running analysis with {lookback_days} days lookback period..."):
        return _run_analysis_(symbols=symbols, lookback_days=lookback_days, investment_per_stock=investment_per_stock, market_index=market_index)



def create_plotly_chart(result):
    symbol = result['Symbol']
    df = result['Data'].copy()
    short_sma_period = result['Short SMA Period']
    long_sma_period = result['Long SMA Period']
    currency_symbol = result['Currency']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        subplot_titles=(f'{symbol} Price Analysis', 'Z-Score'),
                        row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='royalblue', width=2)),
        row=1, col=1
    )

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

    if 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VWAP'], name='VWAP',
                      line=dict(color='purple', width=1.5, dash='dot')),
            row=1, col=1
        )

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

    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper_Band'], 
                  fill='tonexty', fillcolor='rgba(0,128,0,0.1)',
                  line=dict(color='rgba(0,0,0,0)'), showlegend=False),
        row=1, col=1
    )

    current_price = result['Current Price']
    stop_loss = result['Stop Loss']
    take_profit = result['Take Profit']

    fig.add_hline(y=current_price, line_width=1, line_color="black", line_dash="solid",
                 annotation_text="Current", row=1, col=1)

    fig.add_hline(y=stop_loss, line_width=1.5, line_color="red", line_dash="dash",
                 annotation_text="Stop Loss", row=1, col=1)

    fig.add_hline(y=take_profit, line_width=1.5, line_color="green", line_dash="dash",
                 annotation_text="Take Profit", row=1, col=1)

    if result['Action'] != "HOLD":
        marker_color = 'green' if result['Action'] == 'BUY' else 'red'
        marker_symbol = 'triangle-up' if result['Action'] == 'BUY' else 'triangle-down'

        fig.add_trace(
            go.Scatter(x=[df.index[-1]], y=[current_price], 
                      mode='markers', marker=dict(color=marker_color, size=15, symbol=marker_symbol),
                      name=result['Action']),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['Z_Score'], name='Z-Score', line=dict(color='purple', width=2)),
        row=2, col=1
    )

    z_threshold = result['Z-Score Threshold']
    fig.add_hline(y=0, line_width=1, line_color="black", line_dash="solid", row=2, col=1)
    fig.add_hline(y=z_threshold, line_width=1, line_color="red", line_dash="dash", row=2, col=1)
    fig.add_hline(y=-z_threshold, line_width=1, line_color="green", line_dash="dash", row=2, col=1)

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


## Frontend Streamlit
def main():
    st.title("📈 Day Trading Mean Reversion Strategy Analyzer")
    st.markdown("This tool analyzes stocks using a mean reversion strategy optimized for day trading. It provides buy/sell signals based on price deviations from moving averages.")

    st.sidebar.header("Configuration")

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

    default_stocks = {
        "India National Stock Exchange": ["SBI", "RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
        "US Stock Market": ["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META", "TESLA", "NVIDIA", "JPMORGAN", "VISA", "WALMART"],
        "Bombay Stock Exchange": ["RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI", "ITC", "KOTAK", "AXIS", "L&T", "BHARTI"],
        "Malaysian Stock Exchange": ["MAYBANK", "CIMB", "PBBANK", "TENAGA", "PCHEM", "IOICORP", "KLK", "SIME", "GENTING", "HAPSENG"]
    }

    # Initialize session state for stock list
    if "stock_list" not in st.session_state:
        st.session_state.stock_list = default_stocks[selected_market].copy()

    # Create a container for the stock input area
    with st.sidebar:
        st.subheader("Selected Stocks")
        
        # Text area for the stock list
        stock_input = st.text_area(
            "Enter stock symbols (one per line)", 
            value="\n".join(st.session_state.stock_list),
            height=150,
            key="stock_input"
        )

    # Update session state with current input
    st.session_state.stock_list = [symbol.strip().upper() for symbol in stock_input.split("\n") if symbol.strip()]
    entered_symbols = st.session_state.stock_list

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

    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_button = st.button("Run Analysis", type="primary")
    with col2:
        clear_button = st.button("Clear Results")

    if clear_button:
        st.session_state.results = None
        st.rerun()  # Updated from experimental_rerun

    if "results" not in st.session_state:
        st.session_state.results = None


    ####### Run Analysis: ############
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
    
    
    ### Displaying the results of the call 
    if st.session_state.results:
        results = st.session_state.results

        df_display = pd.DataFrame([{k: v for k, v in result.items() if k != 'Data'} for result in results])
        df_display = df_display.set_index('Symbol')

        df_display['Abs_Z_Score'] = abs(df_display['Z-Score'])
        df_display = df_display.sort_values('Abs_Z_Score', ascending=False)
        df_display = df_display.drop('Abs_Z_Score', axis=1)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💹 Trade Signals", "📈 Charts", "📉 Price History"])

        with tab1:
            st.header("Analysis Overview")

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
                            currency = row.get('Currency', '₹')  # Get currency symbol from result
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
                        currency = selected_result.get('Currency', '₹')  # Get currency symbol from result
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
        st.write("""
        ### How to use this tool:
        1. Enter stock symbols in the sidebar (one per line)
        2. Adjust the lookback period (10-60 days recommended for day trading)
        3. Set your investment amount per stock
        4. Click 'Run Analysis' to get results
        """)

        st.write("""
        ### Mean Reversion Strategy:
        This strategy identifies stocks that have deviated significantly from their historical average prices and are likely to revert back. The analysis uses:
        - Dynamic moving averages
        - Z-Scores to measure deviation
        - Bollinger Bands
        - Volume-weighted average price (VWAP)
        """)

if __name__ == "__main__":
    main()