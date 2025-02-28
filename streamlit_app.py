import streamlit as st
import pandas as pd
from ai_swing_analysis import AISwingAnalysis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize analyzer
swing_analyzer = AISwingAnalysis()

# Custom CSS for futuristic design
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1A1A3D, #2D2D6F);
        color: white;
    }
    .stTitle {
        color: #00FFFF !important; /* Neon cyan */
        font-family: 'Orbitron', sans-serif;
        font-size: 36px;
    }
    .stHeader {
        color: #FF00FF !important; /* Neon magenta */
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
    }
    .stButton>button {
        background-color: #00FFFF;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: 2px solid #FF00FF;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px #00FFFF;
    }
    .stSelectbox, .stMultiselect, .stTextInput {
        background-color: #2D2D6F;
        color: white;
        border: 2px solid #00FFFF;
        border-radius: 10px;
        font-size: 14px;
    }
    .stPlotlyChart {
        border: 2px solid #FF00FF;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description (futuristic, mobile-friendly)
st.title("Swing Trading Suite")
st.write("Track and analyse short-term trading opportunities with real-time AI insights. Tip: Add tickers to monitor signals.")

# Load or initialize watchlist
if 'swing_watchlist' not in st.session_state:
    st.session_state.swing_watchlist = ["AAPL", "TSLA"]

# Mobile-friendly watchlist (use columns for better layout)
st.header("Swing Watchlist")
col1, col2 = st.columns(2)
with col1:
    watchlist = st.multiselect("Select tickers to monitor:", st.session_state.swing_watchlist, default=st.session_state.swing_watchlist, key="watchlist_select", help="Choose stocks to track for swing trading opportunities.")
with col2:
    if st.button("Update Watchlist", key="update", help="Refresh all ticker data for the latest prices and signals."):
        st.session_state.swing_watchlist = watchlist
        st.success("Watchlist updated with the latest information. Tip: Check for fresh prices and signals.")
        for ticker in watchlist:
            signal, conf = swing_analyzer.get_signal(ticker)
            st.write(f"{ticker}: {signal} (Confidence: {conf:.2%})")

# Add ticker (mobile-friendly input)
new_ticker = st.text_input("Add Ticker (e.g., AAPL):", key="add_ticker", help="Enter a stock ticker symbol (e.g., TSLA for Tesla).").upper()
if st.button("Add Ticker", key="add", help="Add a new ticker to your watchlist for analysis."):
    if new_ticker and new_ticker not in st.session_state.swing_watchlist:
        st.session_state.swing_watchlist.append(new_ticker)
        st.success(f"Added {new_ticker} to watchlist. Tip: Analyse for detailed insights.")

# Remove ticker (mobile-friendly dropdown)
remove_ticker = st.selectbox("Remove Ticker:", [""] + st.session_state.swing_watchlist, key="remove_select", help="Select a ticker to remove from your watchlist.")
if st.button("Remove Ticker", key="remove", help="Remove the selected ticker from your watchlist."):
    if remove_ticker and remove_ticker in st.session_state.swing_watchlist:
        st.session_state.swing_watchlist.remove(remove_ticker)
        st.success(f"Removed {remove_ticker} from watchlist.")

# Analyse selected ticker (mobile-friendly)
selected_ticker = st.selectbox("Analyse Ticker:", [f"{t} - {swing_analyzer.analyze_ticker(t)['company_name']}" for t in watchlist] if watchlist else [], key="analyse_select", help="Select a ticker to get a detailed AI analysis and chart.")
if st.button("Analyse Selected", key="analyse", help="Generate a detailed analysis, including signals, trends, and a chart for the selected ticker."):
    if selected_ticker:
        ticker = selected_ticker.split(" - ")[0]
        result = st.cache_data(lambda: swing_analyzer.analyze_ticker(ticker))()  # Cached analysis
        st.info(result['summary'])
        st.write("Detailed Report:")
        st.write(result['report'])
        # Embed Plotly chart directly in Streamlit for mobile compatibility
        fig = go.Figure(eval(result['visualization'].replace('.html', '')))
        st.plotly_chart(fig, use_container_width=True, height=600, help="Interactive chart showing price trends and indicators.")

# Cache expensive operations for performance
@st.cache_data
def fetch_and_analyze(ticker):
    return swing_analyzer.analyze_ticker(ticker)

# Add futuristic particle animation (optional, via HTML/CSS or Lottie)
st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
        <svg width="50" height="50">
            <circle cx="25" cy="25" r="10" fill="#00FFFF" opacity="0.5">
                <animate attributeName="r" from="10" to="15" dur="2s" repeatCount="indefinite"/>
                <animate attributeName="opacity" from="0.5" to="0.1" dur="2s" repeatCount="indefinite"/>
            </circle>
        </svg>
    </div>
""", unsafe_allow_html=True)