"""
ChatGPT Launch Portfolio Analyzer
What would have happened to YOUR portfolio after Nov 30, 2022?

Analyzes:
- Portfolio performance vs benchmark
- Sector concentration (entropy) changes
- Disruption cascade impact
- Recommendations based on creative destruction
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="ChatGPT Launch Portfolio Analyzer", layout="wide")

# ============================================================
# CONSTANTS
# ============================================================

CHATGPT_LAUNCH = "2022-11-30"
ANALYSIS_END = datetime.now().strftime("%Y-%m-%d")

# Pre-defined ticker categories
WINNERS = {
    "NVDA": "Nvidia - AI chips",
    "MSFT": "Microsoft - OpenAI partner",
    "META": "Meta - AI investment",
    "GOOGL": "Google - AI race",
    "AMZN": "Amazon - AWS AI",
}

LOSERS = {
    "CHGG": "Chegg - Education disrupted",
    "PRCT": "Procept - Knowledge work",
    "TAL": "TAL Education",
    "EDU": "New Oriental Education",
}

SECTORS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Education": ["CHGG", "DUOL", "TWOU", "LRN"],
    "Healthcare": ["UNH", "JNJ", "PFE", "ABBV"],
    "Financials": ["JPM", "BAC", "GS", "MS"],
    "Consumer": ["AMZN", "TSLA", "HD", "NKE"],
    "Energy": ["XOM", "CVX", "COP"],
}

# ============================================================
# ENTROPY CALCULATIONS
# ============================================================

def shannon_entropy(weights: np.ndarray) -> float:
    """Calculate Shannon entropy of portfolio weights."""
    weights = np.array(weights)
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 0.0
    weights = weights / weights.sum()
    return -np.sum(weights * np.log2(weights))


def max_entropy(n: int) -> float:
    """Maximum entropy for n assets."""
    if n <= 1:
        return 0.0
    return np.log2(n)


def concentration_ratio(weights: np.ndarray, top_n: int = 3) -> float:
    """What % of portfolio is in top N holdings."""
    weights = np.array(sorted(weights, reverse=True))
    return weights[:top_n].sum() / weights.sum() * 100


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=3600)
def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch close prices for tickers."""
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)

        if data.empty:
            return pd.DataFrame()

        # yfinance now always returns MultiIndex: (Price Type, Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Get Close prices
            prices = data["Close"].copy()
            # If single ticker, it might still be a Series
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
        else:
            # Fallback for older yfinance versions
            if "Close" in data.columns:
                prices = data[["Close"]]
                prices.columns = tickers[:1]
            else:
                prices = data

        return prices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative returns from prices."""
    return (prices / prices.iloc[0] - 1) * 100


# ============================================================
# PORTFOLIO ANALYSIS
# ============================================================

def analyze_portfolio(tickers: list, weights: list, start: str, end: str) -> dict:
    """Complete portfolio analysis."""

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Fetch data
    prices = fetch_prices(tickers + ["SPY"], start, end)

    if prices.empty:
        return None

    # Calculate returns
    returns = calculate_returns(prices)

    # Portfolio value over time
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in zip(tickers, weights):
        if ticker in returns.columns:
            portfolio_returns += returns[ticker] * weight

    # Final values
    final_portfolio = portfolio_returns.iloc[-1]
    final_spy = returns["SPY"].iloc[-1] if "SPY" in returns.columns else 0

    # Entropy analysis
    initial_entropy = shannon_entropy(weights)

    # Calculate ending weights (after price changes)
    ending_values = []
    for ticker, weight in zip(tickers, weights):
        if ticker in prices.columns:
            price_change = prices[ticker].iloc[-1] / prices[ticker].iloc[0]
            ending_values.append(weight * price_change)
        else:
            ending_values.append(weight)

    ending_weights = np.array(ending_values) / sum(ending_values)
    ending_entropy = shannon_entropy(ending_weights)

    return {
        "portfolio_returns": portfolio_returns,
        "spy_returns": returns["SPY"] if "SPY" in returns.columns else None,
        "individual_returns": returns,
        "final_portfolio_return": final_portfolio,
        "final_spy_return": final_spy,
        "alpha": final_portfolio - final_spy,
        "initial_weights": weights,
        "ending_weights": ending_weights,
        "initial_entropy": initial_entropy,
        "ending_entropy": ending_entropy,
        "entropy_change": ending_entropy - initial_entropy,
        "initial_concentration": concentration_ratio(weights),
        "ending_concentration": concentration_ratio(ending_weights),
        "tickers": tickers,
    }


def classify_holdings(tickers: list) -> dict:
    """Classify holdings as winners, losers, or neutral."""
    classification = {"winners": [], "losers": [], "neutral": []}

    for ticker in tickers:
        if ticker in WINNERS:
            classification["winners"].append(ticker)
        elif ticker in LOSERS:
            classification["losers"].append(ticker)
        else:
            classification["neutral"].append(ticker)

    return classification


def generate_recommendations(analysis: dict) -> list:
    """Generate recommendations based on disruption cascade."""
    recs = []

    classification = classify_holdings(analysis["tickers"])

    # Check for AI winners exposure
    winner_weight = sum(
        w for t, w in zip(analysis["tickers"], analysis["ending_weights"])
        if t in WINNERS
    )

    if winner_weight < 0.20:
        recs.append({
            "type": "warning",
            "title": "Low AI Winner Exposure",
            "text": f"Only {winner_weight*100:.1f}% in AI winners. Consider adding NVDA, MSFT, or META for AI infrastructure exposure."
        })

    # Check for disrupted sector exposure
    loser_weight = sum(
        w for t, w in zip(analysis["tickers"], analysis["ending_weights"])
        if t in LOSERS
    )

    if loser_weight > 0.05:
        recs.append({
            "type": "danger",
            "title": "Disruption Risk",
            "text": f"{loser_weight*100:.1f}% exposed to disrupted sectors (education, knowledge work). These faced -90%+ drawdowns."
        })

    # Entropy/concentration warning
    if analysis["entropy_change"] < -0.5:
        recs.append({
            "type": "info",
            "title": "Concentration Increased",
            "text": f"Portfolio entropy dropped {abs(analysis['entropy_change']):.2f} bits. Your portfolio is more concentrated now—winners pulled ahead."
        })

    # Alpha analysis
    if analysis["alpha"] > 50:
        recs.append({
            "type": "success",
            "title": "Strong Outperformance",
            "text": f"Your portfolio beat SPY by {analysis['alpha']:.1f}%. You were positioned for the AI regime shift."
        })
    elif analysis["alpha"] < -20:
        recs.append({
            "type": "danger",
            "title": "Significant Underperformance",
            "text": f"Your portfolio lagged SPY by {abs(analysis['alpha']):.1f}%. Consider rebalancing toward AI infrastructure."
        })

    # Sample space expansion insight
    recs.append({
        "type": "info",
        "title": "Sample Space Expansion",
        "text": "ChatGPT didn't just shift probabilities—it created a new asset class (AI infrastructure). The investment universe itself changed on Nov 30, 2022."
    })

    return recs


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("ChatGPT Launch Portfolio Analyzer")
st.caption("What would have happened to YOUR portfolio after November 30, 2022?")

# Sidebar for portfolio input
st.sidebar.header("Your Portfolio (Nov 2022)")

# Pre-set portfolio options
portfolio_preset = st.sidebar.selectbox(
    "Start from preset",
    ["Custom", "60/40 Traditional", "Tech Heavy", "Diversified", "Education Sector"]
)

if portfolio_preset == "60/40 Traditional":
    default_tickers = ["SPY", "AGG"]
    default_weights = [60, 40]
elif portfolio_preset == "Tech Heavy":
    default_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    default_weights = [25, 25, 20, 15, 15]
elif portfolio_preset == "Diversified":
    default_tickers = ["SPY", "AAPL", "MSFT", "JPM", "XOM", "JNJ"]
    default_weights = [30, 15, 15, 15, 15, 10]
elif portfolio_preset == "Education Sector":
    default_tickers = ["CHGG", "DUOL", "LRN", "TWOU"]
    default_weights = [40, 30, 20, 10]
else:
    default_tickers = ["AAPL", "MSFT", "NVDA"]
    default_weights = [40, 35, 25]

# Custom input - SIMPLER VERSION
st.sidebar.subheader("Holdings")

# Dynamic input for each holding
num_holdings = st.sidebar.number_input("Number of holdings", min_value=1, max_value=10, value=len(default_tickers))

tickers = []
weights = []

for i in range(int(num_holdings)):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        default_t = default_tickers[i] if i < len(default_tickers) else ""
        ticker = st.text_input(f"Ticker {i+1}", value=default_t, key=f"ticker_{i}")
        if ticker:
            tickers.append(ticker.strip().upper())
    with col2:
        default_w = default_weights[i] if i < len(default_weights) else 10
        weight = st.number_input(f"Weight %", value=float(default_w), min_value=0.0, max_value=100.0, key=f"weight_{i}")
        weights.append(weight)

# Filter out empty tickers
valid_pairs = [(t, w) for t, w in zip(tickers, weights) if t]
if valid_pairs:
    tickers, weights = zip(*valid_pairs)
    tickers = list(tickers)
    weights = list(weights)
else:
    tickers = []
    weights = []

# Validate
if len(tickers) == 0:
    st.sidebar.warning("Enter at least one ticker")
    st.stop()

# Normalize weights display
total_weight = sum(weights)
st.sidebar.write(f"**Total:** {total_weight:.0f}% (will normalize to 100%)")

# Analyze button
if st.sidebar.button("Analyze Portfolio", type="primary"):
    with st.spinner("Fetching data and analyzing..."):
        analysis = analyze_portfolio(tickers, weights, CHATGPT_LAUNCH, ANALYSIS_END)

    if analysis is None:
        st.error("Could not fetch data. Check ticker symbols.")
        st.stop()

    # Store in session state
    st.session_state["analysis"] = analysis

# Display results
if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]

    # Key metrics
    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if analysis["final_portfolio_return"] > 0 else "inverse"
        st.metric(
            "Your Portfolio",
            f"{analysis['final_portfolio_return']:.1f}%",
            delta=f"Since ChatGPT launch"
        )

    with col2:
        st.metric(
            "S&P 500",
            f"{analysis['final_spy_return']:.1f}%",
            delta="Benchmark"
        )

    with col3:
        alpha = analysis["alpha"]
        st.metric(
            "Alpha",
            f"{alpha:+.1f}%",
            delta="Outperformance" if alpha > 0 else "Underperformance"
        )

    with col4:
        st.metric(
            "Entropy Change",
            f"{analysis['entropy_change']:+.2f} bits",
            delta="More concentrated" if analysis['entropy_change'] < 0 else "More diversified"
        )

    # Charts
    st.subheader("Performance Chart")

    chart_data = pd.DataFrame({
        "Your Portfolio": analysis["portfolio_returns"],
        "S&P 500": analysis["spy_returns"]
    })
    st.line_chart(chart_data)

    # Mark ChatGPT launch
    st.caption("Starting point: November 30, 2022 (ChatGPT launch)")

    # Individual holdings
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Individual Holdings")

        holdings_data = []
        for ticker, init_w, end_w in zip(
            analysis["tickers"],
            analysis["initial_weights"],
            analysis["ending_weights"]
        ):
            if ticker in analysis["individual_returns"].columns:
                ret = analysis["individual_returns"][ticker].iloc[-1]
            else:
                ret = 0

            # Classification
            if ticker in WINNERS:
                status = "🚀 Winner"
            elif ticker in LOSERS:
                status = "💀 Disrupted"
            else:
                status = "—"

            holdings_data.append({
                "Ticker": ticker,
                "Initial %": f"{init_w*100:.1f}%",
                "Current %": f"{end_w*100:.1f}%",
                "Return": f"{ret:.1f}%",
                "Status": status
            })

        st.dataframe(pd.DataFrame(holdings_data), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Concentration Analysis")

        # Entropy comparison
        entropy_data = pd.DataFrame({
            "Metric": ["Initial", "Current"],
            "Entropy (bits)": [analysis["initial_entropy"], analysis["ending_entropy"]],
            "Top 3 Concentration": [
                f"{analysis['initial_concentration']:.1f}%",
                f"{analysis['ending_concentration']:.1f}%"
            ]
        })
        st.dataframe(entropy_data, hide_index=True, use_container_width=True)

        # Visual
        st.write("**Weight Distribution**")
        weight_df = pd.DataFrame({
            "Ticker": analysis["tickers"],
            "Initial": analysis["initial_weights"] * 100,
            "Current": analysis["ending_weights"] * 100
        }).set_index("Ticker")
        st.bar_chart(weight_df)

    # Recommendations
    st.subheader("Disruption Cascade Analysis")

    recommendations = generate_recommendations(analysis)

    for rec in recommendations:
        if rec["type"] == "success":
            st.success(f"**{rec['title']}:** {rec['text']}")
        elif rec["type"] == "warning":
            st.warning(f"**{rec['title']}:** {rec['text']}")
        elif rec["type"] == "danger":
            st.error(f"**{rec['title']}:** {rec['text']}")
        else:
            st.info(f"**{rec['title']}:** {rec['text']}")

    # The lesson
    st.divider()
    st.subheader("The Key Insight")

    st.markdown("""
    **ChatGPT's launch on November 30, 2022 was a "sample space expansion" event.**

    | Concept | Before ChatGPT | After ChatGPT |
    |---------|----------------|---------------|
    | Investment universe | Tech = FAANG | Tech = FAANG + "AI Infrastructure" |
    | Education sector | Stable, growing | Existential threat |
    | Portfolio question | "What's your tech allocation?" | "What's your AI exposure?" |

    **This wasn't a regime shift (probabilities changing)—it was the game itself changing.**

    - **Winners (NVDA +700%):** Owned the new infrastructure
    - **Losers (CHGG -99%):** Business model became obsolete overnight
    - **Lesson:** When sample space expands, you must re-evaluate every holding
    """)

else:
    # Default view
    st.info("👈 Enter your November 2022 portfolio in the sidebar and click **Analyze Portfolio**")

    st.subheader("What This Tool Shows")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Portfolio Performance**
        - Your returns vs S&P 500 benchmark
        - Individual holding performance
        - Alpha calculation
        """)

    with col2:
        st.markdown("""
        **Entropy Analysis**
        - Portfolio concentration (Shannon entropy)
        - How AI winners changed your allocation
        - Diversification impact
        """)

    st.subheader("The Creative Destruction")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Winners (Sample Space Expansion)**
        - NVDA: +700% (AI chips)
        - MSFT: +100%+ (OpenAI partner)
        - META: +400%+ (AI pivot)
        """)

    with col2:
        st.error("""
        **Losers (Disrupted)**
        - CHGG: -99% (Chegg - education)
        - Knowledge work companies
        - Content creation middlemen
        """)

# Footer
st.divider()
st.caption("Built with DRIVER framework | Case 2: ChatGPT Launch - Sample Space Expansion")
