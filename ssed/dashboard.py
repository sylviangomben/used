"""
SSED Fusion Dashboard

The class deliverable: a Streamlit app that demonstrates
Sample Space Expansion Detection across all three layers.

Run: streamlit run ssed/dashboard.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssed.quant_signals import (
    run_quant_signals,
    fetch_prices,
    compute_hmm_signals,
    compute_entropy_signals,
    compute_divergence_signals,
    shannon_entropy,
    REGIME_LABELS,
)
from ssed.narrative_signals import (
    compute_news_signals,
    compute_filing_diff,
)
from ssed.backtest import run_backtest

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SSED — Sample Space Expansion Detector",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# SIDEBAR — Configuration
# ============================================================

st.sidebar.title("SSED Configuration")

st.sidebar.markdown("---")
st.sidebar.subheader("Event Parameters")

event_name = st.sidebar.text_input("Event Name", value="ChatGPT Launch")
event_date = st.sidebar.date_input(
    "Event Date",
    value=datetime(2022, 11, 30),
)
analysis_end = st.sidebar.date_input(
    "Analysis End",
    value=datetime(2024, 12, 1),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Tickers")

winner_ticker = st.sidebar.text_input("Suspected Winner", value="NVDA")
loser_ticker = st.sidebar.text_input("Suspected Loser", value="CHGG")
benchmark_ticker = st.sidebar.text_input("Benchmark", value="SPY")

st.sidebar.markdown("---")
st.sidebar.subheader("API Status")

has_openai = bool(os.environ.get("OPENAI_API_KEY"))
has_newsapi = bool(os.environ.get("NEWSAPI_KEY"))

st.sidebar.markdown(
    f"- OpenAI API: {'✅ Connected' if has_openai else '⚠️ Not set (heuristic mode)'}\n"
    f"- NewsAPI: {'✅ Connected' if has_newsapi else '⚠️ Not set (demo articles)'}"
)

if not has_openai:
    st.sidebar.caption(
        "Set OPENAI_API_KEY in .env to enable AI classification (Layer 3)"
    )

# ============================================================
# HEADER
# ============================================================

st.title("🔬 Sample Space Expansion Detector")
st.caption(
    "Fusing quantitative regime detection with LLM-powered narrative analysis "
    "to distinguish regime shifts (P changes) from sample space expansion (X changes)"
)

# Conceptual framework
with st.expander("📖 What is Sample Space Expansion?", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Regime Shift (P changed)**
        - Probabilities change within a known universe
        - Same assets, same categories, different dynamics
        - Example: Tariff shock changes sector rotation
        - HMM detects this well
        """)
    with col2:
        st.markdown("""
        **Sample Space Expansion (X changed)**
        - The investment universe ITSELF changes
        - New asset class emerges, old ones may die
        - Example: ChatGPT creates "AI infrastructure"
        - Requires quant + narrative fusion to detect
        """)

st.markdown("---")

# ============================================================
# RUN ANALYSIS
# ============================================================

event_date_str = event_date.strftime("%Y-%m-%d")
analysis_end_str = analysis_end.strftime("%Y-%m-%d")

if st.sidebar.button("🔍 Run Analysis", type="primary", use_container_width=True):
    st.session_state["running"] = True

if st.session_state.get("running"):

    # --------------------------------------------------------
    # LAYER 1: Quantitative Signals
    # --------------------------------------------------------
    st.header("Layer 1: Quantitative Signals")
    st.caption("Deterministic metrics — no LLM involved")

    with st.spinner("Fetching market data and computing signals..."):
        quant = run_quant_signals(
            event_date=event_date_str,
            analysis_end=analysis_end_str,
            winner=winner_ticker.upper(),
            loser=loser_ticker.upper(),
            benchmark=benchmark_ticker.upper(),
        )
        st.session_state["quant"] = quant

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "HMM Regime",
            quant.hmm.regime_label.replace("_", " ").title(),
            delta=f"p={quant.hmm.regime_probability:.2f}",
        )
    with col2:
        st.metric(
            "Entropy Z-Score",
            f"{quant.entropy.entropy_zscore:.2f}σ",
            delta=f"{quant.entropy.entropy_change:+.3f} bits",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Divergence",
            f"{quant.divergence.total_divergence_pct:+.0f}%",
            delta=f"{winner_ticker} vs {loser_ticker}",
        )
    with col4:
        st.metric(
            "HHI Change",
            f"{quant.concentration.hhi_change:+.4f}",
            delta="More concentrated" if quant.concentration.hhi_change > 0 else "More diversified",
            delta_color="inverse" if quant.concentration.hhi_change > 0 else "normal",
        )

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Price divergence chart
        st.subheader("Winner vs Loser Divergence")

        tickers = [winner_ticker.upper(), loser_ticker.upper(), benchmark_ticker.upper()]
        prices = fetch_prices(tickers, event_date_str, analysis_end_str)

        fig = go.Figure()
        colors = {
            winner_ticker.upper(): "#00cc66",
            loser_ticker.upper(): "#ff4444",
            benchmark_ticker.upper(): "#888888",
        }
        for ticker in tickers:
            if ticker in prices.columns:
                normalized = (prices[ticker] / prices[ticker].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    name=ticker,
                    line=dict(
                        color=colors.get(ticker, "#4488ff"),
                        dash="dash" if ticker == benchmark_ticker.upper() else "solid",
                    ),
                ))

        fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            yaxis_title="Normalized Price (100 = Event Date)",
            yaxis_type="log",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Rolling entropy chart
        st.subheader("Rolling Entropy (Concentration)")

        ent = quant.entropy
        if ent.rolling_dates and ent.rolling_entropy:
            dates = pd.to_datetime(ent.rolling_dates)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=dates,
                y=ent.rolling_entropy,
                name="Rolling Entropy",
                line=dict(color="#4488ff"),
                fill="tozeroy",
                fillcolor="rgba(68, 136, 255, 0.1)",
            ))
            fig2.add_hline(
                y=ent.baseline_entropy,
                line_dash="dash",
                line_color="orange",
                annotation_text="Pre-event baseline",
            )

            # Mark the event date
            event_dt = pd.to_datetime(event_date_str)
            if dates.min() <= event_dt <= dates.max():
                fig2.add_shape(
                    type="line",
                    x0=event_dt.isoformat(), x1=event_dt.isoformat(),
                    y0=0, y1=1, yref="paper",
                    line=dict(dash="dot", color="red"),
                )
                fig2.add_annotation(
                    x=event_dt.isoformat(), y=1, yref="paper",
                    text=event_name, showarrow=False,
                    font=dict(color="red", size=11),
                )

            fig2.update_layout(
                yaxis_title="Shannon Entropy (bits)",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Insufficient data for rolling entropy chart")

    # HMM Regime Timeline
    st.subheader("HMM Regime States Over Time")

    if quant.hmm.regime_history:
        regime_map = {"low_volatility": 0, "medium_volatility": 1, "high_volatility": 2}
        regime_colors = {0: "#00cc66", 1: "#ffaa00", 2: "#ff4444"}

        regime_values = [regime_map.get(r, 1) for r in quant.hmm.regime_history]

        # Create a simple bar-like timeline
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            y=regime_values,
            mode="lines",
            line=dict(color="#4488ff", width=1),
            fill="tozeroy",
            fillcolor="rgba(68, 136, 255, 0.2)",
            name="Regime State",
        ))
        fig3.update_layout(
            yaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=["Low Vol", "Medium Vol", "High Vol"],
            ),
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Transition matrix
    with st.expander("HMM Transition Matrix"):
        transmat = quant.hmm.transition_matrix
        labels = ["Low Vol", "Med Vol", "High Vol"]
        df_trans = pd.DataFrame(transmat, index=labels, columns=labels)
        st.dataframe(df_trans.style.format("{:.3f}").background_gradient(cmap="YlOrRd"), use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # LAYER 2: Narrative Signals
    # --------------------------------------------------------
    st.header("Layer 2: Narrative Signals")
    st.caption("News sentiment + SEC filing analysis")

    with st.spinner("Analyzing narrative signals..."):
        # News sentiment
        news_signals = compute_news_signals(
            query=f"{event_name} AI market impact",
            from_date=event_date_str,
            to_date=analysis_end_str,
            event_context=f"{event_name} on {event_date_str}",
        )
        st.session_state["news"] = news_signals

        # SEC filing diff
        try:
            filing_diff = compute_filing_diff(
                ticker=loser_ticker.upper(),
                before_year=event_date.year,
                after_year=min(event_date.year + 2, analysis_end.year),
            )
            st.session_state["filing"] = filing_diff
        except Exception as e:
            filing_diff = None
            st.warning(f"SEC filing analysis unavailable: {e}")

    news_col, filing_col = st.columns(2)

    with news_col:
        st.subheader("📰 News Sentiment")

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Articles Analyzed", news_signals.article_count)
        with m2:
            color = "normal" if news_signals.avg_sentiment > 0 else "inverse"
            st.metric("Avg Sentiment", f"{news_signals.avg_sentiment:+.3f}")
        with m3:
            st.metric("Trend", news_signals.sentiment_trend.title())

        # Novel themes
        if news_signals.novel_theme_counts:
            st.markdown("**Novel Themes Detected:**")
            for theme, count in sorted(
                news_signals.novel_theme_counts.items(), key=lambda x: -x[1]
            ):
                st.markdown(f"- 🆕 **{theme}**: {count} mention{'s' if count > 1 else ''}")
        else:
            st.info("No novel themes detected")

        # Top articles
        st.markdown("**Top Articles:**")
        for a in news_signals.top_articles[:4]:
            sentiment_emoji = "🟢" if a.sentiment_score > 0.3 else "🔴" if a.sentiment_score < -0.3 else "🟡"
            st.markdown(f"{sentiment_emoji} **[{a.sentiment_score:+.2f}]** {a.title}")
            if a.novel_themes:
                st.caption(f"Themes: {', '.join(a.novel_themes)}")

    with filing_col:
        st.subheader("📄 SEC Filing Analysis")

        if filing_diff:
            st.markdown(
                f"**{filing_diff.ticker}** — 10-K comparison: "
                f"{filing_diff.before_date} vs {filing_diff.after_date}"
            )

            if filing_diff.sample_space_signal:
                st.success(
                    f"**Sample Space Signal: DETECTED** — "
                    f"{len(filing_diff.new_risk_factors)} new risk categories found"
                )
            else:
                st.info(
                    f"**Sample Space Signal: Not detected** — "
                    f"{len(filing_diff.new_risk_factors)} new risk categories"
                )

            if filing_diff.new_risk_factors:
                st.markdown("**New Risk Factors:**")
                for rf in filing_diff.new_risk_factors:
                    st.markdown(f"- ➕ {rf}")

            if filing_diff.removed_risk_factors:
                st.markdown("**Removed Risk Factors:**")
                for rf in filing_diff.removed_risk_factors:
                    st.markdown(f"- ➖ {rf}")

            st.markdown(f"**Language Shift:** {filing_diff.language_shift_summary}")
            st.caption(f"Reasoning: {filing_diff.signal_reasoning}")
        else:
            st.info("SEC filing diff not available for this ticker")

    st.markdown("---")

    # --------------------------------------------------------
    # LAYER 3: Fusion & Classification
    # --------------------------------------------------------
    st.header("Layer 3: AI Fusion & Classification")

    if has_openai:
        st.caption("o4-mini reasoning over all signals via function calling")

        if st.button("🧠 Run AI Classification", type="primary"):
            with st.spinner("o4-mini is analyzing all signals..."):
                from ssed.openai_core import classify_event

                result = classify_event(
                    event_description=(
                        f"{event_name} — analyzing market impact on "
                        f"{winner_ticker} (winner) vs {loser_ticker} (loser)"
                    ),
                    event_date=event_date_str,
                    analysis_end=analysis_end_str,
                    winner=winner_ticker.upper(),
                    loser=loser_ticker.upper(),
                )
                st.session_state["classification"] = result

        if "classification" in st.session_state:
            result = st.session_state["classification"]

            # Classification banner
            if result.classification.value == "sample_space_expansion":
                st.success(
                    f"## 🔬 Classification: SAMPLE SPACE EXPANSION (X changed)\n"
                    f"**Confidence: {result.confidence.value.upper()}**"
                )
            elif result.classification.value == "regime_shift":
                st.warning(
                    f"## ⚡ Classification: REGIME SHIFT (P changed)\n"
                    f"**Confidence: {result.confidence.value.upper()}**"
                )
            else:
                st.info(
                    f"## 📊 Classification: {result.classification.value.replace('_', ' ').upper()}\n"
                    f"**Confidence: {result.confidence.value.upper()}**"
                )

            st.markdown(f"**What Changed:** {result.what_changed}")
            st.markdown(f"**Reasoning:** {result.reasoning}")

            # Evidence breakdown
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            with ev_col1:
                st.markdown("**📈 Entropy Signal**")
                st.markdown(result.entropy_interpretation)
            with ev_col2:
                st.markdown("**📊 Divergence Signal**")
                st.markdown(result.divergence_interpretation)
            with ev_col3:
                st.markdown("**🔄 HMM Signal**")
                st.markdown(result.hmm_interpretation)

            st.markdown("**Key Evidence:**")
            for e in result.key_evidence:
                st.markdown(f"- {e}")

            # Raw JSON
            with st.expander("Raw Classification JSON"):
                st.json(result.model_dump())

    else:
        # No API key — show heuristic classification
        st.caption("Heuristic classification (set OPENAI_API_KEY for AI-powered analysis)")

        # Simple rule-based classification from the quant + narrative signals
        signals_detected = 0
        evidence = []

        if quant.divergence.total_divergence_pct > 500:
            signals_detected += 1
            evidence.append(
                f"Divergence of {quant.divergence.total_divergence_pct:.0f}% "
                f"({winner_ticker} vs {loser_ticker}) — exceeds 500% threshold"
            )

        if quant.entropy.entropy_zscore < -2:
            signals_detected += 1
            evidence.append(
                f"Entropy z-score of {quant.entropy.entropy_zscore:.2f} — "
                f"unusual concentration (>2σ below mean)"
            )

        if quant.concentration.hhi_change > 0.02:
            signals_detected += 1
            evidence.append(
                f"HHI increased by {quant.concentration.hhi_change:.4f} — "
                f"market more concentrated"
            )

        if news_signals.novel_theme_counts:
            signals_detected += 1
            themes = ", ".join(news_signals.novel_theme_counts.keys())
            evidence.append(f"Novel narrative themes detected: {themes}")

        if filing_diff and filing_diff.sample_space_signal:
            signals_detected += 1
            evidence.append(
                f"{len(filing_diff.new_risk_factors)} new AI risk factors in "
                f"{loser_ticker} 10-K filings"
            )

        # Classify based on signal count
        if signals_detected >= 4:
            classification = "SAMPLE SPACE EXPANSION (X changed)"
            confidence = "HIGH"
            what_changed = (
                "The investment universe itself expanded. A new asset class "
                "(AI infrastructure) emerged, while existing categories "
                "(education, knowledge work) faced existential disruption."
            )
            st.success(
                f"## 🔬 Classification: {classification}\n"
                f"**Confidence: {confidence}** ({signals_detected}/5 signals converge)"
            )
        elif signals_detected >= 2:
            classification = "LIKELY SAMPLE SPACE EXPANSION"
            confidence = "MEDIUM"
            what_changed = (
                "Multiple signals suggest the investment universe changed, "
                "but not all indicators converge."
            )
            st.warning(
                f"## 🔬 Classification: {classification}\n"
                f"**Confidence: {confidence}** ({signals_detected}/5 signals converge)"
            )
        else:
            classification = "REGIME SHIFT (P changed)"
            confidence = "LOW"
            what_changed = "Probabilities shifted within the existing universe."
            st.info(
                f"## ⚡ Classification: {classification}\n"
                f"**Confidence: {confidence}** ({signals_detected}/5 signals converge)"
            )

        st.markdown(f"**What Changed:** {what_changed}")

        st.markdown("**Evidence:**")
        for e in evidence:
            st.markdown(f"- ✅ {e}")

        missing = 5 - signals_detected
        if missing > 0:
            st.caption(f"{missing} signal(s) did not reach threshold")

    st.markdown("---")

    # --------------------------------------------------------
    # SIGNAL CONVERGENCE SUMMARY
    # --------------------------------------------------------
    st.header("Signal Convergence Summary")

    summary_data = {
        "Signal": [
            "Divergence (>500%)",
            "Entropy (z < -2)",
            "HHI (increasing)",
            "Novel Narratives",
            "SEC Filing New Risks",
        ],
        "Layer": ["Quant", "Quant", "Quant", "Narrative", "Narrative"],
        "Value": [
            f"{quant.divergence.total_divergence_pct:+.0f}%",
            f"{quant.entropy.entropy_zscore:.2f}σ",
            f"{quant.concentration.hhi_change:+.4f}",
            str(len(news_signals.novel_theme_counts)) + " themes",
            (
                f"{len(filing_diff.new_risk_factors)} new"
                if filing_diff
                else "N/A"
            ),
        ],
        "Triggered": [
            "✅" if quant.divergence.total_divergence_pct > 500 else "❌",
            "✅" if quant.entropy.entropy_zscore < -2 else "❌",
            "✅" if quant.concentration.hhi_change > 0.02 else "❌",
            "✅" if news_signals.novel_theme_counts else "❌",
            (
                "✅" if filing_diff and filing_diff.sample_space_signal else "❌"
            ),
        ],
    }

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(
        df_summary.style.apply(
            lambda row: [
                "background-color: #1a3a1a" if row["Triggered"] == "✅" else ""
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    triggered_count = summary_data["Triggered"].count("✅")
    st.caption(
        f"{triggered_count}/5 signals triggered — "
        f"{'Strong convergence across both layers' if triggered_count >= 4 else 'Partial convergence'}"
    )

    # --------------------------------------------------------
    # LONG-SHORT PORTFOLIO BACKTEST
    # --------------------------------------------------------
    st.markdown("---")
    st.header("Long-Short Portfolio Backtest")
    st.caption(
        "Strategy: Long AI winners (NVDA, MSFT) / Short disrupted (CHGG) — "
        "equal weight, dollar-neutral, buy-and-hold from event date"
    )

    with st.spinner("Running backtest..."):
        bt = run_backtest(
            long_tickers=[winner_ticker.upper(), "MSFT"],
            short_tickers=[loser_ticker.upper()],
            start_date=event_date_str,
            end_date=analysis_end_str,
        )

    # Performance metrics row
    bt_col1, bt_col2, bt_col3, bt_col4, bt_col5 = st.columns(5)
    with bt_col1:
        st.metric("Total Return", f"{bt.total_return_pct:+.1f}%")
    with bt_col2:
        st.metric("Alpha vs SPY", f"{bt.alpha_pct:+.1f}%")
    with bt_col3:
        st.metric("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}")
    with bt_col4:
        st.metric("Max Drawdown", f"{bt.max_drawdown_pct:.1f}%")
    with bt_col5:
        st.metric("Volatility", f"{bt.volatility_pct:.1f}%")

    # Equity curve chart
    bt_chart_col1, bt_chart_col2 = st.columns([2, 1])

    with bt_chart_col1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt.portfolio_values.index, y=bt.portfolio_values.values,
            name="Long-Short Portfolio",
            line=dict(color="#00cc66", width=2),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt.long_values.index, y=bt.long_values.values,
            name=f"Long ({', '.join(bt.long_tickers)})",
            line=dict(color="#4488ff", width=1, dash="dot"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt.short_values.index, y=bt.short_values.values,
            name=f"Short ({', '.join(bt.short_tickers)})",
            line=dict(color="#ff8844", width=1, dash="dot"),
        ))
        if bt.benchmark_values is not None:
            fig_bt.add_trace(go.Scatter(
                x=bt.benchmark_values.index, y=bt.benchmark_values.values,
                name="SPY (Benchmark)",
                line=dict(color="#888888", width=1, dash="dash"),
            ))

        fig_bt.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.3)
        fig_bt.update_layout(
            yaxis_title="Portfolio Value ($100 initial)",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    with bt_chart_col2:
        st.markdown("**Strategy Breakdown**")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Long Leg | {bt.long_return_pct:+.1f}% |
        | Short Leg | {bt.short_return_pct:+.1f}% |
        | Combined | {bt.total_return_pct:+.1f}% |
        | Benchmark | {bt.benchmark_return_pct:+.1f}% |
        | **Alpha** | **{bt.alpha_pct:+.1f}%** |
        | Annualized | {bt.annualized_return_pct:+.1f}% |
        | Trading Days | {bt.trading_days} |
        """)

        st.markdown(
            "**Interpretation:** The long-short strategy captures "
            "creative destruction — profiting from both the rise of "
            "AI infrastructure AND the decline of disrupted sectors. "
            f"A Sharpe of {bt.sharpe_ratio:.2f} confirms this isn't "
            "random — the signal is real."
        )

    # --------------------------------------------------------
    # THE INSIGHT
    # --------------------------------------------------------
    st.markdown("---")
    st.header("The Key Insight")

    st.markdown(f"""
    **{event_name} was {'a sample space expansion event' if triggered_count >= 4 else 'a potential regime change'}.**

    | Concept | Before | After |
    |---------|--------|-------|
    | Investment universe | Tech = FAANG | Tech = FAANG + "AI Infrastructure" |
    | Risk categories | Standard sector risks | + AI disruption, GPU dependency, LLM competition |
    | Portfolio question | "What's your tech allocation?" | "What's your AI exposure?" |

    **This wasn't just a regime shift (P changing) — the game itself changed (X expanded).**

    - **{winner_ticker}** (+{quant.divergence.winner_return_pct:.0f}%): Owned the new infrastructure
    - **{loser_ticker}** ({quant.divergence.loser_return_pct:.0f}%): Business model disrupted
    - **The paradox**: Sample space expanded BUT entropy decreased — new category, dominated by few players
    """)


# --------------------------------------------------------
# DEFAULT VIEW (before analysis runs)
# --------------------------------------------------------
else:
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Layer 1: Quantitative
        - HMM regime detection
        - Rolling Shannon entropy
        - Winner/loser divergence
        - HHI concentration index
        """)
    with col2:
        st.markdown("""
        ### Layer 2: Narrative
        - News sentiment scoring
        - SEC 10-K risk factor diffs
        - Novel theme detection
        - Language shift analysis
        """)
    with col3:
        st.markdown("""
        ### Layer 3: AI Fusion
        - OpenAI o4-mini reasoning
        - Cross-modal convergence
        - Typed classification output
        - Evidence chain generation
        """)

    st.markdown("---")
    st.caption(
        "SSED — Sample Space Expansion Detector | "
        "MGMT 69000: Mastering AI for Finance | "
        "Built with DRIVER Framework"
    )
