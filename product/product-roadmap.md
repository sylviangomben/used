# Roadmap

Building on: Existing entropy/divergence code (portfolio_analyzer.py, validate_thesis.py), Streamlit UI, yfinance pipeline, cascade mapping structure (starter_template.py).

## Sections

### 1. Quant Signal Engine
Wire up hmmlearn for regime detection on top of existing entropy/divergence code. Output a structured `QuantSignals` dict that Layer 3 can consume. Demo: ChatGPT launch period showing HMM regime transitions + entropy spike + divergence acceleration on a timeline.

### 2. OpenAI Function Calling Core
Build the tool-calling loop where o4-mini reasons over quant signals. Define tools (`get_entropy`, `get_divergence`, `get_hmm_regime`) with `strict: True`, Pydantic structured output for `RegimeClassification`. Demo: "Was ChatGPT launch a regime shift or sample space expansion?" — typed, evidence-backed answer from quant data alone.

### 3. Narrative Signal Layer
Add news sentiment via NewsAPI + GPT-4.1-nano bulk scoring. Add SEC filing diff analysis via edgartools (Item 1A Risk Factors year-over-year). New tools: `get_news_sentiment`, `search_sec_filings`. Demo: narrative signals (novel "AI infrastructure" language) appear alongside quant signals.

### 4. Fusion Dashboard
Streamlit UI bringing it all together — historical backtest view (ChatGPT demo), signal timeline (Layer 1 + Layer 2 side by side), AI analysis panel (Layer 3 classification with evidence chain and confidence scoring). This is the class deliverable.
