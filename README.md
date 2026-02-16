# Sample Space Expansion Detector (SSED)

**Detecting when the investment universe itself changes — not just probabilities within it.**

MGMT 69000 — AI in Finance | Purdue University

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## What This Does

Traditional regime detection (HMMs) catches probability shifts within a known universe (*P changes*). This tool goes further — it detects when the investment universe itself changes (*X changes*, or **sample space expansion**).

**Case study: ChatGPT's launch (Nov 30, 2022)**
- NVDA went from a gaming GPU company to AI infrastructure backbone (+800% in 2 years)
- CHGG went from an education leader to disrupted incumbent (-90%)
- This wasn't a normal market regime shift — AI created entirely new categories of winners and losers

## Architecture

```
Layer 1: Quantitative Signals (deterministic, runs locally)
  ├── HMM regime detection (hmmlearn) — 3 volatility states
  ├── Shannon entropy — rolling concentration/novelty measurement
  ├── Winner/loser divergence — spread velocity + acceleration
  └── HHI concentration index — market structure shifts

Layer 2: Narrative Signals (OpenAI API)
  ├── News sentiment (GPT-4.1-nano bulk scoring)
  ├── SEC filing language shifts (10-K Risk Factor diffs)
  └── Novel risk factor detection ("AI" appearing in CHGG's 10-K)

Layer 3: Fusion & Classification (OpenAI o4-mini reasoning)
  ├── 6 function-calling tools with strict: True
  ├── Structured output: RegimeClassification (Pydantic)
  └── P change vs X change classification with evidence chain
```

## Long-Short Portfolio Backtest

The system's thesis is validated through a dollar-neutral long-short portfolio:
- **Long:** NVDA, MSFT (AI infrastructure winners)
- **Short:** CHGG (disrupted education incumbent)
- **Period:** Nov 2022 – Dec 2024 (post-ChatGPT launch)

| Metric | Value |
|--------|-------|
| Total Return | +243% |
| Alpha vs SPY | +191% |
| Sharpe Ratio | 2.35 |
| Max Drawdown | -17.3% |

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/ssed.git
cd ssed
pip install -r requirements.txt

# Run the dashboard (works without API keys — demo mode)
streamlit run ssed/dashboard.py
```

### Optional: Enable AI Features

```bash
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...     (enables Layer 2+3 AI analysis)
#   NEWSAPI_KEY=...           (enables live news sentiment)
```

## Three Operating Modes

| Mode | What You Need | What Works |
|------|--------------|------------|
| **Demo** (no keys) | Nothing | Quant signals, backtest, heuristic classification |
| **OpenAI only** | `OPENAI_API_KEY` | + AI classification, sentiment scoring, filing analysis |
| **Full** | Both keys | + Live news feed from NewsAPI |

## Project Structure

```
ssed/
├── quant_signals.py      # Layer 1: HMM, entropy, divergence, HHI
├── narrative_signals.py  # Layer 2: news sentiment, SEC filing diffs
├── openai_core.py        # Layer 3: function calling + fusion classifier
├── backtest.py           # Long-short portfolio backtest engine
└── dashboard.py          # Streamlit UI — all layers visualized
```

## Key Design Decisions

1. **LLM as orchestrator, not calculator** — All numbers come from deterministic code; OpenAI interprets and reasons
2. **Function calling with `strict: True`** — 6 tools that return real data; LLM never generates numbers
3. **Structured outputs via Pydantic** — Every LLM response is typed, never free-form text parsing
4. **Graceful degradation** — Works without any API keys using heuristic fallbacks
5. **Direct APIs over frameworks** — OpenAI function calling directly (no LangChain), SEC EDGAR API directly (no heavy wrappers)

## Tech Stack

- **UI:** Streamlit
- **Quant:** pandas, numpy, hmmlearn, scikit-learn, scipy
- **AI:** OpenAI API (o4-mini for reasoning, GPT-4.1-nano for bulk sentiment)
- **Data:** yfinance, NewsAPI, SEC EDGAR
- **Visualization:** Plotly

## References

- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models"
- Herfindahl-Hirschman Index (HHI) — U.S. Department of Justice
- OpenAI Function Calling — [docs.openai.com](https://platform.openai.com/docs/guides/function-calling)
