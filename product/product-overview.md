# Sample Space Expansion Detector (SSED)

## The Problem
The existing thesis proves sample space expansion happened *retroactively* — after the fact, using historical data. But there's no tool that detects it *as it unfolds*. Traditional regime detection (HMMs) catches probability shifts within a known universe (P changes). It misses when the investment universe itself changes (X changes).

**The gap confirmed by research:** No existing open-source project fuses quantitative regime detection with LLM-powered narrative analysis. FinGPT does sentiment. hmmlearn does regimes. Nobody combines them to distinguish P changes from X changes.

## Success Looks Like
A Streamlit dashboard that monitors markets and flags:
> "Warning: potential sample space expansion detected in [sector] — new asset class forming, concentration increasing, divergence accelerating."

- Demo it on the ChatGPT launch period (Nov 2022 - Dec 2024) as proof of concept
- Show it running on current data for live monitoring
- Class deliverable for MGMT 69000 that demonstrates the thesis as a working AI system
- Clean enough for a GitHub portfolio piece (README, .env.example, reproducible)

## Building On (Existing Foundations)

### From This Project
- **Shannon entropy calculations** — portfolio/market concentration measurement (`portfolio_analyzer.py`)
- **Divergence metrics** — winner/loser spread tracking, NVDA vs CHGG (`validate_thesis.py`)
- **HHI concentration index** — market concentration quantification
- **Streamlit UI** — interactive dashboard framework
- **yfinance integration** — free price data pipeline
- **Cascade mapping structure** — trigger -> direct -> second-order -> third-order (`starter_template.py`)

### From the Ecosystem (Confirmed via Research)
- **hmmlearn** (~3K stars) — Gaussian HMM for regime state estimation on returns
- **OpenAI API (function calling + structured outputs)** — LLM reasons over tool-returned data, never generates numbers
- **edgartools** + **sec-edgar-downloader** — SEC filing section extraction (Item 1A Risk Factors)
- **NewsAPI** (free tier: 100 req/day) — financial news feed
- **Instructor** (~8K stars) — Pydantic-based structured output from OpenAI (typed regime classifications)
- **FinBERT** (`ProsusAI/finbert`) — fast batch sentiment pre-filter before GPT deep analysis

### Considered but Not Using
- **LangChain/LangGraph** — Unnecessary abstraction for our use case; direct OpenAI function calling is simpler and gives full control
- **FinGPT** — Good data pipelines but we don't need fine-tuned models; OpenAI API is sufficient
- **OpenBB** — Overkill for our data needs; yfinance + NewsAPI + EDGAR covers it

## The Unique Part
Fusing quantitative regime detection with LLM-powered narrative analysis to distinguish:
- **Regime shifts** (P changes) — probabilities change within a known universe
- **Sample space expansion** (X changes) — the investment universe itself changes

The detection signal: **when both statistical measures (entropy spike, HMM log-likelihood deterioration) and narrative measures (novel risk factors, unprecedented terminology in filings/news) simultaneously indicate the historical model is breaking down.**

This distinction is the original intellectual contribution. No existing tool makes it.

## Architecture

```
Layer 1: Quantitative Signals (deterministic, runs locally)
  ├── Shannon entropy over rolling windows (from existing code)
  ├── Divergence acceleration (winner/loser spread velocity)
  ├── Concentration shift (HHI, top-N weight changes)
  ├── HMM regime state transitions + log-likelihood monitoring (hmmlearn)
  └── Output: structured dict of quant signals

Layer 2: Narrative Signals (OpenAI + multi-source data)
  ├── News sentiment (GPT-4.1-nano bulk scoring via NewsAPI feed)
  ├── SEC filing language shifts (edgartools → Item 1A diff → o4-mini analysis)
  ├── "New category" detection (novel terminology emergence across filings)
  └── Output: structured MarketNarrativeSignals (Pydantic model)

Layer 3: Fusion & Classification (OpenAI o4-mini reasoning)
  ├── Receives Layer 1 + Layer 2 as tool results
  ├── Regime shift vs. sample space expansion classifier
  ├── Structured output: RegimeClassification (Pydantic)
  │   ├── classification: regime_shift | sample_space_expansion | mean_reversion | inconclusive
  │   ├── confidence: high | medium | low
  │   ├── key_evidence: list[str]
  │   └── reasoning: str
  └── Natural language alert generation

UI: Streamlit dashboard (extends existing portfolio analyzer)
  ├── Historical backtest view (ChatGPT launch demo)
  ├── Signal dashboard (Layer 1 + Layer 2 visualized)
  └── AI analysis panel (Layer 3 classification + evidence chain)
```

## Tiered Build Plan

| Tier | What | Demo Value | OpenAI Cost |
|------|------|------------|-------------|
| **Tier 1 (Core)** | Price data + HMM + entropy + OpenAI reasoning via function calling | Detects ChatGPT event retroactively with AI explanation | Low (~$0.50/analysis) |
| **Tier 2 (Impressive)** | + news sentiment analysis (GPT-4.1-nano batch) | Detects narrative shifts alongside quantitative signals | Moderate (~$2-5 for bulk news) |
| **Tier 3 (Wow Factor)** | + SEC filings diff analysis + cross-company novel risk detection | Full multi-source fusion — "AI infrastructure" appears in 10-Ks as proof | Moderate (~$5-10 for filing analysis) |

## Tech Stack
- **UI:** Streamlit (extend existing app)
- **Quant:** pandas, numpy, hmmlearn, scipy
- **AI:** OpenAI API
  - `o4-mini` (reasoning_effort="high") — Layer 3 fusion/classification
  - `GPT-4.1-nano` — Layer 2 bulk sentiment scoring
  - Structured Outputs via `client.beta.chat.completions.parse()` + Pydantic models
  - Function calling with `strict: True` on all tool definitions
- **Data:** yfinance, NewsAPI (free tier), SEC EDGAR (edgartools + sec-edgar-downloader)
- **Structured Output:** Pydantic models (or Instructor library)
- **Config:** python-dotenv for API key management

## Key Design Principles
1. **LLM as orchestrator, not calculator** — All numbers come from deterministic code; OpenAI interprets and reasons
2. **Function calling pattern** — Define tools (get_entropy, get_divergence, get_hmm_regime, get_news_sentiment, search_sec_filings) that return real data
3. **Structured outputs everywhere** — Every LLM response is typed via Pydantic, never free-form text parsing
4. **Two-tier sentiment** — GPT-4.1-nano for bulk scoring, o4-mini only for deep analysis and fusion
5. **Temporal integrity** — Point-in-time data discipline, no lookahead bias
6. **Chat Completions, not Assistants API** — Full control over context, multi-model routing, lower latency

## Open Questions
- How far back should the historical backtest window extend?
- Real-time monitoring frequency (daily? weekly?) for live demo
- Whether to include a "what-if" simulator for hypothetical expansion events
- Reddit/social signals: add in Tier 3 or skip? (PRAW API complexity vs. signal value)
