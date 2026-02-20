"""
SSED Sector Scanner & Heatmap

Scans S&P 500 sectors for expansion/contraction signals using
entropy, divergence, and momentum metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

from ssed.quant_signals import fetch_prices


# Sector ETFs representing S&P 500 sectors
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Disc.": "XLY",
    "Communication": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

# Top holdings per sector for divergence analysis
SECTOR_STOCKS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "PFE"],
    "Financials": ["BRK-B", "JPM", "V", "MA", "BAC"],
    "Consumer Disc.": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Communication": ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
    "Industrials": ["GE", "CAT", "UNP", "RTX", "BA"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities": ["NEE", "SO", "DUK", "CEG", "SRE"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "PSA"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM"],
}


@dataclass
class SectorSignal:
    """Signals for a single sector."""
    sector: str
    etf: str
    return_pct: float
    volatility: float
    entropy_score: float  # normalized 0-1, lower = more concentrated
    divergence_score: float  # spread between top and bottom stock
    momentum_score: float  # recent vs longer-term performance
    expansion_score: float  # composite signal (0-1, higher = more expansion-like)


def _shannon_entropy(returns: pd.Series, bins: int = 20) -> float:
    """Compute Shannon entropy of return distribution."""
    counts, _ = np.histogram(returns.dropna(), bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def scan_sectors(
    lookback_days: int = 180,
    momentum_short: int = 30,
) -> list[SectorSignal]:
    """
    Scan all sectors and compute expansion signals.

    Returns list of SectorSignal sorted by expansion_score (highest first).
    """
    # Fetch all ETFs via financialdatasets.ai
    etf_tickers = list(SECTOR_ETFS.values())
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")
    try:
        etf_prices = fetch_prices(etf_tickers, start_date, end_date)
    except Exception:
        return []

    if etf_prices.empty:
        return []

    signals = []

    for sector, etf in SECTOR_ETFS.items():
        if etf not in etf_prices.columns:
            continue

        prices = etf_prices[etf].dropna()
        if len(prices) < 60:
            continue

        returns = prices.pct_change().dropna()

        # Return
        total_return = (prices.iloc[-1] / prices.iloc[-lookback_days] - 1) * 100 if len(prices) > lookback_days else 0

        # Volatility
        vol = float(returns.tail(lookback_days).std() * np.sqrt(252) * 100)

        # Entropy (normalized by max possible)
        ent = _shannon_entropy(returns.tail(lookback_days))
        max_ent = np.log2(20)  # max entropy for 20 bins
        entropy_score = ent / max_ent if max_ent > 0 else 0.5

        # Momentum: short-term vs long-term
        short_ret = (prices.iloc[-1] / prices.iloc[-momentum_short] - 1) if len(prices) > momentum_short else 0
        long_ret = (prices.iloc[-1] / prices.iloc[-lookback_days] - 1) if len(prices) > lookback_days else 0
        momentum_score = float(short_ret - long_ret / (lookback_days / momentum_short)) if long_ret != 0 else 0

        # Divergence: fetch top stocks in sector
        stocks = SECTOR_STOCKS.get(sector, [])
        divergence_score = 0.0
        if stocks:
            try:
                stock_prices = fetch_prices(stocks, start_date, end_date)
                if not stock_prices.empty:
                    stock_returns = {}
                    for s in stocks:
                        if s in stock_prices.columns:
                            sp = stock_prices[s].dropna()
                            if len(sp) > lookback_days:
                                stock_returns[s] = (sp.iloc[-1] / sp.iloc[-lookback_days] - 1) * 100

                    if stock_returns:
                        vals = list(stock_returns.values())
                        divergence_score = max(vals) - min(vals)
            except Exception:
                pass

        # Composite expansion score (0-1)
        # High divergence + low entropy + high momentum = expansion signal
        norm_div = min(divergence_score / 200, 1.0)  # normalize: 200% divergence = max
        inv_entropy = 1 - entropy_score  # lower entropy = higher signal
        norm_mom = min(max(momentum_score * 10, 0), 1.0)  # normalize momentum

        expansion_score = (norm_div * 0.4 + inv_entropy * 0.3 + norm_mom * 0.3)

        signals.append(SectorSignal(
            sector=sector,
            etf=etf,
            return_pct=round(float(total_return), 2),
            volatility=round(vol, 2),
            entropy_score=round(float(entropy_score), 3),
            divergence_score=round(float(divergence_score), 1),
            momentum_score=round(float(momentum_score), 4),
            expansion_score=round(float(expansion_score), 3),
        ))

    # Sort by expansion score
    signals.sort(key=lambda s: s.expansion_score, reverse=True)
    return signals


def scan_market_movers(
    lookback_days: int = 90,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Scan a broad set of stocks for the biggest divergences.
    Returns top movers with their signals.
    """
    # Broad universe: top stocks across sectors
    all_stocks = []
    for stocks in SECTOR_STOCKS.values():
        all_stocks.extend(stocks)
    all_stocks = list(set(all_stocks))

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    try:
        prices = fetch_prices(all_stocks, start_date, end_date)
    except Exception:
        return pd.DataFrame()

    if prices.empty:
        return pd.DataFrame()

    results = []
    for ticker in all_stocks:
        if ticker not in prices.columns:
            continue
        p = prices[ticker].dropna()
        if len(p) < lookback_days:
            continue

        ret = (p.iloc[-1] / p.iloc[-lookback_days] - 1) * 100
        vol = float(p.pct_change().dropna().tail(lookback_days).std() * np.sqrt(252) * 100)
        ent = _shannon_entropy(p.pct_change().dropna().tail(lookback_days))

        # Find which sector
        sector = "Unknown"
        for s, stocks in SECTOR_STOCKS.items():
            if ticker in stocks:
                sector = s
                break

        results.append({
            "Ticker": ticker,
            "Sector": sector,
            "Return": round(float(ret), 1),
            "Volatility": round(vol, 1),
            "Entropy": round(float(ent), 3),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("Return", ascending=False)

    # Top winners and losers
    winners = df.head(top_n // 2)
    losers = df.tail(top_n // 2)
    return pd.concat([winners, losers]).reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("SSED Sector Scanner")
    print("=" * 60)

    signals = scan_sectors()
    for s in signals:
        flag = "🔴" if s.expansion_score > 0.6 else "🟡" if s.expansion_score > 0.4 else "🟢"
        print(
            f"{flag} {s.sector:20s} | Return: {s.return_pct:+6.1f}% | "
            f"Divergence: {s.divergence_score:6.1f}% | "
            f"Entropy: {s.entropy_score:.3f} | "
            f"Expansion: {s.expansion_score:.3f}"
        )
