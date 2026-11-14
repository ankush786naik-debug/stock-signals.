"""
Free stock screener: fundamental + technical scoring -> BUY / HOLD / SELL
Markets: India (NSE/BSE), US (S&P 500), or any custom list.

Install:  pip install yfinance pandas numpy lxml
Run:
  python stock_screener.py --market india --limit 100
  python stock_screener.py --market us --limit 100
  python stock_screener.py --tickers RELIANCE.NS TCS.NS INFY.NS AAPL MSFT

Educational use only. NOT financial advice.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---- Value-dip strategy settings (edit these) ----
MIN_FUND = 65     # company quality score needed (0-100)
MIN_DROP = 15     # price must be at least this % below 52-week high
MAX_DROP = 45     # deeper than this = "falling knife", only WATCH
MAX_RSI = 45      # RSI must show weakness / oversold
STOP_PCT = 10     # stop-loss % below buy price

# Fallback list if the NSE download fails
NIFTY_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "NESTLEIND", "WIPRO",
    "ULTRACEMCO", "ONGC", "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL",
]


# ---------------------------------------------------------------- universes
def india_universe():
    """Nifty 500 first (liquid, quality stocks), then all NSE, then fallback."""
    sources = [
        ("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv", "Symbol"),
        ("https://archives.nseindia.com/content/equities/EQUITY_L.csv", "SYMBOL"),
    ]
    for url, col in sources:
        try:
            df = pd.read_csv(url, storage_options=HEADERS)
            df.columns = [c.strip() for c in df.columns]
            syms = [str(s).strip() + ".NS" for s in df[col].dropna()]
            if len(syms) > 50:
                print(f"Loaded {len(syms)} Indian stocks from {url}")
                return syms
        except Exception as e:
            print(f"Source failed ({url}): {e}")
    print("Using fallback Nifty list (25 stocks).")
    return [s + ".NS" for s in NIFTY_FALLBACK]


def us_universe():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url, storage_options=HEADERS)[0]
        return [s.replace(".", "-") for s in df["Symbol"]]
    except Exception as e:
        print(f"S&P list failed ({e}); using small default list.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "XOM"]


# ---------------------------------------------------------------- helpers
def g(d, key):
    v = d.get(key)
    return v if isinstance(v, (int, float)) and not np.isnan(v) else None


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down
    return float((100 - 100 / (1 + rs)).iloc[-1])


def add(points, cond, pts, bucket):
    """Add points to a bucket if the metric exists (cond is None = missing)."""
    if cond is None:
        return
    bucket["score"] += pts if cond else 0
    bucket["max"] += pts


# ---------------------------------------------------------------- analysis
def analyze(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="1y")
        if hist.empty or len(hist) < 200 or not info:
            return None

        close = hist["Close"]
        price = float(close.iloc[-1])

        pe, pb = g(info, "trailingPE"), g(info, "priceToBook")
        peg = g(info, "trailingPegRatio") or g(info, "pegRatio")
        roe, margin = g(info, "returnOnEquity"), g(info, "profitMargins")
        rev_g, earn_g = g(info, "revenueGrowth"), g(info, "earningsGrowth")
        de, cur = g(info, "debtToEquity"), g(info, "currentRatio")
        fcf = g(info, "freeCashflow")

        # --- Fundamental scoring (each bucket tracks points earned / possible)
        val = {"score": 0, "max": 0}
        add(val, None if pe is None else 0 < pe < 25, 15, val)
        add(val, None if pb is None else 0 < pb < 4, 10, val)
        add(val, None if peg is None else 0 < peg < 1.5, 10, val)

        prof = {"score": 0, "max": 0}
        add(prof, None if roe is None else roe > 0.15, 15, prof)
        add(prof, None if margin is None else margin > 0.10, 10, prof)

        growth = {"score": 0, "max": 0}
        add(growth, None if rev_g is None else rev_g > 0.10, 10, growth)
        add(growth, None if earn_g is None else earn_g > 0.10, 10, growth)

        health = {"score": 0, "max": 0}
        # yfinance reports debtToEquity in percent (50 = 0.5x)
        add(health, None if de is None else de < 100, 10, health)
        add(health, None if cur is None else cur > 1.2, 5, health)
        add(health, None if fcf is None else fcf > 0, 5, health)

        buckets = [val, prof, growth, health]
        got = sum(b["score"] for b in buckets)
        poss = sum(b["max"] for b in buckets)
        if poss < 40:  # too little data to judge
            return None
        fund_score = round(100 * got / poss, 1)

        # --- Technical timing
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        r = rsi(close)
        hi, lo = close.max(), close.min()
        pos52 = (price - lo) / (hi - lo) if hi > lo else 0.5
        uptrend = price > sma200 and sma50 > sma200

        # --- Value-dip logic: good company + price has fallen
        drop = (hi - price) / hi * 100            # % below 52-week high
        stabilising = price > close.iloc[-6]      # last 5 days not falling further
        dip_ok = MIN_DROP <= drop <= MAX_DROP and r <= MAX_RSI

        if fund_score >= MIN_FUND and dip_ok and stabilising:
            signal = "BUY DIP"
        elif fund_score >= MIN_FUND and dip_ok:
            signal = "WATCH (still falling - wait for stabilising)"
        elif fund_score >= MIN_FUND and drop > MAX_DROP:
            signal = "WATCH (fell too much - check why)"
        elif fund_score >= MIN_FUND:
            signal = "WAIT (good company, not cheap yet)"
        elif fund_score < 40:
            signal = "AVOID"
        else:
            signal = "HOLD"

        # Trade plan: buy only if price moves ABOVE yesterday's high (confirmation)
        tick = lambda x: round(round(x * 20) / 20, 2)   # NSE tick size 0.05
        entry = float(hist["High"].iloc[-1]) * 1.002
        swing_low = float(hist["Low"].iloc[-10:].min())
        stop = max(swing_low * 0.99, entry * (1 - STOP_PCT / 100))
        stop = min(stop, entry * 0.97)                  # at least 3% risk room
        t1 = entry + 0.5 * (hi - entry)                 # halfway back to high
        t2 = hi                                         # full recovery
        rr = (t1 - entry) / (entry - stop) if entry > stop else 0

        return {
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Sector": info.get("sector"),
            "Price": round(price, 2),
            "Drop%": round(drop, 1),
            "Entry": tick(entry),
            "StopLoss": tick(stop),
            "Target1": tick(t1),
            "Target2": tick(t2),
            "RR": round(rr, 1),
            "PE": pe, "PB": pb, "PEG": peg,
            "ROE%": None if roe is None else round(roe * 100, 1),
            "Margin%": None if margin is None else round(margin * 100, 1),
            "RevGrowth%": None if rev_g is None else round(rev_g * 100, 1),
            "D/E": de,
            "RSI": round(r, 1),
            "FundScore": fund_score,
            "Signal": signal,
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["india", "us"], default="india")
    ap.add_argument("--tickers", nargs="*", help="custom list, e.g. TCS.NS AAPL")
    ap.add_argument("--limit", type=int, default=100, help="max stocks to scan")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    tickers = a.tickers or (india_universe() if a.market == "india" else us_universe())
    tickers = tickers[: a.limit]
    print(f"Scanning {len(tickers)} stocks...")

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(analyze, tickers) if r]

    if not rows:
        print("No data returned. Check internet / ticker symbols.")
        return

    df = pd.DataFrame(rows).sort_values("FundScore", ascending=False)
    df.to_csv("screener_results.csv", index=False)

    cols = ["Ticker", "Price", "Drop%", "RSI", "FundScore", "Entry", "StopLoss", "Target1"]
    pd.set_option("display.width", 200)
    print("\nBUY DIP candidates:")
    print(df[df.Signal == "BUY DIP"][cols].head(15).to_string(index=False))
    print("\nFull results saved to screener_results.csv")
    print("Educational only - not financial advice. Verify before investing.")


if __name__ == "__main__":
    main()
