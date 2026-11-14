"""
Daily scan -> Telegram. Needs stock_screener.py in the same folder.
Env vars: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, optional MARKET (india/us), LIMIT
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from stock_screener import analyze, india_universe, us_universe

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
MARKET = os.getenv("MARKET", "india")
LIMIT = int(os.getenv("LIMIT", "300"))


def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT_ID, "text": text[:4000]}, timeout=30)
    r.raise_for_status()


def main():
    tickers = (india_universe() if MARKET == "india" else us_universe())[:LIMIT]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(analyze, tickers) if r]

    if not rows:
        send("Scan returned no data today (data source may be down).")
        return

    df = pd.DataFrame(rows).sort_values(["FundScore", "Drop%"], ascending=False)
    df.to_csv("screener_results.csv", index=False)

    buys = df[df.Signal == "BUY DIP"].head(10)
    watch = df[df.Signal.str.startswith("WATCH")].head(5)
    cur = "₹" if MARKET == "india" else "$"

    msg = f"📊 Value-dip signals {date.today()} ({len(df)} scanned)\n"
    msg += "Good company, price down, 3-month hold.\nBuy ONLY if price trades above the entry level.\n\n"
    msg += "🟢 BUY DIP:\n"
    if buys.empty:
        msg += "None today - wait\n"
    for _, r in buys.iterrows():
        name = r.Ticker.replace(".NS", "")
        msg += (f"BUY {name} above {cur}{r.Entry:,.2f}\n"
                f"SL {cur}{r.StopLoss:,.2f} | T1 {cur}{r.Target1:,.2f} "
                f"| T2 {cur}{r.Target2:,.2f}\n"
                f"Down {r['Drop%']}% from high | Score {r.FundScore} "
                f"| RSI {r.RSI} | R:R 1:{r.RR}\n\n")
    if not watch.empty:
        msg += "\n👀 Watchlist (not yet):\n"
        for _, r in watch.iterrows():
            msg += f"• {r.Ticker}  down {r['Drop%']}%  score {r.FundScore}\n"
    msg += "\nNot financial advice. Check news/results before buying."
    send(msg)


if __name__ == "__main__":
    main()
