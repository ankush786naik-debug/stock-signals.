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

    df = pd.DataFrame(rows).sort_values("FundScore", ascending=False)
    df.to_csv("screener_results.csv", index=False)

    buys = df[df.Signal == "BUY"].head(15)
    sells = df[df.Signal.str.startswith("SELL")].head(5)

    msg = f"📊 Daily signals {date.today()} ({len(df)} stocks scanned)\n\n"
    msg += "🟢 BUY candidates:\n"
    if buys.empty:
        msg += "None today\n"
    for _, r in buys.iterrows():
        msg += f"• {r.Ticker}  ₹/$ {r.Price}  score {r.FundScore}  RSI {r.RSI}\n"
    if not sells.empty:
        msg += "\n🔴 Avoid / Sell:\n"
        for _, r in sells.iterrows():
            msg += f"• {r.Ticker}  score {r.FundScore}\n"
    msg += "\nNot financial advice. Do your own research."
    send(msg)


if __name__ == "__main__":
    main()
