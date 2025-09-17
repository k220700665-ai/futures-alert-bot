import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import ta
import time
import logging
import requests
import os

# === CONFIGURATION ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
LOG_FILE = "hot_stream_signals.log"
REFRESH_INTERVAL = 3600  # 1 hour in seconds
MAX_PAIRS = 50

# === Setup Logging ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Telegram Alert Function ===
def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}
    try:
        requests.post(url, data=payload)
        logging.info(f"Telegram alert sent: {message}")
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")

# === Fetch Hot Perpetual Pairs ===
def get_hot_perpetual_symbols(limit=MAX_PAIRS):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"Error fetching Binance data: {e}")
        return []
    if not isinstance(data, list):
        logging.error("Unexpected response format from Binance API")
        return []
    sorted_by_volume = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)
    top_volume = [d['symbol'].lower() for d in sorted_by_volume if d['symbol'].endswith('USDT')][:limit]
    sorted_by_gain = sorted(data, key=lambda x: float(x['priceChangePercent']), reverse=True)
    top_gainers = [d['symbol'].lower() for d in sorted_by_gain if d['symbol'].endswith('USDT')][:limit]
    sorted_by_loss = sorted(data, key=lambda x: float(x['priceChangePercent']))
    top_losers = [d['symbol'].lower() for d in sorted_by_loss if d['symbol'].endswith('USDT')][:limit]
    combined = list(set(top_volume + top_gainers + top_losers))
    return combined

# === Async Retry Decorator ===
def async_retry(max_retries=3, delay=2):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logging.warning(f"Exception on attempt {attempt + 1} for {func.__name__}: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
            logging.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator

# === WebSocket Connection with Retry ===
@async_retry(max_retries=5, delay=3)
async def connect_with_retry(url):
    return await websockets.connect(url)

# === WebSocket Handler ===
async def handle_stream(symbol, signal_cache):
    url = f"wss://fstream.binance.com/ws/{symbol}@kline_1m"
    df = fetch_historical_klines(symbol)
    try:
        ws = await connect_with_retry(url)
    except Exception as e:
        logging.error(f"Failed to connect websocket for {symbol}: {e}")
        return
    async with ws:
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                kline = data['k']
                new_row = {
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df = df.tail(100)
                funding_rate = fetch_latest_funding_rate(symbol)
                signal, entry_price, reason, tp, sl = evaluate_signal(df, funding_rate, symbol)
                if signal in ["LONG", "SHORT"] and signal_cache.get(symbol) != signal:
                    alert_msg = (
                        f"📢 {symbol.upper()} Signal: {signal}\n"
                        f"Entry Price: {entry_price:.4f}\n"
                        f"Current Price: {df.iloc[-1]['close']:.4f}\n"
                        f"TP: {tp:.4f} | SL: {sl:.4f}\n"
                        f"Reason: {reason}"
                    )
                    send_telegram_alert(BOT_TOKEN, CHAT_ID, alert_msg)
                    signal_cache[symbol] = signal
            except Exception as e:
                logging.error(f"Error in streaming data for {symbol}: {e}")
                break

# === Main Async Runner ===
async def main():
    signal_cache = {}
    while True:
        hot_symbols = get_hot_perpetual_symbols()
        logging.info(f"Streaming hot pairs: {hot_symbols}")
        MAX_STREAMS = 10  # concurrency limit
        tasks = []
        for i, symbol in enumerate(hot_symbols):
            if i >= MAX_STREAMS:
                break
            tasks.append(handle_stream(symbol, signal_cache))
        await asyncio.gather(*tasks)
        logging.info(f"Sleeping for {REFRESH_INTERVAL} seconds before next scan...")
        await asyncio.sleep(REFRESH_INTERVAL)

# === Run the bot ===
if __name__ == "__main__":
    asyncio.run(main())






