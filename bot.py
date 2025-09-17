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

# === Retry Logic for WebSocket Connection ===
async def connect_with_retry(url, retries=5, delay=5):
    for attempt in range(retries):
        try:
            return await websockets.connect(url)
        except Exception as e:
            logging.warning(f"WebSocket connection failed (attempt {attempt + 1}/{retries}): {e}")
            await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
    raise ConnectionError(f"Failed to connect to WebSocket after {retries} attempts.")

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
# === Fetch Historical Candles ===
def fetch_historical_klines(symbol, interval='1m', limit=100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame([
        {
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        }
        for k in data
    ])
    return df

def fetch_higher_timeframe_klines(symbol, interval='1h', limit=50):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame([
        {
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        }
        for k in data
    ])
    return df

# === Fetch Funding Rate ===
def fetch_latest_funding_rate(symbol):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {'symbol': symbol.upper(), 'limit': 1}
    response = requests.get(url, params=params)
    data = response.json()
    if isinstance(data, list) and len(data) > 0:
        return float(data[0]['fundingRate'])
    return 0.0

# === Strategy Logic ===
def evaluate_signal(df, funding_rate, symbol):
    if len(df) < 20:
        return "NO SIGNAL", None, "Not enough data", None, None
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 1.5 * df['avg_volume']
    df['volume_drop'] = df['volume'] < 0.5 * df['avg_volume']
    N = 10
    slope = 0
    if len(df) >= N:
        x = np.arange(N)
        y = df['volume'].tail(N).values
        slope = np.polyfit(x, y, 1)[0]
    volume_trend = "Stable"
    if slope > 50:
        volume_trend = "Gradual Increase"
    elif slope < -50:
        volume_trend = "Gradual Decrease"
    df['funding_rate'] = funding_rate
    htf_df = fetch_higher_timeframe_klines(symbol)
    htf_df['atr'] = ta.volatility.average_true_range(htf_df['high'], htf_df['low'], htf_df['close'], window=14)
    htf_atr = htf_df['atr'].iloc[-1]
    row = df.iloc[-1]
    signal = "NO SIGNAL"
    reason = ""
    tp = None
    sl = None
    entry_price = row['close']
    risk_factor = 1.5
    reward_factor = 3.0
    if (
        row['ema_9'] > row['ema_21'] and
        55 < row['rsi'] < 70 and
        row['close'] > row['bb_upper'] and
        row['volume_spike'] and
        volume_trend == "Gradual Increase" and
        row['funding_rate'] < 0.005
    ):
        signal = "LONG"
        sl = entry_price - htf_atr * risk_factor
        tp = entry_price + htf_atr * reward_factor
        reason = (
            f"EMA9 ({row['ema_9']:.2f}) > EMA21 ({row['ema_21']:.2f}), "
            f"RSI: {row['rsi']:.2f}, Close > BB Upper ({row['bb_upper']:.2f}), "
            f"Volume Spike, Volume Trend: {volume_trend}, "
            f"Funding Rate: {row['funding_rate']:.5f}, ATR(1h): {htf_atr:.4f}"
        )
    elif (
        row['ema_9'] < row['ema_21'] and
        30 < row['rsi'] < 45 and
        row['close'] < row['bb_lower'] and
        row['volume_spike'] and
        volume_trend == "Gradual Decrease" and
        row['funding_rate'] > -0.005
    ):
        signal = "SHORT"
        sl = entry_price + htf_atr * risk_factor
        tp = entry_price - htf_atr * reward_factor
        reason = (
            f"EMA9 ({row['ema_9']:.2f}) < EMA21 ({row['ema_21']:.2f}), "
            f"RSI: {row['rsi']:.2f}, Close < BB Lower ({row['bb_lower']:.2f}), "
            f"Volume Spike, Volume Trend: {volume_trend}, "
            f"Funding Rate: {row['funding_rate']:.5f}, ATR(1h): {htf_atr:.4f}"
        )
    if signal in ["LONG", "SHORT"]:
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        if reward / risk < 1.5:
            signal = "NO SIGNAL"
            reason += "\nRejected due to poor risk-reward ratio."
    return signal, round(entry_price, 4), reason, round(tp, 4) if tp else None, round(sl, 4) if sl else None

# === WebSocket Handler ===
async def handle_stream(symbol, signal_cache):
    url = f"wss://fstream.binance.com/ws/{symbol}@kline_1m"

    print(f"[{symbol}] Fetching historical candles...")
    df = fetch_historical_klines(symbol)
    print(f"[{symbol}] Historical candles fetched. Connecting to WebSocket...")

    ws = await connect_with_retry(url)
    print(f"[{symbol}] Connected to WebSocket.")

    async with ws:
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                logging.warning(f"[{symbol}] Timeout while waiting for WebSocket message.")
                continue

            print(f"[{symbol}] Received WebSocket message.")
            data = json.loads(msg)
            kline = data['k']
            new_row = {
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).tail(100)
            print(f"[{symbol}] Updated dataframe with latest kline.")

            funding_rate = fetch_latest_funding_rate(symbol)
            print(f"[{symbol}] Funding rate fetched: {funding_rate}")

            signal, entry_price, reason, tp, sl = evaluate_signal(df, funding_rate, symbol)
            print(f"[{symbol}] Signal evaluated: {signal}")

            if signal in ["LONG", "SHORT"] and signal_cache.get(symbol) != signal:
                alert_msg = (
                    f"📢 {symbol.upper()} Signal: {signal}\n"
                    f"Entry Price: {entry_price:.4f}\n"
                    f"Current Price: {df.iloc[-1]['close']:.4f}\n"
                    f"TP: {tp:.4f} \nSL: {sl:.4f}\n"
                    f"Reason: {reason}"
                )
                print(f"[{symbol}] Sending Telegram alert...")
                send_telegram_alert(BOT_TOKEN, CHAT_ID, alert_msg)
                signal_cache[symbol] = signal
                print(f"[{symbol}] Signal cached.")

# === Main Async Runner with Retry Logic ===
async def run_with_retry(signal_cache, max_iterations=24):
    iteration = 0
    while iteration < max_iterations:
        hot_symbols = get_hot_perpetual_symbols()
        print(f"Streaming hot pairs: {hot_symbols}")
        MAX_STREAMS = 10
        tasks = []
        for i, symbol in enumerate(hot_symbols):
            if i >= MAX_STREAMS:
                break
            tasks.append(handle_stream(symbol, signal_cache))
        await asyncio.gather(*tasks)
        iteration += 1
        print(f"Iteration {iteration} done. Sleeping for {REFRESH_INTERVAL}s ...")
        await asyncio.sleep(REFRESH_INTERVAL)
    print("One day run complete, exiting.")

# === Run the bot ===
if __name__ == "__main__":
    signal_cache = {}
    asyncio.run(run_with_retry(signal_cache, max_iterations=24))









