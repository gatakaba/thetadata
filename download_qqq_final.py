"""
QQQ データダウンロード（Parquet形式）
- option: Quote+Trade統合
- stock_quote: 1秒足
- stock_trade: ティック
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from io import StringIO
from tqdm import tqdm

BASE_URL = "http://localhost:25503"

def get_expirations(symbol):
    url = f"{BASE_URL}/v3/option/list/expirations?symbol={symbol}"
    resp = requests.get(url)
    df = pd.read_csv(StringIO(resp.text))
    return pd.to_datetime(df['expiration']).tolist()

def get_strikes(symbol, expiration):
    exp_str = expiration.strftime('%Y%m%d')
    url = f"{BASE_URL}/v3/option/list/strikes?symbol={symbol}&expiration={exp_str}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    df = pd.read_csv(StringIO(resp.text))
    return sorted(df['strike'].tolist())

def get_stock_price(symbol, date):
    date_str = date.strftime('%Y%m%d')
    url = f"{BASE_URL}/v3/stock/history/eod?symbol={symbol}&start_date={date_str}&end_date={date_str}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    df = pd.read_csv(StringIO(resp.text))
    if len(df) == 0:
        return None
    return df['close'].iloc[0]

def get_option_quote(symbol, expiration, strike, right, date):
    exp_str = expiration.strftime('%Y%m%d')
    date_str = date.strftime('%Y%m%d')
    url = f"{BASE_URL}/v3/option/history/quote?symbol={symbol}&expiration={exp_str}&strike={strike}&right={right}&date={date_str}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(StringIO(resp.text))
        return df if len(df) > 0 else None
    except:
        return None

def get_option_trade(symbol, expiration, strike, right, date):
    exp_str = expiration.strftime('%Y%m%d')
    date_str = date.strftime('%Y%m%d')
    url = f"{BASE_URL}/v3/option/history/trade?symbol={symbol}&expiration={exp_str}&strike={strike}&right={right}&date={date_str}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(StringIO(resp.text))
        return df if len(df) > 0 else None
    except:
        return None

def get_stock_quote(symbol, date):
    date_str = date.strftime('%Y%m%d')
    url = f"{BASE_URL}/v3/stock/history/quote?symbol={symbol}&date={date_str}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(StringIO(resp.text))
        if len(df) > 0:
            df['symbol'] = symbol
        return df if len(df) > 0 else None
    except:
        return None

def get_stock_trade(symbol, date):
    date_str = date.strftime('%Y%m%d')
    url = f"{BASE_URL}/v3/stock/history/trade?symbol={symbol}&date={date_str}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(StringIO(resp.text))
        if len(df) > 0:
            df['symbol'] = symbol
        return df if len(df) > 0 else None
    except:
        return None

def merge_option_quote_trade(quote_df, trade_df):
    """QuoteとTradeをタイムスタンプ（秒単位）でマージ"""
    if quote_df is None:
        return None

    quote_df = quote_df.copy()
    quote_df['ts_sec'] = pd.to_datetime(quote_df['timestamp'], format='ISO8601').dt.floor('s')

    if trade_df is not None and len(trade_df) > 0:
        trade_df = trade_df.copy()
        trade_df['ts_sec'] = pd.to_datetime(trade_df['timestamp'], format='ISO8601').dt.floor('s')
        trade_agg = trade_df.groupby(['symbol', 'expiration', 'strike', 'right', 'ts_sec']).agg({
            'price': 'mean', 'size': 'sum', 'sequence': 'count'
        }).reset_index()
        trade_agg.columns = ['symbol', 'expiration', 'strike', 'right', 'ts_sec',
                            'trade_price', 'trade_size', 'trade_count']
        merged = quote_df.merge(trade_agg[['ts_sec', 'trade_price', 'trade_size', 'trade_count']],
                                on='ts_sec', how='left')
    else:
        merged = quote_df.copy()
        merged['trade_price'] = None
        merged['trade_size'] = None
        merged['trade_count'] = 0

    merged = merged.drop(columns=['ts_sec'], errors='ignore')
    merged['trade_count'] = merged['trade_count'].fillna(0).astype(int)
    return merged

def find_atm_strikes(strikes, price, n=3):
    if not strikes or price is None:
        return []
    closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - price))
    start_idx = max(0, closest_idx - n)
    end_idx = min(len(strikes), closest_idx + n + 1)
    return strikes[start_idx:end_idx]

def download_day(args):
    symbol, date, all_expirations, output_dir = args
    date_str = date.strftime('%Y%m%d')
    option_file = f"{output_dir}/{date_str}_option.parquet"
    stock_quote_file = f"{output_dir}/{date_str}_stock_quote.parquet"
    stock_trade_file = f"{output_dir}/{date_str}_stock_trade.parquet"

    # 全ファイル存在ならスキップ
    if os.path.exists(option_file) and os.path.exists(stock_quote_file) and os.path.exists(stock_trade_file):
        return {"date": date_str, "status": "skip", "rows": 0}

    try:
        price = get_stock_price(symbol, date)
        if price is None:
            return {"date": date_str, "status": "no price", "rows": 0}

        total_rows = 0

        # === Stock Quote ===
        if not os.path.exists(stock_quote_file):
            df = get_stock_quote(symbol, date)
            if df is not None:
                df.to_parquet(stock_quote_file, index=False)
                total_rows += len(df)

        # === Stock Trade ===
        if not os.path.exists(stock_trade_file):
            df = get_stock_trade(symbol, date)
            if df is not None:
                df.to_parquet(stock_trade_file, index=False)
                total_rows += len(df)

        # === Option ===
        if not os.path.exists(option_file):
            valid_exps = [exp for exp in all_expirations
                          if timedelta(days=7) <= (exp.date() - date) <= timedelta(days=60)]
            valid_exps = sorted(valid_exps)[:3]

            if valid_exps:
                day_data = []
                for exp in valid_exps:
                    strikes = get_strikes(symbol, exp)
                    atm_strikes = find_atm_strikes(strikes, price, n=3)

                    for strike in atm_strikes:
                        for right in ['put', 'call']:
                            quote_df = get_option_quote(symbol, exp, strike, right, date)
                            trade_df = get_option_trade(symbol, exp, strike, right, date)
                            merged = merge_option_quote_trade(quote_df, trade_df)
                            if merged is not None:
                                day_data.append(merged)

                if day_data:
                    combined = pd.concat(day_data, ignore_index=True)
                    combined.to_parquet(option_file, index=False)
                    total_rows += len(combined)

        return {"date": date_str, "status": "ok", "rows": total_rows}

    except Exception as e:
        return {"date": date_str, "status": f"error: {e}", "rows": 0}

def get_trading_days(start_date, end_date):
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days

def main():
    symbol = "QQQ"
    output_dir = f"/mnt/c/Users/kaba/Desktop/theta_data/{symbol}_data"
    os.makedirs(output_dir, exist_ok=True)

    # 1年分
    end_date = datetime(2024, 12, 17).date()
    start_date = end_date - timedelta(days=365)

    print(f"=== {symbol} データダウンロード (Parquet) ===")
    print(f"期間: {start_date} ~ {end_date}")
    print(f"出力: {output_dir}")
    print(f"並列数: 5")
    print()

    print("満期日リスト取得中...")
    all_expirations = get_expirations(symbol)
    print(f"満期日: {len(all_expirations)}件")

    trading_days = get_trading_days(start_date, end_date)
    print(f"取引日: {len(trading_days)}日")
    print()

    tasks = [(symbol, date, all_expirations, output_dir) for date in trading_days]

    total_rows = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_day, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="ダウンロード"):
            result = future.result()
            total_rows += result["rows"]

    print()
    print(f"=== 完了: {total_rows:,}行 ===")

if __name__ == "__main__":
    main()
