"""
QQQ オプション戦略バックテスト（メモリ効率版）
- 1日ずつ処理してメモリ節約
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/mnt/c/Users/kaba/Desktop/theta_data/QQQ_data")
USD_JPY = 155


def process_day(date_str, params):
    """1日分のデータを処理"""
    option_file = DATA_DIR / f"{date_str}_option.parquet"
    stock_file = DATA_DIR / f"{date_str}_stock_quote.parquet"

    if not option_file.exists() or not stock_file.exists():
        return None

    # データ読み込み
    option = pd.read_parquet(option_file)
    stock = pd.read_parquet(stock_file)

    option['timestamp'] = pd.to_datetime(option['timestamp'])
    stock['timestamp'] = pd.to_datetime(stock['timestamp'])

    # 株価特徴量
    stock = stock.sort_values('timestamp').reset_index(drop=True)
    stock['mid'] = (stock['bid'] + stock['ask']) / 2

    window = params['window']
    hold_time = params['hold_time']
    threshold = params['threshold']
    direction = params['direction']

    stock[f'change_{window}s'] = stock['mid'].diff(window)
    stock[f'future_{hold_time}s'] = stock['mid'].shift(-hold_time) - stock['mid']

    # オプションスプレッド計算
    option['ts_key'] = option['timestamp'].dt.floor('s')
    option['spread'] = option['ask'] - option['bid']

    # 株価をオプションにマージしてATM判定
    stock['ts_key'] = stock['timestamp'].dt.floor('s')
    stock_mid = stock[['ts_key', 'mid']].drop_duplicates('ts_key')
    option = option.merge(stock_mid, on='ts_key', suffixes=('', '_stock'))
    option['atm_dist'] = (option['strike'] - option['mid']).abs()

    # ATMスプレッド取得
    idx = option.groupby(['ts_key', 'right'])['atm_dist'].idxmin()
    atm = option.loc[idx][['ts_key', 'right', 'spread']].copy()

    put_spread = atm[atm['right'] == 'PUT'][['ts_key', 'spread']].rename(columns={'spread': 'put_spread'})
    call_spread = atm[atm['right'] == 'CALL'][['ts_key', 'spread']].rename(columns={'spread': 'call_spread'})

    spreads = put_spread.merge(call_spread, on='ts_key', how='outer')

    # 株価にスプレッドをマージ
    df = stock.merge(spreads, on='ts_key', how='left')

    change_col = f'change_{window}s'
    future_col = f'future_{hold_time}s'

    df = df.dropna(subset=[change_col, future_col, 'put_spread', 'call_spread'])

    if len(df) == 0:
        return None

    # シグナル
    if direction == 'mean_revert':
        long_mask = df[change_col] < -threshold
        short_mask = df[change_col] > threshold
    else:
        long_mask = df[change_col] > threshold
        short_mask = df[change_col] < -threshold

    delta = 0.5
    trades = []

    # ロング
    long_df = df[long_mask].copy()
    if len(long_df) > 0:
        long_df['gross_pnl'] = long_df[future_col] * delta * 100
        long_df['net_pnl'] = long_df['gross_pnl'] - long_df['put_spread'] * 100
        long_df['type'] = 'long'
        long_df['spread_used'] = long_df['put_spread']
        trades.append(long_df[['timestamp', 'type', 'mid', 'spread_used', 'gross_pnl', 'net_pnl']])

    # ショート
    short_df = df[short_mask].copy()
    if len(short_df) > 0:
        short_df['gross_pnl'] = -short_df[future_col] * delta * 100
        short_df['net_pnl'] = short_df['gross_pnl'] - short_df['call_spread'] * 100
        short_df['type'] = 'short'
        short_df['spread_used'] = short_df['call_spread']
        trades.append(short_df[['timestamp', 'type', 'mid', 'spread_used', 'gross_pnl', 'net_pnl']])

    if not trades:
        return None

    result = pd.concat(trades, ignore_index=True)
    result['date'] = date_str
    result['pnl_jpy'] = result['net_pnl'] * USD_JPY

    return result


def backtest(params, name):
    """バックテスト実行"""
    files = sorted(DATA_DIR.glob("*_option.parquet"))
    dates = [f.stem.split('_')[0] for f in files]

    all_trades = []
    for date_str in tqdm(dates, desc=name):
        result = process_day(date_str, params)
        if result is not None:
            all_trades.append(result)

    if not all_trades:
        return pd.DataFrame()

    return pd.concat(all_trades, ignore_index=True)


def analyze(trades, name):
    """結果分析"""
    if len(trades) == 0:
        print(f"{name}: トレードなし")
        return None

    stats = {
        'name': name,
        'trades': len(trades),
        'win_rate': (trades['net_pnl'] > 0).mean() * 100,
        'avg_pnl': trades['pnl_jpy'].mean(),
        'total_pnl': trades['pnl_jpy'].sum(),
        'avg_spread': trades['spread_used'].mean(),
    }

    print(f"\n=== {name} ===")
    print(f"トレード数: {stats['trades']:,}")
    print(f"勝率: {stats['win_rate']:.1f}%")
    print(f"平均損益: {stats['avg_pnl']:+,.0f}円")
    print(f"累計損益: {stats['total_pnl']:+,.0f}円")
    print(f"平均スプレッド: ${stats['avg_spread']:.3f}")

    daily = trades.groupby('date')['pnl_jpy'].sum()
    print(f"プラス日: {(daily > 0).sum()}/{len(daily)}日")

    return stats


def main():
    print("=" * 60)
    print("QQQ オプション戦略バックテスト")
    print("=" * 60)

    files = list(DATA_DIR.glob("*_option.parquet"))
    print(f"\nデータ: {len(files)}日分")

    strategies = [
        {'name': '逆張り15s→30s', 'params': {'window': 15, 'threshold': 0.2, 'hold_time': 30, 'direction': 'mean_revert'}},
        {'name': '逆張り30s→60s', 'params': {'window': 30, 'threshold': 0.3, 'hold_time': 60, 'direction': 'mean_revert'}},
        {'name': '逆張り60s→120s', 'params': {'window': 60, 'threshold': 0.5, 'hold_time': 120, 'direction': 'mean_revert'}},
        {'name': '順張り30s→60s', 'params': {'window': 30, 'threshold': 0.3, 'hold_time': 60, 'direction': 'momentum'}},
    ]

    results = []
    for strat in strategies:
        trades = backtest(strat['params'], strat['name'])
        stats = analyze(trades, strat['name'])
        if stats:
            results.append(stats)

    print("\n" + "=" * 60)
    print("サマリー")
    print("=" * 60)
    for r in results:
        print(f"{r['name']}: {r['trades']:,}件, 勝率{r['win_rate']:.1f}%, 累計{r['total_pnl']:+,.0f}円")


if __name__ == "__main__":
    main()
