"""
opening_range戦略のデバッグ
"""
import pandas as pd
import numpy as np
from optuna_backtest import load_day_data, prepare_features, get_date_splits
from advanced_strategies import strategy_opening_range

# テスト用パラメータ
test_params = {
    'range_minutes': 15,
    'hold_time': 120,
    'max_spread': 0.10,
}

# 1日分のデータを読み込み
splits = get_date_splits()
first_date = splits['train'][0]

print(f"デバッグ対象日: {first_date}")
print(f"パラメータ: {test_params}")
print("="*60)

stock, option = load_day_data(first_date)
if stock is None or option is None:
    print("データ読み込み失敗")
    exit(1)

print(f"株価データ: {len(stock)} 行")
print(f"オプションデータ: {len(option)} 行")
print(f"時間範囲: {stock['timestamp'].min()} ~ {stock['timestamp'].max()}")

df = prepare_features(stock, option)
print(f"特徴量追加後: {len(df)} 行")

# オープニングレンジを計算
first_time = df['timestamp'].iloc[0]
range_end = first_time + pd.Timedelta(minutes=test_params['range_minutes'])

opening_data = df[df['timestamp'] <= range_end]
print(f"\nオープニングレンジデータ: {len(opening_data)} 行")
print(f"期間: {first_time} ~ {range_end}")

if len(opening_data) > 0:
    opening_high = opening_data['mid'].max()
    opening_low = opening_data['mid'].min()
    print(f"オープニングレンジ: ${opening_high:.2f} - ${opening_low:.2f}")
    print(f"レンジ幅: ${opening_high - opening_low:.2f}")

# レンジ後のデータ
after_range = df[df['timestamp'] > range_end].copy()
print(f"\nレンジ後のデータ: {len(after_range)} 行")

if len(after_range) > 0:
    # スプレッドチェック
    valid_spread = (
        (after_range['put_spread'] <= test_params['max_spread']) &
        (after_range['call_spread'] <= test_params['max_spread'])
    ).sum()
    print(f"有効なスプレッド: {valid_spread} / {len(after_range)} 行")

    # ブレイクアウト候補
    above_high = (after_range['mid'] > opening_high).sum()
    below_low = (after_range['mid'] < opening_low).sum()
    print(f"高値ブレイク: {above_high} 回")
    print(f"安値ブレイク: {below_low} 回")

# 実際に戦略を実行
print("\n" + "="*60)
print("戦略実行")
print("="*60)
trades = strategy_opening_range(df, test_params)
print(f"生成された取引: {len(trades)} 件")

if len(trades) > 0:
    print("\n最初の5件:")
    print(trades.head())
    print(f"\nロング: {(trades['type'] == 'long').sum()} 件")
    print(f"ショート: {(trades['type'] == 'short').sum()} 件")
    print(f"\n平均PnL: {trades['net_pnl'].mean():.2f}")
    print(f"総PnL: {trades['net_pnl'].sum():.2f}")
else:
    print("取引が生成されませんでした")
    print("\n原因の可能性:")
    print("1. オープニングレンジデータが不足 (< 10行)")
    print("2. レンジ後にfuture値が計算できない")
    print("3. スプレッドが広すぎる")
    print("4. ブレイクアウトが発生していない")
