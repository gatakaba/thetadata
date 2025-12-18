"""
VWAP戦略のみを高速最適化
"""
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import warnings
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')

DATA_DIR = Path("/mnt/c/Users/kaba/Desktop/theta_data/QQQ_data")
USD_JPY = 155
CAPITAL = 5_000_000

# グローバルキャッシュ
DATA_CACHE = {}

def get_date_splits():
    files = sorted(DATA_DIR.glob("*_option.parquet"))
    dates = sorted([f.stem.split('_')[0] for f in files])
    n = len(dates)
    train_end = int(n * 0.6)
    valid_end = int(n * 0.8)
    return {
        'train': dates[:train_end],
        'valid': dates[train_end:valid_end],
        'test': dates[valid_end:]
    }

def load_all_data(dates):
    """全データを事前にロード"""
    global DATA_CACHE

    for date_str in tqdm(dates, desc="Loading data"):
        if date_str in DATA_CACHE:
            continue

        option_file = DATA_DIR / f"{date_str}_option.parquet"
        stock_file = DATA_DIR / f"{date_str}_stock_quote.parquet"

        if not option_file.exists() or not stock_file.exists():
            continue

        stock = pd.read_parquet(stock_file)
        option = pd.read_parquet(option_file)

        stock['timestamp'] = pd.to_datetime(stock['timestamp'])
        option['timestamp'] = pd.to_datetime(option['timestamp'])

        stock = stock.sort_values('timestamp').reset_index(drop=True)
        stock['mid'] = (stock['bid'] + stock['ask']) / 2

        # VWAP計算
        stock['vwap'] = stock['mid'].expanding().mean()
        stock['vwap_dev'] = (stock['mid'] - stock['vwap']) / stock['vwap'] * 100

        # スプレッド計算
        option['ts_key'] = option['timestamp'].dt.floor('s')
        option['spread'] = option['ask'] - option['bid']
        stock['ts_key'] = stock['timestamp'].dt.floor('s')

        stock_mid = stock[['ts_key', 'mid']].drop_duplicates('ts_key')
        option = option.merge(stock_mid, on='ts_key', suffixes=('', '_stock'))
        option['atm_dist'] = (option['strike'] - option['mid']).abs()

        idx = option.groupby(['ts_key', 'right'])['atm_dist'].idxmin()
        atm = option.loc[idx][['ts_key', 'right', 'spread']].copy()

        put_spread = atm[atm['right'] == 'PUT'][['ts_key', 'spread']].rename(columns={'spread': 'put_spread'})
        call_spread = atm[atm['right'] == 'CALL'][['ts_key', 'spread']].rename(columns={'spread': 'call_spread'})
        spreads = put_spread.merge(call_spread, on='ts_key', how='outer')

        df = stock.merge(spreads, on='ts_key', how='left')
        DATA_CACHE[date_str] = df

def fast_backtest(params, dates):
    """高速バックテスト"""
    vwap_threshold = params.get('vwap_threshold', 0.1)
    hold_time = params.get('hold_time', 60)
    direction = params.get('direction', 'mean_revert')
    max_spread = params.get('max_spread', 0.10)

    all_pnl = []

    for date_str in dates:
        if date_str not in DATA_CACHE:
            continue

        df = DATA_CACHE[date_str].copy()

        df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
        df = df.dropna(subset=['vwap_dev', f'future_{hold_time}s', 'put_spread', 'call_spread'])

        if len(df) == 0:
            continue

        spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

        # VWAP戦略シグナル
        if direction == 'mean_revert':
            long_mask = (df['vwap_dev'] < -vwap_threshold) & spread_ok
            short_mask = (df['vwap_dev'] > vwap_threshold) & spread_ok
        else:  # momentum
            long_mask = (df['vwap_dev'] > vwap_threshold) & spread_ok
            short_mask = (df['vwap_dev'] < -vwap_threshold) & spread_ok

        delta = 0.5
        future_col = f'future_{hold_time}s'

        # PnL計算
        long_pnl = df.loc[long_mask, future_col] * delta * 100 - df.loc[long_mask, 'put_spread'] * 100
        short_pnl = -df.loc[short_mask, future_col] * delta * 100 - df.loc[short_mask, 'call_spread'] * 100

        day_pnl = long_pnl.sum() + short_pnl.sum()
        day_trades = len(long_pnl) + len(short_pnl)

        if day_trades > 0:
            all_pnl.append({
                'date': date_str,
                'pnl': day_pnl * USD_JPY,
                'trades': day_trades,
                'win': (long_pnl > 0).sum() + (short_pnl > 0).sum()
            })

    if not all_pnl:
        return {'sharpe': -10, 'total_pnl': 0, 'total_trades': 0, 'win_rate': 0, 'max_dd': 100}

    pnl_df = pd.DataFrame(all_pnl)
    daily_pnl = pnl_df['pnl']

    # メトリクス計算
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
    total_pnl = daily_pnl.sum()
    total_win = pnl_df['win'].sum()
    total_trades = pnl_df['trades'].sum()
    win_rate = total_win / total_trades * 100 if total_trades > 0 else 0

    cumsum = daily_pnl.cumsum()
    max_dd = (cumsum.cummax() - cumsum).max() / CAPITAL * 100 if len(cumsum) > 0 else 0

    return {
        'sharpe': sharpe,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'n_days': len(pnl_df)
    }

def create_objective(train_dates, valid_dates):
    """Optuna目的関数"""
    def objective(trial):
        params = {
            'vwap_threshold': trial.suggest_float('vwap_threshold', 0.05, 0.5, step=0.05),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'direction': trial.suggest_categorical('direction', ['mean_revert', 'momentum']),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }

        train_stats = fast_backtest(params, train_dates)

        if train_stats['total_trades'] < 50:
            return -10
        if train_stats['max_dd'] > 30:
            return -10

        valid_stats = fast_backtest(params, valid_dates)

        if valid_stats['total_trades'] < 20:
            return -5

        score = (train_stats['sharpe'] + valid_stats['sharpe']) / 2

        if valid_stats['sharpe'] > 0 and train_stats['sharpe'] > 0:
            score += 0.5
        if abs(train_stats['sharpe'] - valid_stats['sharpe']) > 1:
            score -= 0.5

        return score

    return objective

def main():
    splits = get_date_splits()
    all_dates = splits['train'] + splits['valid'] + splits['test']

    print(f"\n{'='*60}")
    print(f"VWAP戦略最適化")
    print(f"Train: {len(splits['train'])}日, Valid: {len(splits['valid'])}日, Test: {len(splits['test'])}日")
    print('='*60)

    print("\nデータをロード中...")
    load_all_data(all_dates)

    print("\n最適化開始...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    objective = create_objective(splits['train'], splits['valid'])
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    if study.best_trial is None:
        print("最適化失敗")
        return

    best_params = study.best_params

    # 最終評価
    train_stats = fast_backtest(best_params, splits['train'])
    valid_stats = fast_backtest(best_params, splits['valid'])
    test_stats = fast_backtest(best_params, splits['test'])

    result = {
        'strategy': 'vwap',
        'best_params': best_params,
        'best_score': study.best_value,
        'train': train_stats,
        'valid': valid_stats,
        'test': test_stats
    }

    # 結果保存
    output_file = 'result_vwap.json'
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(result), f, indent=2, ensure_ascii=False)

    # 結果表示
    print(f"\n{'='*60}")
    print(f"VWAP戦略最適化結果")
    print('='*60)
    print(f"最良パラメータ: {best_params}")
    print(f"最良スコア: {study.best_value:.2f}")
    print(f"\n--- Train ---")
    print(f"  Sharpe: {train_stats['sharpe']:.2f}")
    print(f"  PnL: {train_stats['total_pnl']:+,.0f}円")
    print(f"  勝率: {train_stats['win_rate']:.1f}%")
    print(f"  トレード数: {train_stats['total_trades']}")
    print(f"  Max DD: {train_stats['max_dd']:.1f}%")
    print(f"\n--- Valid ---")
    print(f"  Sharpe: {valid_stats['sharpe']:.2f}")
    print(f"  PnL: {valid_stats['total_pnl']:+,.0f}円")
    print(f"  勝率: {valid_stats['win_rate']:.1f}%")
    print(f"  トレード数: {valid_stats['total_trades']}")
    print(f"  Max DD: {valid_stats['max_dd']:.1f}%")
    print(f"\n--- Test ---")
    print(f"  Sharpe: {test_stats['sharpe']:.2f}")
    print(f"  PnL: {test_stats['total_pnl']:+,.0f}円")
    print(f"  勝率: {test_stats['win_rate']:.1f}%")
    print(f"  トレード数: {test_stats['total_trades']}")
    print(f"  Max DD: {test_stats['max_dd']:.1f}%")

    # 有効性判定
    is_valid = (
        test_stats['sharpe'] > 0.5 and
        test_stats['total_pnl'] > 0 and
        test_stats['win_rate'] > 40
    )

    if is_valid:
        print(f"\n✓ VWAP戦略は有効です！")
    else:
        print(f"\n✗ VWAP戦略は有効ではありません")

    print(f"\n結果を {output_file} に保存しました")

if __name__ == "__main__":
    main()
