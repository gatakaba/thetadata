"""
最適化v2: 2分割 (Train 80% / Test 20%)
- Trainのみでパラメータ探索
- Testで最終評価
- 全14戦略対応
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
import sys

sys.path.insert(0, '/mnt/c/Users/kaba/Desktop/ib')
from optuna_backtest import load_day_data, prepare_features, STRATEGIES as BASE_STRATEGIES
from advanced_strategies import ADVANCED_STRATEGIES

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("/mnt/c/Users/kaba/Desktop/theta_data/QQQ_data")
CAPITAL = 5_000_000
USD_JPY = 155

# 全戦略を統合
ALL_STRATEGIES = {**BASE_STRATEGIES, **ADVANCED_STRATEGIES}

# データキャッシュ
DATA_CACHE = {}

def load_all_data():
    global DATA_CACHE
    files = sorted(DATA_DIR.glob("*_option.parquet"))
    all_dates = sorted([f.stem.split('_')[0] for f in files])

    print(f"データ読み込み: {len(all_dates)}日分")
    for date_str in tqdm(all_dates, desc="Loading"):
        if date_str not in DATA_CACHE:
            stock, option = load_day_data(date_str)
            if stock is not None:
                df = prepare_features(stock, option)
                DATA_CACHE[date_str] = df

    print(f"キャッシュ完了: {len(DATA_CACHE)}日分")
    return list(DATA_CACHE.keys())

def get_splits_v2():
    """2分割: Train 80% / Test 20%"""
    files = sorted(DATA_DIR.glob("*_option.parquet"))
    all_dates = sorted([f.stem.split('_')[0] for f in files])

    n = len(all_dates)
    train_end = int(n * 0.8)

    return {
        'train': all_dates[:train_end],
        'test': all_dates[train_end:],
    }

def fast_backtest(strategy_func, params, dates):
    all_trades = []
    for date_str in dates:
        if date_str not in DATA_CACHE:
            continue
        df = DATA_CACHE[date_str]
        try:
            trades = strategy_func(df.copy(), params)
            if trades is not None and len(trades) > 0:
                trades['date'] = date_str
                all_trades.append(trades)
        except Exception:
            continue

    if not all_trades:
        return pd.DataFrame()

    result = pd.concat(all_trades, ignore_index=True)
    result['pnl_jpy'] = result['net_pnl'] * USD_JPY
    return result

def evaluate_trades(trades):
    if trades is None or len(trades) == 0:
        return {'sharpe': -10, 'total_pnl': 0, 'total_trades': 0, 'win_rate': 0, 'max_dd': 100}

    trades = trades.copy()
    trades['pnl_jpy'] = trades['net_pnl'] * USD_JPY

    total_pnl = trades['pnl_jpy'].sum()
    total_trades = len(trades)
    win_rate = (trades['net_pnl'] > 0).mean() * 100

    daily_pnl = trades.groupby('date')['pnl_jpy'].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    else:
        sharpe = 0

    cumsum = trades['pnl_jpy'].cumsum()
    max_dd = (cumsum.cummax() - cumsum).max() / CAPITAL * 100 if len(cumsum) > 0 else 0

    return {
        'sharpe': sharpe,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'n_days': len(daily_pnl),
    }

def get_params(strategy_name, trial):
    """戦略ごとのパラメータを定義"""

    # 基本戦略
    if strategy_name in ['momentum', 'mean_revert']:
        return {
            'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60, 120]),
            'threshold': trial.suggest_float('threshold', 0.3, 2.0, step=0.1),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'volatility_breakout':
        return {
            'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60]),
            'vol_mult': trial.suggest_float('vol_mult', 1.0, 3.0, step=0.2),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'rsi':
        return {
            'oversold': trial.suggest_int('oversold', 20, 40),
            'overbought': trial.suggest_int('overbought', 60, 80),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'time_filter':
        return {
            'window': trial.suggest_categorical('window', [5, 10, 15, 30]),
            'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
            'start_hour': trial.suggest_int('start_hour', 9, 11),
            'end_hour': trial.suggest_int('end_hour', 14, 16),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'vwap':
        return {
            'vwap_threshold': trial.suggest_float('vwap_threshold', 0.1, 0.5, step=0.05),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'direction': trial.suggest_categorical('direction', ['mean_revert', 'momentum']),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'combined':
        return {
            'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60]),
            'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
            'rsi_filter': trial.suggest_int('rsi_filter', 5, 25),
            'vol_filter': trial.suggest_float('vol_filter', 0.3, 0.7, step=0.1),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'high_threshold':
        return {
            'window': trial.suggest_categorical('window', [5, 10, 15, 30]),
            'threshold': trial.suggest_float('threshold', 1.0, 3.0, step=0.2),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }

    # 高度な戦略
    elif strategy_name == 'spread_dynamics':
        return {
            'spread_lookback': trial.suggest_categorical('spread_lookback', [30, 60, 120, 300]),
            'spread_threshold': trial.suggest_float('spread_threshold', 0.7, 0.95, step=0.05),
            'price_threshold': trial.suggest_float('price_threshold', 0.3, 1.5, step=0.1),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'window': trial.suggest_categorical('window', [5, 10, 15, 30]),
        }
    elif strategy_name == 'multi_timeframe':
        return {
            'short_window': trial.suggest_categorical('short_window', [5, 10, 15]),
            'long_window': trial.suggest_categorical('long_window', [60, 120]),
            'threshold': trial.suggest_float('threshold', 0.2, 1.0, step=0.1),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'volatility_regime':
        return {
            'window': trial.suggest_categorical('window', [10, 15, 30, 60]),
            'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
            'vol_percentile': trial.suggest_float('vol_percentile', 0.3, 0.7, step=0.1),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'price_level':
        return {
            'lookback': trial.suggest_categorical('lookback', [300, 600, 1200, 1800]),
            'level_threshold': trial.suggest_float('level_threshold', 0.1, 0.3, step=0.05),
            'hold_time': trial.suggest_categorical('hold_time', [60, 120, 300, 600]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
        }
    elif strategy_name == 'momentum_filter':
        return {
            'window': trial.suggest_categorical('window', [10, 15, 30, 60]),
            'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
            'rsi_confirm': trial.suggest_int('rsi_confirm', 5, 20),
            'vol_confirm': trial.suggest_float('vol_confirm', 0.3, 0.7, step=0.1),
            'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
        }
    elif strategy_name == 'opening_range':
        return {
            'range_minutes': trial.suggest_categorical('range_minutes', [5, 10, 15, 30]),
            'hold_time': trial.suggest_categorical('hold_time', [60, 120, 300, 600]),
            'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def create_objective(strategy_name, train_dates):
    """Trainのみでスコアリング"""
    strategy_func = ALL_STRATEGIES[strategy_name]

    def objective(trial):
        params = get_params(strategy_name, trial)

        # Trainのみでバックテスト
        trades = fast_backtest(strategy_func, params, train_dates)
        stats = evaluate_trades(trades)

        # 制約
        if stats['total_trades'] < 50:
            return -10
        if stats['max_dd'] > 30:
            return -10

        # スコア = Train Sharpeのみ
        return stats['sharpe']

    return objective

def optimize_strategy(strategy_name, train_dates, n_trials=100):
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
    )

    objective = create_objective(strategy_name, train_dates)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study

def main():
    # データ読み込み
    load_all_data()
    splits = get_splits_v2()

    print(f"\n=== 2分割 (全{len(ALL_STRATEGIES)}戦略) ===")
    print(f"Train: {len(splits['train'])}日 ({splits['train'][0]} ~ {splits['train'][-1]})")
    print(f"Test:  {len(splits['test'])}日 ({splits['test'][0]} ~ {splits['test'][-1]})")

    results = {}

    for strategy_name in ALL_STRATEGIES.keys():
        print(f"\n{'='*60}")
        print(f"最適化: {strategy_name}")
        print('='*60)

        try:
            study = optimize_strategy(strategy_name, splits['train'], n_trials=100)
        except Exception as e:
            print(f"✗ {strategy_name}: エラー ({e})")
            results[strategy_name] = {'status': 'error', 'error': str(e)}
            continue

        if study.best_value <= -5:
            print(f"✗ {strategy_name}: 有効なパラメータなし (best_score={study.best_value:.2f})")
            results[strategy_name] = {'status': 'failed', 'best_score': study.best_value}
            continue

        best_params = study.best_params
        strategy_func = ALL_STRATEGIES[strategy_name]

        # Train評価
        train_trades = fast_backtest(strategy_func, best_params, splits['train'])
        train_stats = evaluate_trades(train_trades)

        # Test評価
        test_trades = fast_backtest(strategy_func, best_params, splits['test'])
        test_stats = evaluate_trades(test_trades)

        results[strategy_name] = {
            'status': 'success',
            'best_score': study.best_value,
            'best_params': best_params,
            'train': train_stats,
            'test': test_stats,
        }

        print(f"\n最良パラメータ: {best_params}")
        print(f"Train: Sharpe={train_stats['sharpe']:.2f}, PnL={train_stats['total_pnl']:+,.0f}円, 勝率={train_stats['win_rate']:.1f}%")
        print(f"Test:  Sharpe={test_stats['sharpe']:.2f}, PnL={test_stats['total_pnl']:+,.0f}円, 勝率={test_stats['win_rate']:.1f}%")

        # 有効性判定
        is_valid = (
            train_stats['sharpe'] > 0.5 and
            test_stats['sharpe'] > 0 and
            test_stats['total_pnl'] > 0
        )
        print(f"\n{'✓ 有効' if is_valid else '✗ 無効'}")

    # 結果保存
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open('results_v2.json', 'w') as f:
        json.dump(convert(results), f, indent=2, default=str)

    print("\n" + "="*60)
    print("最終結果サマリー")
    print("="*60)

    valid_count = 0
    for name, data in results.items():
        if data['status'] in ['failed', 'error']:
            print(f"{name}: 失敗")
        else:
            t = data['test']
            tr = data['train']
            is_valid = tr['sharpe'] > 0.5 and t['sharpe'] > 0 and t['total_pnl'] > 0
            status = "✓" if is_valid else "✗"
            if is_valid:
                valid_count += 1
            print(f"{name}: Train Sharpe={tr['sharpe']:.2f}, Test Sharpe={t['sharpe']:.2f}, Test PnL={t['total_pnl']:+,.0f}円 {status}")

    print(f"\n有効な戦略: {valid_count}/{len(ALL_STRATEGIES)}")

if __name__ == "__main__":
    main()
