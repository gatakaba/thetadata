"""
高度戦略の高速最適化
- データを事前にメモリにキャッシュ
- 各戦略を順次最適化
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler

# 基本インポート
from optuna_backtest import (
    load_day_data, prepare_features, generate_trades,
    evaluate_trades, get_date_splits,
    CAPITAL, USD_JPY, DATA_DIR
)
from advanced_strategies import ADVANCED_STRATEGIES

optuna.logging.set_verbosity(optuna.logging.WARNING)

# グローバルデータキャッシュ
DATA_CACHE = {}

def load_all_data():
    """全データを事前にメモリにロード"""
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
    return DATA_CACHE

def fast_backtest(strategy_func, params, dates):
    """キャッシュを使った高速バックテスト"""
    all_trades = []

    for date_str in dates:
        if date_str not in DATA_CACHE:
            continue

        df = DATA_CACHE[date_str]
        trades = strategy_func(df.copy(), params)

        if len(trades) > 0:
            trades['date'] = date_str
            all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    result = pd.concat(all_trades, ignore_index=True)
    result['pnl_jpy'] = result['net_pnl'] * USD_JPY
    return result

def create_fast_objective(strategy_name, train_dates, valid_dates):
    """高速objective関数"""
    strategy_func = ADVANCED_STRATEGIES[strategy_name]

    def objective(trial):
        if strategy_name == 'spread_dynamics':
            params = {
                'spread_lookback': trial.suggest_categorical('spread_lookback', [30, 60, 120, 300]),
                'spread_threshold': trial.suggest_float('spread_threshold', 0.7, 0.95, step=0.05),
                'price_threshold': trial.suggest_float('price_threshold', 0.3, 1.5, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'window': trial.suggest_categorical('window', [5, 10, 15, 30]),
            }
        elif strategy_name == 'multi_timeframe':
            params = {
                'short_window': trial.suggest_categorical('short_window', [5, 10, 15]),
                'long_window': trial.suggest_categorical('long_window', [60, 120]),
                'threshold': trial.suggest_float('threshold', 0.2, 1.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
        elif strategy_name == 'volatility_regime':
            params = {
                'window': trial.suggest_categorical('window', [10, 15, 30, 60]),
                'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
                'vol_percentile': trial.suggest_float('vol_percentile', 0.3, 0.7, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
        elif strategy_name == 'price_level':
            params = {
                'lookback': trial.suggest_categorical('lookback', [300, 600, 1200, 1800]),
                'level_threshold': trial.suggest_float('level_threshold', 0.1, 0.3, step=0.05),
                'hold_time': trial.suggest_categorical('hold_time', [60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
        elif strategy_name == 'momentum_filter':
            params = {
                'window': trial.suggest_categorical('window', [10, 15, 30, 60]),
                'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
                'rsi_confirm': trial.suggest_int('rsi_confirm', 5, 20),
                'vol_confirm': trial.suggest_float('vol_confirm', 0.3, 0.7, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
            }
        elif strategy_name == 'opening_range':
            params = {
                'range_minutes': trial.suggest_categorical('range_minutes', [5, 10, 15, 30]),
                'hold_time': trial.suggest_categorical('hold_time', [60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Train
        trades = fast_backtest(strategy_func, params, train_dates)
        train_stats = evaluate_trades(trades)

        if train_stats['total_trades'] < 30:
            return -10
        if train_stats['max_dd'] > 50:
            return -10

        # Valid
        valid_trades = fast_backtest(strategy_func, params, valid_dates)
        valid_stats = evaluate_trades(valid_trades)

        if valid_stats['total_trades'] < 10:
            return -5

        score = (train_stats['sharpe'] + valid_stats['sharpe']) / 2

        if valid_stats['sharpe'] > 0 and train_stats['sharpe'] > 0:
            score += 0.5
        if abs(train_stats['sharpe'] - valid_stats['sharpe']) > 2:
            score -= 0.5

        return score

    return objective

def optimize_all_advanced():
    """全高度戦略を最適化"""
    # データロード
    load_all_data()

    splits = get_date_splits()
    results = {}

    strategies = list(ADVANCED_STRATEGIES.keys())

    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"最適化: {strategy_name}")
        print('='*60)

        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
            )

            objective = create_fast_objective(strategy_name, splits['train'], splits['valid'])
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            if study.best_trial is None or study.best_value <= -5:
                print(f"✗ {strategy_name}: 最適化失敗")
                results[strategy_name] = {
                    'best_params': {},
                    'train': {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100},
                    'valid': {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100},
                    'test': {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100},
                }
                continue

            best_params = study.best_params
            strategy_func = ADVANCED_STRATEGIES[strategy_name]

            # 全期間評価
            split_results = {}
            for split_name, dates in splits.items():
                trades = fast_backtest(strategy_func, best_params, dates)
                stats = evaluate_trades(trades)
                split_results[split_name] = stats

            results[strategy_name] = {
                'best_params': best_params,
                'train': split_results['train'],
                'valid': split_results['valid'],
                'test': split_results['test'],
            }

            # 結果表示
            print(f"\n最良パラメータ: {best_params}")
            print(f"Train Sharpe: {split_results['train']['sharpe']:.2f}, PnL: {split_results['train']['total_pnl']:+,.0f}円")
            print(f"Valid Sharpe: {split_results['valid']['sharpe']:.2f}, PnL: {split_results['valid']['total_pnl']:+,.0f}円")
            print(f"Test Sharpe: {split_results['test']['sharpe']:.2f}, PnL: {split_results['test']['total_pnl']:+,.0f}円")

            is_valid = (
                split_results['test']['sharpe'] > 0.5 and
                split_results['test']['total_pnl'] > 0 and
                split_results['test']['win_rate'] > 40
            )
            print(f"\n{'✓' if is_valid else '✗'} {strategy_name} {'は有効' if is_valid else 'は無効'}")

            # 個別保存
            with open(f'result_{strategy_name}.json', 'w') as f:
                json.dump(convert_numpy(results[strategy_name]), f, indent=2)

        except Exception as e:
            print(f"✗ {strategy_name}: エラー - {e}")
            results[strategy_name] = {
                'best_params': {},
                'train': {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100},
                'valid': {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100},
                'test': {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100},
            }

    # 全結果保存
    with open('advanced_results.json', 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    return results

def convert_numpy(obj):
    """numpy型をPython標準型に変換"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

def print_summary(results):
    """結果サマリー"""
    print("\n" + "="*80)
    print("高度戦略 最適化結果サマリー")
    print("="*80)

    valid_strategies = []

    for name, data in results.items():
        test = data.get('test', {})
        sharpe = test.get('sharpe', -10)
        pnl = test.get('total_pnl', 0)
        win_rate = test.get('win_rate', 0)

        is_valid = sharpe > 0.5 and pnl > 0 and win_rate > 40
        status = "✓ 有効" if is_valid else "✗ 無効"

        print(f"{name:20s}: Test Sharpe={sharpe:+7.2f}, PnL={pnl:+12,.0f}円, 勝率={win_rate:5.1f}% {status}")

        if is_valid:
            valid_strategies.append({
                'name': name,
                'sharpe': sharpe,
                'pnl': pnl,
                'params': data.get('best_params', {})
            })

    print("\n" + "="*80)
    print(f"有効な高度戦略: {len(valid_strategies)}個")
    print("="*80)

    return valid_strategies

if __name__ == "__main__":
    results = optimize_all_advanced()
    valid = print_summary(results)

    if valid:
        print("\n★ 有効な高度戦略 ★")
        for v in valid:
            print(f"  - {v['name']}: Sharpe={v['sharpe']:.2f}, PnL={v['pnl']:+,.0f}円")
            print(f"    パラメータ: {v['params']}")
