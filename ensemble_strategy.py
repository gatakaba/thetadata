"""
アンサンブル戦略
- 複数の有効な戦略を組み合わせる
- ウォークフォワード検証
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from optuna_backtest import (
    load_day_data, prepare_features, STRATEGIES,
    backtest_strategy, evaluate_trades, get_date_splits,
    CAPITAL, USD_JPY, DATA_DIR
)
from advanced_strategies import ADVANCED_STRATEGIES
import optuna
from optuna.samplers import TPESampler

ALL_STRATEGIES = {**STRATEGIES, **ADVANCED_STRATEGIES}

def load_best_params():
    """最適化済みパラメータを読み込み"""
    params = {}
    for f in Path('.').glob('result_*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                name = f.stem.replace('result_', '')
                if 'best_params' in data:
                    params[name] = data['best_params']
        except:
            pass
    return params

def ensemble_vote(signals_list):
    """複数の戦略のシグナルを投票で統合"""
    if not signals_list:
        return pd.Series(dtype=float)

    # 全シグナルを結合
    combined = pd.concat(signals_list, axis=1)

    # 過半数が同意した場合のみシグナル
    vote_threshold = len(signals_list) / 2
    long_votes = (combined == 1).sum(axis=1)
    short_votes = (combined == -1).sum(axis=1)

    result = pd.Series(0, index=combined.index)
    result[long_votes > vote_threshold] = 1
    result[short_votes > vote_threshold] = -1

    return result

def ensemble_weighted(signals_list, weights):
    """重み付きアンサンブル"""
    if not signals_list:
        return pd.Series(dtype=float)

    combined = pd.concat(signals_list, axis=1)
    weighted = combined.multiply(weights, axis=1)

    score = weighted.sum(axis=1)

    result = pd.Series(0, index=combined.index)
    result[score > 0.5] = 1
    result[score < -0.5] = -1

    return result

def get_strategy_signals(df, strategy_name, params):
    """戦略からシグナルを取得"""
    if strategy_name not in ALL_STRATEGIES:
        return None

    strategy_func = ALL_STRATEGIES[strategy_name]

    # パラメータから必要な情報を抽出
    window = params.get('window', 15)
    threshold = params.get('threshold', 0.5)
    hold_time = params.get('hold_time', 60)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return None

    # シグナル生成（戦略に依存）
    signals = pd.Series(0, index=df.index)

    if strategy_name in ['mean_revert', 'high_threshold', 'time_filter', 'combined']:
        # 逆張り系
        signals[df[change_col] < -threshold] = 1
        signals[df[change_col] > threshold] = -1
    elif strategy_name in ['momentum', 'momentum_filter', 'multi_timeframe']:
        # 順張り系
        signals[df[change_col] > threshold] = 1
        signals[df[change_col] < -threshold] = -1
    elif strategy_name == 'rsi':
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        if 'rsi_14' in df.columns:
            signals[df['rsi_14'] < oversold] = 1
            signals[df['rsi_14'] > overbought] = -1
    elif strategy_name == 'vwap':
        vwap_threshold = params.get('vwap_threshold', 0.1)
        direction = params.get('direction', 'mean_revert')
        if 'vwap_dev' in df.columns:
            if direction == 'mean_revert':
                signals[df['vwap_dev'] < -vwap_threshold] = 1
                signals[df['vwap_dev'] > vwap_threshold] = -1
            else:
                signals[df['vwap_dev'] > vwap_threshold] = 1
                signals[df['vwap_dev'] < -vwap_threshold] = -1

    return signals

def backtest_ensemble(strategy_names, all_params, dates, method='vote', weights=None):
    """アンサンブル戦略のバックテスト"""
    all_trades = []

    for date_str in dates:
        stock, option = load_day_data(date_str)
        if stock is None:
            continue

        df = prepare_features(stock, option)

        # 各戦略のシグナルを取得
        signals_list = []
        for name in strategy_names:
            if name in all_params:
                sig = get_strategy_signals(df, name, all_params[name])
                if sig is not None:
                    signals_list.append(sig)

        if not signals_list:
            continue

        # シグナル統合
        if method == 'vote':
            final_signal = ensemble_vote(signals_list)
        elif method == 'weighted':
            final_signal = ensemble_weighted(signals_list, weights or [1]*len(signals_list))
        else:
            continue

        # 最初の戦略のhold_timeを使用
        hold_time = all_params[strategy_names[0]].get('hold_time', 60)
        max_spread = 0.10

        df = df.copy()
        df['signal'] = final_signal
        df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
        df = df.dropna(subset=[f'future_{hold_time}s', 'put_spread', 'call_spread'])

        spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

        long_mask = (df['signal'] == 1) & spread_ok
        short_mask = (df['signal'] == -1) & spread_ok

        # トレード生成
        future_col = f'future_{hold_time}s'
        delta = 0.5
        trades = []

        long_df = df[long_mask].copy()
        if len(long_df) > 0:
            long_df['gross_pnl'] = long_df[future_col] * delta * 100
            long_df['net_pnl'] = long_df['gross_pnl'] - long_df['put_spread'] * 100
            long_df['type'] = 'long'
            long_df['spread_used'] = long_df['put_spread']
            trades.append(long_df[['timestamp', 'type', 'mid', 'spread_used', 'gross_pnl', 'net_pnl']])

        short_df = df[short_mask].copy()
        if len(short_df) > 0:
            short_df['gross_pnl'] = -short_df[future_col] * delta * 100
            short_df['net_pnl'] = short_df['gross_pnl'] - short_df['call_spread'] * 100
            short_df['type'] = 'short'
            short_df['spread_used'] = short_df['call_spread']
            trades.append(short_df[['timestamp', 'type', 'mid', 'spread_used', 'gross_pnl', 'net_pnl']])

        if trades:
            day_trades = pd.concat(trades, ignore_index=True)
            day_trades['date'] = date_str
            all_trades.append(day_trades)

    if not all_trades:
        return pd.DataFrame()

    result = pd.concat(all_trades, ignore_index=True)
    result['pnl_jpy'] = result['net_pnl'] * USD_JPY
    return result

def walk_forward_validation(strategy_func, params, n_folds=5):
    """ウォークフォワード検証"""
    files = sorted(DATA_DIR.glob("*_option.parquet"))
    all_dates = sorted([f.stem.split('_')[0] for f in files])

    fold_size = len(all_dates) // n_folds
    results = []

    for i in range(n_folds - 1):
        # 学習期間
        train_end = (i + 1) * fold_size
        train_dates = all_dates[:train_end]

        # テスト期間
        test_start = train_end
        test_end = min(test_start + fold_size, len(all_dates))
        test_dates = all_dates[test_start:test_end]

        # テスト
        trades = backtest_strategy(strategy_func, params, test_dates)
        stats = evaluate_trades(trades)
        stats['fold'] = i + 1
        stats['train_size'] = len(train_dates)
        stats['test_size'] = len(test_dates)
        results.append(stats)

    return results

def find_valid_strategies():
    """有効な戦略を見つける"""
    valid = []

    for f in Path('.').glob('result_*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                name = f.stem.replace('result_', '')
                test = data.get('test', {})

                # 異なるキー名に対応
                sharpe = test.get('sharpe', 0)
                pnl = test.get('total_pnl', test.get('pnl', 0))
                win_rate = test.get('win_rate', 0)

                # 有効性判定: Sharpe > 0.5, PnL > 0, 勝率 > 40%
                if sharpe > 0.5 and pnl > 0 and win_rate > 40:
                    valid.append({
                        'name': name,
                        'sharpe': sharpe,
                        'pnl': pnl,
                        'params': data.get('best_params', {})
                    })
        except:
            pass

    return sorted(valid, key=lambda x: x['sharpe'], reverse=True)

def optimize_ensemble():
    """有効な戦略のアンサンブルを最適化"""
    valid_strategies = find_valid_strategies()

    if len(valid_strategies) < 2:
        print("有効な戦略が2つ未満です")
        return None

    splits = get_date_splits()
    all_params = load_best_params()

    # トップ3戦略でアンサンブル
    top_strategies = [s['name'] for s in valid_strategies[:3]]
    print(f"アンサンブル対象: {top_strategies}")

    # 各手法でテスト
    methods = ['vote']  # weightedは重みの最適化が必要

    best_result = None
    best_sharpe = -10

    for method in methods:
        trades = backtest_ensemble(
            top_strategies,
            all_params,
            splits['test'],
            method=method
        )
        stats = evaluate_trades(trades)

        print(f"\n{method}法:")
        print(f"  Sharpe: {stats['sharpe']:.2f}")
        print(f"  PnL: {stats['total_pnl']:+,.0f}円")
        print(f"  勝率: {stats['win_rate']:.1f}%")

        if stats['sharpe'] > best_sharpe:
            best_sharpe = stats['sharpe']
            best_result = {
                'method': method,
                'strategies': top_strategies,
                'stats': stats
            }

    return best_result

if __name__ == "__main__":
    print("="*60)
    print("アンサンブル戦略の検証")
    print("="*60)

    # 有効な戦略を確認
    valid = find_valid_strategies()
    print(f"\n有効な戦略: {len(valid)}個")

    for v in valid:
        print(f"  - {v['name']}: Sharpe={v['sharpe']:.2f}, PnL={v['pnl']:+,.0f}円")

    if len(valid) >= 2:
        print("\nアンサンブル最適化を実行...")
        result = optimize_ensemble()

        if result:
            print("\n" + "="*60)
            print("★ アンサンブル結果 ★")
            print("="*60)
            print(f"手法: {result['method']}")
            print(f"使用戦略: {result['strategies']}")
            print(f"Test Sharpe: {result['stats']['sharpe']:.2f}")
            print(f"Test PnL: {result['stats']['total_pnl']:+,.0f}円")
    else:
        print("\n有効な戦略が不足しています")
        print("個別戦略の最適化を先に実行してください")
