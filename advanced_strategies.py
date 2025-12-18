"""
追加の高度な戦略
- マーケットメイキング風戦略
- オプションスプレッド分析戦略
- マルチタイムフレーム戦略
"""
import pandas as pd
import numpy as np
from optuna_backtest import (
    load_day_data, prepare_features, generate_trades,
    backtest_strategy, evaluate_trades, get_date_splits,
    CAPITAL, USD_JPY
)
import optuna
from optuna.samplers import TPESampler

def strategy_spread_dynamics(df, params):
    """オプションスプレッドのダイナミクス戦略
    - スプレッドが縮小した時にエントリー（流動性改善シグナル）
    """
    spread_lookback = params['spread_lookback']
    spread_threshold = params['spread_threshold']
    price_threshold = params['price_threshold']
    hold_time = params['hold_time']
    window = params['window']

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    # スプレッドの変化を計算
    df['avg_spread'] = (df['put_spread'] + df['call_spread']) / 2
    df['spread_ma'] = df['avg_spread'].rolling(spread_lookback).mean()
    df['spread_ratio'] = df['avg_spread'] / df['spread_ma']

    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[change_col, 'spread_ratio', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    # スプレッドが平均より狭い時のみ取引
    spread_ok = df['spread_ratio'] < spread_threshold

    long_mask = (df[change_col] < -price_threshold) & spread_ok
    short_mask = (df[change_col] > price_threshold) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_multi_timeframe(df, params):
    """マルチタイムフレーム戦略
    - 短期と長期のトレンドが一致した時にエントリー
    """
    short_window = params['short_window']
    long_window = params['long_window']
    threshold = params['threshold']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.12)

    short_col = f'change_{short_window}s'
    long_col = f'change_{long_window}s'

    if short_col not in df.columns or long_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[short_col, long_col, f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    # 短期・長期ともに同方向
    long_mask = (
        (df[short_col] < -threshold) &
        (df[long_col] < 0) &  # 長期も下落
        spread_ok
    )
    short_mask = (
        (df[short_col] > threshold) &
        (df[long_col] > 0) &  # 長期も上昇
        spread_ok
    )

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_volatility_regime(df, params):
    """ボラティリティレジーム戦略
    - 低ボラ時は平均回帰、高ボラ時はモメンタム
    """
    window = params['window']
    threshold = params['threshold']
    vol_percentile = params['vol_percentile']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.12)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']

    # ボラティリティのパーセンタイル
    vol_threshold = df['volatility_60s'].quantile(vol_percentile)
    df['high_vol'] = df['volatility_60s'] > vol_threshold

    df = df.dropna(subset=[change_col, f'future_{hold_time}s', 'put_spread', 'call_spread', 'volatility_60s'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    # 低ボラ: 平均回帰
    low_vol_long = (~df['high_vol']) & (df[change_col] < -threshold) & spread_ok
    low_vol_short = (~df['high_vol']) & (df[change_col] > threshold) & spread_ok

    # 高ボラ: モメンタム
    high_vol_long = (df['high_vol']) & (df[change_col] > threshold * 1.5) & spread_ok
    high_vol_short = (df['high_vol']) & (df[change_col] < -threshold * 1.5) & spread_ok

    long_mask = low_vol_long | high_vol_long
    short_mask = low_vol_short | high_vol_short

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_price_level(df, params):
    """価格レベル戦略
    - 日中の高値/安値からの距離でエントリー
    """
    lookback = params['lookback']
    level_threshold = params['level_threshold']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.12)

    df = df.copy()

    # ローリングの高値/安値
    df['rolling_high'] = df['mid'].rolling(lookback).max()
    df['rolling_low'] = df['mid'].rolling(lookback).min()
    df['range'] = df['rolling_high'] - df['rolling_low']

    # 高値/安値からの距離（%）
    df['dist_from_high'] = (df['rolling_high'] - df['mid']) / df['range']
    df['dist_from_low'] = (df['mid'] - df['rolling_low']) / df['range']

    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=['dist_from_high', 'dist_from_low', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    # 安値近くでロング、高値近くでショート
    long_mask = (df['dist_from_low'] < level_threshold) & spread_ok
    short_mask = (df['dist_from_high'] < level_threshold) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_momentum_filter(df, params):
    """フィルター付きモメンタム
    - 複数の確認条件を満たした場合のみエントリー
    """
    window = params['window']
    threshold = params['threshold']
    rsi_confirm = params['rsi_confirm']
    vol_confirm = params['vol_confirm']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.10)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']

    # ボラティリティ確認
    vol_threshold = df['volatility_60s'].quantile(vol_confirm)

    df = df.dropna(subset=[change_col, 'rsi_14', 'volatility_60s', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)
    vol_ok = df['volatility_60s'] > vol_threshold

    # ロング: 価格上昇 + RSI上昇トレンド + ボラ確認
    long_mask = (
        (df[change_col] > threshold) &
        (df['rsi_14'] > 50 + rsi_confirm) &
        vol_ok &
        spread_ok
    )

    # ショート: 価格下落 + RSI下落トレンド + ボラ確認
    short_mask = (
        (df[change_col] < -threshold) &
        (df['rsi_14'] < 50 - rsi_confirm) &
        vol_ok &
        spread_ok
    )

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_opening_range(df, params):
    """オープニングレンジブレイクアウト
    - 最初のN分のレンジをブレイクしたらエントリー
    """
    range_minutes = params['range_minutes']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.10)

    df = df.copy()

    # 最初のN分間のデータを特定
    first_time = df['timestamp'].iloc[0]
    range_end = first_time + pd.Timedelta(minutes=range_minutes)

    opening_data = df[df['timestamp'] <= range_end]
    if len(opening_data) < 10:
        return pd.DataFrame()

    opening_high = opening_data['mid'].max()
    opening_low = opening_data['mid'].min()

    # レンジ後のデータ
    after_range = df[df['timestamp'] > range_end].copy()
    if len(after_range) == 0:
        return pd.DataFrame()

    after_range[f'future_{hold_time}s'] = after_range['mid'].shift(-hold_time) - after_range['mid']
    after_range = after_range.dropna(subset=[f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (after_range['put_spread'] <= max_spread) & (after_range['call_spread'] <= max_spread)

    # ブレイクアウト - 前の値と比較して「ブレイクした瞬間」のみ
    after_range['prev_mid'] = after_range['mid'].shift(1)

    # 上方ブレイク: 前回は範囲内 or 以下、今回は範囲より上
    long_mask = (
        (after_range['prev_mid'] <= opening_high) &
        (after_range['mid'] > opening_high) &
        spread_ok
    )

    # 下方ブレイク: 前回は範囲内 or 以上、今回は範囲より下
    short_mask = (
        (after_range['prev_mid'] >= opening_low) &
        (after_range['mid'] < opening_low) &
        spread_ok
    )

    return generate_trades(after_range, long_mask, short_mask, hold_time)


# 追加戦略の登録
ADVANCED_STRATEGIES = {
    'spread_dynamics': strategy_spread_dynamics,
    'multi_timeframe': strategy_multi_timeframe,
    'volatility_regime': strategy_volatility_regime,
    'price_level': strategy_price_level,
    'momentum_filter': strategy_momentum_filter,
    'opening_range': strategy_opening_range,
}

def create_advanced_objective(strategy_name, train_dates, valid_dates):
    """高度な戦略用のOptuna目的関数"""
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

        # Trainでバックテスト
        trades = backtest_strategy(strategy_func, params, train_dates)
        train_stats = evaluate_trades(trades)

        if train_stats['total_trades'] < 50:
            return -10
        if train_stats['max_dd'] > 30:
            return -10

        # Validでバックテスト
        valid_trades = backtest_strategy(strategy_func, params, valid_dates)
        valid_stats = evaluate_trades(valid_trades)

        if valid_stats['total_trades'] < 20:
            return -5

        score = (train_stats['sharpe'] + valid_stats['sharpe']) / 2

        if valid_stats['sharpe'] > 0 and train_stats['sharpe'] > 0:
            score += 0.5
        if abs(train_stats['sharpe'] - valid_stats['sharpe']) > 1:
            score -= 0.5

        return score

    return objective

def optimize_advanced_strategy(strategy_name, n_trials=100):
    """高度な戦略を最適化"""
    splits = get_date_splits()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f'{strategy_name}_optimization'
    )

    objective = create_advanced_objective(strategy_name, splits['train'], splits['valid'])
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <strategy_name> [n_trials]")
        print(f"Available strategies: {list(ADVANCED_STRATEGIES.keys())}")
        sys.exit(1)

    strategy_name = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if strategy_name not in ADVANCED_STRATEGIES:
        print(f"Unknown strategy: {strategy_name}")
        sys.exit(1)

    splits = get_date_splits()
    print(f"\n{'='*60}")
    print(f"最適化: {strategy_name} ({n_trials} trials)")
    print('='*60)

    study = optimize_advanced_strategy(strategy_name, n_trials=n_trials)

    if study.best_trial is None:
        print("最適化失敗")
        sys.exit(1)

    best_params = study.best_params
    strategy_func = ADVANCED_STRATEGIES[strategy_name]

    # 最終評価
    results = {}
    for split_name, dates in splits.items():
        trades = backtest_strategy(strategy_func, best_params, dates)
        stats = evaluate_trades(trades)
        results[split_name] = stats

    print(f"\n最良パラメータ: {best_params}")
    print(f"Train Sharpe: {results['train']['sharpe']:.2f}, PnL: {results['train']['total_pnl']:+,.0f}円")
    print(f"Valid Sharpe: {results['valid']['sharpe']:.2f}, PnL: {results['valid']['total_pnl']:+,.0f}円")
    print(f"Test Sharpe: {results['test']['sharpe']:.2f}, PnL: {results['test']['total_pnl']:+,.0f}円")

    # 結果保存
    output = {
        'strategy': strategy_name,
        'best_params': best_params,
        'train': results['train'],
        'valid': results['valid'],
        'test': results['test']
    }

    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(f'result_{strategy_name}.json', 'w') as f:
        json.dump(convert(output), f, indent=2, ensure_ascii=False)

    is_valid = results['test']['sharpe'] > 0.5 and results['test']['total_pnl'] > 0
    print(f"\n{'✓' if is_valid else '✗'} {strategy_name} {'は有効' if is_valid else 'は無効'}")
