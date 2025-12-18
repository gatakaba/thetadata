"""
QQQ オプション戦略 Optuna最適化フレームワーク
- Train/Valid/Test分割
- 複数戦略対応
- 並列最適化
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import optuna
from optuna.samplers import TPESampler
import warnings
import json
warnings.filterwarnings('ignore')

DATA_DIR = Path("/mnt/c/Users/kaba/Desktop/theta_data/QQQ_data")
USD_JPY = 155
CAPITAL = 5_000_000  # 500万円

# データ分割
def get_date_splits():
    """Train/Valid/Test分割を取得"""
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

def load_day_data(date_str):
    """1日分のデータを読み込み"""
    option_file = DATA_DIR / f"{date_str}_option.parquet"
    stock_file = DATA_DIR / f"{date_str}_stock_quote.parquet"

    if not option_file.exists() or not stock_file.exists():
        return None, None

    option = pd.read_parquet(option_file)
    stock = pd.read_parquet(stock_file)

    option['timestamp'] = pd.to_datetime(option['timestamp'])
    stock['timestamp'] = pd.to_datetime(stock['timestamp'])

    return stock, option

def prepare_features(stock, option, lookback=60):
    """特徴量を計算"""
    stock = stock.sort_values('timestamp').reset_index(drop=True)
    stock['mid'] = (stock['bid'] + stock['ask']) / 2

    # 価格変化
    for w in [5, 10, 15, 30, 60, 120]:
        stock[f'change_{w}s'] = stock['mid'].diff(w)
        stock[f'ret_{w}s'] = stock['mid'].pct_change(w) * 100

    # ボラティリティ
    stock['volatility_60s'] = stock['mid'].rolling(60).std()
    stock['volatility_300s'] = stock['mid'].rolling(300).std()

    # RSI
    delta = stock['mid'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    stock['rsi_14'] = 100 - (100 / (1 + rs))

    # VWAP (簡易版 - 出来高がないので時間加重)
    stock['vwap'] = stock['mid'].expanding().mean()
    stock['vwap_dev'] = (stock['mid'] - stock['vwap']) / stock['vwap'] * 100

    # 時間帯
    stock['hour'] = stock['timestamp'].dt.hour
    stock['minute'] = stock['timestamp'].dt.minute

    # オプションスプレッド計算
    option['ts_key'] = option['timestamp'].dt.floor('s')
    option['spread'] = option['ask'] - option['bid']

    stock['ts_key'] = stock['timestamp'].dt.floor('s')
    stock_mid = stock[['ts_key', 'mid']].drop_duplicates('ts_key')
    option = option.merge(stock_mid, on='ts_key', suffixes=('', '_stock'))
    option['atm_dist'] = (option['strike'] - option['mid']).abs()

    # ATMスプレッド
    idx = option.groupby(['ts_key', 'right'])['atm_dist'].idxmin()
    atm = option.loc[idx][['ts_key', 'right', 'spread']].copy()

    put_spread = atm[atm['right'] == 'PUT'][['ts_key', 'spread']].rename(columns={'spread': 'put_spread'})
    call_spread = atm[atm['right'] == 'CALL'][['ts_key', 'spread']].rename(columns={'spread': 'call_spread'})

    spreads = put_spread.merge(call_spread, on='ts_key', how='outer')
    df = stock.merge(spreads, on='ts_key', how='left')

    return df

# ======== 戦略定義 ========

def strategy_mean_revert(df, params):
    """平均回帰戦略"""
    window = params['window']
    threshold = params['threshold']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.15)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[change_col, f'future_{hold_time}s', 'put_spread', 'call_spread'])

    # スプレッドフィルター
    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    long_mask = (df[change_col] < -threshold) & spread_ok
    short_mask = (df[change_col] > threshold) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_momentum(df, params):
    """モメンタム戦略"""
    window = params['window']
    threshold = params['threshold']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.15)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[change_col, f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    # モメンタム: トレンド方向にエントリー
    long_mask = (df[change_col] > threshold) & spread_ok
    short_mask = (df[change_col] < -threshold) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_volatility_breakout(df, params):
    """ボラティリティブレイクアウト戦略"""
    lookback = params['lookback']
    mult = params['mult']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.15)

    df = df.copy()
    df['rolling_std'] = df['mid'].rolling(lookback).std()
    df['upper'] = df['mid'].rolling(lookback).mean() + mult * df['rolling_std']
    df['lower'] = df['mid'].rolling(lookback).mean() - mult * df['rolling_std']

    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=['upper', 'lower', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    long_mask = (df['mid'] > df['upper']) & spread_ok
    short_mask = (df['mid'] < df['lower']) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_rsi(df, params):
    """RSI逆張り戦略"""
    oversold = params['oversold']
    overbought = params['overbought']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.15)

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=['rsi_14', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    long_mask = (df['rsi_14'] < oversold) & spread_ok
    short_mask = (df['rsi_14'] > overbought) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_time_filter(df, params):
    """時間帯フィルター付き平均回帰"""
    window = params['window']
    threshold = params['threshold']
    hold_time = params['hold_time']
    start_hour = params['start_hour']
    end_hour = params['end_hour']
    max_spread = params.get('max_spread', 0.15)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[change_col, f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)
    time_ok = (df['hour'] >= start_hour) & (df['hour'] < end_hour)

    long_mask = (df[change_col] < -threshold) & spread_ok & time_ok
    short_mask = (df[change_col] > threshold) & spread_ok & time_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_vwap(df, params):
    """VWAP乖離戦略"""
    threshold = params['vwap_threshold']
    hold_time = params['hold_time']
    direction = params['direction']  # 'mean_revert' or 'momentum'
    max_spread = params.get('max_spread', 0.15)

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=['vwap_dev', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    if direction == 'mean_revert':
        long_mask = (df['vwap_dev'] < -threshold) & spread_ok
        short_mask = (df['vwap_dev'] > threshold) & spread_ok
    else:
        long_mask = (df['vwap_dev'] > threshold) & spread_ok
        short_mask = (df['vwap_dev'] < -threshold) & spread_ok

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_combined(df, params):
    """複合シグナル戦略"""
    window = params['window']
    threshold = params['threshold']
    rsi_filter = params['rsi_filter']
    vol_filter = params['vol_filter']
    hold_time = params['hold_time']
    max_spread = params.get('max_spread', 0.15)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[change_col, 'rsi_14', 'volatility_60s', f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)
    vol_ok = df['volatility_60s'] > df['volatility_60s'].quantile(vol_filter)

    # ロング: 価格下落 + RSI低い + ボラ高い
    long_mask = (
        (df[change_col] < -threshold) &
        (df['rsi_14'] < 50 - rsi_filter) &
        vol_ok &
        spread_ok
    )

    # ショート: 価格上昇 + RSI高い + ボラ高い
    short_mask = (
        (df[change_col] > threshold) &
        (df['rsi_14'] > 50 + rsi_filter) &
        vol_ok &
        spread_ok
    )

    return generate_trades(df, long_mask, short_mask, hold_time)

def strategy_high_threshold(df, params):
    """高閾値平均回帰（トレード数削減）"""
    window = params['window']
    threshold = params['threshold']  # 高い閾値
    hold_time = params['hold_time']
    cooldown = params['cooldown']  # 連続エントリー制限（秒）
    max_spread = params.get('max_spread', 0.10)

    change_col = f'change_{window}s'
    if change_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
    df = df.dropna(subset=[change_col, f'future_{hold_time}s', 'put_spread', 'call_spread'])

    spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

    long_signal = (df[change_col] < -threshold) & spread_ok
    short_signal = (df[change_col] > threshold) & spread_ok

    # クールダウン適用
    long_mask = apply_cooldown(long_signal, cooldown)
    short_mask = apply_cooldown(short_signal, cooldown)

    return generate_trades(df, long_mask, short_mask, hold_time)

def apply_cooldown(signal, cooldown):
    """連続シグナルにクールダウンを適用"""
    result = signal.copy()
    last_signal_idx = -cooldown - 1

    for i in range(len(signal)):
        if signal.iloc[i]:
            if i - last_signal_idx > cooldown:
                result.iloc[i] = True
                last_signal_idx = i
            else:
                result.iloc[i] = False

    return result

def generate_trades(df, long_mask, short_mask, hold_time):
    """トレード結果を生成"""
    future_col = f'future_{hold_time}s'
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
        return pd.DataFrame()

    return pd.concat(trades, ignore_index=True)

# ======== バックテスト実行 ========

def backtest_strategy(strategy_func, params, dates):
    """指定した戦略と日付でバックテスト"""
    all_trades = []

    for date_str in dates:
        stock, option = load_day_data(date_str)
        if stock is None:
            continue

        df = prepare_features(stock, option)
        trades = strategy_func(df, params)

        if len(trades) > 0:
            trades['date'] = date_str
            all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    result = pd.concat(all_trades, ignore_index=True)
    result['pnl_jpy'] = result['net_pnl'] * USD_JPY
    return result

def evaluate_trades(trades, capital=CAPITAL):
    """トレード結果を評価"""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'sharpe': -10,
            'win_rate': 0,
            'total_pnl': 0,
            'max_dd': 1,
            'avg_daily_trades': 0,
            'profit_factor': 0
        }

    daily_pnl = trades.groupby('date')['pnl_jpy'].sum()
    n_days = len(daily_pnl)

    # シャープレシオ（年率換算）
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # 最大ドローダウン
    cumsum = daily_pnl.cumsum()
    running_max = cumsum.cummax()
    dd = (running_max - cumsum) / capital
    max_dd = dd.max() if len(dd) > 0 else 0

    # プロフィットファクター
    gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'total_trades': len(trades),
        'sharpe': sharpe,
        'win_rate': (trades['net_pnl'] > 0).mean() * 100,
        'total_pnl': trades['pnl_jpy'].sum(),
        'max_dd': max_dd * 100,
        'avg_daily_trades': len(trades) / n_days if n_days > 0 else 0,
        'profit_factor': profit_factor,
        'n_days': n_days,
        'plus_days': (daily_pnl > 0).sum(),
        'avg_spread': trades['spread_used'].mean()
    }

# ======== Optuna最適化 ========

STRATEGIES = {
    'mean_revert': strategy_mean_revert,
    'momentum': strategy_momentum,
    'volatility_breakout': strategy_volatility_breakout,
    'rsi': strategy_rsi,
    'time_filter': strategy_time_filter,
    'vwap': strategy_vwap,
    'combined': strategy_combined,
    'high_threshold': strategy_high_threshold,
}

def create_objective(strategy_name, train_dates, valid_dates):
    """Optuna用の目的関数を作成"""
    strategy_func = STRATEGIES[strategy_name]

    def objective(trial):
        # パラメータサンプリング
        if strategy_name == 'mean_revert':
            params = {
                'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60, 120]),
                'threshold': trial.suggest_float('threshold', 0.3, 2.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'momentum':
            params = {
                'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60, 120]),
                'threshold': trial.suggest_float('threshold', 0.3, 2.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'volatility_breakout':
            params = {
                'lookback': trial.suggest_categorical('lookback', [30, 60, 120, 300, 600]),
                'mult': trial.suggest_float('mult', 1.5, 3.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'rsi':
            params = {
                'oversold': trial.suggest_int('oversold', 20, 40),
                'overbought': trial.suggest_int('overbought', 60, 80),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'time_filter':
            params = {
                'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60]),
                'threshold': trial.suggest_float('threshold', 0.3, 2.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'start_hour': trial.suggest_int('start_hour', 9, 12),
                'end_hour': trial.suggest_int('end_hour', 13, 16),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'vwap':
            params = {
                'vwap_threshold': trial.suggest_float('vwap_threshold', 0.05, 0.5, step=0.05),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300, 600]),
                'direction': trial.suggest_categorical('direction', ['mean_revert', 'momentum']),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'combined':
            params = {
                'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60]),
                'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
                'rsi_filter': trial.suggest_int('rsi_filter', 5, 25),
                'vol_filter': trial.suggest_float('vol_filter', 0.3, 0.7, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.15, step=0.01),
            }
        elif strategy_name == 'high_threshold':
            params = {
                'window': trial.suggest_categorical('window', [10, 15, 30, 60]),
                'threshold': trial.suggest_float('threshold', 0.8, 3.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [60, 120, 300, 600]),
                'cooldown': trial.suggest_categorical('cooldown', [60, 120, 300, 600]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Trainでバックテスト
        trades = backtest_strategy(strategy_func, params, train_dates)
        train_stats = evaluate_trades(trades)

        # 制約チェック
        if train_stats['total_trades'] < 100:
            return -10  # トレード数が少なすぎる
        if train_stats['max_dd'] > 30:
            return -10  # DDが大きすぎる
        if train_stats['avg_daily_trades'] > 200:
            return -10  # トレード数が多すぎる

        # Validでバックテスト
        valid_trades = backtest_strategy(strategy_func, params, valid_dates)
        valid_stats = evaluate_trades(valid_trades)

        # Train/Validの整合性チェック
        if valid_stats['total_trades'] < 30:
            return -5

        # 目的関数: Sharpeの平均（オーバーフィット防止）
        score = (train_stats['sharpe'] + valid_stats['sharpe']) / 2

        # ボーナス/ペナルティ
        if valid_stats['sharpe'] > 0 and train_stats['sharpe'] > 0:
            score += 0.5  # 両方プラスならボーナス
        if abs(train_stats['sharpe'] - valid_stats['sharpe']) > 1:
            score -= 0.5  # 乖離が大きいとペナルティ

        return score

    return objective

def optimize_strategy(strategy_name, n_trials=100):
    """指定した戦略を最適化"""
    splits = get_date_splits()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f'{strategy_name}_optimization'
    )

    objective = create_objective(strategy_name, splits['train'], splits['valid'])
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study

def final_evaluation(strategy_name, params):
    """最終評価（Test期間）"""
    splits = get_date_splits()
    strategy_func = STRATEGIES[strategy_name]

    results = {}
    for split_name, dates in splits.items():
        trades = backtest_strategy(strategy_func, params, dates)
        stats = evaluate_trades(trades)
        stats['split'] = split_name
        results[split_name] = stats

    return results

def run_all_optimizations(n_trials=50):
    """全戦略を最適化"""
    results = {}

    for strategy_name in STRATEGIES.keys():
        print(f"\n{'='*60}")
        print(f"最適化中: {strategy_name}")
        print('='*60)

        study = optimize_strategy(strategy_name, n_trials=n_trials)

        if study.best_trial is not None:
            best_params = study.best_params
            best_value = study.best_value

            # 最終評価
            final = final_evaluation(strategy_name, best_params)

            results[strategy_name] = {
                'best_params': best_params,
                'best_score': best_value,
                'train': final['train'],
                'valid': final['valid'],
                'test': final['test']
            }

            print(f"\n最良パラメータ: {best_params}")
            print(f"Train Sharpe: {final['train']['sharpe']:.2f}")
            print(f"Valid Sharpe: {final['valid']['sharpe']:.2f}")
            print(f"Test Sharpe: {final['test']['sharpe']:.2f}")
            print(f"Test PnL: {final['test']['total_pnl']:+,.0f}円")

    return results

if __name__ == "__main__":
    print("="*60)
    print("QQQ オプション戦略 Optuna最適化")
    print("="*60)

    splits = get_date_splits()
    print(f"\nTrain: {len(splits['train'])}日")
    print(f"Valid: {len(splits['valid'])}日")
    print(f"Test: {len(splits['test'])}日")

    # 全戦略最適化
    results = run_all_optimizations(n_trials=50)

    # 結果保存
    with open('optimization_results.json', 'w') as f:
        # numpy/pandas型をPython型に変換
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        json.dump(convert(results), f, indent=2, ensure_ascii=False)

    # サマリー表示
    print("\n" + "="*60)
    print("最終結果サマリー")
    print("="*60)

    valid_strategies = []
    for name, r in results.items():
        test = r['test']
        if test['sharpe'] > 0 and test['total_pnl'] > 0:
            valid_strategies.append({
                'name': name,
                'sharpe': test['sharpe'],
                'pnl': test['total_pnl'],
                'win_rate': test['win_rate'],
                'trades': test['total_trades']
            })
            print(f"\n✓ {name}")
            print(f"  Test Sharpe: {test['sharpe']:.2f}")
            print(f"  Test PnL: {test['total_pnl']:+,.0f}円")
            print(f"  勝率: {test['win_rate']:.1f}%")
            print(f"  トレード数: {test['total_trades']}")

    if not valid_strategies:
        print("\n有効な戦略が見つかりませんでした")
    else:
        print(f"\n{len(valid_strategies)}個の有効な戦略を発見")
