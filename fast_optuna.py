"""
高速Optunaバックテスト
- データを事前にメモリにロード
- 軽量なシグナル生成
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

        # 特徴量計算
        for w in [5, 10, 15, 30, 60, 120]:
            stock[f'change_{w}s'] = stock['mid'].diff(w)

        stock['volatility_60s'] = stock['mid'].rolling(60).std()

        delta = stock['mid'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        stock['rsi_14'] = 100 - (100 / (1 + rs))

        stock['vwap'] = stock['mid'].expanding().mean()
        stock['vwap_dev'] = (stock['mid'] - stock['vwap']) / stock['vwap'] * 100

        stock['hour'] = stock['timestamp'].dt.hour

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

def fast_backtest(params, dates, strategy_type='mean_revert'):
    """高速バックテスト"""
    window = params.get('window', 15)
    threshold = params.get('threshold', 0.5)
    hold_time = params.get('hold_time', 60)
    max_spread = params.get('max_spread', 0.10)

    all_pnl = []
    trade_count = 0

    for date_str in dates:
        if date_str not in DATA_CACHE:
            continue

        df = DATA_CACHE[date_str].copy()

        change_col = f'change_{window}s'
        if change_col not in df.columns:
            continue

        df[f'future_{hold_time}s'] = df['mid'].shift(-hold_time) - df['mid']
        df = df.dropna(subset=[change_col, f'future_{hold_time}s', 'put_spread', 'call_spread'])

        if len(df) == 0:
            continue

        spread_ok = (df['put_spread'] <= max_spread) & (df['call_spread'] <= max_spread)

        # 戦略別シグナル
        if strategy_type == 'mean_revert':
            long_mask = (df[change_col] < -threshold) & spread_ok
            short_mask = (df[change_col] > threshold) & spread_ok
        elif strategy_type == 'momentum':
            long_mask = (df[change_col] > threshold) & spread_ok
            short_mask = (df[change_col] < -threshold) & spread_ok
        elif strategy_type == 'rsi':
            oversold = params.get('oversold', 30)
            overbought = params.get('overbought', 70)
            long_mask = (df['rsi_14'] < oversold) & spread_ok
            short_mask = (df['rsi_14'] > overbought) & spread_ok
        elif strategy_type == 'vwap':
            vwap_threshold = params.get('vwap_threshold', 0.1)
            direction = params.get('direction', 'mean_revert')
            if direction == 'mean_revert':
                long_mask = (df['vwap_dev'] < -vwap_threshold) & spread_ok
                short_mask = (df['vwap_dev'] > vwap_threshold) & spread_ok
            else:
                long_mask = (df['vwap_dev'] > vwap_threshold) & spread_ok
                short_mask = (df['vwap_dev'] < -vwap_threshold) & spread_ok
        elif strategy_type == 'time_filter':
            start_hour = params.get('start_hour', 10)
            end_hour = params.get('end_hour', 15)
            time_ok = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
            long_mask = (df[change_col] < -threshold) & spread_ok & time_ok
            short_mask = (df[change_col] > threshold) & spread_ok & time_ok
        elif strategy_type == 'combined':
            rsi_filter = params.get('rsi_filter', 10)
            vol_filter = params.get('vol_filter', 0.5)
            vol_threshold = df['volatility_60s'].quantile(vol_filter)
            vol_ok = df['volatility_60s'] > vol_threshold
            long_mask = (df[change_col] < -threshold) & (df['rsi_14'] < 50 - rsi_filter) & vol_ok & spread_ok
            short_mask = (df[change_col] > threshold) & (df['rsi_14'] > 50 + rsi_filter) & vol_ok & spread_ok
        elif strategy_type == 'high_threshold':
            cooldown = params.get('cooldown', 120)
            base_long = (df[change_col] < -threshold) & spread_ok
            base_short = (df[change_col] > threshold) & spread_ok
            # 簡易クールダウン（毎N行のみ）
            long_mask = base_long & (df.index % cooldown == 0)
            short_mask = base_short & (df.index % cooldown == 0)
        elif strategy_type == 'volatility_breakout':
            lookback = params.get('lookback', 120)
            mult = params.get('mult', 2.0)
            df['rolling_std'] = df['mid'].rolling(lookback).std()
            df['upper'] = df['mid'].rolling(lookback).mean() + mult * df['rolling_std']
            df['lower'] = df['mid'].rolling(lookback).mean() - mult * df['rolling_std']
            df = df.dropna(subset=['upper', 'lower'])
            long_mask = (df['mid'] > df['upper']) & spread_ok
            short_mask = (df['mid'] < df['lower']) & spread_ok
        else:
            continue

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
            trade_count += day_trades

    if not all_pnl:
        return {'sharpe': -10, 'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 100}

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
        'pnl': total_pnl,
        'trades': total_trades,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'n_days': len(pnl_df)
    }

def create_objective(strategy_type, train_dates, valid_dates):
    """Optuna目的関数"""
    def objective(trial):
        if strategy_type in ['mean_revert', 'momentum', 'time_filter']:
            params = {
                'window': trial.suggest_categorical('window', [5, 10, 15, 30, 60, 120]),
                'threshold': trial.suggest_float('threshold', 0.3, 2.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
            if strategy_type == 'time_filter':
                params['start_hour'] = trial.suggest_int('start_hour', 9, 12)
                params['end_hour'] = trial.suggest_int('end_hour', 13, 16)
        elif strategy_type == 'rsi':
            params = {
                'oversold': trial.suggest_int('oversold', 20, 40),
                'overbought': trial.suggest_int('overbought', 60, 80),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
        elif strategy_type == 'vwap':
            params = {
                'vwap_threshold': trial.suggest_float('vwap_threshold', 0.05, 0.5, step=0.05),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'direction': trial.suggest_categorical('direction', ['mean_revert', 'momentum']),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
        elif strategy_type == 'combined':
            params = {
                'window': trial.suggest_categorical('window', [5, 10, 15, 30]),
                'threshold': trial.suggest_float('threshold', 0.3, 1.5, step=0.1),
                'rsi_filter': trial.suggest_int('rsi_filter', 5, 25),
                'vol_filter': trial.suggest_float('vol_filter', 0.3, 0.7, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
            }
        elif strategy_type == 'high_threshold':
            params = {
                'window': trial.suggest_categorical('window', [10, 15, 30, 60]),
                'threshold': trial.suggest_float('threshold', 0.8, 3.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [60, 120, 300]),
                'cooldown': trial.suggest_categorical('cooldown', [60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.10, step=0.01),
            }
        elif strategy_type == 'volatility_breakout':
            params = {
                'lookback': trial.suggest_categorical('lookback', [60, 120, 300, 600]),
                'mult': trial.suggest_float('mult', 1.5, 3.0, step=0.1),
                'hold_time': trial.suggest_categorical('hold_time', [30, 60, 120, 300]),
                'max_spread': trial.suggest_float('max_spread', 0.05, 0.12, step=0.01),
            }
        else:
            return -10

        train_stats = fast_backtest(params, train_dates, strategy_type)

        if train_stats['trades'] < 50:
            return -10
        if train_stats['max_dd'] > 30:
            return -10

        valid_stats = fast_backtest(params, valid_dates, strategy_type)

        if valid_stats['trades'] < 20:
            return -5

        score = (train_stats['sharpe'] + valid_stats['sharpe']) / 2

        if valid_stats['sharpe'] > 0 and train_stats['sharpe'] > 0:
            score += 0.5
        if abs(train_stats['sharpe'] - valid_stats['sharpe']) > 1:
            score -= 0.5

        return score

    return objective

def optimize_all():
    """全戦略を最適化"""
    splits = get_date_splits()
    all_dates = splits['train'] + splits['valid'] + splits['test']

    print("データをロード中...")
    load_all_data(all_dates)

    strategies = [
        'mean_revert', 'momentum', 'rsi', 'vwap',
        'time_filter', 'combined', 'high_threshold', 'volatility_breakout'
    ]

    results = {}

    for strategy_type in strategies:
        print(f"\n{'='*60}")
        print(f"最適化: {strategy_type}")
        print('='*60)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        objective = create_objective(strategy_type, splits['train'], splits['valid'])

        study.optimize(objective, n_trials=50, show_progress_bar=True)

        if study.best_trial is None:
            print(f"{strategy_type}: 最適化失敗")
            continue

        best_params = study.best_params

        # 最終評価
        train_stats = fast_backtest(best_params, splits['train'], strategy_type)
        valid_stats = fast_backtest(best_params, splits['valid'], strategy_type)
        test_stats = fast_backtest(best_params, splits['test'], strategy_type)

        results[strategy_type] = {
            'best_params': best_params,
            'train': train_stats,
            'valid': valid_stats,
            'test': test_stats
        }

        print(f"\n最良パラメータ: {best_params}")
        print(f"Train: Sharpe={train_stats['sharpe']:.2f}, PnL={train_stats['pnl']:+,.0f}円")
        print(f"Valid: Sharpe={valid_stats['sharpe']:.2f}, PnL={valid_stats['pnl']:+,.0f}円")
        print(f"Test: Sharpe={test_stats['sharpe']:.2f}, PnL={test_stats['pnl']:+,.0f}円")

        is_valid = test_stats['sharpe'] > 0 and test_stats['pnl'] > 0
        print(f"{'✓ 有効' if is_valid else '✗ 無効'}")

        # 結果保存
        with open(f'result_{strategy_type}.json', 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj
            json.dump(convert(results[strategy_type]), f, indent=2, ensure_ascii=False)

    return results

def print_summary(results):
    """サマリー表示"""
    print("\n" + "="*80)
    print("最適化結果サマリー")
    print("="*80)

    valid_strategies = []

    for name, r in results.items():
        test = r['test']
        is_valid = test['sharpe'] > 0 and test['pnl'] > 0

        status = "✓" if is_valid else "✗"
        print(f"{status} {name}: Sharpe={test['sharpe']:.2f}, PnL={test['pnl']:+,.0f}円, 勝率={test['win_rate']:.1f}%")

        if is_valid:
            valid_strategies.append({
                'name': name,
                'sharpe': test['sharpe'],
                'pnl': test['pnl'],
                'params': r['best_params']
            })

    print(f"\n有効な戦略: {len(valid_strategies)}個")

    if valid_strategies:
        best = max(valid_strategies, key=lambda x: x['sharpe'])
        print(f"\n★ 最良戦略: {best['name']}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  PnL: {best['pnl']:+,.0f}円")
        print(f"  パラメータ: {best['params']}")

    return valid_strategies

if __name__ == "__main__":
    results = optimize_all()
    valid = print_summary(results)

    # 全結果保存
    with open('all_results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(results), f, indent=2, ensure_ascii=False)

    print("\n結果をall_results.jsonに保存しました")
