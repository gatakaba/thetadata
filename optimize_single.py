"""
単一戦略のOptuna最適化
Usage: python optimize_single.py <strategy_name> <n_trials>
"""
import sys
import json
from optuna_backtest import (
    optimize_strategy, final_evaluation, get_date_splits, STRATEGIES
)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <strategy_name> [n_trials]")
        print(f"Available strategies: {list(STRATEGIES.keys())}")
        sys.exit(1)

    strategy_name = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if strategy_name not in STRATEGIES:
        print(f"Unknown strategy: {strategy_name}")
        print(f"Available strategies: {list(STRATEGIES.keys())}")
        sys.exit(1)

    splits = get_date_splits()
    print(f"\n{'='*60}")
    print(f"最適化: {strategy_name} ({n_trials} trials)")
    print(f"Train: {len(splits['train'])}日, Valid: {len(splits['valid'])}日, Test: {len(splits['test'])}日")
    print('='*60)

    study = optimize_strategy(strategy_name, n_trials=n_trials)

    if study.best_trial is None:
        print("最適化失敗")
        sys.exit(1)

    best_params = study.best_params
    best_value = study.best_value

    # 最終評価
    final = final_evaluation(strategy_name, best_params)

    result = {
        'strategy': strategy_name,
        'best_params': best_params,
        'best_score': best_value,
        'train': final['train'],
        'valid': final['valid'],
        'test': final['test']
    }

    # 結果保存
    output_file = f'result_{strategy_name}.json'
    with open(output_file, 'w') as f:
        def convert(obj):
            import numpy as np
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
    print(f"結果: {strategy_name}")
    print('='*60)
    print(f"最良パラメータ: {best_params}")
    print(f"\n--- Train ---")
    print(f"  Sharpe: {final['train']['sharpe']:.2f}")
    print(f"  PnL: {final['train']['total_pnl']:+,.0f}円")
    print(f"  勝率: {final['train']['win_rate']:.1f}%")
    print(f"  トレード数: {final['train']['total_trades']}")
    print(f"\n--- Valid ---")
    print(f"  Sharpe: {final['valid']['sharpe']:.2f}")
    print(f"  PnL: {final['valid']['total_pnl']:+,.0f}円")
    print(f"  勝率: {final['valid']['win_rate']:.1f}%")
    print(f"  トレード数: {final['valid']['total_trades']}")
    print(f"\n--- Test ---")
    print(f"  Sharpe: {final['test']['sharpe']:.2f}")
    print(f"  PnL: {final['test']['total_pnl']:+,.0f}円")
    print(f"  勝率: {final['test']['win_rate']:.1f}%")
    print(f"  トレード数: {final['test']['total_trades']}")

    # 有効性判定
    is_valid = (
        final['test']['sharpe'] > 0.5 and
        final['test']['total_pnl'] > 0 and
        final['test']['win_rate'] > 40
    )

    if is_valid:
        print(f"\n✓ {strategy_name} は有効な戦略です！")
    else:
        print(f"\n✗ {strategy_name} は有効な戦略ではありません")

    return result

if __name__ == "__main__":
    main()
