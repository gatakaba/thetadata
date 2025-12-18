"""
opening_range戦略の統計を確認
"""
from optuna_backtest import get_date_splits, backtest_strategy, evaluate_trades
from advanced_strategies import strategy_opening_range

test_params = {
    'range_minutes': 15,
    'hold_time': 120,
    'max_spread': 0.10,
}

splits = get_date_splits()
print(f"パラメータ: {test_params}")
print("="*60)

# Trainで実行
print("\nTrain期間でバックテスト...")
trades = backtest_strategy(strategy_opening_range, test_params, splits['train'])
stats = evaluate_trades(trades)

print(f"\n【Train統計】")
print(f"総取引数: {stats['total_trades']}")
print(f"総PnL: {stats['total_pnl']:,.0f}円")
print(f"勝率: {stats['win_rate']:.1%}")
print(f"Sharpe比率: {stats['sharpe']:.2f}")
print(f"最大ドローダウン: {stats['max_dd']:.1f}%")
print(f"平均PnL: {stats['avg_pnl']:.2f}円")

# なぜ-10が返されるか確認
if stats['total_trades'] < 50:
    print(f"\n✗ 取引数不足: {stats['total_trades']} < 50")
elif stats['max_dd'] > 30:
    print(f"\n✗ ドローダウン超過: {stats['max_dd']:.1f}% > 30%")
else:
    print(f"\n✓ 基本条件クリア")
