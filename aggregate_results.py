"""
全戦略の最適化結果を集約し、有効な戦略をレポート
"""
import json
from pathlib import Path
import pandas as pd

def load_all_results():
    """全ての結果ファイルを読み込み"""
    results = {}
    result_files = list(Path('.').glob('result_*.json'))

    for f in result_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                strategy_name = f.stem.replace('result_', '')
                results[strategy_name] = data
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return results

def evaluate_strategy(result):
    """戦略の有効性を評価"""
    test = result.get('test', {})

    sharpe = test.get('sharpe', -10)
    pnl = test.get('total_pnl', 0)
    win_rate = test.get('win_rate', 0)
    max_dd = test.get('max_dd', 100)
    trades = test.get('total_trades', 0)

    # 有効性スコア
    score = 0

    # Sharpe評価
    if sharpe > 1.5:
        score += 3
    elif sharpe > 1.0:
        score += 2
    elif sharpe > 0.5:
        score += 1
    elif sharpe > 0:
        score += 0.5

    # PnL評価
    if pnl > 1000000:
        score += 3
    elif pnl > 500000:
        score += 2
    elif pnl > 100000:
        score += 1
    elif pnl > 0:
        score += 0.5

    # 勝率評価
    if win_rate > 55:
        score += 2
    elif win_rate > 50:
        score += 1
    elif win_rate > 45:
        score += 0.5

    # ドローダウン評価
    if max_dd < 5:
        score += 2
    elif max_dd < 10:
        score += 1
    elif max_dd < 20:
        score += 0.5

    # トレード数評価（過剰取引ペナルティ）
    if 50 <= trades <= 500:
        score += 1
    elif trades > 1000:
        score -= 1

    return {
        'score': score,
        'sharpe': sharpe,
        'pnl': pnl,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'trades': trades,
        'is_valid': sharpe > 0 and pnl > 0 and win_rate > 40
    }

def generate_report(results):
    """レポートを生成"""
    evaluations = []

    for name, result in results.items():
        eval_result = evaluate_strategy(result)
        eval_result['name'] = name
        eval_result['params'] = result.get('best_params', {})

        # Train/Valid情報も追加
        train = result.get('train', {})
        valid = result.get('valid', {})
        eval_result['train_sharpe'] = train.get('sharpe', 0)
        eval_result['valid_sharpe'] = valid.get('sharpe', 0)
        eval_result['train_pnl'] = train.get('total_pnl', 0)
        eval_result['valid_pnl'] = valid.get('total_pnl', 0)

        evaluations.append(eval_result)

    # スコアでソート
    evaluations.sort(key=lambda x: x['score'], reverse=True)

    return evaluations

def print_report(evaluations):
    """レポートを表示"""
    print("=" * 80)
    print("QQQ オプション戦略 最適化結果レポート")
    print("=" * 80)

    # 有効な戦略
    valid_strategies = [e for e in evaluations if e['is_valid']]

    print(f"\n### 発見された有効戦略: {len(valid_strategies)}個 ###\n")

    for i, e in enumerate(valid_strategies, 1):
        print(f"【{i}位】{e['name']} (スコア: {e['score']:.1f})")
        print(f"  Test Sharpe: {e['sharpe']:.2f}")
        print(f"  Test PnL: {e['pnl']:+,.0f}円")
        print(f"  勝率: {e['win_rate']:.1f}%")
        print(f"  最大DD: {e['max_dd']:.1f}%")
        print(f"  トレード数: {e['trades']}")
        print(f"  Train→Valid→Test Sharpe: {e['train_sharpe']:.2f} → {e['valid_sharpe']:.2f} → {e['sharpe']:.2f}")
        print(f"  パラメータ: {e['params']}")
        print()

    # 無効な戦略
    invalid_strategies = [e for e in evaluations if not e['is_valid']]

    print(f"\n### 無効な戦略: {len(invalid_strategies)}個 ###\n")

    for e in invalid_strategies:
        print(f"✗ {e['name']}: Sharpe={e['sharpe']:.2f}, PnL={e['pnl']:+,.0f}円, 勝率={e['win_rate']:.1f}%")

    return valid_strategies

def save_report(evaluations, filename='strategy_report.json'):
    """レポートをJSONで保存"""
    # パラメータを文字列に変換（JSON互換性のため）
    for e in evaluations:
        e['params'] = str(e['params'])

    with open(filename, 'w') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)

    print(f"\nレポートを {filename} に保存しました")

def main():
    results = load_all_results()

    if not results:
        print("結果ファイルが見つかりません")
        print("result_*.json ファイルを確認してください")
        return

    print(f"読み込んだ戦略数: {len(results)}")

    evaluations = generate_report(results)
    valid_strategies = print_report(evaluations)

    save_report(evaluations)

    # 最終的な推奨戦略
    if valid_strategies:
        best = valid_strategies[0]
        print("\n" + "=" * 80)
        print("★ 推奨戦略 ★")
        print("=" * 80)
        print(f"戦略名: {best['name']}")
        print(f"Test Sharpe: {best['sharpe']:.2f}")
        print(f"Test PnL: {best['pnl']:+,.0f}円")
        print(f"パラメータ: {best['params']}")
    else:
        print("\n" + "=" * 80)
        print("有効な戦略が見つかりませんでした")
        print("追加の戦略探索が必要です")
        print("=" * 80)

if __name__ == "__main__":
    main()
