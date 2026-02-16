"""Black-Scholes オプション価格計算"""
import math
from datetime import datetime, timedelta

def norm_cdf(x):
    """標準正規分布の累積分布関数（近似）"""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)

def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes PUT価格
    S: 原資産価格
    K: ストライク
    T: 残存期間（年）
    r: 無リスク金利
    sigma: ボラティリティ（IV）
    """
    if T <= 0:
        return max(K - S, 0)  # 満期時は本質的価値のみ

    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return put

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes CALL価格"""
    if T <= 0:
        return max(S - K, 0)

    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call

def calc_greeks(S, K, T, r, sigma, option_type='P'):
    """グリークス計算"""
    if T <= 0:
        return {'delta': -1 if option_type == 'P' else 1, 'gamma': 0, 'theta': 0, 'vega': 0}

    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Delta
    if option_type == 'P':
        delta = norm_cdf(d1) - 1
    else:
        delta = norm_cdf(d1)

    # Gamma
    gamma = math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(2 * math.pi * T))

    # Theta (per day)
    theta_part1 = -S * sigma * math.exp(-d1**2 / 2) / (2 * math.sqrt(2 * math.pi * T))
    if option_type == 'P':
        theta = (theta_part1 + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365
    else:
        theta = (theta_part1 - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365

    # Vega (per 1% IV change)
    vega = S * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi) / 100

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def main():
    # パラメータ設定
    S = 622.0       # 原資産価格（QQQ）
    K = 620.0       # ストライク
    days = 4        # 残存日数（月曜→金曜）
    T = days / 365  # 年換算
    r = 0.05        # 無リスク金利 5%

    # ポジション情報
    qty = 90
    avg_cost = 3.32  # 平均購入価格（50枚@$3.48 + 40枚@$3.10）
    total_cost = avg_cost * qty * 100

    print("=" * 60)
    print("QQQ PUT $620 オプション価格計算")
    print("=" * 60)
    print(f"原資産価格: ${S:.2f}")
    print(f"ストライク: ${K:.2f}")
    print(f"残存日数: {days}日")
    print(f"保有数量: {qty}枚")
    print(f"平均購入価格: ${avg_cost:.2f}")
    print(f"投資コスト: ${total_cost:,.0f}")
    print()

    print("-" * 60)
    print(f"{'IV':>6} | {'PUT価格':>8} | {'90枚価値':>12} | {'損益':>12} | {'損益%':>8}")
    print("-" * 60)

    for iv in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        put_price = black_scholes_put(S, K, T, r, iv)
        position_value = put_price * qty * 100
        pnl = position_value - total_cost
        pnl_pct = (pnl / total_cost) * 100

        print(f"{iv*100:>5.0f}% | ${put_price:>7.2f} | ${position_value:>10,.0f} | ${pnl:>+10,.0f} | {pnl_pct:>+7.1f}%")

    print("-" * 60)
    print()

    # 代表的なIV（25%）でのグリークス
    iv_base = 0.25
    greeks = calc_greeks(S, K, T, r, iv_base, 'P')
    put_price = black_scholes_put(S, K, T, r, iv_base)

    print(f"グリークス (IV={iv_base*100:.0f}%):")
    print(f"  Delta: {greeks['delta']:.4f} (90枚: {greeks['delta']*qty*100:.0f})")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Theta: ${greeks['theta']:.4f}/日 (90枚: ${greeks['theta']*qty*100:.2f}/日)")
    print(f"  Vega:  ${greeks['vega']:.4f}/1%IV (90枚: ${greeks['vega']*qty*100:.2f}/1%IV)")
    print()

    # 価格シナリオ分析
    print("=" * 60)
    print("価格シナリオ分析 (IV=25%)")
    print("=" * 60)
    print(f"{'QQQ価格':>10} | {'PUT価格':>8} | {'90枚価値':>12} | {'損益':>12}")
    print("-" * 60)

    for price in [615, 618, 620, 622, 624, 626, 628, 630]:
        put_price = black_scholes_put(price, K, T, r, iv_base)
        position_value = put_price * qty * 100
        pnl = position_value - total_cost
        marker = " <-- 現在" if price == 622 else ""
        print(f"${price:>9} | ${put_price:>7.2f} | ${position_value:>10,.0f} | ${pnl:>+10,.0f}{marker}")

    print("-" * 60)
    print()

    # 損益分岐点
    print("損益分岐点分析:")
    # 二分探索で損益分岐点を見つける
    low, high = 600, 640
    while high - low > 0.01:
        mid = (low + high) / 2
        put_price = black_scholes_put(mid, K, T, r, iv_base)
        position_value = put_price * qty * 100
        if position_value > total_cost:
            low = mid
        else:
            high = mid
    breakeven = (low + high) / 2
    print(f"  IV 25%での損益分岐点: QQQ ${breakeven:.2f}")
    print(f"  現在価格からの距離: ${S - breakeven:+.2f}")
    print()

    # 円換算
    usd_jpy = 155
    print(f"円換算 (USD/JPY={usd_jpy}):")
    put_price_25 = black_scholes_put(S, K, T, r, 0.25)
    position_value = put_price_25 * qty * 100
    pnl = position_value - total_cost
    print(f"  現在価値: ¥{position_value * usd_jpy:,.0f}")
    print(f"  投資コスト: ¥{total_cost * usd_jpy:,.0f}")
    print(f"  含み損益: ¥{pnl * usd_jpy:+,.0f}")

if __name__ == '__main__':
    main()
