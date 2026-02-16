"""
マージン・アカウント状態確認スクリプト
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB


def main():
    ib = IB()
    ib.connect('172.24.32.1', 4001, clientId=22)
    print("IB接続完了\n")

    # アカウントサマリー
    print("=" * 50)
    print("アカウント状態")
    print("=" * 50)

    account_values = ib.accountValues()

    # 重要な項目を抽出
    important_keys = [
        'NetLiquidation',      # 純資産
        'EquityWithLoanValue', # 証拠金計算用の資産
        'FullInitMarginReq',   # 必要証拠金（初期）
        'FullMaintMarginReq',  # 必要証拠金（維持）
        'AvailableFunds',      # 利用可能資金
        'BuyingPower',         # 購買力
        'ExcessLiquidity',     # 余剰流動性
        'Cushion',             # クッション（%）
    ]

    for key in important_keys:
        for av in account_values:
            if av.tag == key and av.currency in ['JPY', 'USD', '']:
                if av.currency:
                    print(f"  {key}: {av.value} {av.currency}")
                else:
                    print(f"  {key}: {av.value}")

    # 全ポジション
    print("\n" + "=" * 50)
    print("全ポジション")
    print("=" * 50)
    positions = ib.positions()

    if positions:
        for pos in positions:
            c = pos.contract
            if c.secType == 'OPT':
                right = 'CALL' if c.right == 'C' else 'PUT'
                print(f"  {c.symbol} {right} ${c.strike} 満期:{c.lastTradeDateOrContractMonth}")
                print(f"    数量: {int(pos.position)}枚, 平均コスト: ${pos.avgCost / 100:.2f}")
            elif c.secType == 'STK':
                print(f"  {c.symbol} 株式")
                print(f"    数量: {int(pos.position)}株, 平均コスト: ${pos.avgCost:.2f}")
            else:
                print(f"  {c.symbol} ({c.secType})")
                print(f"    数量: {pos.position}, コスト: {pos.avgCost}")
            print()
    else:
        print("  ポジションなし")

    ib.disconnect()


if __name__ == '__main__':
    main()
