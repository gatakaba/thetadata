"""
ポジション・注文確認スクリプト
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB


def main():
    ib = IB()
    ib.connect('172.24.32.1', 4001, clientId=21)
    print("IB接続完了\n")

    # オープン注文確認
    print("=" * 50)
    print("オープン注文")
    print("=" * 50)
    orders = ib.openOrders()
    if orders:
        for order in orders:
            print(f"  {order.action} {order.totalQuantity}枚 @ ${order.lmtPrice}")
    else:
        print("  なし")

    # 現在のポジション
    print("\n" + "=" * 50)
    print("現在のポジション")
    print("=" * 50)
    positions = ib.positions()
    qqq_options = [p for p in positions if p.contract.symbol == 'QQQ' and p.contract.secType == 'OPT']

    if qqq_options:
        for pos in qqq_options:
            c = pos.contract
            right = 'CALL' if c.right == 'C' else 'PUT'
            print(f"  {c.localSymbol}")
            print(f"    {right} ${c.strike} 満期:{c.lastTradeDateOrContractMonth}")
            print(f"    数量: {int(pos.position)}枚")
            print(f"    平均コスト: ${pos.avgCost / 100:.2f}")
            print()
    else:
        print("  QQQオプションポジションなし")

    # 約定履歴（本日）
    print("=" * 50)
    print("本日の約定")
    print("=" * 50)
    fills = ib.fills()
    qqq_fills = [f for f in fills if f.contract.symbol == 'QQQ']

    if qqq_fills:
        for fill in qqq_fills:
            c = fill.contract
            e = fill.execution
            print(f"  {e.time.strftime('%H:%M:%S')} {e.side} {int(e.shares)}枚 @ ${e.price:.2f}")
            print(f"    {c.localSymbol}")
            print()
    else:
        print("  なし")

    ib.disconnect()


if __name__ == '__main__':
    main()
