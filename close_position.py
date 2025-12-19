"""
ポジションクローズスクリプト
Usage: python close_position.py [--dry-run]
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
from ib_insync import IB, LimitOrder


def parse_args():
    parser = argparse.ArgumentParser(description='Close QQQ option position')
    parser.add_argument('--dry-run', action='store_true', help='Price check only, no order')
    return parser.parse_args()


class PositionCloser:
    def __init__(self, args):
        self.args = args
        self.ib = IB()

    def connect(self):
        self.ib.connect('172.24.32.1', 4001, clientId=22)
        print("IB接続完了\n")

    def disconnect(self):
        self.ib.disconnect()

    def get_qqq_option_positions(self):
        """QQQオプションポジション取得"""
        positions = self.ib.positions()
        return [p for p in positions if p.contract.symbol == 'QQQ' and p.contract.secType == 'OPT' and p.position > 0]

    def get_current_price(self, contract):
        """現在価格取得"""
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(2)
        bid = ticker.bid if ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask > 0 else 0
        last = ticker.last if ticker.last > 0 else 0
        self.ib.cancelMktData(contract)
        return bid, ask, last

    def place_sell_order(self, contract, qty: int, price: float) -> bool:
        """指値売り注文、リトライあり"""
        order = LimitOrder('SELL', qty, round(price, 2))
        trade = self.ib.placeOrder(contract, order)
        print(f"売り注文発行: {qty}枚 @ ${price:.2f}")

        # 最大60秒待機、10秒ごとにリトライ
        for attempt in range(6):
            self.ib.sleep(10)

            if trade.isDone():
                if trade.orderStatus.status == 'Filled':
                    avg = trade.orderStatus.avgFillPrice
                    print(f"✓ 約定完了: {qty}枚 @ ${avg:.2f}")
                    print(f"  売却額: ${avg * qty * 100:,.0f}")
                    return True
                else:
                    print(f"✗ 注文失敗: {trade.orderStatus.status}")
                    return False

            # リトライ - Bid更新
            self.ib.cancelOrder(order)
            self.ib.sleep(1)

            bid, ask, last = self.get_current_price(contract)
            new_price = bid if bid > 0 else price * 0.99

            order = LimitOrder('SELL', qty, round(new_price, 2))
            trade = self.ib.placeOrder(contract, order)
            print(f"再発注 #{attempt+2}: @ ${new_price:.2f} (Bid: ${bid:.2f})")

        self.ib.cancelOrder(order)
        print("✗ タイムアウト")
        return False

    def run(self):
        """メイン実行"""
        self.connect()

        try:
            # ポジション取得
            positions = self.get_qqq_option_positions()

            if not positions:
                print("クローズ対象のQQQオプションポジションなし")
                return False

            print("=" * 60)
            print("ポジションクローズ")
            print("=" * 60)

            success_all = True

            for pos in positions:
                contract = pos.contract
                contract.exchange = 'SMART'  # 必須: exchange設定
                qty = int(pos.position)
                avg_cost = pos.avgCost / 100  # 1株あたりに変換

                right = 'CALL' if contract.right == 'C' else 'PUT'
                print(f"\n{contract.localSymbol}")
                print(f"  {right} ${contract.strike} 満期:{contract.lastTradeDateOrContractMonth}")
                print(f"  数量: {qty}枚")
                print(f"  取得単価: ${avg_cost:.2f}")
                print("-" * 40)

                # 現在価格取得
                bid, ask, last = self.get_current_price(contract)
                print(f"  Bid: ${bid:.2f}")
                print(f"  Ask: ${ask:.2f}")
                print(f"  Last: ${last:.2f}")

                # P&L計算
                current = bid if bid > 0 else last
                pnl = (current - avg_cost) * qty * 100
                pnl_pct = (current / avg_cost - 1) * 100 if avg_cost > 0 else 0
                print(f"  P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)")
                print()

                if self.args.dry_run:
                    print(">>> DRY-RUN: 発注スキップ <<<")
                    continue

                # 売り注文
                sell_price = bid if bid > 0 else last * 0.98
                print("=" * 40)
                print(">>> 売り発注実行 <<<")
                print("=" * 40)
                success = self.place_sell_order(contract, qty, sell_price)
                if not success:
                    success_all = False

            return success_all

        finally:
            self.disconnect()


def main():
    args = parse_args()
    closer = PositionCloser(args)
    success = closer.run()
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
