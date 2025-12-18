"""
トレーリングストップ自動売却
高値から指定%下落で自動売却
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
from datetime import datetime
from ib_insync import IB, LimitOrder


def parse_args():
    parser = argparse.ArgumentParser(description='Trailing stop auto sell')
    parser.add_argument('--trail', type=float, default=0.5, help='Trail percent from high (default 0.5)')
    parser.add_argument('--stop-loss', type=float, default=10.0, help='Stop loss percent from entry (default 10)')
    parser.add_argument('--dry-run', action='store_true', help='Monitor only, no sell')
    return parser.parse_args()


class TrailingStop:
    def __init__(self, args):
        self.args = args
        self.ib = IB()
        self.trail_pct = args.trail
        self.stop_loss_pct = args.stop_loss
        self.high_price = 0.0
        self.entry_price = 0.0

    def connect(self):
        self.ib.connect('172.24.32.1', 4001, clientId=30)
        print("IB接続完了")

    def disconnect(self):
        self.ib.disconnect()

    def get_position(self):
        positions = self.ib.positions()
        for p in positions:
            if p.contract.symbol == 'QQQ' and p.contract.secType == 'OPT' and p.position > 0:
                return p
        return None

    def sell(self, contract, qty, price):
        contract.exchange = 'SMART'
        order = LimitOrder('SELL', qty, round(price, 2))
        trade = self.ib.placeOrder(contract, order)
        print(f"\n{'='*50}")
        print(f">>> 売り注文発行: {qty}枚 @ ${price:.2f}")
        print(f"{'='*50}")

        for attempt in range(6):
            self.ib.sleep(10)
            if trade.isDone():
                if trade.orderStatus.status == 'Filled':
                    avg = trade.orderStatus.avgFillPrice
                    print(f"✓ 約定完了: {qty}枚 @ ${avg:.2f}")
                    return True
                else:
                    print(f"Status: {trade.orderStatus.status}")
                    break

            # リトライ
            self.ib.cancelOrder(order)
            self.ib.sleep(1)
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)
            new_price = ticker.bid if ticker.bid > 0 else price * 0.99
            self.ib.cancelMktData(contract)

            order = LimitOrder('SELL', qty, round(new_price, 2))
            trade = self.ib.placeOrder(contract, order)
            print(f"再発注 #{attempt+2}: @ ${new_price:.2f}")

        return False

    def run(self):
        self.connect()

        try:
            pos = self.get_position()
            if not pos:
                print("ポジションなし。終了。")
                return

            contract = pos.contract
            contract.exchange = 'SMART'
            qty = int(pos.position)
            self.entry_price = pos.avgCost / 100

            right = 'CALL' if contract.right == 'C' else 'PUT'
            print(f"\n監視開始: {contract.localSymbol}")
            print(f"  {right} ${contract.strike} x {qty}枚")
            print(f"  取得単価: ${self.entry_price:.2f}")
            print(f"  トレーリング: -{self.trail_pct}% (高値から)")
            print(f"  損切り: -{self.stop_loss_pct}% (取得単価から)")
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct / 100)
            print(f"  損切りライン: ${stop_loss_price:.2f}")
            if self.args.dry_run:
                print("  [DRY-RUN MODE]")
            print(f"\n{'='*50}\n")

            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(2)

            while True:
                self.ib.sleep(1)

                bid = ticker.bid if ticker.bid > 0 else 0
                ask = ticker.ask if ticker.ask > 0 else 0
                last = ticker.last if ticker.last > 0 else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last

                if mid <= 0:
                    continue

                # 高値更新
                if mid > self.high_price:
                    self.high_price = mid

                # トレーリングストップ価格
                stop_price = self.high_price * (1 - self.trail_pct / 100)

                # 損切りライン（取得単価から計算）
                stop_loss_price = self.entry_price * (1 - self.stop_loss_pct / 100)

                # P&L計算
                pnl_pct = (mid / self.entry_price - 1) * 100 if self.entry_price > 0 else 0
                high_pnl = (self.high_price / self.entry_price - 1) * 100 if self.entry_price > 0 else 0

                now = datetime.now().strftime('%H:%M:%S')
                print(f"{now} 現在:${mid:.2f} 高値:${self.high_price:.2f} Trail:${stop_price:.2f} SL:${stop_loss_price:.2f} P&L:{pnl_pct:+.1f}%")

                # 損切り判定（最優先 - 取得単価から-X%で強制売却）
                if pnl_pct <= -self.stop_loss_pct:
                    print(f"\n>>> 損切り発動! 取得単価${self.entry_price:.2f}から-{self.stop_loss_pct}% <<<")

                    if self.args.dry_run:
                        print("[DRY-RUN] 売却スキップ")
                        return

                    sell_price = bid if bid > 0 else mid * 0.99
                    success = self.sell(contract, qty, sell_price)

                    if success:
                        final_pnl = (sell_price / self.entry_price - 1) * 100
                        print(f"\n最終P&L: {final_pnl:+.1f}%")
                    return

                # トレーリングストップ判定（利益確定用 - 高値から-X%で売却）
                if mid <= stop_price and self.high_price > self.entry_price:
                    print(f"\n>>> トレーリングストップ発動! 高値${self.high_price:.2f}から-{self.trail_pct}% <<<")

                    if self.args.dry_run:
                        print("[DRY-RUN] 売却スキップ")
                        return

                    sell_price = bid if bid > 0 else mid * 0.99
                    success = self.sell(contract, qty, sell_price)

                    if success:
                        final_pnl = (sell_price / self.entry_price - 1) * 100
                        print(f"\n最終P&L: {final_pnl:+.1f}%")
                    return

        except KeyboardInterrupt:
            print("\n中断")
        finally:
            self.disconnect()


def main():
    args = parse_args()
    ts = TrailingStop(args)
    ts.run()


if __name__ == '__main__':
    main()
