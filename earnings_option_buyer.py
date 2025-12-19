"""
決算後オプション購入スクリプト
Usage: python earnings_put_buyer.py --call [--dry-run] [--budget JPY] [--otm PCT]
       python earnings_put_buyer.py --put  [--dry-run] [--budget JPY] [--otm PCT]
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
import sys
from datetime import datetime, timedelta
from ib_insync import IB, Stock, Option, LimitOrder


def parse_args():
    parser = argparse.ArgumentParser(description='決算後オプション購入')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--call', action='store_true', help='CALL purchase (bullish)')
    group.add_argument('--put', action='store_true', help='PUT purchase (bearish)')
    parser.add_argument('--dry-run', action='store_true', help='Price check only, no order')
    parser.add_argument('--qty', type=int, default=None, help='Exact quantity (overrides budget)')
    parser.add_argument('--budget', type=int, default=5_000_000, help='Budget in JPY, default 5M')
    parser.add_argument('--otm', type=float, default=0, help='OTM pct, 0=ATM')
    parser.add_argument('--expiry', type=str, default=None, help='Expiry YYYYMMDD, default this Friday')
    parser.add_argument('--usd-jpy', type=float, default=150.0, help='USD/JPY rate')
    return parser.parse_args()


def get_friday_expiry() -> str:
    """今週金曜日の日付を取得"""
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0 and today.hour >= 16:
        days_until_friday = 7
    friday = today + timedelta(days=days_until_friday)
    return friday.strftime('%Y%m%d')


def calc_strike(price: float, otm_pct: float, is_call: bool) -> float:
    """ストライク計算（1ドル刻み）"""
    if is_call:
        target = price * (1 + otm_pct / 100)  # CALL: OTMは上
    else:
        target = price * (1 - otm_pct / 100)  # PUT: OTMは下
    return round(target)


class EarningsOptionBuyer:
    def __init__(self, args):
        self.args = args
        self.ib = IB()
        self.budget_usd = args.budget / args.usd_jpy
        self.is_call = args.call
        self.right = 'C' if args.call else 'P'
        self.direction = 'CALL' if args.call else 'PUT'

    def connect(self):
        self.ib.connect('172.24.32.1', 4001, clientId=20)
        print(f"IB接続完了")

    def disconnect(self):
        self.ib.disconnect()

    def get_qqq_price(self) -> float:
        """QQQ現在価格取得"""
        qqq = Stock('QQQ', 'SMART', 'USD')
        self.ib.qualifyContracts(qqq)
        ticker = self.ib.reqMktData(qqq)
        self.ib.sleep(2)

        price = ticker.last or ticker.close
        self.ib.cancelMktData(qqq)
        return price

    def get_option(self, strike: float, expiry: str):
        """オプション契約取得"""
        opt = Option('QQQ', expiry, strike, self.right, 'SMART', currency='USD', tradingClass='QQQ')
        qualified = self.ib.qualifyContracts(opt)
        if not qualified:
            return None, None

        ticker = self.ib.reqMktData(opt)
        self.ib.sleep(3)
        return opt, ticker

    def calc_qty(self, price: float) -> int:
        """購入数量計算"""
        # --qty指定があればそれを使用
        if self.args.qty is not None:
            return self.args.qty
        if price <= 0:
            return 0
        cost_per_contract = price * 100
        qty = int(self.budget_usd / cost_per_contract)
        return max(1, qty)

    def place_order(self, contract, qty: int, price: float) -> bool:
        """指値注文実行"""
        order = LimitOrder('BUY', qty, round(price, 2))
        trade = self.ib.placeOrder(contract, order)
        print(f"注文発行: BUY {qty}枚 @ ${price:.2f}")

        # 最大60秒待機、10秒ごとにリトライ
        for attempt in range(6):
            self.ib.sleep(10)

            if trade.isDone():
                if trade.orderStatus.status == 'Filled':
                    avg = trade.orderStatus.avgFillPrice
                    print(f"✓ 約定完了: {qty}枚 @ ${avg:.2f}")
                    print(f"  総コスト: ${avg * qty * 100:,.0f} (約{avg * qty * 100 * self.args.usd_jpy:,.0f}円)")
                    return True
                else:
                    print(f"✗ 注文失敗: {trade.orderStatus.status}")
                    return False

            # リトライ
            self.ib.cancelOrder(order)
            self.ib.sleep(1)

            # 最新Ask取得
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)
            new_price = ticker.ask if ticker.ask > 0 else price * 1.01
            self.ib.cancelMktData(contract)

            order = LimitOrder('BUY', qty, round(new_price, 2))
            trade = self.ib.placeOrder(contract, order)
            print(f"再発注 #{attempt+2}: @ ${new_price:.2f}")

        self.ib.cancelOrder(order)
        print("✗ タイムアウト")
        return False

    def run(self):
        """メイン実行"""
        print("=" * 60)
        print(f"決算後 {self.direction} オプション購入")
        print("=" * 60)
        print(f"方向: {self.direction} {'(上昇予想)' if self.is_call else '(下落予想)'}")
        if self.args.qty:
            print(f"数量: {self.args.qty}枚（固定）")
        else:
            print(f"予算: {self.args.budget:,}円 (${self.budget_usd:,.0f})")
        print(f"OTM: {self.args.otm}%")
        print(f"Dry-run: {self.args.dry_run}")
        print()

        self.connect()

        try:
            # QQQ価格取得
            qqq_price = self.get_qqq_price()
            print(f"QQQ現在値: ${qqq_price:.2f}")

            # ストライク計算
            strike = calc_strike(qqq_price, self.args.otm, self.is_call)
            print(f"ストライク: ${strike} ({'ATM' if self.args.otm == 0 else f'OTM {self.args.otm}%'})")

            # 満期日
            expiry = self.args.expiry or get_friday_expiry()
            expiry_display = f"{expiry[:4]}/{expiry[4:6]}/{expiry[6:]}"
            print(f"満期日: {expiry_display}")
            print()

            # オプション取得
            opt, ticker = self.get_option(strike, expiry)
            if not opt:
                print("✗ オプション契約取得失敗")
                return False

            print(f"契約: {opt.localSymbol}")
            print("-" * 40)

            # 価格情報
            bid = ticker.bid if ticker.bid > 0 else 0
            ask = ticker.ask if ticker.ask > 0 else 0
            last = ticker.last if ticker.last > 0 else 0

            iv = None
            delta = None
            if ticker.modelGreeks:
                iv = ticker.modelGreeks.impliedVol * 100 if ticker.modelGreeks.impliedVol else None
                delta = ticker.modelGreeks.delta if ticker.modelGreeks.delta else None

            print(f"Bid:  ${bid:.2f}")
            print(f"Ask:  ${ask:.2f}")
            print(f"Last: ${last:.2f}")
            print(f"スプレッド: ${ask - bid:.2f} ({(ask/bid - 1)*100:.1f}%)" if bid > 0 else "")
            if iv:
                print(f"IV: {iv:.1f}%")
            if delta:
                print(f"Delta: {delta:.3f}")
            print()

            # 購入数量計算
            buy_price = ask if ask > 0 else last
            qty = self.calc_qty(buy_price)
            total_cost = buy_price * qty * 100

            print(f"購入価格: ${buy_price:.2f} (Ask)")
            print(f"購入数量: {qty}枚")
            print(f"総コスト: ${total_cost:,.0f} (約{total_cost * self.args.usd_jpy:,.0f}円)")
            print(f"最大損失: ${total_cost:,.0f} (プレミアム全額)")
            print()

            if self.args.dry_run:
                print(">>> DRY-RUN: 発注はスキップ <<<")
                return True

            # 発注
            print("=" * 40)
            print(f">>> {self.direction} 発注実行 <<<")
            print("=" * 40)
            success = self.place_order(opt, qty, buy_price)
            return success

        finally:
            self.disconnect()


def main():
    args = parse_args()
    buyer = EarningsOptionBuyer(args)
    success = buyer.run()
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
