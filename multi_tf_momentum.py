#!/usr/bin/env python3
"""複数時間枠モメンタム戦略 - QQQ株ロングのみ

ロジック:
- 10分足でトレンド方向を判定（close→close）
- 2分足で+5bp以上の動きでエントリー（close→close）
- 次の2分足境界でエグジット

ポジションサイジング:
- ハーフケリー: 7.14%
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
from datetime import datetime
from collections import deque
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock, MarketOrder
import numpy as np


# パラメータ
FAST_SEC = 120  # 2分
SLOW_SEC = 600  # 10分
THRESH_BP = 5   # 5bp
HALF_KELLY = 0.0714  # 7.14%

# タイムゾーン
ET = ZoneInfo("America/New_York")


def log(msg: str):
    now = datetime.now(ET)
    print(f"{now.strftime('%H:%M:%S.%f')[:-3]} | {msg}", flush=True)


def get_bar_boundary(dt: datetime, interval_sec: int) -> datetime:
    """指定した間隔のバー境界を計算"""
    interval_min = interval_sec // 60
    minute = (dt.minute // interval_min) * interval_min
    return dt.replace(minute=minute, second=0, microsecond=0)


def is_market_hours(dt: datetime) -> bool:
    """RTH判定（9:30-16:00 ET）"""
    if dt.weekday() >= 5:  # 土日
        return False
    t = dt.time()
    from datetime import time as dt_time
    return dt_time(9, 30) <= t < dt_time(16, 0)


class MultiTFMomentum:
    def __init__(self, ib: IB, dry_run: bool = False):
        self.ib = ib
        self.dry_run = dry_run

        # QQQ株
        self.stock = Stock('QQQ', 'SMART', 'USD')
        ib.qualifyContracts(self.stock)

        # マーケットデータを常時購読
        self.ticker = ib.reqMktData(self.stock, '', False, False)

        # 価格履歴（2分足用）
        self.fast_prices = deque(maxlen=10)
        self.fast_times = deque(maxlen=10)

        # 価格履歴（10分足用）
        self.slow_prices = deque(maxlen=10)
        self.slow_times = deque(maxlen=10)

        # 現在のバー
        self.current_fast_boundary = None
        self.current_fast_prices = []
        self.current_slow_boundary = None
        self.current_slow_prices = []

        # ポジション状態
        self.in_position = False
        self.entry_time = None
        self.entry_price = 0.0
        self.position_qty = 0
        self.exit_boundary = None  # エグジット予定の2分足境界

        # 二重エントリー防止
        self.last_entry_bar = None

        # 口座情報
        self.account_value = 0.0

    def get_current_price(self) -> tuple[float, float, float]:
        """現在のbid/ask/midを取得（常時購読から）"""
        self.ib.sleep(0.1)  # データ更新を待つ

        bid = self.ticker.bid if self.ticker.bid and self.ticker.bid > 0 else 0
        ask = self.ticker.ask if self.ticker.ask and self.ticker.ask > 0 else 0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

        return bid, ask, mid

    def update_bars(self, now: datetime, mid: float) -> bool:
        """バーを更新。2分足が確定したらTrueを返す"""
        fast_boundary = get_bar_boundary(now, FAST_SEC)
        slow_boundary = get_bar_boundary(now, SLOW_SEC)

        fast_bar_closed = False

        # 2分足の更新
        if self.current_fast_boundary is None:
            self.current_fast_boundary = fast_boundary
            self.current_fast_prices = [mid]
        elif fast_boundary > self.current_fast_boundary:
            # 前の2分足を確定
            if self.current_fast_prices:
                close_price = self.current_fast_prices[-1]
                self.fast_prices.append(close_price)
                self.fast_times.append(self.current_fast_boundary)
                fast_bar_closed = True
            # 新しい2分足を開始
            self.current_fast_boundary = fast_boundary
            self.current_fast_prices = [mid]
        else:
            self.current_fast_prices.append(mid)

        # 10分足の更新
        if self.current_slow_boundary is None:
            self.current_slow_boundary = slow_boundary
            self.current_slow_prices = [mid]
        elif slow_boundary > self.current_slow_boundary:
            # 前の10分足を確定
            if self.current_slow_prices:
                close_price = self.current_slow_prices[-1]
                self.slow_prices.append(close_price)
                self.slow_times.append(self.current_slow_boundary)
            # 新しい10分足を開始
            self.current_slow_boundary = slow_boundary
            self.current_slow_prices = [mid]
        else:
            self.current_slow_prices.append(mid)

        return fast_bar_closed

    def check_signal(self) -> bool:
        """ロングシグナルをチェック"""
        if len(self.fast_prices) < 2 or len(self.slow_prices) < 2:
            return False

        # 二重エントリー防止
        current_bar = self.fast_times[-1] if self.fast_times else None
        if current_bar and current_bar == self.last_entry_bar:
            return False

        # 10分足の方向（close→close）
        slow_ret = np.log(self.slow_prices[-1] / self.slow_prices[-2])
        slow_dir = np.sign(slow_ret)

        # 2分足のリターン（close→close）
        fast_ret = np.log(self.fast_prices[-1] / self.fast_prices[-2])

        threshold = THRESH_BP / 10000

        # ロングシグナル
        if fast_ret > threshold and slow_dir > 0:
            log(f"シグナル検出: fast_ret={fast_ret*10000:.1f}bp, slow_dir={slow_dir:+.0f}")
            return True

        return False

    def get_account_value(self) -> float:
        """口座の総資産を取得（USD換算）"""
        jpy_value = 0.0
        usd_value = 0.0

        for av in self.ib.accountSummary():
            if av.tag == 'NetLiquidation':
                if av.currency == 'JPY':
                    jpy_value = float(av.value)
                elif av.currency == 'USD':
                    usd_value = float(av.value)

        # USD建てがあればそれを使用、なければJPYを変換
        if usd_value > 0:
            return usd_value
        elif jpy_value > 0:
            # 円建てをドルに変換（概算レート）
            usdjpy = 155.0  # TODO: 実際のレートを取得
            return jpy_value / usdjpy

        return 0.0

    def calculate_position_size(self, ask_price: float) -> int:
        """ハーフケリーでポジションサイズを計算"""
        self.account_value = self.get_account_value()
        if self.account_value <= 0:
            return 0

        position_value = self.account_value * HALF_KELLY
        shares = int(position_value / ask_price)

        return max(1, shares)

    def enter_position(self, ask_price: float, now: datetime):
        """ポジションをエントリー"""
        qty = self.calculate_position_size(ask_price)

        # 次の2分足境界でエグジット
        next_boundary = get_bar_boundary(now, FAST_SEC)
        next_boundary = next_boundary.replace(
            minute=next_boundary.minute + (FAST_SEC // 60)
        )
        # 分のオーバーフロー処理
        if next_boundary.minute >= 60:
            next_boundary = next_boundary.replace(
                hour=next_boundary.hour + 1,
                minute=next_boundary.minute - 60
            )

        log(f"エントリー: {qty}株 @ ${ask_price:.2f}")
        log(f"  口座: ${self.account_value:,.0f}, ハーフケリー: {HALF_KELLY*100:.2f}%")
        log(f"  エグジット予定: {next_boundary.strftime('%H:%M:%S')}")

        if self.dry_run:
            log("  [DRY RUN] 注文スキップ")
            self.in_position = True
            self.entry_time = now
            self.entry_price = ask_price
            self.position_qty = qty
            self.exit_boundary = next_boundary
            self.last_entry_bar = self.fast_times[-1] if self.fast_times else None
            return

        # 成行買い
        order = MarketOrder('BUY', qty)
        trade = self.ib.placeOrder(self.stock, order)

        for _ in range(20):
            self.ib.sleep(1)
            if trade.orderStatus.status == 'Filled':
                avg = trade.orderStatus.avgFillPrice
                log(f"  約定: {qty}株 @ ${avg:.2f}")
                self.in_position = True
                self.entry_time = now
                self.entry_price = avg
                self.position_qty = qty
                self.exit_boundary = next_boundary
                self.last_entry_bar = self.fast_times[-1] if self.fast_times else None
                return
            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                log(f"  失敗: {trade.orderStatus.status}")
                return

        log(f"  タイムアウト: {trade.orderStatus.status}")

    def exit_position(self, bid_price: float):
        """ポジションをエグジット"""
        if not self.in_position:
            return

        pnl = (bid_price - self.entry_price) * self.position_qty
        ret_bp = (bid_price / self.entry_price - 1) * 10000

        log(f"エグジット: {self.position_qty}株 @ ${bid_price:.2f}")
        log(f"  PnL: ${pnl:+,.2f} ({ret_bp:+.1f}bp)")

        if self.dry_run:
            log("  [DRY RUN] 注文スキップ")
            self._reset_position()
            return

        order = MarketOrder('SELL', self.position_qty)
        trade = self.ib.placeOrder(self.stock, order)

        for _ in range(20):
            self.ib.sleep(1)
            if trade.orderStatus.status == 'Filled':
                avg = trade.orderStatus.avgFillPrice
                actual_pnl = (avg - self.entry_price) * self.position_qty
                log(f"  約定: {self.position_qty}株 @ ${avg:.2f}, PnL: ${actual_pnl:+,.2f}")
                self._reset_position()
                return
            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                log(f"  失敗: {trade.orderStatus.status}")
                return

        log(f"  タイムアウト: {trade.orderStatus.status}")

    def _reset_position(self):
        """ポジション状態をリセット"""
        self.in_position = False
        self.entry_time = None
        self.entry_price = 0.0
        self.position_qty = 0
        self.exit_boundary = None

    def run(self, interval_sec: float = 1.0):
        """メインループ"""
        log("複数時間枠モメンタム戦略 開始")
        log(f"設定: fast={FAST_SEC}s, slow={SLOW_SEC}s, thresh={THRESH_BP}bp")
        log(f"ポジションサイジング: ハーフケリー {HALF_KELLY*100:.2f}%")
        if self.dry_run:
            log("[DRY RUN モード]")

        try:
            while True:
                now = datetime.now(ET)

                # RTH外はスキップ
                if not is_market_hours(now):
                    self.ib.sleep(interval_sec)
                    continue

                # 価格取得
                bid, ask, mid = self.get_current_price()
                if mid <= 0:
                    self.ib.sleep(interval_sec)
                    continue

                # バーを更新
                fast_bar_closed = self.update_bars(now, mid)

                # ポジション保有中
                if self.in_position:
                    # 次の2分足境界でエグジット
                    if now >= self.exit_boundary:
                        self.exit_position(bid)
                    else:
                        unrealized = (bid - self.entry_price) * self.position_qty
                        remaining = (self.exit_boundary - now).total_seconds()
                        log(f"保有中: 含み益${unrealized:+,.2f}, 残り{remaining:.0f}s")

                # ポジションなし
                else:
                    # 2分足確定時のみシグナルチェック
                    if fast_bar_closed and self.check_signal():
                        self.enter_position(ask, now)

                self.ib.sleep(interval_sec)

        except KeyboardInterrupt:
            log("終了")
            if self.in_position:
                log("ポジションをクローズします...")
                bid, _, _ = self.get_current_price()
                self.exit_position(bid)

    def cleanup(self):
        """クリーンアップ"""
        self.ib.cancelMktData(self.stock)


def main():
    parser = argparse.ArgumentParser(description='複数時間枠モメンタム戦略')
    parser.add_argument('--dry-run', action='store_true', help='ドライラン')
    parser.add_argument('--interval', type=float, default=1.0, help='監視間隔（秒）')
    args = parser.parse_args()

    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=80)
        log("IBKRに接続しました")

        ib.sleep(2)  # アカウント情報の取得を待つ
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    strategy = MultiTFMomentum(ib, dry_run=args.dry_run)

    try:
        strategy.run(interval_sec=args.interval)
    finally:
        strategy.cleanup()
        ib.disconnect()
        log("切断しました")


if __name__ == '__main__':
    main()
