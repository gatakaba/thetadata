"""
QQQ オプション自動売買システム v2
クラス設計版
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from ib_insync import IB, Stock, Option, LimitOrder, Ticker, Contract


@dataclass
class Config:
    """設定"""
    ib_ip: str = '172.24.32.1'
    ib_port: int = 4001
    client_id: int = 10
    symbol: str = 'QQQ'
    budget_usd: float = 50000

    # エントリー条件
    entry_change_pct: float = 2.0
    entry_iv_max: float = 20.0

    # 出口条件
    take_profit_half: float = 50.0
    take_profit_full: float = 100.0
    stop_loss: float = -30.0

    # 注文設定
    order_timeout_sec: int = 60
    order_retry_interval: int = 10


@dataclass
class Position:
    """ポジション"""
    contract: Contract
    entry_price: float
    qty: int
    ticker: Ticker
    half_sold: bool = False

    @property
    def right(self) -> str:
        return self.contract.right

    @property
    def direction(self) -> str:
        return "CALL" if self.right == 'C' else "PUT"

    def calc_pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price / self.entry_price - 1) * 100


class IBConnection:
    """IB接続管理"""

    def __init__(self, config: Config):
        self.config = config
        self.ib = IB()

    def connect(self):
        self.ib.connect(
            self.config.ib_ip,
            self.config.ib_port,
            clientId=self.config.client_id
        )
        print(f"IB接続完了: {self.config.ib_ip}:{self.config.ib_port}")

    def disconnect(self):
        self.ib.disconnect()
        print("IB切断完了")

    def sleep(self, sec: float):
        self.ib.sleep(sec)


class ContractFactory:
    """契約生成"""

    def __init__(self, ib: IB, symbol: str):
        self.ib = ib
        self.symbol = symbol

    def create_stock(self) -> Stock:
        stock = Stock(self.symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        return stock

    def create_option(self, strike: float, right: str, expiry: str) -> Optional[Contract]:
        contract = Option(
            self.symbol, expiry, strike, right,
            'SMART', currency='USD', tradingClass=self.symbol
        )
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            return None
        return contract

    @staticmethod
    def get_next_monthly_expiry() -> str:
        """次の月次満期日（第3金曜日）"""
        today = datetime.now()
        first_day = today.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)

        if third_friday <= today:
            next_month = today.replace(day=28) + timedelta(days=4)
            first_day = next_month.replace(day=1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            third_friday = first_friday + timedelta(weeks=2)

        return third_friday.strftime('%Y%m%d')

    @staticmethod
    def get_atm_strike(price: float) -> float:
        return round(price)


class MarketData:
    """市場データ管理"""

    def __init__(self, ib: IB):
        self.ib = ib
        self._tickers: dict[Contract, Ticker] = {}

    def subscribe(self, contract: Contract) -> Ticker:
        ticker = self.ib.reqMktData(contract)
        self._tickers[contract] = ticker
        return ticker

    def unsubscribe(self, contract: Contract):
        if contract in self._tickers:
            self.ib.cancelMktData(contract)
            del self._tickers[contract]

    def unsubscribe_all(self):
        for contract in list(self._tickers.keys()):
            self.unsubscribe(contract)

    @staticmethod
    def get_iv(ticker: Ticker) -> Optional[float]:
        if ticker.modelGreeks and ticker.modelGreeks.impliedVol:
            return ticker.modelGreeks.impliedVol * 100
        return None


class OrderManager:
    """注文管理"""

    def __init__(self, ib: IB, config: Config):
        self.ib = ib
        self.config = config

    def calc_qty(self, price: float) -> int:
        if price <= 0:
            return 0
        cost_per_contract = price * 100
        qty = int(self.config.budget_usd / cost_per_contract)
        return max(1, qty)

    def place_limit_order(
        self,
        contract: Contract,
        action: str,
        qty: int,
        price: float
    ) -> Optional[float]:
        """指値注文、リトライあり"""
        max_attempts = self.config.order_timeout_sec // self.config.order_retry_interval

        order = LimitOrder(action, qty, round(price, 2))
        trade = self.ib.placeOrder(contract, order)
        print(f"注文発行: {action} {qty}枚 @ {price:.2f}")

        for attempt in range(max_attempts):
            self.ib.sleep(self.config.order_retry_interval)

            if trade.isDone():
                if trade.orderStatus.status == 'Filled':
                    avg_price = trade.orderStatus.avgFillPrice
                    print(f"約定完了: {qty}枚 @ {avg_price}")
                    return avg_price
                else:
                    print(f"注文失敗: {trade.orderStatus.status}")
                    return None

            # キャンセル＆再発注
            self.ib.cancelOrder(order)
            self.ib.sleep(1)

            new_price = self._get_updated_price(contract, action, price)
            order = LimitOrder(action, qty, round(new_price, 2))
            trade = self.ib.placeOrder(contract, order)
            print(f"再発注 #{attempt+2}: {action} {qty}枚 @ {new_price:.2f}")

        self.ib.cancelOrder(order)
        print("注文タイムアウト")
        return None

    def _get_updated_price(
        self,
        contract: Contract,
        action: str,
        fallback: float
    ) -> float:
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)
        self.ib.cancelMktData(contract)

        if action == 'BUY':
            return ticker.ask if ticker.ask > 0 else fallback * 1.01
        else:
            return ticker.bid if ticker.bid > 0 else fallback * 0.99


class SignalDetector:
    """シグナル検出"""

    def __init__(self, config: Config):
        self.config = config

    def check_entry(self, change_pct: float, iv: Optional[float]) -> Optional[str]:
        """エントリーシグナル判定（順張り）"""
        if iv is None or iv >= self.config.entry_iv_max:
            return None

        if change_pct >= self.config.entry_change_pct:
            return 'C'
        elif change_pct <= -self.config.entry_change_pct:
            return 'P'

        return None

    def check_exit(self, pnl_pct: float, half_sold: bool) -> Optional[str]:
        """出口シグナル判定"""
        if pnl_pct >= self.config.take_profit_full:
            return 'full_profit'
        elif pnl_pct >= self.config.take_profit_half and not half_sold:
            return 'half_profit'
        elif pnl_pct <= self.config.stop_loss:
            return 'stop_loss'
        return None


class AutoTrader:
    """自動売買メインクラス"""

    def __init__(self, config: Config):
        self.config = config
        self.conn = IBConnection(config)
        self.contracts = ContractFactory(self.conn.ib, config.symbol)
        self.market_data = MarketData(self.conn.ib)
        self.orders = OrderManager(self.conn.ib, config)
        self.signals = SignalDetector(config)

        self.qqq_ticker: Optional[Ticker] = None
        self.close_price: float = 0.0
        self.position: Optional[Position] = None

    def start(self):
        """初期化"""
        self.conn.connect()

        stock = self.contracts.create_stock()
        self.qqq_ticker = self.market_data.subscribe(stock)
        self.conn.sleep(3)

        self.close_price = self.qqq_ticker.close or self.qqq_ticker.last
        print(f"QQQ監視開始 - 終値: {self.close_price}")
        self._print_config()

    def _print_config(self):
        print("\n" + "=" * 50)
        print("自動売買開始")
        print(f"エントリー: ±{self.config.entry_change_pct}% & IV<{self.config.entry_iv_max}%")
        print(f"利確: +{self.config.take_profit_half}%半分 / +{self.config.take_profit_full}%全量")
        print(f"損切り: {self.config.stop_loss}%")
        print(f"予算: ${self.config.budget_usd:,.0f}")
        print("=" * 50 + "\n")

    def entry(self, right: str) -> bool:
        """エントリー"""
        price = self.qqq_ticker.last
        strike = self.contracts.get_atm_strike(price)
        expiry = self.contracts.get_next_monthly_expiry()

        contract = self.contracts.create_option(strike, right, expiry)
        if not contract:
            print(f"契約取得失敗: {strike} {right} {expiry}")
            return False

        print(f"オプション契約: {contract.localSymbol}")

        ticker = self.market_data.subscribe(contract)
        self.conn.sleep(2)

        ask_price = ticker.ask
        if not ask_price or ask_price <= 0:
            print("Ask価格取得失敗")
            self.market_data.unsubscribe(contract)
            return False

        qty = self.orders.calc_qty(ask_price)
        direction = "CALL" if right == 'C' else "PUT"
        print(f"エントリー: {direction} {qty}枚 Ask:{ask_price}")

        fill_price = self.orders.place_limit_order(contract, 'BUY', qty, ask_price)

        if fill_price:
            self.position = Position(contract, fill_price, qty, ticker)
            print(f"ポジション確立: {contract.localSymbol} {qty}枚 @ {fill_price}")
            return True
        else:
            self.market_data.unsubscribe(contract)
            return False

    def exit(self, qty: int, reason: str) -> bool:
        """決済"""
        if not self.position:
            return False

        ticker = self.position.ticker
        bid_price = ticker.bid if ticker.bid > 0 else ticker.last * 0.98

        print(f"{reason}: {qty}枚売却 Bid:{bid_price:.2f}")

        fill_price = self.orders.place_limit_order(
            self.position.contract, 'SELL', qty, bid_price
        )

        if fill_price:
            self.position.qty -= qty
            if self.position.qty <= 0:
                self.market_data.unsubscribe(self.position.contract)
                self.position = None
                print("ポジション解消完了")
            return True
        return False

    def run(self):
        """メインループ"""
        self.start()

        while True:
            self.conn.sleep(1)

            price = self.qqq_ticker.last
            if not price or price <= 0:
                continue

            change_pct = (price / self.close_price - 1) * 100
            now = datetime.now().strftime('%H:%M:%S')

            if self.position is None:
                self._handle_no_position(now, price, change_pct)
            else:
                self._handle_position(now, price, change_pct)

    def _handle_no_position(self, now: str, price: float, change_pct: float):
        """ポジションなし時の処理"""
        # ATM CALLのIVを取得
        strike = self.contracts.get_atm_strike(price)
        expiry = self.contracts.get_next_monthly_expiry()
        temp_contract = self.contracts.create_option(strike, 'C', expiry)

        iv = None
        if temp_contract:
            temp_ticker = self.market_data.subscribe(temp_contract)
            self.conn.sleep(2)
            iv = self.market_data.get_iv(temp_ticker)
            self.market_data.unsubscribe(temp_contract)

        status = f"{now} QQQ:{price:.2f} ({change_pct:+.2f}%)"
        if iv:
            status += f" IV:{iv:.1f}%"
        print(status)

        signal = self.signals.check_entry(change_pct, iv)
        if signal:
            direction = "CALL" if signal == 'C' else "PUT"
            print(f"\n>>> {direction} ENTRY SIGNAL <<<")
            self.entry(signal)
            print()

    def _handle_position(self, now: str, price: float, change_pct: float):
        """ポジションあり時の処理"""
        opt_price = self.position.ticker.last
        if not opt_price or opt_price <= 0:
            return

        pnl_pct = self.position.calc_pnl_pct(opt_price)

        print(f"{now} QQQ:{price:.2f} ({change_pct:+.2f}%) | "
              f"{self.position.direction}:{opt_price:.2f} P&L:{pnl_pct:+.1f}%")

        exit_signal = self.signals.check_exit(pnl_pct, self.position.half_sold)

        if exit_signal == 'full_profit':
            print(f"\n>>> +{self.config.take_profit_full}% 全量利確 <<<")
            self.exit(self.position.qty, "利確")
            print()

        elif exit_signal == 'half_profit':
            print(f"\n>>> +{self.config.take_profit_half}% 半分利確 <<<")
            half_qty = self.position.qty // 2
            if half_qty > 0:
                self.exit(half_qty, "半利確")
                self.position.half_sold = True
            print()

        elif exit_signal == 'stop_loss':
            print(f"\n>>> {self.config.stop_loss}% 損切り <<<")
            self.exit(self.position.qty, "損切り")
            print()

    def shutdown(self):
        """終了処理"""
        if self.position:
            print("緊急決済中...")
            self.exit(self.position.qty, "シャットダウン")
        self.market_data.unsubscribe_all()
        self.conn.disconnect()


def main():
    config = Config()
    trader = AutoTrader(config)

    try:
        trader.run()
    except KeyboardInterrupt:
        print("\n終了シグナル受信")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        trader.shutdown()


if __name__ == '__main__':
    main()
