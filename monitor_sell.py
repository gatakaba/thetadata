"""
QQQ監視・自動売却スクリプト
$621超えでCALLポジションを売却
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
from datetime import datetime
from ib_insync import IB, Stock, LimitOrder

# バッファなし出力
def log(msg):
    print(msg, flush=True)

def main():
    target_price = 621.0
    check_interval = 300  # 5分

    ib = IB()
    ib.connect('172.24.32.1', 4001, clientId=88)

    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)

    log("=" * 50)
    log(f"監視開始: QQQ > ${target_price} で自動売却")
    log("5分ごとにチェック")
    log("=" * 50)

    try:
        while True:
            ticker = ib.reqMktData(qqq)
            ib.sleep(2)
            price = ticker.last or ticker.close
            ib.cancelMktData(qqq)

            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log(f"{now} | QQQ: ${price:.2f}")

            if price > target_price:
                log("")
                log("=" * 50)
                log(f">>> QQQ ${target_price} 超え! 売却実行 <<<")
                log("=" * 50)

                # ポジション取得して売却
                positions = ib.positions()
                for pos in positions:
                    if pos.contract.symbol == 'QQQ' and pos.contract.secType == 'OPT' and pos.position > 0:
                        contract = pos.contract
                        contract.exchange = 'SMART'
                        qty = int(pos.position)

                        # 現在価格取得
                        opt_ticker = ib.reqMktData(contract)
                        ib.sleep(2)
                        bid = opt_ticker.bid if opt_ticker.bid > 0 else 0
                        ib.cancelMktData(contract)

                        log(f"ポジション: {contract.localSymbol} x {qty}")
                        log(f"Bid: ${bid:.2f}")

                        # 売却
                        if bid > 0:
                            order = LimitOrder('SELL', qty, round(bid, 2))
                            trade = ib.placeOrder(contract, order)
                            log(f"売り注文: {qty}枚 @ ${bid:.2f}")
                            ib.sleep(10)
                            log(f"Status: {trade.orderStatus.status}")

                log("監視終了")
                break

            # 5分待機
            ib.sleep(check_interval)

    finally:
        ib.disconnect()

if __name__ == '__main__':
    main()
