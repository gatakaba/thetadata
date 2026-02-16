#!/usr/bin/env python3
"""自動損切りスクリプト
- QQQが指定価格以下で全CALLポジション成行売り
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
from datetime import datetime
from ib_insync import IB, Stock, MarketOrder

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='自動損切り')
    parser.add_argument('--stop', type=float, required=True, help='損切りライン（QQQ価格）')
    args = parser.parse_args()

    STOP_PRICE = args.stop

    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=110)
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    # QQQ購読
    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)
    qqq_ticker = ib.reqMktData(qqq)
    ib.sleep(2)

    log("=" * 60)
    log(f"自動損切り監視開始")
    log(f"損切りライン: QQQ <= ${STOP_PRICE}")
    log("=" * 60)

    try:
        while True:
            ib.sleep(0.5)
            qqq_price = qqq_ticker.last if qqq_ticker.last and qqq_ticker.last > 0 else None
            if qqq_price is None:
                continue

            if qqq_price <= STOP_PRICE:
                log(f"損切り発動! QQQ ${qqq_price:.2f} <= ${STOP_PRICE}")
                
                # 全CALLポジション取得
                positions = ib.positions()
                for pos in positions:
                    c = pos.contract
                    if c.symbol == 'QQQ' and c.secType == 'OPT' and c.right == 'C' and pos.position > 0:
                        qty = int(pos.position)
                        c.exchange = 'SMART'
                        ib.qualifyContracts(c)
                        
                        log(f"売却: {c.localSymbol} x {qty}枚")
                        order = MarketOrder('SELL', qty)
                        trade = ib.placeOrder(c, order)
                        
                        for _ in range(20):
                            ib.sleep(1)
                            if trade.orderStatus.status == 'Filled':
                                log(f"  約定: {qty}枚 @ ${trade.orderStatus.avgFillPrice:.2f}")
                                break
                            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                                log(f"  失敗: {trade.orderStatus.status}")
                                break
                
                log("損切り完了")
                break
            else:
                log(f"監視中... QQQ: ${qqq_price:.2f} (ストップ: <=${STOP_PRICE})")
                ib.sleep(0.5)

    except KeyboardInterrupt:
        log("\n中断 (Ctrl+C)")
    finally:
        ib.disconnect()
        log("切断完了")

if __name__ == '__main__':
    main()
