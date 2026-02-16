#!/usr/bin/env python3
"""CALL成行売りスクリプト"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
from datetime import datetime
from ib_insync import IB, Option, MarketOrder

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='CALL成行売り')
    parser.add_argument('--strike', type=float, required=True, help='ストライク価格')
    parser.add_argument('--qty', type=int, required=True, help='枚数')
    parser.add_argument('--expiry', type=str, default=None, help='満期日 (YYYYMMDD)、省略時は当日')
    args = parser.parse_args()

    # 満期日（省略時は当日）
    if args.expiry:
        expiry = args.expiry
    else:
        expiry = datetime.now().strftime('%Y%m%d')

    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=72)
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    # CALL契約
    call = Option('QQQ', expiry, args.strike, 'C', 'SMART')
    ib.qualifyContracts(call)

    log(f"CALL ${args.strike} ({expiry}) x {args.qty}枚 成行売り")

    # 成行売り
    order = MarketOrder('SELL', args.qty)
    trade = ib.placeOrder(call, order)

    for _ in range(20):
        ib.sleep(1)
        if trade.orderStatus.status == 'Filled':
            avg = trade.orderStatus.avgFillPrice
            total = avg * args.qty * 100
            log(f"約定: {args.qty}枚 @ ${avg:.2f}")
            log(f"合計: ${total:,.0f}")
            break
        elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
            log(f"失敗: {trade.orderStatus.status}")
            break
        log(f"待機中... {trade.orderStatus.status}")
    else:
        log(f"タイムアウト: {trade.orderStatus.status}")

    ib.disconnect()

if __name__ == '__main__':
    main()
