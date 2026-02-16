#!/usr/bin/env python3
"""レンジブレイクで売却するスクリプト
QQQが指定レンジから外れたら成行売り
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
from datetime import datetime
from ib_insync import IB, Stock, Option, MarketOrder

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='レンジブレイク売り')
    parser.add_argument('--lower', type=float, required=True, help='下限価格')
    parser.add_argument('--upper', type=float, required=True, help='上限価格')
    parser.add_argument('--strike', type=float, required=True, help='ストライク価格')
    parser.add_argument('--qty', type=int, required=True, help='売却枚数')
    parser.add_argument('--expiry', type=str, required=True, help='満期日 (YYYYMMDD)')
    parser.add_argument('--right', type=str, default='C', choices=['C', 'P'], help='C=CALL, P=PUT')
    args = parser.parse_args()

    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=111)
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    # QQQ購読
    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)
    qqq_ticker = ib.reqMktData(qqq)
    ib.sleep(2)

    log("=" * 60)
    log(f"レンジブレイク監視開始")
    log(f"レンジ: ${args.lower} - ${args.upper}")
    log(f"対象: {'CALL' if args.right == 'C' else 'PUT'} ${args.strike} ({args.expiry}) x {args.qty}枚")
    log(f"条件: QQQ < ${args.lower} または QQQ > ${args.upper} で成行売り")
    log("=" * 60)

    try:
        while True:
            ib.sleep(0.5)
            qqq_price = qqq_ticker.last if qqq_ticker.last and qqq_ticker.last > 0 else None
            if qqq_price is None:
                continue

            if qqq_price < args.lower:
                log(f"下限ブレイク! QQQ ${qqq_price:.2f} < ${args.lower}")
                execute_sell(ib, args)
                break
            elif qqq_price > args.upper:
                log(f"上限ブレイク! QQQ ${qqq_price:.2f} > ${args.upper}")
                execute_sell(ib, args)
                break
            else:
                log(f"監視中... QQQ: ${qqq_price:.2f} (レンジ: ${args.lower}-${args.upper})")
                ib.sleep(0.5)

    except KeyboardInterrupt:
        log("\n中断 (Ctrl+C)")
    finally:
        ib.disconnect()
        log("切断完了")


def execute_sell(ib, args):
    """成行売りを実行"""
    opt = Option('QQQ', args.expiry, args.strike, args.right, 'SMART')
    ib.qualifyContracts(opt)

    log(f"売却実行: {'CALL' if args.right == 'C' else 'PUT'} ${args.strike} x {args.qty}枚")
    order = MarketOrder('SELL', args.qty)
    trade = ib.placeOrder(opt, order)

    for _ in range(20):
        ib.sleep(1)
        if trade.orderStatus.status == 'Filled':
            avg = trade.orderStatus.avgFillPrice
            total = avg * args.qty * 100
            log(f"約定: {args.qty}枚 @ ${avg:.2f}")
            log(f"合計: ${total:,.0f}")
            return
        elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
            log(f"失敗: {trade.orderStatus.status}")
            return
        log(f"待機中... {trade.orderStatus.status}")

    log(f"タイムアウト: {trade.orderStatus.status}")


if __name__ == '__main__':
    main()
