#!/usr/bin/env python3
"""PUT $625 監視スクリプト
利確: PUT >= $21.28 または QQQ <= $627
損切: PUT <= $14.80 または QQQ >= $648
期限: 2026/02/05 強制手仕舞い
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

from datetime import datetime, date
from ib_insync import IB, Stock, Option, MarketOrder

# 設定
TAKE_PROFIT_PUT = 21.28   # 利確: PUT価格
TAKE_PROFIT_QQQ = 627.0   # 利確: QQQ価格
STOP_LOSS_PUT = 14.80     # 損切: PUT価格
STOP_LOSS_QQQ = 648.0     # 損切: QQQ価格
DEADLINE = date(2026, 2, 5)  # 強制手仕舞い日

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def main():
    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=120)
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    # QQQ購読
    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)
    qqq_ticker = ib.reqMktData(qqq)

    # PUT $625 購読
    put = Option('QQQ', '20260430', 625, 'P', 'SMART')
    ib.qualifyContracts(put)
    put_ticker = ib.reqMktData(put)

    ib.sleep(2)

    log("=" * 60)
    log("PUT $625 監視開始")
    log(f"利確: PUT >= ${TAKE_PROFIT_PUT} または QQQ <= ${TAKE_PROFIT_QQQ}")
    log(f"損切: PUT <= ${STOP_LOSS_PUT} または QQQ >= ${STOP_LOSS_QQQ}")
    log(f"期限: {DEADLINE} 強制手仕舞い")
    log("=" * 60)

    try:
        while True:
            ib.sleep(1)

            qqq_price = qqq_ticker.last if qqq_ticker.last and qqq_ticker.last > 0 else None
            put_bid = put_ticker.bid if put_ticker.bid and put_ticker.bid > 0 else None

            if qqq_price is None or put_bid is None:
                continue

            today = date.today()
            reason = None

            # 期限チェック
            if today >= DEADLINE:
                reason = f"期限到達 ({DEADLINE})"

            # 利確チェック
            elif put_bid >= TAKE_PROFIT_PUT:
                reason = f"利確 (PUT ${put_bid:.2f} >= ${TAKE_PROFIT_PUT})"
            elif qqq_price <= TAKE_PROFIT_QQQ:
                reason = f"利確 (QQQ ${qqq_price:.2f} <= ${TAKE_PROFIT_QQQ})"

            # 損切チェック
            elif put_bid <= STOP_LOSS_PUT:
                reason = f"損切 (PUT ${put_bid:.2f} <= ${STOP_LOSS_PUT})"
            elif qqq_price >= STOP_LOSS_QQQ:
                reason = f"損切 (QQQ ${qqq_price:.2f} >= ${STOP_LOSS_QQQ})"

            if reason:
                log(f"発動: {reason}")

                # ポジション確認して売却
                positions = ib.positions()
                for pos in positions:
                    c = pos.contract
                    if (c.symbol == 'QQQ' and c.secType == 'OPT' and
                        c.right == 'P' and c.strike == 625.0 and
                        '20260430' in c.lastTradeDateOrContractMonth and
                        pos.position > 0):

                        qty = int(pos.position)
                        c.exchange = 'SMART'
                        ib.qualifyContracts(c)

                        log(f"売却: {c.localSymbol} x {qty}枚")
                        order = MarketOrder('SELL', qty)
                        trade = ib.placeOrder(c, order)

                        for _ in range(30):
                            ib.sleep(1)
                            if trade.orderStatus.status == 'Filled':
                                log(f"約定: {qty}枚 @ ${trade.orderStatus.avgFillPrice:.2f}")
                                pnl = (trade.orderStatus.avgFillPrice - 18.50) * qty * 100
                                log(f"損益: ${pnl:.2f} (¥{pnl * 156:,.0f})")
                                break
                            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                                log(f"失敗: {trade.orderStatus.status}")
                                break

                log("完了")
                break
            else:
                log(f"監視中 | QQQ: ${qqq_price:.2f} | PUT Bid: ${put_bid:.2f}")
                ib.sleep(4)

    except KeyboardInterrupt:
        log("\n中断 (Ctrl+C)")
    finally:
        ib.disconnect()
        log("切断完了")

if __name__ == '__main__':
    main()
