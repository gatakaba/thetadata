"""$624 売却トリガー監視スクリプト
QQQ < $624 でPUTポジションを全量売却
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

from datetime import datetime
from ib_insync import IB, Stock, MarketOrder

def log(msg):
    print(msg, flush=True)

def main():
    trigger_price = 624.0
    check_interval = 3  # 3秒ごとにチェック

    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=225)
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)

    log("=" * 50)
    log(f"🎯 $624 売却トリガー監視開始")
    log(f"条件: QQQ < ${trigger_price:.2f} でPUT全量売却")
    log(f"間隔: {check_interval}秒")
    log("=" * 50)

    try:
        while True:
            # 現在価格取得
            tickers = ib.reqTickers(qqq)
            if not tickers:
                log("価格取得失敗、待機中...")
                ib.sleep(2)
                continue

            ticker = tickers[0]
            price = ticker.last if ticker.last > 0 else (ticker.close if ticker.close > 0 else None)

            if price is None:
                if ticker.bid > 0 and ticker.ask > 0:
                    price = (ticker.bid + ticker.ask) / 2
                else:
                    log("価格データ待機中...")
                    ib.sleep(check_interval)
                    continue

            now = datetime.now().strftime('%H:%M:%S')

            # 距離表示
            diff = price - trigger_price
            status = "待機中" if diff >= 0 else "!!! トリガー !!!"

            log(f"{now} | QQQ: ${price:.2f} | 距離: ${diff:+.2f} | {status}")

            # 判定: $624を下回ったら売る
            if price < trigger_price:
                log("")
                log("=" * 50)
                log(f"🚨 QQQ ${trigger_price} 下回り! 売却実行 🚨")
                log("=" * 50)

                positions = ib.positions()
                executed = False

                for pos in positions:
                    # QQQのPUTオプションを探す
                    if pos.contract.symbol == 'QQQ' and pos.contract.secType == 'OPT':
                        if pos.contract.right == 'P' and pos.position > 0:
                            contract = pos.contract
                            contract.exchange = 'SMART'
                            qty = int(pos.position)

                            log(f"売却対象: {contract.localSymbol} x {qty}枚")

                            # 成行売り（確実な約定優先）
                            order = MarketOrder('SELL', qty)
                            trade = ib.placeOrder(contract, order)

                            # 約定確認ループ
                            log("注文送信中...")
                            for _ in range(15):
                                ib.sleep(1)
                                status = trade.orderStatus.status
                                if status == 'Filled':
                                    log(f"✅ 約定完了: {qty}枚 @ ${trade.orderStatus.avgFillPrice:.2f}")
                                    executed = True
                                    break
                                elif status in ['Cancelled', 'Inactive']:
                                    log(f"❌ 注文無効: {status}")
                                    break

                            if not executed:
                                log(f"⚠️ 注文ステータス: {trade.orderStatus.status}")

                if executed:
                    log("全ポジション売却完了。停止します。")
                    break
                else:
                    log("売却対象のPUTポジションが見つかりませんでした。")
                    break

            ib.sleep(check_interval)

    except KeyboardInterrupt:
        log("\n監視停止 (Ctrl+C)")
    except Exception as e:
        log(f"\n予期せぬエラー: {e}")
    finally:
        ib.disconnect()

if __name__ == '__main__':
    main()
