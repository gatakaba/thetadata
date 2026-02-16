"""アダプティブ指値売却スクリプト
- Ask × (1+割合) で指値、10秒ごとに下げる
- 10% → 0%、5分間、30ステップ
- QQQ > $623 で即成行に切り替え
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

from datetime import datetime
from ib_insync import IB, Stock, LimitOrder, MarketOrder

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def main():
    # パラメータ
    START_PREMIUM = 0.10  # 開始時 +10%
    END_PREMIUM = 0.00    # 終了時 +0% (=Ask)
    STEPS = 30            # ステップ数
    INTERVAL = 10         # 秒
    QQQ_STOP = 623.0      # QQQがこれを超えたら即成行

    ib = IB()
    try:
        ib.connect('172.24.32.1', 4001, clientId=300)
    except Exception as e:
        log(f"接続エラー: {e}")
        return

    # QQQ取得
    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)

    # ポジション取得
    positions = ib.positions()
    put_contract = None
    put_qty = 0

    for pos in positions:
        c = pos.contract
        if c.symbol == 'QQQ' and c.secType == 'OPT' and c.right == 'P' and pos.position > 0:
            put_contract = c
            put_contract.exchange = 'SMART'
            put_qty = int(pos.position)
            log(f"対象ポジション: {c.localSymbol} x {put_qty}枚")
            break

    if not put_contract:
        log("売却対象のPUTポジションが見つかりません")
        ib.disconnect()
        return

    # オプションの市場データ購読
    ib.qualifyContracts(put_contract)
    opt_ticker = ib.reqMktData(put_contract)
    qqq_ticker = ib.reqMktData(qqq)
    ib.sleep(2)

    log("=" * 60)
    log(f"アダプティブ売却開始")
    log(f"開始プレミアム: +{START_PREMIUM*100:.0f}%")
    log(f"終了プレミアム: +{END_PREMIUM*100:.0f}%")
    log(f"ステップ: {STEPS}回 x {INTERVAL}秒 = {STEPS*INTERVAL/60:.1f}分")
    log(f"QQQストップ: ${QQQ_STOP} 超えで即成行")
    log("=" * 60)

    # Ask価格が取得できるまで待機（市場オープン待ち）
    log("オプション市場オープン待機中...")
    while True:
        ib.sleep(1)
        # QQQストップチェック
        qqq_price = qqq_ticker.last if qqq_ticker.last and qqq_ticker.last > 0 else None
        if qqq_price is None and qqq_ticker.bid and qqq_ticker.ask:
            qqq_price = (qqq_ticker.bid + qqq_ticker.ask) / 2
        if qqq_price and qqq_price > QQQ_STOP:
            log(f"QQQ ${qqq_price:.2f} > ${QQQ_STOP} - 市場オープン前にストップ!")
            ib.disconnect()
            return
        # Ask価格チェック
        ask = opt_ticker.ask if opt_ticker.ask and opt_ticker.ask > 0 else None
        if ask:
            log(f"Ask取得: ${ask:.2f} - アダプティブ売却開始")
            break
        # 10秒ごとにステータス表示
        qqq_str = f"${qqq_price:.2f}" if qqq_price else "N/A"
        log(f"待機中... QQQ: {qqq_str} | Ask: 未取得")
        ib.sleep(9)  # 合計10秒待機

    current_order = None
    current_trade = None
    emergency_stop = False

    def do_market_sell(reason):
        """成行売り実行"""
        nonlocal current_order, current_trade, put_qty
        log("")
        log("!" * 60)
        log(f"🚨 {reason}")
        log("!" * 60)

        # 既存注文キャンセル
        if current_order:
            ib.cancelOrder(current_order)
            ib.sleep(1)

        # 成行売り
        order = MarketOrder('SELL', put_qty)
        trade = ib.placeOrder(put_contract, order)
        log(f"成行注文送信: {put_qty}枚")

        for _ in range(15):
            ib.sleep(1)
            if trade.orderStatus.status == 'Filled':
                log(f"✅ 約定完了: {put_qty}枚 @ ${trade.orderStatus.avgFillPrice:.2f}")
                return True
            elif trade.orderStatus.status in ['Cancelled', 'Inactive']:
                log(f"❌ 注文失敗: {trade.orderStatus.status}")
                return False
        log(f"⚠️ ステータス: {trade.orderStatus.status}")
        return False

    try:
        step = 0
        while step <= STEPS:
            # 現在のプレミアム計算
            premium = START_PREMIUM - (START_PREMIUM - END_PREMIUM) * step / STEPS

            # QQQ価格チェック
            qqq_price = qqq_ticker.last if qqq_ticker.last and qqq_ticker.last > 0 else None
            if qqq_price is None and qqq_ticker.bid and qqq_ticker.ask:
                qqq_price = (qqq_ticker.bid + qqq_ticker.ask) / 2

            if qqq_price and qqq_price > QQQ_STOP:
                do_market_sell(f"QQQ ${qqq_price:.2f} > ${QQQ_STOP} - 即成行に切り替え!")
                return

            # Ask価格取得
            ask = opt_ticker.ask if opt_ticker.ask and opt_ticker.ask > 0 else None
            bid = opt_ticker.bid if opt_ticker.bid and opt_ticker.bid > 0 else None

            if ask is None:
                log(f"Step {step}/{STEPS} | Ask一時的に取得不可、待機...")
                ib.sleep(1)
                # ステップを進めずにリトライ
                continue

            # 指値価格計算
            limit_price = round(ask * (1 + premium), 2)

            # 既存注文キャンセル
            if current_trade:
                # 約定済みチェック
                if current_trade.orderStatus.status == 'Filled':
                    filled_qty = current_trade.orderStatus.filled
                    avg_price = current_trade.orderStatus.avgFillPrice
                    log("")
                    log("=" * 60)
                    log(f"✅ 約定完了: {int(filled_qty)}枚 @ ${avg_price:.2f}")
                    log(f"合計: ${filled_qty * avg_price * 100:,.0f}")
                    log("=" * 60)
                    return

                # 部分約定チェック
                if current_trade.orderStatus.filled > 0:
                    filled = int(current_trade.orderStatus.filled)
                    remaining = put_qty - filled
                    log(f"部分約定: {filled}枚、残り{remaining}枚")
                    put_qty = remaining

                ib.cancelOrder(current_order)
                ib.sleep(0.5)

            # 新規指値注文
            current_order = LimitOrder('SELL', put_qty, limit_price)
            current_trade = ib.placeOrder(put_contract, current_order)

            qqq_str = f"${qqq_price:.2f}" if qqq_price else "N/A"
            bid_str = f"${bid:.2f}" if bid else "N/A"
            ask_str = f"${ask:.2f}" if ask else "N/A"
            log(f"Step {step:>2}/{STEPS} | QQQ: {qqq_str} | Bid: {bid_str} | Ask: {ask_str} | 指値: ${limit_price:.2f} (+{premium*100:.1f}%)")

            # 0.5秒ごとにQQQチェックしながら待機
            if step < STEPS:
                for _ in range(int(INTERVAL / 0.5)):
                    ib.sleep(0.5)
                    # QQQストップチェック
                    qqq_price = qqq_ticker.last if qqq_ticker.last and qqq_ticker.last > 0 else None
                    if qqq_price is None and qqq_ticker.bid and qqq_ticker.ask:
                        qqq_price = (qqq_ticker.bid + qqq_ticker.ask) / 2
                    if qqq_price and qqq_price > QQQ_STOP:
                        do_market_sell(f"待機中にQQQ ${qqq_price:.2f} > ${QQQ_STOP} 検出!")
                        emergency_stop = True
                        break
                if emergency_stop:
                    break

            step += 1

        # 緊急停止で抜けた場合は終了
        if emergency_stop:
            return

        # 最終ステップ後、約定確認
        log("")
        log("5分経過 - 最終確認中...")
        if current_trade:
            for _ in range(10):
                ib.sleep(1)
                if current_trade.orderStatus.status == 'Filled':
                    log(f"✅ 約定完了: {put_qty}枚 @ ${current_trade.orderStatus.avgFillPrice:.2f}")
                    return

        # まだ約定していなければ成行に切り替え
        log("指値で約定せず - 成行に切り替え")
        if current_order:
            ib.cancelOrder(current_order)
            ib.sleep(1)

        order = MarketOrder('SELL', put_qty)
        trade = ib.placeOrder(put_contract, order)

        for _ in range(15):
            ib.sleep(1)
            if trade.orderStatus.status == 'Filled':
                log(f"✅ 成行約定: {put_qty}枚 @ ${trade.orderStatus.avgFillPrice:.2f}")
                return

        log(f"⚠️ 最終ステータス: {trade.orderStatus.status}")

    except KeyboardInterrupt:
        log("\n中断 (Ctrl+C)")
        if current_order:
            ib.cancelOrder(current_order)
            log("注文キャンセル済み")
    except Exception as e:
        log(f"エラー: {e}")
    finally:
        ib.disconnect()
        log("切断完了")

if __name__ == '__main__':
    main()
