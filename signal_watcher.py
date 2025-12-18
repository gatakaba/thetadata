import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import subprocess
from datetime import datetime
from ib_insync import IB, Stock, Option


def connect():
    ib = IB()
    ib.connect('172.24.32.1', 4001, clientId=10)
    return ib


def setup_tickers(ib):
    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)
    qqq_ticker = ib.reqMktData(qqq)

    opt = Option('QQQ', '20260130', 610, 'C', 'SMART', currency='USD', tradingClass='QQQ')
    ib.qualifyContracts(opt)
    opt_ticker = ib.reqMktData(opt)

    ib.sleep(3)
    return qqq_ticker, opt_ticker


def get_market_data(qqq_ticker, opt_ticker, close_price):
    price = qqq_ticker.last
    change_pct = (price / close_price - 1) * 100
    opt_price = opt_ticker.last

    iv = None
    if opt_ticker.modelGreeks and opt_ticker.modelGreeks.impliedVol:
        iv = opt_ticker.modelGreeks.impliedVol * 100

    return price, change_pct, opt_price, iv


def call_claude(prompt):
    subprocess.run(['claude', '-p', prompt])


def buy_option(opt_type, price, iv):
    prompt = f'''{opt_type}エントリーシグナル。現在値:{price:.2f} IV:{iv:.1f}%。
予算500万円でQQQ 1/30満期 ATM{opt_type}を買え。ib_insync使用。IP:172.24.32.1 ポート:4001。
手順: 1.Ask取得→指値注文 2.10秒待機 3.未約定ならキャンセル→再注文 4.全量約定まで繰り返す
完了したら /tmp/position.txt に entry_price,qty を書け'''
    call_claude(prompt)


def sell_option(opt_type, qty, pnl_pct, reason):
    prompt = f'''{reason}シグナル。P&L:{pnl_pct:.1f}%。
QQQ 1/30 {opt_type} を売却しろ。{qty}枚。
ib_insync使用。IP:172.24.32.1 ポート:4001。
手順: 1.Bid取得→指値注文 2.10秒待機 3.未約定ならキャンセル→再注文 4.全量約定まで繰り返す'''
    call_claude(prompt)


def read_position(opt_type):
    try:
        with open('/tmp/position.txt') as f:
            data = f.read().strip().split(',')
            return {'type': opt_type, 'entry_price': float(data[0]), 'qty': int(data[1])}
    except:
        return None


def check_entry_signal(iv, change_pct):
    # 順張り（トレンドフォロー）
    # CALL: 上昇トレンド + IV低め（安いプレミアム）
    if change_pct >= 2 and iv < 20:
        return 'CALL'
    # PUT: 下落トレンド + IV低め
    elif change_pct <= -2 and iv < 20:
        return 'PUT'
    return None


def check_exit_signal(pnl_pct, position):
    if pnl_pct >= 100:
        return 'full_profit'
    elif pnl_pct >= 50 and not position.get('half_sold'):
        return 'half_profit'
    elif pnl_pct <= -30:
        return 'stop_loss'
    return None


def main():
    ib = connect()
    qqq_ticker, opt_ticker = setup_tickers(ib)
    close_price = qqq_ticker.close
    position = None

    print(f'監視開始 - QQQ終値: {close_price}')
    print('エントリー（順張り）:')
    print('  CALL: +2%以上 & IV<20%（上昇トレンドに乗る）')
    print('  PUT:  -2%以下 & IV<20%（下落トレンドに乗る）')
    print('出口: +50%半分 / +100%全部 / -30%損切り')

    while True:
        ib.sleep(1)

        price, change_pct, opt_price, iv = get_market_data(qqq_ticker, opt_ticker, close_price)

        now = datetime.now().strftime('%H:%M:%S')
        print(f'{now} QQQ:{price:.2f} ({change_pct:+.2f}%) IV:{iv:.1f}% OPT:{opt_price}' if iv else f'{now} QQQ:{price:.2f}')

        if iv is None:
            continue

        # エントリー監視
        if position is None:
            signal = check_entry_signal(iv, change_pct)
            if signal:
                print(f'>>> {signal} ENTRY <<<')
                buy_option(signal, price, iv)
                ib.sleep(5)
                position = read_position(signal)
                if position:
                    print(f'ポジション: {position}')

        # 出口監視
        else:
            pnl_pct = (opt_price / position['entry_price'] - 1) * 100
            print(f'  P&L: {pnl_pct:+.1f}%')

            exit_signal = check_exit_signal(pnl_pct, position)

            if exit_signal == 'full_profit':
                print('>>> EXIT: +100% <<<')
                sell_option(position['type'], position['qty'], pnl_pct, '利確')
                position = None

            elif exit_signal == 'half_profit':
                print('>>> EXIT: +50% 半分 <<<')
                half_qty = position['qty'] // 2
                sell_option(position['type'], half_qty, pnl_pct, '半利確')
                position['qty'] -= half_qty
                position['half_sold'] = True

            elif exit_signal == 'stop_loss':
                print('>>> EXIT: -30% 損切り <<<')
                sell_option(position['type'], position['qty'], pnl_pct, '損切り')
                position = None


if __name__ == '__main__':
    main()
