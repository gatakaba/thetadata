import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import csv
from datetime import datetime, timedelta
from ib_insync import IB, Stock, Option

ib = IB()
ib.connect('172.24.32.1', 4001, clientId=1)

qqq = Stock('QQQ', 'SMART', 'USD')
ib.qualifyContracts(qqq)
qqq_ticker = ib.reqMktData(qqq)
ib.sleep(2)

current_price = qqq_ticker.last
print(f'QQQ現在値: {current_price}')

# 行使価格: ATM周辺5本
atm = round(current_price / 5) * 5
strikes = [atm - 10, atm - 5, atm, atm + 5, atm + 10]

# 満期: 約1週間後、1ヶ月後、3ヶ月後
expiries = ['20251226', '20260130', '20260320']

print(f'行使価格: {strikes}')
print(f'満期: {expiries}')
print(f'契約数: {len(strikes) * len(expiries) * 2}')

options = {}
for expiry in expiries:
    for strike in strikes:
        for right in ['C', 'P']:
            opt = Option('QQQ', expiry, strike, right, 'SMART', currency='USD', tradingClass='QQQ')
            try:
                ib.qualifyContracts(opt)
                key = f'{expiry}_{strike}_{right}'
                options[key] = {
                    'expiry': expiry,
                    'strike': strike,
                    'right': 'CALL' if right == 'C' else 'PUT',
                    'ticker': ib.reqMktData(opt)
                }
            except:
                print(f'スキップ: {expiry} {strike} {right}')

print(f'購読成功: {len(options)}契約')

filename = f'qqq_options_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'qqq_last',
        'expiry', 'strike', 'type', 'bid', 'ask', 'last',
        'bidSize', 'askSize', 'volume',
        'delta', 'gamma', 'theta', 'vega', 'iv', 'undPrice'
    ])

stop_time = datetime.now().replace(hour=6, minute=15, second=0, microsecond=0)
if datetime.now().hour >= 6:
    stop_time = stop_time + timedelta(days=1)

print(f'ロギング開始: {filename}')
print(f'自動停止: {stop_time.strftime("%Y-%m-%d %H:%M:%S")}')

while datetime.now() < stop_time:
    ib.sleep(1)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    qqq_last = qqq_ticker.last

    rows = []
    for key, opt in options.items():
        ticker = opt['ticker']
        g = ticker.modelGreeks
        rows.append([
            now, qqq_last, opt['expiry'], opt['strike'], opt['right'],
            ticker.bid, ticker.ask, ticker.last,
            ticker.bidSize, ticker.askSize, ticker.volume,
            g.delta if g else '', g.gamma if g else '',
            g.theta if g else '', g.vega if g else '',
            f'{g.impliedVol:.4f}' if g and g.impliedVol else '',
            g.undPrice if g else ''
        ])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'{now} QQQ:{qqq_last:.2f} {len(rows)}件記録')

print('ロギング終了')
ib.disconnect()
