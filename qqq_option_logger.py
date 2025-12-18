import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import csv
from datetime import datetime
from ib_insync import IB, Stock, Option

ib = IB()
ib.connect('172.24.32.1', 4001, clientId=1)

qqq = Stock('QQQ', 'SMART', 'USD')
ib.qualifyContracts(qqq)
qqq_ticker = ib.reqMktData(qqq)

strike = 610
expiries = ['20260130']

options = {}
for expiry in expiries:
    call = Option('QQQ', expiry, strike, 'C', 'SMART', currency='USD', tradingClass='QQQ')
    put = Option('QQQ', expiry, strike, 'P', 'SMART', currency='USD', tradingClass='QQQ')
    ib.qualifyContracts(call, put)
    options[expiry] = {
        'call': ib.reqMktData(call),
        'put': ib.reqMktData(put)
    }

filename = f'qqq_options_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'qqq_last',
        'expiry', 'type', 'bid', 'ask', 'last', 'delta', 'gamma', 'theta', 'vega', 'iv'
    ])

stop_time = datetime.now().replace(hour=6, minute=15, second=0, microsecond=0)
if datetime.now().hour >= 6:
    stop_time = stop_time.replace(day=stop_time.day + 1)

print(f'ロギング開始: {filename}')
print(f'自動停止: {stop_time.strftime("%Y-%m-%d %H:%M:%S")}')

while datetime.now() < stop_time:
    ib.sleep(1)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    qqq_last = qqq_ticker.last

    rows = []
    for expiry, opt in options.items():
        for opt_type, ticker in [('CALL', opt['call']), ('PUT', opt['put'])]:
            g = ticker.modelGreeks
            rows.append([
                now, qqq_last, expiry, opt_type,
                ticker.bid, ticker.ask, ticker.last,
                g.delta if g else '', g.gamma if g else '',
                g.theta if g else '', g.vega if g else '',
                f'{g.impliedVol:.4f}' if g and g.impliedVol else ''
            ])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'{now} QQQ:{qqq_last:.2f} 記録完了')

print('ロギング終了')
ib.disconnect()
