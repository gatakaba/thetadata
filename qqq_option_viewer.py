#!/usr/bin/env python3
"""QQQ Viewer - リアルタイム価格ビューア"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
import os
from datetime import datetime, timedelta
from ib_insync import IB, Stock, Option

def get_next_friday():
    """直近の金曜日を取得"""
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    return next_friday.strftime('%Y%m%d')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ANSIカラーコード
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def arrow_indicator(value, thresholds):
    """値を矢印で表現（閾値リスト: [弱, 中, 強]）"""
    abs_val = abs(value)
    if abs_val < thresholds[0]:
        return '→'
    elif abs_val < thresholds[1]:
        arrow = '↑' if value > 0 else '↓'
        color = GREEN if value > 0 else RED
        return f'{color}{arrow}{RESET}'
    elif abs_val < thresholds[2]:
        arrow = '↑↑' if value > 0 else '↓↓'
        color = GREEN if value > 0 else RED
        return f'{color}{arrow}{RESET}'
    else:
        arrow = '↑↑↑' if value > 0 else '↓↓↓'
        color = GREEN if value > 0 else RED
        return f'{color}{arrow}{RESET}'

def main():
    parser = argparse.ArgumentParser(description='QQQ Realtime Viewer')
    parser.add_argument('--host', default='172.24.32.1', help='IB Gateway host')
    parser.add_argument('--port', type=int, default=4001, help='IB Gateway port')
    parser.add_argument('--interval', type=float, default=0.5, help='更新間隔(秒)')
    args = parser.parse_args()

    ib = IB()
    print(f'Connecting to {args.host}:{args.port}...')
    ib.connect(args.host, args.port, clientId=11)

    # QQQ現在値取得
    qqq = Stock('QQQ', 'SMART', 'USD')
    ib.qualifyContracts(qqq)
    qqq_ticker = ib.reqMktData(qqq)
    ib.sleep(2)

    # ATMオプション取得（$1刻み、動的更新）
    current_price = qqq_ticker.last
    atm_strike = round(current_price)  # $1刻み
    expiry = get_next_friday()

    def subscribe_atm(strike):
        """ATMオプションを購読"""
        call_opt = Option('QQQ', expiry, strike, 'C', 'SMART', currency='USD', tradingClass='QQQ')
        put_opt = Option('QQQ', expiry, strike, 'P', 'SMART', currency='USD', tradingClass='QQQ')
        try:
            ib.qualifyContracts(call_opt, put_opt)
            c_ticker = ib.reqMktData(call_opt)
            p_ticker = ib.reqMktData(put_opt)
            return c_ticker, p_ticker
        except:
            return None, None

    call_ticker, put_ticker = subscribe_atm(atm_strike)
    print(f'ATM Options: {expiry} ${atm_strike}')

    # ポジション取得・購読
    positions = ib.positions()
    pos_tickers = {}
    for pos in positions:
        c = pos.contract
        if c.secType == 'OPT' and c.symbol == 'QQQ':
            c.exchange = 'SMART'
            ib.qualifyContracts(c)
            pos_tickers[c.conId] = {
                'contract': c,
                'ticker': ib.reqMktData(c),
                'qty': int(pos.position),
                'avg_cost': pos.avgCost / 100  # 1株あたりに変換
            }

    # 口座情報取得用
    account_info = {'net_liq': None, 'available': None}
    for av in ib.accountSummary():
        if av.tag == 'NetLiquidation':
            account_info['net_liq'] = f"{float(av.value):,.0f} {av.currency}"
        if av.tag == 'AvailableFunds':
            account_info['available'] = f"{float(av.value):,.0f} {av.currency}"

    print('Starting viewer... (Ctrl+C to quit)\n')
    ib.sleep(1)

    # EMA用変数
    ema_d1 = 0.0
    ema_d2 = 0.0
    prev_price = qqq_ticker.last
    prev_d1 = 0.0
    prev_time = datetime.now()
    alpha = 2 / (10 / args.interval + 1)  # 10秒相当のEMA係数

    # 移動平均用変数
    ema_20s = qqq_ticker.last  # 20秒MA
    ema_60s = qqq_ticker.last  # 60秒MA
    alpha_20s = 2 / (20 / args.interval + 1)
    alpha_60s = 2 / (60 / args.interval + 1)

    try:
        while True:
            clear_screen()
            now = datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            qqq_last = qqq_ticker.last

            # 移動平均更新
            ema_20s = alpha_20s * qqq_last + (1 - alpha_20s) * ema_20s
            ema_60s = alpha_60s * qqq_last + (1 - alpha_60s) * ema_60s

            dt = (now - prev_time).total_seconds()
            if dt > 0:
                # 瞬間1階微分
                instant_d1 = (qqq_last - prev_price) / dt
                # EMA適用
                ema_d1 = alpha * instant_d1 + (1 - alpha) * ema_d1

                # 瞬間2階微分
                instant_d2 = (ema_d1 - prev_d1) / dt
                # EMA適用
                ema_d2 = alpha * instant_d2 + (1 - alpha) * ema_d2

            prev_price = qqq_last
            prev_d1 = ema_d1
            prev_time = now

            # d1: [0.025, 0.15, 0.30], d2: [0.0025, 0.015, 0.030]
            d1_arrow = arrow_indicator(ema_d1, [0.025, 0.15, 0.30])
            d2_arrow = arrow_indicator(ema_d2, [0.0025, 0.015, 0.030])

            # 移動平均との差分（閾値0.1で色付け）
            diff_20s = qqq_last - ema_20s
            diff_60s = qqq_last - ema_60s

            if abs(diff_20s) < 0.1:
                diff_20s_str = f'{diff_20s:+.2f}'
            elif diff_20s > 0:
                diff_20s_str = f'{GREEN}+{diff_20s:.2f}{RESET}'
            else:
                diff_20s_str = f'{RED}{diff_20s:.2f}{RESET}'

            if abs(diff_60s) < 0.1:
                diff_60s_str = f'{diff_60s:+.2f}'
            elif diff_60s > 0:
                diff_60s_str = f'{GREEN}+{diff_60s:.2f}{RESET}'
            else:
                diff_60s_str = f'{RED}{diff_60s:.2f}{RESET}'

            print(f'{BOLD}{CYAN}QQQ  {qqq_last:.2f}{RESET}')
            print(f'd1: {d1_arrow}  d2: {d2_arrow}')
            print()
            print(f'MA20: {ema_20s:.2f} ({diff_20s_str})  MA60: {ema_60s:.2f} ({diff_60s_str})')
            print(f'{now_str}')
            print()

            # ATMストライク動的更新（チャタリング防止: ATMから$1以上離れたら更新）
            if abs(qqq_last - atm_strike) >= 1.0:
                atm_strike = round(qqq_last)
                call_ticker, put_ticker = subscribe_atm(atm_strike)

            # ATMオプション表示
            if call_ticker and put_ticker:
                exp_str = f'{expiry[:4]}-{expiry[4:6]}-{expiry[6:]}'
                print(f'ATM Options ({exp_str} ${atm_strike})')
                print('-' * 40)

                c_bid = call_ticker.bid if call_ticker.bid and call_ticker.bid != -1 else '-'
                c_ask = call_ticker.ask if call_ticker.ask and call_ticker.ask != -1 else '-'
                p_bid = put_ticker.bid if put_ticker.bid and put_ticker.bid != -1 else '-'
                p_ask = put_ticker.ask if put_ticker.ask and put_ticker.ask != -1 else '-'

                if c_bid != '-':
                    c_bid = f'{c_bid:.2f}'
                if c_ask != '-':
                    c_ask = f'{c_ask:.2f}'
                if p_bid != '-':
                    p_bid = f'{p_bid:.2f}'
                if p_ask != '-':
                    p_ask = f'{p_ask:.2f}'

                print(f'CALL  Bid: {c_bid:>6}  Ask: {c_ask:>6}')
                print(f'PUT   Bid: {p_bid:>6}  Ask: {p_ask:>6}')

            # ポジション更新（新規ポジションを検出）
            current_positions = ib.positions()
            for pos in current_positions:
                c = pos.contract
                if c.secType == 'OPT' and c.symbol == 'QQQ':
                    if c.conId not in pos_tickers:
                        c.exchange = 'SMART'
                        ib.qualifyContracts(c)
                        pos_tickers[c.conId] = {
                            'contract': c,
                            'ticker': ib.reqMktData(c),
                            'qty': int(pos.position),
                            'avg_cost': pos.avgCost / 100
                        }
                    else:
                        # 数量更新
                        pos_tickers[c.conId]['qty'] = int(pos.position)
                        pos_tickers[c.conId]['avg_cost'] = pos.avgCost / 100

            # 決済されたポジションを削除
            current_conIds = {pos.contract.conId for pos in current_positions if pos.contract.secType == 'OPT'}
            for conId in list(pos_tickers.keys()):
                if conId not in current_conIds:
                    del pos_tickers[conId]

            # ポジション表示
            if pos_tickers:
                print()
                print(f'{BOLD}Positions{RESET}')
                print('-' * 50)
                total_pnl = 0
                for conId, data in pos_tickers.items():
                    c = data['contract']
                    t = data['ticker']
                    qty = data['qty']
                    avg_cost = data['avg_cost']

                    # 損益分岐点計算
                    if c.right == 'C':
                        breakeven = c.strike + avg_cost
                    else:
                        breakeven = c.strike - avg_cost

                    # 現在価格（mid）
                    if t.bid and t.bid > 0 and t.ask and t.ask > 0:
                        mid = (t.bid + t.ask) / 2
                        current_val = mid * qty * 100
                        cost_val = avg_cost * qty * 100
                        pnl = current_val - cost_val
                        total_pnl += pnl

                        pnl_color = GREEN if pnl >= 0 else RED
                        pnl_jpy = pnl * 156  # 固定レート
                        pnl_str = f'{pnl_color}¥{pnl_jpy:+,.0f}{RESET}'

                        # 損益分岐点までの距離
                        be_diff = qqq_last - breakeven
                        if c.right == 'C':
                            be_color = GREEN if be_diff > 0 else RED
                        else:
                            be_color = GREEN if be_diff < 0 else RED

                        exp_str = c.lastTradeDateOrContractMonth
                        print(f'  {c.right} ${c.strike:.0f} ({exp_str}) x {qty}枚')
                        print(f'    Avg: ${avg_cost:.2f} | Now: ${mid:.2f} | PnL: {pnl_str}')
                        print(f'    BE: ${breakeven:.2f} ({be_color}{be_diff:+.2f}{RESET})')
                    else:
                        exp_str = c.lastTradeDateOrContractMonth
                        print(f'  {c.right} ${c.strike:.0f} ({exp_str}) x {qty}枚')
                        print(f'    Avg: ${avg_cost:.2f} | Now: N/A | BE: ${breakeven:.2f}')

                if total_pnl != 0:
                    pnl_color = GREEN if total_pnl >= 0 else RED
                    total_pnl_jpy = total_pnl * 156
                    print(f'  {BOLD}Total PnL: {pnl_color}¥{total_pnl_jpy:+,.0f}{RESET}')

            # 口座情報表示
            print()
            print(f'{BOLD}Account{RESET}')
            print('-' * 50)
            if account_info['net_liq']:
                print(f'  Net Liq:   {account_info["net_liq"]}')
            if account_info['available']:
                print(f'  Available: {account_info["available"]}')

            ib.sleep(args.interval)

    except KeyboardInterrupt:
        print('\nDisconnecting...')
    finally:
        ib.disconnect()
        print('Done.')

if __name__ == '__main__':
    main()
