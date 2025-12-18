# IB Trading Tools

QQQオプション取引ツール for Interactive Brokers

## セットアップ

```bash
uv sync
```

## ツール一覧

### check_position.py
ポジション・注文確認
```bash
uv run python check_position.py
```

### earnings_option_buyer.py
決算/経済指標後のオプション購入
```bash
# CALL購入（上昇予想）
uv run python earnings_option_buyer.py --call --budget 5000000

# PUT購入（下落予想）
uv run python earnings_option_buyer.py --put --budget 5000000

# dry-run（発注なし）
uv run python earnings_option_buyer.py --call --dry-run
```

### close_position.py
ポジションクローズ
```bash
uv run python close_position.py
uv run python close_position.py --dry-run
```

### auto_trailing_stop.py
トレーリングストップ自動売却
```bash
# デフォルト: 損切り-10%, トレーリング-0.5%
uv run python auto_trailing_stop.py

# カスタム設定
uv run python auto_trailing_stop.py --stop-loss 5 --trail 1

# dry-run
uv run python auto_trailing_stop.py --dry-run
```

## IB接続設定

- Host: `172.24.32.1`
- Port: `4001`
- TWS/IB Gatewayで「Enable ActiveX and Socket Clients」を有効化
