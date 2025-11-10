## Wyckoff 点数图（Point & Figure）可视化（OKX 4H 示例）

一个使用 OKX 历史K线，生成比特币点数图（Point & Figure, P&F）的简洁脚本。支持：
- 以“千 USD”为单位显示价格（105000 显示为 105）
- 格值（box_size）与转向格（reversal_boxes）可配置
- 使用每根K线的高低价进行柱内延伸/反转，列数更贴近真实波动
- 自动分页输出多张图，每张图标题与文件名包含起始时间

### 功能概览
- 数据源：OKX REST `market/history-candles`
- 周期：默认 `4H`
- 开始时间：默认 `2025-06-10T00:00:00Z`
- 分页抓取：支持向过去翻页获取更多历史数据
- 图表：
  - P&F 点数图（横轴为列，纵轴为价格格子）
  - 价格标签显示为“千 USD”，字号可调，避免重叠

---

## 快速开始

### 1) 环境
- Python 3.12+（其他版本通常也可）

### 2) 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3) 运行

```bash
python3 analyse.py
```

运行后会在当前目录生成：
- `wyckoff_pnf_XX_YYYYMMDD-HHMM.png`（当列数较多时输出多张）
- 如果列数不超过分页设置，也可能生成单张 `wyckoff_pnf.png`

---

## 关键参数（在 `analyse.py` 顶部）
- `symbol`: 交易对，例如 `"BTC-USDT"`
- `bar`: 周期，例如 `"4H"`
- `start_time`: 开始时间（ISO8601，UTC），例如 `"2025-06-10T00:00:00Z"`
- `price_scale_thousand`: 是否以“千”为单位显示价格（True 推荐）
- `box_size`: 格值（单位与显示一致；当以“千”为单位时，`1.0` 即 1000 USD）
- `reversal_boxes`: 转向格（常用 2 或 3；越小越敏感、列数越多）
- `columns_per_image`: 每张图最多多少列，超出自动分页

建议组合：
- 更平滑：`box_size=1.0`，`reversal_boxes=3`
- 更敏感：`box_size=0.5`，`reversal_boxes=2`

---

## 注意事项
- 中国大陆网络环境通常无法直连 OKX API，需：
  - 开启 VPN；或
  - 在 `analyse.py` 中配置 `proxies`（HTTP/HTTPS 代理），示例已在代码注释中提供
- 若出现 `No route to host`、`SSLError: UNEXPECTED_EOF_WHILE_READING`、`429` 等网络错误：
  - 切换/开启 VPN 或代理
  - 稍等重试（脚本内已添加重试与分页间隔）
  - 将 `time.sleep(0.2)` 调大到 `0.5` 以降低触发风控概率

---

## 常见问题
- 数字重叠：已将价格标签字号降至 5，可按需再改；或调大 `columns_per_image`、减小 `box_size`。
- 列数太少：减小 `box_size` 或将 `reversal_boxes` 调为 `2`，并确保 `start_time` 足够早、分页获取足够多历史数据。
- 中文字体缺失告警：脚本已设置常见中文字体优先级（如 `PingFang SC`）。若系统缺少，警告不影响图像生成。

---

## 许可
MIT License


