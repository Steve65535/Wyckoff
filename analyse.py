import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# ========== 基础参数 ==========
symbol = "BTC-USDT"
bar = "4H"
limit = 300  # 每页拉取多少根K线（OKX上限通常为300）
start_time = "2025-06-10T00:00:00Z"  # 你想从哪个时间点开始
price_scale_thousand = True  # True 时以“千”为单位显示（105000 显示为 105）
# 点数图参数
box_size = 1.0  # 格值（与显示单位一致：当 price_scale_thousand=True 时，单位为“千USD”；1.0即1000 USD；可改为0.5）
reversal_boxes = 2  # 转向格（2格更敏感，列更多）
# 分图参数：每张图最多多少列
columns_per_image = 80
# 可选代理（如开本地代理端口时启用）
proxies = None
# proxies = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

# 解决中文字体警告（按可用字体优先）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 获取K线数据 ==========
def fetch_ohlcv(symbol, bar, start_time, limit=300, max_pages=200):
    url = "https://www.okx.com/api/v5/market/history-candles"
    params = {"instId": symbol, "bar": bar, "limit": limit}
    all_rows = []
    before = None
    start_ts = pd.Timestamp(start_time).tz_convert("UTC").value // 10**6  # ms
    # 带重试的 Session，缓解偶发 SSL EOF / 429 等
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    # 分页向过去拉取，直到覆盖 start_time 或达到页数上限
    for page in range(1, max_pages + 1):
        if before is not None:
            params["before"] = before
        r = session.get(url, params=params, headers=headers, timeout=10, proxies=proxies)
        resp = r.json()
        data = resp.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        print(f"分页 {page}: 获取 {len(data)} 条，累计 {len(all_rows)} 条")
        # OKX 返回通常按新->旧排序，拿最后一条的 ts 继续向过去翻页
        earliest_ts = int(data[-1][0])  # ms
        # 若已到达或早于目标起点，停止继续翻页
        if earliest_ts <= start_ts:
            before = None
            break
        # 继续向更早处翻页，减 1ms 避免重复
        before = earliest_ts - 1
        # 降低触发风控/网络抖动概率
        time.sleep(0.2)

    # 官方返回顺序：
    # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "volCcy", "volCcyQuote", "confirm"
    ])

    # 只保留前6个关键列
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[df["timestamp"] >= pd.Timestamp(start_time)]
    df = df.reset_index(drop=True)
    print(f"共获取K线条数: {len(df)}")
    return df

# ========== 枢轴点/三点识别（简化版） ==========
def detect_three_point(df, window=3):
    """
    使用局部枢轴点（pivot）寻找最近的三点（高低交替），用于画三点确认示意。
    window: 两侧比较的窗口大小，越大越严格。
    返回: 最近三个枢轴点列表 [(ts, price, 'H'/'L'), ...]（最多3个）
    """
    highs = []
    lows = []
    for i in range(window, len(df) - window):
        c = float(df.loc[i, "close"])
        left = df.loc[i - window:i - 1, "close"].astype(float)
        right = df.loc[i + 1:i + window, "close"].astype(float)
        if c >= left.max() and c >= right.max():
            highs.append((df.loc[i, "timestamp"], c, "H", i))
        if c <= left.min() and c <= right.min():
            lows.append((df.loc[i, "timestamp"], c, "L", i))
    # 合并并按索引排序
    pivots = highs + lows
    pivots.sort(key=lambda x: x[3])
    # 取末尾若干，确保类型交替
    seq = []
    for p in reversed(pivots):
        if not seq:
            seq.append(p)
        else:
            if p[2] != seq[-1][2]:  # 高低交替
                seq.append(p)
        if len(seq) >= 3:
            break
    seq = list(reversed(seq))
    # 仅返回(ts, price, type)
    return [(ts, price, t) for (ts, price, t, _) in seq]

# ========== 绘图保存 ==========
def plot_three_point(df, three_points, tz_display="Asia/Shanghai"):
    """
    价格K线 + 成交量 + 三点标注（不引入新依赖，使用原生 matplotlib）
    """
    # 显示用时区（不影响内部计算）
    df_plot = df.copy()
    df_plot["timestamp"] = df_plot["timestamp"].dt.tz_convert(tz_display)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)

    # ---- 画简易蜡烛图 ----
    ts = mdates.date2num(df_plot["timestamp"].to_pydatetime())
    opens = df_plot["open"].astype(float).values
    highs = df_plot["high"].astype(float).values
    lows = df_plot["low"].astype(float).values
    closes = df_plot["close"].astype(float).values
    volumes = df_plot["volume"].astype(float).values

    width = 0.03  # 蜡烛实体宽度
    up_color = "#26a69a"
    down_color = "#ef5350"

    for x, o, h, l, c in zip(ts, opens, highs, lows, closes):
        color = up_color if c >= o else down_color
        ax_price.vlines(x, l, h, color=color, linewidth=1)
        rect_bottom = min(o, c)
        rect_height = abs(c - o) if abs(c - o) > 1e-8 else 1e-8
        ax_price.add_patch(plt.Rectangle((x - width / 2, rect_bottom), width, rect_height, 
                                         edgecolor=color, facecolor=color, linewidth=1))

    # ---- 三点标注 ----
    if three_points:
        labels = ["①", "②", "③"][-len(three_points):]
        for (label, (t, p, tp)) in zip(labels, three_points):
            t_disp = t.tz_convert(tz_display)
            ax_price.scatter(mdates.date2num(t_disp.to_pydatetime()), p, 
                             s=60, color="#ff9800", zorder=3)
            ax_price.annotate(f"{label}{'H' if tp=='H' else 'L'}",
                              xy=(mdates.date2num(t_disp.to_pydatetime()), p),
                              xytext=(5, 8), textcoords="offset points",
                              fontsize=9, color="#ff9800",
                              bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))
        # 连接线（若有3点）
        if len(three_points) == 3:
            xys = []
            for (t, p, _) in three_points:
                t_disp = t.tz_convert(tz_display)
                xys.append((mdates.date2num(t_disp.to_pydatetime()), p))
            xs, ys = zip(*xys)
            ax_price.plot(xs, ys, color="#ffa726", linewidth=1.2, linestyle="--", alpha=0.9)

    # ---- 价格轴/时间轴样式 ----
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.2)
    ax_price.yaxis.set_major_locator(mticker.MaxNLocator(7))
    ax_price.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))

    ax_price.xaxis_date()
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=df_plot["timestamp"].dt.tz))
    for label in ax_price.get_xticklabels():
        label.set_visible(False)  # 顶部不显示x刻度，由底部共享

    # ---- 成交量 ----
    colors = [up_color if c >= o else down_color for o, c in zip(opens, closes)]
    ax_vol.bar(ts, volumes, width=width, color=colors, alpha=0.7)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, linestyle="--", alpha=0.2)
    ax_vol.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax_vol.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax_vol.xaxis_date()
    ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=df_plot["timestamp"].dt.tz))
    plt.setp(ax_vol.get_xticklabels(), rotation=0, ha="center")

    # ---- 标题与最新行情摘要 ----
    last_row = df_plot.iloc[-1]
    last_info = f"O:{float(last_row['open']):.2f} H:{float(last_row['high']):.2f} " \
                f"L:{float(last_row['low']):.2f} C:{float(last_row['close']):.2f} " \
                f"Vol:{float(last_row['volume']):.0f}"
    ax_price.set_title(f"Wyckoff 三点图（{symbol} {bar}） | {last_info}")

    plt.tight_layout()
    plt.savefig("wyckoff_three_point.png", dpi=150)
    plt.close()
    print("✅ 已保存为 wyckoff_three_point.png")

# ========== 点数图（Point & Figure） ==========
def _floor_to_box(value, box):
    return (value // box) * box

def build_point_and_figure(prices, box_size, reversal_boxes, timestamps=None, highs=None, lows=None):
    """
    根据价格序列生成点数图列数据；若提供 highs/lows，则用每根K的高低价进行“柱内延伸/反转”，列数会更多。
    返回 columns: [ { 'type': 'X'|'O', 'boxes': [level1, level2, ...], 'start_ts': pd.Timestamp|None }, ... ]
    其中 level 按价格从下到上或上到下逐格记录。
    """
    columns = []
    if len(prices) == 0:
        return columns

    # 以首价落到格子上为起点
    first = float(prices[0])
    current_level = _floor_to_box(first, box_size)
    current_col = None  # {'type': 'X'|'O', 'boxes': [levels...], 'top': float, 'bottom': float}

    def start_column(col_type, start_level, end_level, start_idx=None):
        # 生成从 start 到 end 的格子（包含端点），方向取决于 col_type
        boxes = []
        if col_type == 'X':
            lvl = start_level
            while lvl <= end_level:
                boxes.append(lvl)
                lvl += box_size
        else:
            lvl = start_level
            while lvl >= end_level:
                boxes.append(lvl)
                lvl -= box_size
        col = {
            'type': col_type,
            'boxes': boxes,
            'start_ts': timestamps[start_idx] if (timestamps is not None and start_idx is not None) else None
        }
        return col

    # 确定初始方向：直到出现至少1格的突破才建第一列
    i = 1
    while i < len(prices) and current_col is None:
        px = float(prices[i])
        up_break = px >= current_level + box_size
        down_break = px <= current_level - box_size
        if up_break:
            # 新建 X 列，从 current_level 到 达到的最高格
            top_level = _floor_to_box(px, box_size)
            current_col = start_column('X', current_level + box_size, top_level, start_idx=i)
        elif down_break:
            bottom_level = _floor_to_box(px, box_size)
            current_col = start_column('O', current_level - box_size, bottom_level, start_idx=i)
        i += 1

    if current_col is None:
        # 未形成任何列（价格未越过一个格），返回空
        return columns

    columns.append(current_col)

    # 后续价格推进/反转逻辑（如有 highs/lows 则优先用其进行更充分的箱体填充）
    for j in range(i, len(prices)):
        px = float(prices[j])
        hi = float(highs[j]) if highs is not None else px
        lo = float(lows[j]) if lows is not None else px
        col_type = columns[-1]['type']
        boxes = columns[-1]['boxes']
        if col_type == 'X':
            current_top = boxes[-1]
            # 先用当根最高价进行“向上延伸”
            while hi >= current_top + box_size:
                current_top += box_size
                boxes.append(current_top)
            # 使用当根最低价判定反转
            reversal_level = current_top - reversal_boxes * box_size
            if lo <= reversal_level:
                # 新 O 列，从 current_top - box_size 开始向下，直到 low 所在格
                new_bottom = _floor_to_box(lo, box_size)
                start_from = current_top - box_size
                columns.append(start_column('O', start_from, new_bottom, start_idx=j))
        else:
            current_bottom = boxes[-1]
            # 先用当根最低价进行“向下延伸”
            while lo <= current_bottom - box_size:
                current_bottom -= box_size
                boxes.append(current_bottom)
            # 使用当根最高价判定反转
            reversal_level = current_bottom + reversal_boxes * box_size
            if hi >= reversal_level:
                # 新 X 列，从 current_bottom + box_size 开始向上，直到 high 所在格
                new_top = _floor_to_box(hi, box_size)
                start_from = current_bottom + box_size
                columns.append(start_column('X', start_from, new_top, start_idx=j))

    return columns

def plot_point_and_figure(columns, unit_label="千USD", tz_display="Asia/Shanghai", filename="wyckoff_pnf.png", title_suffix=None):
    """
    使用 matplotlib 原生绘制点数图（X/O），横轴为列序号，纵轴为价格格子。
    """
    if not columns:
        print("⚠️ 点数图没有生成任何列（可能价格波动小于格值）。")
        return

    # 计算纵轴范围
    all_levels = [lvl for col in columns for lvl in col['boxes']]
    y_min = min(all_levels) - box_size
    y_max = max(all_levels) + box_size

    fig, ax = plt.subplots(figsize=(10, 8))

    # 画网格（正方形视觉）
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', alpha=0.2)

    # 将 X / O 改为“价格数字（以千为单位）”
    for x, col in enumerate(columns):
        color = '#26a69a' if col['type'] == 'X' else '#ef5350'
        for y in col['boxes']:
            # y 已经是“千”为单位（当 price_scale_thousand=True）
            label = f"{y:.0f}"
            ax.text(x, y, label, color=color, fontsize=5, ha='center', va='center')

    ax.set_xlabel("列（反转后向右推进）")
    ax.set_ylabel(f"价格（{unit_label}，格值={box_size}，转向格={reversal_boxes}）")
    if title_suffix:
        ax.set_title(f"{symbol} 点数图（Point & Figure） | {title_suffix}")
    else:
        ax.set_title(f"{symbol} 点数图（Point & Figure）")

    # y 轴格式化
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=False, prune=None))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ 已保存为 {filename}")

def plot_point_and_figure_paged(columns, unit_label="千USD", tz_display="Asia/Shanghai", columns_per_image=80):
    """
    将列按固定数量分页绘制，多张图片；每张图片标题包含该页起始列的时间（若可用）。
    """
    if not columns:
        print("⚠️ 无列可绘制。")
        return
    total = len(columns)
    pages = (total + columns_per_image - 1) // columns_per_image
    for p in range(pages):
        start = p * columns_per_image
        end = min((p + 1) * columns_per_image, total)
        chunk = columns[start:end]
        # 起始时间（若记录），用于标题和文件名
        start_ts = None
        for c in chunk:
            if c.get('start_ts') is not None:
                start_ts = c['start_ts']
                break
        if start_ts is not None:
            t_disp = start_ts.tz_convert(tz_display)
            stamp = t_disp.strftime("%Y%m%d-%H%M")
            title_suffix = f"起始 {t_disp.strftime('%Y-%m-%d %H:%M %Z')}"
            fname = f"wyckoff_pnf_{p+1:02d}_{stamp}.png"
        else:
            title_suffix = f"第 {p+1}/{pages} 页"
            fname = f"wyckoff_pnf_{p+1:02d}.png"
        plot_point_and_figure(chunk, unit_label=unit_label, tz_display=tz_display, filename=fname, title_suffix=title_suffix)

# ========== 主程序 ==========
if __name__ == "__main__":
    print(f"获取 {symbol} {bar} 数据中...")
    df = fetch_ohlcv(symbol, bar, start_time)
    # 价格序列（收盘），按“千”为单位（若开启）
    close_prices = df["close"].astype(float).values
    high_prices = df["high"].astype(float).values
    low_prices = df["low"].astype(float).values
    ts_series = df["timestamp"]
    if price_scale_thousand:
        close_prices = close_prices / 1000.0
        high_prices = high_prices / 1000.0
        low_prices = low_prices / 1000.0
        unit = "千USD"
    else:
        unit = "USD"

    # 生成点数图列并绘制
    pnf_columns = build_point_and_figure(
        close_prices,
        box_size=box_size,
        reversal_boxes=reversal_boxes,
        timestamps=ts_series,
        highs=high_prices,
        lows=low_prices,
    )
    if len(pnf_columns) > columns_per_image:
        plot_point_and_figure_paged(pnf_columns, unit_label=unit, columns_per_image=columns_per_image)
    else:
        plot_point_and_figure(pnf_columns, unit_label=unit)