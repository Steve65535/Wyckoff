# main_gui_sidepanel_zoomable_unitprice_manual_fixed.py
import sys, os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox,
    QSpinBox, QCheckBox, QFrame
)
from PySide6.QtGui import QDoubleValidator, QPixmap
from PySide6.QtCore import Qt, QThread, Signal
from analyse import fetch_ohlcv, build_point_and_figure, filter_small_columns, plot_point_and_figure

class WorkerThread(QThread):
    log_signal = Signal(str)
    image_signal = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            symbol = self.params['symbol']
            bar = self.params['bar']
            start_time = self.params['start_time']
            box_size = self.params['box_size']
            reversal_boxes = self.params['reversal_boxes']
            price_scale = self.params['price_scale']

            self.log_signal.emit(f"⏳ 获取 {symbol} {bar} 数据中...")
            df = fetch_ohlcv(symbol, bar, start_time)
            if df.empty:
                self.log_signal.emit("⚠️ 数据为空")
                return
            self.log_signal.emit(f"✅ 数据获取完成，共 {len(df)} 条记录")

            close_prices = df["close"].astype(float).values
            high_prices = df["high"].astype(float).values
            low_prices = df["low"].astype(float).values
            ts_series = df["timestamp"]

            unit_label = f"{price_scale} 单位"
            if price_scale != 1:
                close_prices /= price_scale
                high_prices /= price_scale
                low_prices /= price_scale

            self.log_signal.emit("🧩 构建 n 点图...")
            columns = build_point_and_figure(
                close_prices, box_size, reversal_boxes,
                timestamps=ts_series, highs=high_prices, lows=low_prices
            )

            columns = filter_small_columns(columns, box_size, min_boxes=reversal_boxes)
            self.log_signal.emit(f"🧹 已过滤短列，剩余 {len(columns)} 列")

            if not columns:
                self.log_signal.emit("⚠️ 无列可绘制，请调整格子大小或转向格数")
                return

            chart_path = os.path.join(os.getcwd(), "wyckoff_chart.png")
            self.log_signal.emit("🖼️ 绘制并保存图表...")
            plot_point_and_figure(columns, unit_label=unit_label, filename=chart_path)
            if os.path.exists(chart_path):
                self.log_signal.emit(f"✅ 图表生成完成：{chart_path}")
                self.image_signal.emit(chart_path)
            else:
                self.log_signal.emit("⚠️ 图表生成失败")
        except Exception as e:
            self.log_signal.emit(f"❌ 错误: {str(e)}")


class WyckoffApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wyckoff 点数图 GUI")
        self.resize(1200, 750)
        self.current_scale = 1.0
        self.current_pixmap = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # ========== 左侧控制面板 ==========
        control_frame = QFrame()
        control_layout = QVBoxLayout(control_frame)
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_frame.setMinimumWidth(300)

        title = QLabel("📊 参数设置")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 8px;")
        control_layout.addWidget(title)

        # 输入控件
        self.symbol_input = QLineEdit("BTC-USDT")
        self.bar_input = QComboBox()
        self.bar_input.addItems(["1W","3D","1D","4H","1H"])
        self.start_input = QLineEdit("2025-06-10T00:00:00Z")
        self.box_input = QLineEdit("0.5")
        self.box_input.setValidator(QDoubleValidator(0.0001, 1e8, 6))
        self.rev_input = QSpinBox()
        self.rev_input.setRange(1, 10)
        self.rev_input.setValue(2)

        # 单位价格手动输入
        self.unit_price_input = QLineEdit("100")
        self.unit_price_input.setValidator(QDoubleValidator(0.0001, 1e8, 6))
        self.unit_checkbox = QCheckBox("千USD单位")
        self.unit_checkbox.setChecked(True)

        # 添加行函数
        def add_row(label, widget_or_layout):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            if isinstance(widget_or_layout, (QHBoxLayout, QVBoxLayout)):
                row.addLayout(widget_or_layout)
            else:
                row.addWidget(widget_or_layout)
            control_layout.addLayout(row)

        add_row("币种", self.symbol_input)
        add_row("周期", self.bar_input)
        add_row("开始时间", self.start_input)
        add_row("格子大小", self.box_input)
        add_row("转向格数", self.rev_input)

        # 单位格子价格布局
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("单位价格"))
        unit_layout.addWidget(self.unit_price_input)
        unit_layout.addWidget(self.unit_checkbox)
        add_row("单位格子价格", unit_layout)

        # 操作按钮
        self.run_btn = QPushButton("生成并显示图表")
        self.clear_btn = QPushButton("清空日志")
        control_layout.addWidget(self.run_btn)
        control_layout.addWidget(self.clear_btn)

        # 缩放按钮
        zoom_label = QLabel("🔍 缩放控制")
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        control_layout.addWidget(zoom_label)

        self.zoom_in_btn = QPushButton("放大")
        self.zoom_out_btn = QPushButton("缩小")
        self.reset_zoom_btn = QPushButton("重置")
        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        control_layout.addWidget(self.reset_zoom_btn)

        # 日志框
        log_label = QLabel("🧾 日志输出")
        log_label.setAlignment(Qt.AlignCenter)
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        control_layout.addWidget(log_label)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(180)
        control_layout.addWidget(self.log_area)

        control_layout.addStretch()
        main_layout.addWidget(control_frame)

        # ========== 右侧图表显示区 ==========
        self.image_label = QLabel("图表将在此显示")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #fafafa; border: 1px solid #ccc;")
        main_layout.addWidget(self.image_label, stretch=1)

        # ========== 事件绑定 ==========
        self.run_btn.clicked.connect(self.run_task)
        self.clear_btn.clicked.connect(lambda: self.log_area.clear())
        self.zoom_in_btn.clicked.connect(lambda: self.adjust_zoom(1.25))
        self.zoom_out_btn.clicked.connect(lambda: self.adjust_zoom(0.8))
        self.reset_zoom_btn.clicked.connect(lambda: self.set_zoom(1.0))

    def run_task(self):
        try:
            price_scale = float(self.unit_price_input.text())
        except ValueError:
            self.append_log("⚠️ 单位价格输入不合法")
            return

        params = {
            "symbol": self.symbol_input.text(),
            "bar": self.bar_input.currentText(),
            "start_time": self.start_input.text(),
            "box_size": float(self.box_input.text()),
            "reversal_boxes": self.rev_input.value(),
            "price_scale": price_scale
        }
        self.worker = WorkerThread(params)
        self.worker.log_signal.connect(self.append_log)
        self.worker.image_signal.connect(self.show_image)
        self.worker.start()

    def append_log(self, msg):
        self.log_area.append(msg)
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum())

    def show_image(self, path):
        if os.path.exists(path):
            self.current_pixmap = QPixmap(path)
            self.current_scale = 1.0
            self.update_image()
        else:
            self.append_log("⚠️ 图表文件不存在")

    def adjust_zoom(self, factor):
        if self.current_pixmap:
            self.current_scale *= factor
            self.update_image()

    def set_zoom(self, scale):
        if self.current_pixmap:
            self.current_scale = scale
            self.update_image()

    def update_image(self):
        if not self.current_pixmap:
            return
        w = int(self.current_pixmap.width() * self.current_scale)
        h = int(self.current_pixmap.height() * self.current_scale)
        scaled_pixmap = self.current_pixmap.scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WyckoffApp()
    win.show()
    sys.exit(app.exec())