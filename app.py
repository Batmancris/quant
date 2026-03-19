from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "power_multi_factor_strategy"
DOCS_DIR = PROJECT_ROOT / "docs"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = PROJECT_ROOT / "greenpower_demo" / "train.py"
DEFAULT_PREVIEW_PORT = 8765


class DocsPreviewServer:
    def __init__(self, docs_dir: Path, port: int = DEFAULT_PREVIEW_PORT) -> None:
        self.docs_dir = docs_dir
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> str:
        if self.httpd is not None:
            return f"http://127.0.0.1:{self.port}/index.html"

        handler = partial(SimpleHTTPRequestHandler, directory=str(self.docs_dir))
        port = self.port
        while True:
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), handler)
                break
            except OSError:
                port += 1
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.httpd = server
        self.thread = thread
        self.port = port
        return f"http://127.0.0.1:{self.port}/index.html"

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
            self.thread = None


class QuantDesktopApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("电力股多因子量化策略")
        self.root.geometry("1280x860")
        self.root.minsize(1120, 760)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.training_process: subprocess.Popen[str] | None = None
        self.preview_server = DocsPreviewServer(DOCS_DIR)

        self.symbol_var = StringVar(value="000537")
        self.start_date_var = StringVar(value="2015-01-01")
        self.strategy_id_var = StringVar(value="power_multi_factor_strategy")
        self.holding_period_var = IntVar(value=5)
        self.rebalance_var = IntVar(value=5)
        self.top_k_var = IntVar(value=3)
        self.train_ratio_var = DoubleVar(value=0.70)
        self.transaction_cost_var = DoubleVar(value=10.0)
        self.sell_tax_var = DoubleVar(value=5.0)
        self.force_refresh_var = BooleanVar(value=False)

        self.metric_vars: dict[str, StringVar] = {
            "strategy_id": StringVar(value="-"),
            "latest_trade_date": StringVar(value="-"),
            "train_end_date": StringVar(value="-"),
            "annualized_return": StringVar(value="-"),
            "max_drawdown": StringVar(value="-"),
            "excess_return": StringVar(value="-"),
        }

        self._build_ui()
        self._load_summary()
        self.root.after(150, self._poll_logs)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("MetricTitle.TLabel", foreground="#5c6a60")
        style.configure("MetricValue.TLabel", font=("Segoe UI", 14, "bold"))

        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text="电力股多因子量化策略", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Windows 桌面端负责训练和查看摘要，网页端通过 docs/ 静态站部署到 GitHub Pages。",
        ).pack(anchor="w", pady=(4, 12))

        form_frame = ttk.LabelFrame(container, text="策略参数", padding=12)
        form_frame.pack(fill="x")

        fields = [
            ("展示主标的", self.symbol_var),
            ("起始日期", self.start_date_var),
            ("策略 ID", self.strategy_id_var),
        ]
        for index, (label, var) in enumerate(fields):
            ttk.Label(form_frame, text=label).grid(row=0, column=index * 2, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(form_frame, textvariable=var, width=18).grid(row=0, column=index * 2 + 1, sticky="ew", padx=(0, 16), pady=4)

        numeric_fields = [
            ("持有周期", self.holding_period_var, 1),
            ("调仓周期", self.rebalance_var, 3),
            ("持仓数量", self.top_k_var, 5),
            ("训练占比", self.train_ratio_var, 7),
            ("交易成本(bps)", self.transaction_cost_var, 9),
            ("卖出税(bps)", self.sell_tax_var, 11),
        ]
        for label, var, column in numeric_fields:
            ttk.Label(form_frame, text=label).grid(row=1, column=column - 1, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(form_frame, textvariable=var, width=12).grid(row=1, column=column, sticky="ew", padx=(0, 16), pady=4)

        ttk.Checkbutton(form_frame, text="强制刷新数据", variable=self.force_refresh_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 4))

        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=2, column=2, columnspan=10, sticky="e", pady=(8, 4))
        self.train_button = ttk.Button(button_frame, text="训练策略", command=self._start_training)
        self.train_button.pack(side="left", padx=4)
        ttk.Button(button_frame, text="刷新摘要", command=self._load_summary).pack(side="left", padx=4)
        ttk.Button(button_frame, text="打开静态网页", command=self._open_static_site).pack(side="left", padx=4)
        ttk.Button(button_frame, text="打开 docs 目录", command=self._open_docs_folder).pack(side="left", padx=4)

        for column in range(12):
            form_frame.columnconfigure(column, weight=1)

        metrics_frame = ttk.LabelFrame(container, text="策略摘要", padding=12)
        metrics_frame.pack(fill="x", pady=(14, 0))

        metric_labels = [
            ("策略 ID", "strategy_id"),
            ("最新交易日", "latest_trade_date"),
            ("训练截止日", "train_end_date"),
            ("年化收益", "annualized_return"),
            ("最大回撤", "max_drawdown"),
            ("超额收益", "excess_return"),
        ]
        for idx, (title, key) in enumerate(metric_labels):
            block = ttk.Frame(metrics_frame, padding=(8, 2))
            block.grid(row=0, column=idx, sticky="nsew")
            ttk.Label(block, text=title, style="MetricTitle.TLabel").pack(anchor="w")
            ttk.Label(block, textvariable=self.metric_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))
            metrics_frame.columnconfigure(idx, weight=1)

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(14, 0))

        left_panel = ttk.Frame(body)
        right_panel = ttk.Frame(body)
        body.add(left_panel, weight=3)
        body.add(right_panel, weight=2)

        holdings_frame = ttk.LabelFrame(left_panel, text="当前建议持仓", padding=10)
        holdings_frame.pack(fill="both", expand=True)
        self.holdings_tree = ttk.Treeview(holdings_frame, columns=("symbol", "name", "score", "rank"), show="headings", height=9)
        for col, title, width in (
            ("symbol", "代码", 90),
            ("name", "名称", 120),
            ("score", "评分", 90),
            ("rank", "排名", 70),
        ):
            self.holdings_tree.heading(col, text=title)
            self.holdings_tree.column(col, width=width, anchor="center")
        self.holdings_tree.pack(fill="both", expand=True)

        factor_frame = ttk.LabelFrame(left_panel, text="因子权重", padding=10)
        factor_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.factor_tree = ttk.Treeview(factor_frame, columns=("label", "mean_ic", "ic_ir", "weight"), show="headings", height=9)
        for col, title, width in (
            ("label", "因子", 140),
            ("mean_ic", "平均IC", 80),
            ("ic_ir", "ICIR", 80),
            ("weight", "权重", 80),
        ):
            self.factor_tree.heading(col, text=title)
            self.factor_tree.column(col, width=width, anchor="center")
        self.factor_tree.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(right_panel, text="训练日志", padding=10)
        log_frame.pack(fill="both", expand=True)
        self.log_text = ScrolledText(log_frame, wrap="word", height=30, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("end", "桌面应用已启动。点击“训练策略”即可重新生成量化结果和 docs 静态站。\n")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_training_state(self, running: bool) -> None:
        self.train_button.configure(state="disabled" if running else "normal")

    def _build_train_command(self) -> list[str]:
        return [
            str(PYTHON_EXE),
            str(TRAIN_SCRIPT),
            "--symbol",
            self.symbol_var.get().strip() or "000537",
            "--start",
            self.start_date_var.get().strip() or "2015-01-01",
            "--strategy-id",
            self.strategy_id_var.get().strip() or "power_multi_factor_strategy",
            "--holding-period-days",
            str(int(self.holding_period_var.get())),
            "--rebalance-frequency-days",
            str(int(self.rebalance_var.get())),
            "--top-k",
            str(int(self.top_k_var.get())),
            "--train-ratio",
            str(float(self.train_ratio_var.get())),
            "--transaction-cost-bps",
            str(float(self.transaction_cost_var.get())),
            "--sell-tax-bps",
            str(float(self.sell_tax_var.get())),
        ] + (["--force-refresh"] if self.force_refresh_var.get() else [])

    def _start_training(self) -> None:
        if self.training_process is not None and self.training_process.poll() is None:
            messagebox.showinfo("训练进行中", "当前已有训练任务在运行，请等待结束。")
            return
        if not PYTHON_EXE.exists():
            messagebox.showerror("环境缺失", f"未找到虚拟环境解释器：{PYTHON_EXE}")
            return

        command = self._build_train_command()
        self._append_log("\n开始训练量化策略：\n")
        self._append_log(" ".join(command) + "\n\n")
        self._set_training_state(True)

        def _worker() -> None:
            try:
                self.training_process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert self.training_process.stdout is not None
                for line in self.training_process.stdout:
                    self.log_queue.put(line)
                return_code = self.training_process.wait()
                self.log_queue.put(f"\n训练结束，退出码：{return_code}\n")
                self.log_queue.put("__TRAINING_DONE__")
            except Exception as exc:  # pragma: no cover - desktop runtime path
                self.log_queue.put(f"\n训练失败：{exc}\n")
                self.log_queue.put("__TRAINING_DONE__")

        threading.Thread(target=_worker, daemon=True).start()

    def _poll_logs(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__TRAINING_DONE__":
                self._set_training_state(False)
                self._load_summary()
            else:
                self._append_log(item)
        self.root.after(150, self._poll_logs)

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_summary(self) -> None:
        metrics_path = ARTIFACT_DIR / "metrics.json"
        summary_path = ARTIFACT_DIR / "run_summary.json"
        holdings_path = ARTIFACT_DIR / "latest_portfolio.csv"
        factor_path = ARTIFACT_DIR / "factor_weights.csv"

        if not metrics_path.exists() or not summary_path.exists():
            self._append_log("尚未发现策略产物，请先点击“训练策略”。\n")
            return

        metrics = self._load_json(metrics_path)
        summary = self._load_json(summary_path)

        self.metric_vars["strategy_id"].set(summary.get("strategy_id", "-"))
        self.metric_vars["latest_trade_date"].set(summary.get("latest_trade_date", "-"))
        self.metric_vars["train_end_date"].set(summary.get("train_end_date", "-"))
        self.metric_vars["annualized_return"].set(f"{metrics.get('annualized_return', 0.0) * 100:.2f}%")
        self.metric_vars["max_drawdown"].set(f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%")
        self.metric_vars["excess_return"].set(f"{metrics.get('excess_return', 0.0) * 100:.2f}%")

        for tree in (self.holdings_tree, self.factor_tree):
            for item in tree.get_children():
                tree.delete(item)

        if holdings_path.exists():
            holdings = json.loads(pd.read_csv(holdings_path).to_json(orient="records"))
            for row in holdings:
                self.holdings_tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("symbol", ""),
                        row.get("name", ""),
                        f"{float(row.get('score', 0.0)):.3f}",
                        f"{float(row.get('rank', 0.0)):.0f}",
                    ),
                )

        if factor_path.exists():
            factor_frame = pd.read_csv(factor_path)
            for row in factor_frame.itertuples(index=False):
                self.factor_tree.insert(
                    "",
                    "end",
                    values=(
                        getattr(row, "factor_label", getattr(row, "factor", "")),
                        f"{float(getattr(row, 'mean_ic', 0.0)):.3f}",
                        f"{float(getattr(row, 'ic_ir', 0.0)):.3f}",
                        f"{float(getattr(row, 'weight', 0.0)):.3f}",
                    ),
                )

    def _open_static_site(self) -> None:
        if not (DOCS_DIR / "index.html").exists():
            messagebox.showwarning("静态站不存在", "尚未发现 docs/index.html，请先训练策略。")
            return
        url = self.preview_server.start()
        webbrowser.open_new_tab(url)
        self._append_log(f"已在浏览器打开静态站预览：{url}\n")

    def _open_docs_folder(self) -> None:
        if not DOCS_DIR.exists():
            messagebox.showwarning("目录不存在", "尚未发现 docs 目录。")
            return
        os.startfile(str(DOCS_DIR))

    def _on_close(self) -> None:
        self.preview_server.stop()
        if self.training_process is not None and self.training_process.poll() is None:
            if messagebox.askyesno("退出确认", "训练仍在进行中，确定要退出吗？"):
                self.training_process.terminate()
            else:
                return
        self.root.destroy()


def main() -> None:
    root = Tk()
    QuantDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
