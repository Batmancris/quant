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
BUILD_SCRIPT = PROJECT_ROOT / "build_exe.cmd"
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
        self.root.title("A 股多因子量化研究台")
        self.root.geometry("1420x920")
        self.root.minsize(1220, 780)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.training_process: subprocess.Popen[str] | None = None
        self.preview_server = DocsPreviewServer(DOCS_DIR)

        self.symbol_var = StringVar(value="000537")
        self.start_date_var = StringVar(value="2015-01-01")
        self.strategy_id_var = StringVar(value="power_multi_factor_strategy")
        self.holding_period_var = IntVar(value=5)
        self.rebalance_var = IntVar(value=5)
        self.top_k_var = IntVar(value=8)
        self.universe_size_var = IntVar(value=72)
        self.industry_cap_var = IntVar(value=4)
        self.positions_per_industry_var = IntVar(value=2)
        self.train_ratio_var = DoubleVar(value=0.70)
        self.initial_train_days_var = IntVar(value=756)
        self.walk_test_days_var = IntVar(value=63)
        self.walk_step_days_var = IntVar(value=63)
        self.max_single_weight_var = DoubleVar(value=0.18)
        self.max_turnover_var = DoubleVar(value=0.60)
        self.min_market_cap_q_var = DoubleVar(value=0.20)
        self.transaction_cost_var = DoubleVar(value=10.0)
        self.sell_tax_var = DoubleVar(value=5.0)
        self.initial_cash_var = DoubleVar(value=1000000.0)
        self.force_refresh_var = BooleanVar(value=False)

        self.metric_vars: dict[str, StringVar] = {
            "strategy_id": StringVar(value="-"),
            "latest_trade_date": StringVar(value="-"),
            "generated_at": StringVar(value="-"),
            "annualized_return": StringVar(value="-"),
            "max_drawdown": StringVar(value="-"),
            "excess_return": StringVar(value="-"),
            "success_rate": StringVar(value="-"),
            "universe_size": StringVar(value="-"),
            "broker_mode": StringVar(value="-"),
            "paper_equity": StringVar(value="-"),
        }

        self._build_ui()
        self._load_summary()
        self.root.after(150, self._poll_logs)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("MetricTitle.TLabel", foreground="#56616f")
        style.configure("MetricValue.TLabel", font=("Segoe UI", 14, "bold"))

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text="A 股多因子量化研究台", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="更大的股票池、基本面因子、walk-forward 验证、模拟盘与 GitHub Pages 静态展示都从这里驱动。",
        ).pack(anchor="w", pady=(4, 10))

        form_frame = ttk.LabelFrame(container, text="训练参数", padding=10)
        form_frame.pack(fill="x")

        row0 = [
            ("目标股票", self.symbol_var, 14),
            ("开始日期", self.start_date_var, 14),
            ("策略 ID", self.strategy_id_var, 28),
        ]
        for idx, (label, var, width) in enumerate(row0):
            ttk.Label(form_frame, text=label).grid(row=0, column=idx * 2, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(form_frame, textvariable=var, width=width).grid(row=0, column=idx * 2 + 1, sticky="ew", padx=(0, 16), pady=4)

        numeric_fields = [
            ("持有天数", self.holding_period_var),
            ("调仓天数", self.rebalance_var),
            ("持仓数量", self.top_k_var),
            ("样本池规模", self.universe_size_var),
            ("行业入池上限", self.industry_cap_var),
            ("组合行业上限", self.positions_per_industry_var),
            ("训练占比", self.train_ratio_var),
            ("初始训练天数", self.initial_train_days_var),
            ("测试窗口", self.walk_test_days_var),
            ("步进窗口", self.walk_step_days_var),
            ("单票权重上限", self.max_single_weight_var),
            ("换手上限", self.max_turnover_var),
            ("最小市值分位", self.min_market_cap_q_var),
            ("交易成本(bps)", self.transaction_cost_var),
            ("卖出税(bps)", self.sell_tax_var),
            ("模拟盘初始资金", self.initial_cash_var),
        ]
        for idx, (label, var) in enumerate(numeric_fields):
            row = 1 + idx // 6
            col = (idx % 6) * 2
            ttk.Label(form_frame, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(form_frame, textvariable=var, width=12).grid(row=row, column=col + 1, sticky="ew", padx=(0, 16), pady=4)

        ttk.Checkbutton(form_frame, text="强制刷新最新数据", variable=self.force_refresh_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 4))

        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=4, column=2, columnspan=10, sticky="e", pady=(8, 4))
        self.train_button = ttk.Button(button_frame, text="训练策略", command=self._start_training)
        self.train_button.pack(side="left", padx=4)
        ttk.Button(button_frame, text="刷新摘要", command=self._load_summary).pack(side="left", padx=4)
        ttk.Button(button_frame, text="打开静态网页", command=self._open_static_site).pack(side="left", padx=4)
        ttk.Button(button_frame, text="打开产物目录", command=self._open_artifact_dir).pack(side="left", padx=4)
        ttk.Button(button_frame, text="打包 EXE", command=self._build_exe).pack(side="left", padx=4)

        for column in range(12):
            form_frame.columnconfigure(column, weight=1)

        metrics_frame = ttk.LabelFrame(container, text="策略摘要", padding=10)
        metrics_frame.pack(fill="x", pady=(12, 0))
        metric_labels = [
            ("策略 ID", "strategy_id"),
            ("最新交易日", "latest_trade_date"),
            ("生成时间", "generated_at"),
            ("年化收益", "annualized_return"),
            ("最大回撤", "max_drawdown"),
            ("超额收益", "excess_return"),
            ("成功率", "success_rate"),
            ("样本池规模", "universe_size"),
            ("执行模式", "broker_mode"),
            ("模拟盘权益", "paper_equity"),
        ]
        for idx, (title, key) in enumerate(metric_labels):
            block = ttk.Frame(metrics_frame, padding=(8, 2))
            block.grid(row=0, column=idx, sticky="nsew")
            ttk.Label(block, text=title, style="MetricTitle.TLabel").pack(anchor="w")
            ttk.Label(block, textvariable=self.metric_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))
            metrics_frame.columnconfigure(idx, weight=1)

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(12, 0))

        left_panel = ttk.Frame(body)
        right_panel = ttk.Frame(body)
        body.add(left_panel, weight=3)
        body.add(right_panel, weight=2)

        holdings_frame = ttk.LabelFrame(left_panel, text="当前建议持仓", padding=10)
        holdings_frame.pack(fill="both", expand=True)
        self.holdings_tree = ttk.Treeview(
            holdings_frame,
            columns=("symbol", "name", "industry", "score", "weight"),
            show="headings",
            height=9,
        )
        for col, title, width in (
            ("symbol", "代码", 90),
            ("name", "名称", 120),
            ("industry", "行业", 120),
            ("score", "评分", 90),
            ("weight", "权重", 90),
        ):
            self.holdings_tree.heading(col, text=title)
            self.holdings_tree.column(col, width=width, anchor="center")
        self.holdings_tree.pack(fill="both", expand=True)

        orders_frame = ttk.LabelFrame(left_panel, text="模拟盘调仓单", padding=10)
        orders_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.orders_tree = ttk.Treeview(
            orders_frame,
            columns=("symbol", "side", "shares", "price", "notional"),
            show="headings",
            height=9,
        )
        for col, title, width in (
            ("symbol", "代码", 90),
            ("side", "方向", 80),
            ("shares", "股数", 90),
            ("price", "价格", 90),
            ("notional", "金额", 120),
        ):
            self.orders_tree.heading(col, text=title)
            self.orders_tree.column(col, width=width, anchor="center")
        self.orders_tree.pack(fill="both", expand=True)

        factor_frame = ttk.LabelFrame(right_panel, text="因子权重", padding=10)
        factor_frame.pack(fill="both", expand=True)
        self.factor_tree = ttk.Treeview(factor_frame, columns=("label", "mean_ic", "ic_ir", "weight"), show="headings", height=8)
        for col, title, width in (
            ("label", "因子", 150),
            ("mean_ic", "平均IC", 90),
            ("ic_ir", "ICIR", 90),
            ("weight", "权重", 90),
        ):
            self.factor_tree.heading(col, text=title)
            self.factor_tree.column(col, width=width, anchor="center")
        self.factor_tree.pack(fill="both", expand=True)

        fold_frame = ttk.LabelFrame(right_panel, text="Walk-Forward 结果", padding=10)
        fold_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.fold_tree = ttk.Treeview(fold_frame, columns=("fold", "annual", "excess", "success"), show="headings", height=8)
        for col, title, width in (
            ("fold", "Fold", 70),
            ("annual", "年化收益", 100),
            ("excess", "超额收益", 100),
            ("success", "成功率", 100),
        ):
            self.fold_tree.heading(col, text=title)
            self.fold_tree.column(col, width=width, anchor="center")
        self.fold_tree.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(right_panel, text="训练日志", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.log_text = ScrolledText(log_frame, wrap="word", height=14, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("end", "应用已启动。点击“训练策略”将重新生成多因子结果、模拟盘计划和 docs 静态站。\n")
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
            "--universe-size",
            str(int(self.universe_size_var.get())),
            "--universe-per-industry-cap",
            str(int(self.industry_cap_var.get())),
            "--max-positions-per-industry",
            str(int(self.positions_per_industry_var.get())),
            "--train-ratio",
            str(float(self.train_ratio_var.get())),
            "--initial-train-days",
            str(int(self.initial_train_days_var.get())),
            "--walk-forward-test-days",
            str(int(self.walk_test_days_var.get())),
            "--walk-forward-step-days",
            str(int(self.walk_step_days_var.get())),
            "--max-single-weight",
            str(float(self.max_single_weight_var.get())),
            "--max-turnover",
            str(float(self.max_turnover_var.get())),
            "--min-market-cap-quantile",
            str(float(self.min_market_cap_q_var.get())),
            "--transaction-cost-bps",
            str(float(self.transaction_cost_var.get())),
            "--sell-tax-bps",
            str(float(self.sell_tax_var.get())),
            "--initial-cash",
            str(float(self.initial_cash_var.get())),
            "--broker-mode",
            "paper",
        ] + (["--force-refresh"] if self.force_refresh_var.get() else [])

    def _run_background(self, command: list[str], done_token: str) -> None:
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
                self.log_queue.put(f"\n任务结束，退出码：{return_code}\n")
                self.log_queue.put(done_token)
            except Exception as exc:  # pragma: no cover - runtime only
                self.log_queue.put(f"\n任务失败：{exc}\n")
                self.log_queue.put(done_token)

        threading.Thread(target=_worker, daemon=True).start()

    def _start_training(self) -> None:
        if self.training_process is not None and self.training_process.poll() is None:
            messagebox.showinfo("训练进行中", "当前已有训练任务在运行，请等待结束。")
            return
        if not PYTHON_EXE.exists():
            messagebox.showerror("环境缺失", f"未找到虚拟环境解释器：{PYTHON_EXE}")
            return
        command = self._build_train_command()
        self._append_log("\n开始训练策略：\n")
        self._append_log(" ".join(command) + "\n\n")
        self._set_training_state(True)
        self._run_background(command, "__TRAINING_DONE__")

    def _build_exe(self) -> None:
        if not BUILD_SCRIPT.exists():
            messagebox.showwarning("脚本不存在", "未找到 build_exe.cmd，请先生成打包脚本。")
            return
        self._append_log("\n开始打包 Windows EXE...\n")
        self._run_background(["cmd", "/c", str(BUILD_SCRIPT)], "__BUILD_DONE__")

    def _poll_logs(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item in {"__TRAINING_DONE__", "__BUILD_DONE__"}:
                self._set_training_state(False)
                self._load_summary()
            else:
                self._append_log(item)
        self.root.after(150, self._poll_logs)

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _load_summary(self) -> None:
        metrics_path = ARTIFACT_DIR / "metrics.json"
        summary_path = ARTIFACT_DIR / "run_summary.json"
        holdings_path = ARTIFACT_DIR / "latest_portfolio.csv"
        factor_path = ARTIFACT_DIR / "factor_weights.csv"
        fold_path = ARTIFACT_DIR / "walk_forward_folds.csv"
        account_path = ARTIFACT_DIR / "simulation_account.json"
        trade_plan_path = ARTIFACT_DIR / "trade_plan.json"

        if not metrics_path.exists() or not summary_path.exists():
            self._append_log("尚未发现策略产物，请先点击“训练策略”。\n")
            return

        metrics = self._load_json(metrics_path)
        summary = self._load_json(summary_path)
        account = self._load_json(account_path) if account_path.exists() else {"equity": 0.0, "mode": "paper"}
        trade_plan = self._load_json(trade_plan_path) if trade_plan_path.exists() else []

        self.metric_vars["strategy_id"].set(summary.get("strategy_id", "-"))
        self.metric_vars["latest_trade_date"].set(summary.get("latest_trade_date", "-"))
        self.metric_vars["generated_at"].set(summary.get("generated_at", "-"))
        self.metric_vars["annualized_return"].set(f"{metrics.get('annualized_return', 0.0) * 100:.2f}%")
        self.metric_vars["max_drawdown"].set(f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%")
        self.metric_vars["excess_return"].set(f"{metrics.get('excess_return', 0.0) * 100:.2f}%")
        self.metric_vars["success_rate"].set(f"{metrics.get('signal_success_rate', 0.0) * 100:.2f}%")
        self.metric_vars["universe_size"].set(str(len(summary.get("stock_pool", []))))
        self.metric_vars["broker_mode"].set(summary.get("broker_mode", account.get("mode", "paper")))
        self.metric_vars["paper_equity"].set(f"{account.get('equity', 0.0):,.0f}")

        for tree in (self.holdings_tree, self.orders_tree, self.factor_tree, self.fold_tree):
            self._clear_tree(tree)

        if holdings_path.exists():
            holdings = pd.read_csv(holdings_path)
            for row in holdings.itertuples(index=False):
                self.holdings_tree.insert(
                    "",
                    "end",
                    values=(
                        getattr(row, "symbol", ""),
                        getattr(row, "name", ""),
                        getattr(row, "industry", ""),
                        f"{float(getattr(row, 'score', 0.0)):.3f}",
                        f"{float(getattr(row, 'target_weight', 0.0)):.1%}",
                    ),
                )

        for order in trade_plan:
            self.orders_tree.insert(
                "",
                "end",
                values=(
                    order.get("symbol", ""),
                    order.get("side", ""),
                    order.get("shares", 0),
                    f"{float(order.get('price', 0.0)):.2f}",
                    f"{float(order.get('estimated_notional', 0.0)):,.0f}",
                ),
            )

        if factor_path.exists():
            factor_frame = pd.read_csv(factor_path)
            factor_frame = factor_frame.sort_values(["fold_id", "weight"], ascending=[False, False]).drop_duplicates("factor", keep="first")
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

        if fold_path.exists():
            fold_frame = pd.read_csv(fold_path)
            for row in fold_frame.itertuples(index=False):
                self.fold_tree.insert(
                    "",
                    "end",
                    values=(
                        getattr(row, "fold_id", ""),
                        f"{float(getattr(row, 'annualized_return', 0.0)) * 100:.2f}%",
                        f"{float(getattr(row, 'excess_return', 0.0)) * 100:.2f}%",
                        f"{float(getattr(row, 'signal_success_rate', 0.0)) * 100:.2f}%",
                    ),
                )

    def _open_static_site(self) -> None:
        if not (DOCS_DIR / "index.html").exists():
            messagebox.showwarning("静态站不存在", "尚未发现 docs/index.html，请先训练策略。")
            return
        url = self.preview_server.start()
        webbrowser.open_new_tab(url)
        self._append_log(f"已在浏览器打开静态站预览：{url}\n")

    def _open_artifact_dir(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(ARTIFACT_DIR))

    def _on_close(self) -> None:
        self.preview_server.stop()
        if self.training_process is not None and self.training_process.poll() is None:
            if messagebox.askyesno("退出确认", "后台任务仍在执行，确定要退出吗？"):
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
