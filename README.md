# Quant

一个面向 A 股电力板块的多因子量化研究 demo。

这个仓库已经从“单只股票短线涨跌神经网络”收敛为一条更清晰的主线：

- 用一组电力股做横截面打分
- 用多因子排序替代“猜明天涨跌”
- 提供一个 Windows 桌面应用用于本地训练和查看结果
- 提供一个可发布到 GitHub Pages 的静态网页用于公开展示

当前默认展示标的是 `000537 / 绿发电力`，但训练不是只看这一只股票，而是在电力股票池上联合建模后再输出目标股和最新组合建议。

## 仓库地址

- GitHub: <https://github.com/Batmancris/quant>
- GitHub Pages 入口设计为: <https://batmancris.github.io/quant/>

如果你是 fork 后使用，请把上面的地址替换成你自己的仓库地址。

## 项目目标

这个项目的目标不是做“保证盈利”的黑盒预测，而是搭一个研究型量化框架，覆盖下面几件事：

1. 自动抓取 A 股行情与市场数据
2. 构建电力板块横截面因子
3. 根据训练区间的因子 IC 生成权重
4. 在测试区间做组合回测
5. 输出桌面端摘要和静态网页快照

## 当前策略概览

### 股票池

当前默认股票池为 16 只电力股：

- `000027` 深圳能源
- `000537` 绿发电力
- `000539` 粤电力A
- `000543` 皖能电力
- `000600` 建投能源
- `000875` 吉电股份
- `000883` 湖北能源
- `600011` 华能国际
- `600021` 上海电力
- `600023` 浙能电力
- `600025` 华能水电
- `600027` 华电国际
- `600795` 国电电力
- `600900` 长江电力
- `600905` 三峡能源
- `601991` 大唐发电

### 数据来源

数据通过 `AkShare` 抓取，并落地缓存到本地：

- 个股日线
- 上证指数
- 深证成指
- 沪深 300
- 北向资金历史
- A 股交易日历
- 基于股票池自行构造的电力板块代理指数

### 因子框架

当前策略在横截面上使用这些因子：

- `relative_strength_5`：相对电力板块的 5 日强弱
- `trend_strength`：20 日均线偏离
- `macd_trend`：MACD 趋势强度
- `low_volatility`：低波动因子
- `rsi_balance`：RSI 平衡度
- `volume_support`：量能支持
- `turnover_stability`：换手稳定度

训练时会先计算训练区间内每个因子的 `Spearman IC`，再根据 `IC / IC 波动` 生成因子权重。当前实现会优先保留平均 IC 为正的因子，避免把长期反向信号直接混进组合里。

### 回测逻辑

- 默认持有周期：`5` 个交易日
- 默认调仓周期：`5` 个交易日
- 默认持仓数量：`3`
- 组合构建：每次调仓等权买入得分最高的 `Top K`
- 约束：优先规避当日涨停标的，不足时再回退到全样本
- 成本：支持交易成本和卖出印花税
- 基准：股票池等权收益

## 交付形式

### 1. Windows 桌面应用

桌面端入口是根目录的 [app.py](app.py)，用 `Tkinter` 实现，负责：

- 修改训练参数
- 启动训练任务
- 实时查看训练日志
- 查看当前策略摘要
- 查看最新建议持仓
- 打开本地 `docs/` 静态网页

### 2. GitHub Pages 静态网页

静态网页位于：

- [docs/index.html](docs/index.html)
- [docs/.nojekyll](docs/.nojekyll)

训练完成后，会重新生成：

- `docs/site-data.json`

静态网页展示的内容包括：

- 最新策略摘要
- 策略净值与基准曲线
- 因子累计平均 IC 曲线
- 目标股价格与成交量
- 当前建议持仓
- 因子权重表
- 最近调仓记录

## 项目结构

```text
quant/
├─ app.py                         # Windows 桌面应用
├─ run_data.cmd                   # 抓取并缓存数据
├─ run_train.cmd                  # 训练策略并导出产物
├─ run_app.cmd                    # 启动 Windows 桌面应用
├─ requirements.txt               # Python 依赖
├─ greenpower_demo/
│  ├─ __init__.py
│  ├─ config.py                   # 常量、路径、配置 dataclass
│  ├─ data.py                     # AkShare 抓数与缓存
│  ├─ features.py                 # 因子所需特征工程
│  ├─ quant_strategy.py           # 因子打分、回测、产物导出
│  ├─ site_export.py              # 导出 GitHub Pages 静态数据
│  └─ train.py                    # 训练入口
├─ docs/
│  ├─ index.html                  # 静态网页
│  └─ .nojekyll
├─ artifacts/
│  └─ power_multi_factor_strategy/
│     ├─ feature_frame.csv
│     ├─ scored_frame.csv
│     ├─ factor_ic_history.csv
│     ├─ factor_weights.csv
│     ├─ backtest_daily.csv
│     ├─ rebalance_log.csv
│     ├─ latest_portfolio.csv
│     ├─ metrics.json
│     └─ run_summary.json
└─ tests/
   └─ test_quant_strategy.py
```

## 环境要求

- Windows
- Python 3.14
- 已存在的虚拟环境 `.venv`

当前依赖见 [requirements.txt](requirements.txt)：

```text
akshare==1.18.39
pandas==2.3.3
numpy==2.4.3
```

说明：

- `Tkinter` 是 Python 自带的标准库，不单独安装
- 本项目默认直接使用现有 `.venv`

## 快速开始

### 1. 安装依赖

如果 `.venv` 已经存在，直接在根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 只抓取数据

```powershell
.\run_data.cmd
```

等价命令：

```powershell
.\.venv\Scripts\python.exe .\greenpower_demo\data.py --symbol 000537 --start 2015-01-01
```

### 3. 训练策略

```powershell
.\run_train.cmd --holding-period-days 5 --rebalance-frequency-days 5 --top-k 3
```

等价命令：

```powershell
.\.venv\Scripts\python.exe .\greenpower_demo\train.py `
  --symbol 000537 `
  --start 2015-01-01 `
  --strategy-id power_multi_factor_strategy `
  --holding-period-days 5 `
  --rebalance-frequency-days 5 `
  --top-k 3 `
  --train-ratio 0.70 `
  --transaction-cost-bps 10 `
  --sell-tax-bps 5
```

训练完成后，终端会打印一份 JSON 摘要，包括：

- 策略 ID
- 最新交易日
- 训练截止日
- 当前持仓建议
- 回测指标
- 产物目录

### 4. 启动 Windows 桌面应用

```powershell
.\run_app.cmd
```

等价命令：

```powershell
.\.venv\Scripts\python.exe .\app.py
```

### 5. 强制刷新数据后重新训练

```powershell
.\run_train.cmd --force-refresh
```

## 训练参数说明

`greenpower_demo/train.py` 当前支持这些参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--symbol` | `000537` | 展示主标的 |
| `--start` | `2015-01-01` | 起始日期 |
| `--end` | `None` | 截止日期，默认到最新可用交易日 |
| `--strategy-id` | `power_multi_factor_strategy` | 产物目录名称 |
| `--holding-period-days` | `5` | 持有周期 |
| `--rebalance-frequency-days` | `5` | 调仓周期 |
| `--top-k` | `3` | 每次调仓持有数量 |
| `--train-ratio` | `0.70` | 训练区间占比 |
| `--transaction-cost-bps` | `10.0` | 双边交易成本基点 |
| `--sell-tax-bps` | `5.0` | 卖出税费基点 |
| `--force-refresh` | `False` | 强制刷新缓存 |

## 输出产物说明

每次训练会在 `artifacts/<strategy_id>/` 下生成完整产物。

### 核心文件

- `feature_frame.csv`：特征总表
- `scored_frame.csv`：带因子打分结果的总表
- `factor_ic_history.csv`：每个交易日的因子 IC 历史
- `factor_weights.csv`：当前训练得到的因子权重
- `backtest_daily.csv`：每日回测收益和净值
- `rebalance_log.csv`：调仓记录
- `latest_portfolio.csv`：最新建议持仓
- `metrics.json`：汇总指标
- `run_summary.json`：本次训练的摘要信息

### 指标含义

`metrics.json` 当前包含：

- `total_return`
- `benchmark_total_return`
- `annualized_return`
- `benchmark_annualized_return`
- `annualized_volatility`
- `benchmark_annualized_volatility`
- `sharpe`
- `information_ratio`
- `max_drawdown`
- `excess_return`
- `win_rate_vs_benchmark`
- `average_turnover`
- `rebalance_count`

## GitHub Pages 部署

### 静态站生成逻辑

训练完成后，程序会把最新策略快照导出到：

- `docs/site-data.json`

网页本体 `docs/index.html` 会在浏览器里读取这个 JSON 并渲染图表。

### 本地预览

桌面应用中点击“打开静态网页”即可本地预览。  
如果你只想手动预览，也可以先运行桌面应用，再通过内置的小型本地 HTTP 服务打开 `docs/index.html`。

### 发布到 GitHub Pages

1. 确保仓库已经推送到 GitHub
2. 在仓库设置里打开：

```text
Settings -> Pages -> Deploy from branch -> main /docs
```

3. 训练完成后，把静态站相关文件一起提交

如果 `docs/site-data.json` 尚未被 Git 跟踪，请强制加入：

```powershell
git add docs/index.html docs/.nojekyll
git add -f docs/site-data.json
git commit -m "Update static quant dashboard"
git push
```

4. 等待 GitHub Pages 构建完成

如果你的仓库名是 `quant`，用户名是 `Batmancris`，那么地址通常是：

```text
https://batmancris.github.io/quant/
```

## 测试

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前测试覆盖的重点：

- 因子权重归一化
- 打分后能正确生成排名
- 回测能输出净值曲线和调仓日志

## 当前实现取舍

这个仓库故意做了一些取舍，让主线更聚焦：

- 不再保留旧的“单票短线神经网络分类”流程
- 不接券商、不自动下单
- 不做分钟级或高频
- 不接财报数据库、新闻数据库、研报数据库
- 先把“能稳定跑通的研究链路”搭起来，再谈更复杂的 alpha

## 已知局限

这套系统目前仍然是研究 demo，不应被当作实盘建议。主要局限包括：

- 股票池较小，目前只覆盖电力板块 16 只股票
- 因子数量有限，偏技术面和市场行为面
- 回测使用的是日线级近似，不包含更复杂的成交实现细节
- 电力板块代理指数是用股票池自行构造的，不是官方行业指数
- GitHub Pages 展示的是训练后的静态快照，不是实时行情页面

## 下一步适合继续做什么

如果你准备继续迭代，这几个方向最值得做：

1. 引入更多基本面因子，例如股息率、ROE、现金流质量、估值分位
2. 扩大样本池，不只限于 16 只电力股
3. 加行业中性、市值约束、换手约束
4. 增加滚动训练和 walk-forward 验证
5. 引入组合优化，而不只是等权 Top K
6. 把桌面应用打包成 `.exe`

## 风险提示

本项目仅用于学习、研究和工程演示，不构成任何投资建议。  
历史回测结果不代表未来收益，A 股交易存在市场风险、流动性风险、制度风险和模型失效风险。
