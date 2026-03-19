# Quant

这个项目现在只有一条主线：`电力股多因子量化策略`。

它提供两种交付方式：

- `Windows 桌面应用`：本地训练、查看摘要、打开静态网页
- `GitHub Pages 静态网页`：把最新回测结果发布成公开网页

## 功能

- 股票池：16 只 A 股电力股
- 信号：多因子横截面排序，不再做单票短线涨跌神经网络分类
- 组合：等权持有评分最高的前 `K` 只股票
- 调仓：每 `N` 个交易日调仓一次
- 回测：包含换手、交易成本、卖出税和等权行业基准比较
- 输出：桌面端摘要 + `docs/` 静态站

## 运行

抓取数据：

```powershell
.\run_data.cmd
```

训练策略：

```powershell
.\run_train.cmd --holding-period-days 5 --rebalance-frequency-days 5 --top-k 3
```

启动 Windows 桌面应用：

```powershell
.\run_app.cmd
```

## GitHub Pages

每次训练完成后都会导出：

- 静态页面：[docs/index.html](/e:/research/0/finance/docs/index.html)
- 页面数据：[docs/site-data.json](/e:/research/0/finance/docs/site-data.json)

推送到 GitHub 后，在仓库里开启：

`Settings -> Pages -> Deploy from branch -> main /docs`

就可以得到公开网页。

## 目录

- `app.py`：Windows 桌面应用
- `greenpower_demo/train.py`：策略训练入口
- `greenpower_demo/quant_strategy.py`：因子、选股、回测主逻辑
- `docs/`：GitHub Pages 静态站

## 说明

- 这是研究型量化 demo，不是投资建议
- 当前默认策略更偏向“低波动 + 换手稳定”
- 历史回测不代表未来表现
