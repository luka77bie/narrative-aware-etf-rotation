# Narrative-Aware ETF Rotation

[English](README.md) | [简体中文](README.zh-CN.md)

一个面向中国 ETF 的可复现量化轮动研究平台，用于研究和比较：

- 动量策略
- 风险控制
- 市场关注度信号
- 政策叙事数据
- 样本外验证
- 最新 ETF 配置建议

本项目不是自动交易机器人，也不直接连接券商下单。

它的主要作用是：

> 使用历史数据测试每个月应该按什么规则选择 ETF、长期表现如何、风险多大，以及策略结果是否可靠。

---

## 项目能做什么

这个项目可以完成以下工作：

1. 读取 ETF 历史行情数据
2. 计算 20 日和 60 日动量
3. 对 ETF 进行横截面排名
4. 每月选择排名最高的 3 只 ETF
5. 按等权方式构建组合
6. 模拟次日执行
7. 计算交易成本和换手率
8. 评估 CAGR、Sharpe、Sortino 和最大回撤
9. 运行子区间稳健性分析
10. 运行 Walk-Forward 样本外验证
11. 自动生成图表、Markdown 和 HTML 报告
12. 输出最新 Top-3 ETF 配置及换仓建议

---

## 核心策略

当前选定的主策略是：

> **MOM60 月度 Top-3 ETF 轮动**

基本规则：

```text
每月计算一次
→ 计算所有 ETF 的 60 日动量
→ 对 ETF 从高到低排序
→ 选择前 3 只
→ 每只配置约 33.33%
→ 下一个交易日执行
→ 下个月重新计算
```

动量定义：

```text
MOM60 = adjusted_close(t) / adjusted_close(t - 60) - 1
```

---

## 当前研究结论

最终模型状态：

| 组件 | 状态 |
|---|---|
| MOM60 | 当前选定的主模型 |
| MOM60 + 50% 市场关注度 Proxy | 探索性候选模型 |
| 10%–30% Proxy | 已否决 |
| Risk Engine | 已评估 |
| Walk-Forward Validation | 已完成 |
| Policy Narrative V1 | 仅用于流程验证 |
| Narrative Signal V2 | 仅用于流程验证 |
| 完整历史新闻 | 尚未正式接入 |

50% Market Attention Proxy 在全样本中改善了部分指标，但在总体 Walk-Forward 样本外结果中没有稳定超过 MOM60。

因此目前保留：

```text
主模型：
MOM60

探索模型：
MOM60 + 50% Market Attention Proxy

政策叙事信号：
仅用于流程验证
```

---

## 主要历史结果

### 全样本结果

| 模型 | CAGR | Sharpe | Sortino | 最大回撤 | Calmar |
|---|---:|---:|---:|---:|---:|
| MOM60 | 14.93% | 0.644 | 0.861 | -45.70% | 0.327 |
| MOM60 + 50% Proxy | 15.83% | 0.680 | 0.913 | -42.26% | 0.375 |

### Walk-Forward 样本外结果

| 模型 | Folds | OOS CAGR | OOS Sharpe | OOS Sortino | OOS 最大回撤 | OOS Calmar |
|---|---:|---:|---:|---:|---:|---:|
| MOM60 | 4 | 22.21% | 0.859 | 1.196 | -24.34% | 0.913 |
| MOM60 + 50% Proxy | 4 | 21.46% | 0.838 | 1.162 | -25.26% | 0.850 |

总体样本外结果支持 MOM60 继续作为主模型。

---

## 最新 ETF 配置报告

项目可以根据最新有效行情生成当前 Top-3 ETF 配置。

运行：

```bash
python3 scripts/run_momentum_signal.py
python3 scripts/generate_current_allocation.py
```

生成：

```text
outputs/reporting/current_allocation.csv
outputs/reporting/current_allocation.md
outputs/reporting/current_allocation.html
```

macOS 打开：

```bash
open outputs/reporting/current_allocation.html
```

报告会显示：

- 最新数据日期
- 当前 Top-3 ETF
- ETF 代码
- ETF 中文名称
- MOM60 数值
- Momentum Score
- 目标权重
- 上一期持仓
- 新增 ETF
- 剔除 ETF
- 保留 ETF
- 预计单边换手率
- 数据是否过期
- 次日执行说明

示例结构：

```text
Current MOM60 Allocation

Rank  Ticker  ETF Name            Target Weight
1     159516  半导体设备ETF国泰       33.33%
2     159732  消费电子ETF华夏         33.33%
3     588000  科创50ETF华夏          33.33%
```

具体结果以本地最新行情数据为准。

---

## 系统流程

```text
ETF Universe
    ↓
历史行情数据
    ↓
数据验证
    ↓
Momentum Signal
    ↓
Risk Controls
    ↓
Portfolio Construction
    ↓
Backtest
    ↓
Walk-Forward Validation
    ↓
Current Allocation
    ↓
Charts
    ↓
Markdown / HTML Report
```

---

## 安装

### 克隆项目

```bash
git clone git@github.com:luka77bie/narrative-aware-etf-rotation.git
cd narrative-aware-etf-rotation
```

### 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 安装依赖

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

建议使用 Python 3.9 或更高版本。

---

## 运行测试

运行完整测试：

```bash
python3 -m pytest -v
```

检查 Python 文件：

```bash
python3 -m compileall -q main.py src scripts tests
```

---

## 运行完整 Pipeline

查看所有步骤：

```bash
python3 main.py --list-steps
```

完整运行：

```bash
python3 main.py
```

跳过测试：

```bash
python3 main.py --skip-tests
```

从指定步骤继续：

```bash
python3 main.py \
  --skip-tests \
  --from-step "Generate current allocation report"
```

---

## 常用命令

### 生成 Momentum 排名

```bash
python3 scripts/run_momentum_signal.py
```

输出：

```text
outputs/latest_momentum_ranking.csv
outputs/momentum_signal_history.csv
```

### 生成最新配置

```bash
python3 scripts/generate_current_allocation.py
```

### 运行 Momentum 回测

```bash
python3 scripts/run_momentum_backtest.py
```

### 运行市场关注度 Proxy

```bash
python3 scripts/run_narrative_proxy_signal.py
python3 scripts/run_proxy_composite_ablation.py
python3 scripts/run_proxy_robustness.py
```

### 运行 Walk-Forward

```bash
python3 scripts/run_walk_forward_validation.py
```

### 生成研究报告

```bash
python3 scripts/generate_research_charts.py
python3 scripts/generate_research_report.py
python3 scripts/generate_research_html.py
```

打开主报告：

```bash
open outputs/reporting/research_report.html
```

---

## 主要输出文件

### 最新信号

```text
outputs/latest_momentum_ranking.csv
outputs/momentum_signal_history.csv
```

### 最新配置

```text
outputs/reporting/current_allocation.csv
outputs/reporting/current_allocation.md
outputs/reporting/current_allocation.html
```

### 样本外验证

```text
outputs/walk_forward_fold_metrics.csv
outputs/walk_forward_aggregate_metrics.csv
outputs/walk_forward_oos_returns.csv
```

### 主研究报告

```text
outputs/reporting/research_report.md
outputs/reporting/research_report.html
```

### 图表

```text
outputs/reporting/charts/nav_comparison.png
outputs/reporting/charts/drawdown_comparison.png
outputs/reporting/charts/walk_forward_sharpe.png
outputs/reporting/charts/subperiod_cagr.png
```

---

## 项目结构

```text
.
├── .github/
│   └── workflows/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── templates/
├── docs/
├── outputs/
│   └── reporting/
├── scripts/
├── src/
│   ├── backtest/
│   ├── data/
│   ├── evaluation/
│   ├── narrative/
│   ├── portfolio/
│   ├── reporting/
│   ├── risk/
│   └── signals/
├── tests/
├── main.py
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

---

## 数据与回测控制

项目包含以下控制：

- 次日交易执行
- 交易成本
- 换手率
- 最低 ETF 覆盖率
- 重复日期检测
- 价格过期检测
- 固定参数样本外验证
- 非重叠 Walk-Forward 窗口
- Point-in-Time 数据可用性
- 新闻和政策发布时间控制
- Retrieval Time 控制
- Market Close 时间控制
- 防止 Look-Ahead Bias
- 防止 Post-OOS 参数优化

---

## 项目限制

- 历史表现不代表未来表现
- 当前项目不连接券商 API
- 当前项目不会自动下单
- 当前项目不是实时交易系统
- ETF Universe 存在成立日期差异
- 可能存在 Survivorship Bias
- 成交费用和滑点进行了简化
- 当前 Market Attention Proxy 不等于真实新闻情绪
- 当前政策数据不是完整历史 Point-in-Time 数据库
- Walk-Forward fold 数量仍然有限
- Top-3 等权组合存在集中度风险
- 完整 Pipeline 需要本地历史 ETF 数据

---

## 适用人群

本项目适合：

- 学习量化交易的学生
- Quant Research 求职者
- Data / FinTech 求职者
- ETF 轮动研究者
- 想学习 Backtesting 的开发者
- 想研究 Alternative Data 的研究人员

---

## 一句话介绍

> 这个项目会使用历史数据测试：每个月应该选择哪几只 ETF、按什么规则换仓、长期表现如何，以及这个结果是否可靠。

---

## 免责声明

本项目仅用于：

- 研究
- 教学
- 软件演示
- 方法验证

本项目不构成：

- 投资建议
- 买卖推荐
- 收益保证
- 证券发行或交易要约

任何实盘使用都应在独立验证后进行。
