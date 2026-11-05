# NeuroNCAP 重跑（RiskMonitor Table 2，UniAD）实验索引

逐帧标签：UniAD `CollisionLoss`（轴对齐框，自车框加安全距离 δ）> 0。用的是碰撞优化之前的规划，检查前 6 步（3 s），和 NCAP 真值车辆框比较。
完整过程、命令、job 号和每个决定见 [ncap_b4/ncap_rerun_log.md](ncap_b4/ncap_rerun_log.md)。

| # | 实验 | 设置 | 结果文件 |
|---|---|---|---|
| 0 | 数据完整性检查工具 | 检查一个序列的 NCAP / NeuRAD / UniAD 数据是否齐全 | [check_data.py](check_data.py) |
| 1 | 逐帧碰撞率表，关碰撞优化（任务 2） | 20 个组合 × 10 run；δ = 0 / 0.5 / 1.0；6 步和 4 步 | [ncap_b4/rates_nocol_r10.md](ncap_b4/rates_nocol_r10.md) |
| 2 | 逐帧碰撞率表，开碰撞优化 | 同上，另加执行轨迹（优化后）一列，仅作参考 | [ncap_b4/rates_colopt_r10.md](ncap_b4/rates_colopt_r10.md) |
| 3 | B.4 整数规划：数据集划分和每组合 run 数 | 每组合 ≥ 45 run，总数 ≤ 1200，最小化 D；整 scene 按 6:1:3 分到 train / val / test | 采用的规划：[task1_colopt_d1.0.json](ncap_b4/plans/task1_colopt_d1.0.json)、[task3_colopt_d0.0.json](ncap_b4/plans/task3_colopt_d0.0.json)；求解日志（包括不可行的设置）：[ncap_b4/plans/](ncap_b4/plans/) |
| 4 | **任务 1**：碰撞率约 50 % 的数据集上训练和评测 | 开优化，δ = 1.0；1053 run；实际碰撞率：整体 54.8 %，train 48.5 %，test 63.8 % | [ncap_b4/RESULTS.md](ncap_b4/RESULTS.md)（表格和说明）、[table2_task1_colopt_d1.0.csv](ncap_b4/table2_task1_colopt_d1.0.csv)、[score_cv_task1_colopt_d1.0.log](ncap_b4/score_cv_task1_colopt_d1.0.log) |
| 5 | **任务 3**：碰撞率尽量低的数据集上训练和评测（10 % 不可行） | 开优化，δ = 0；1200 run；实际碰撞率：整体 44.5 %，train 43.8 %，test 38.2 % | [ncap_b4/RESULTS.md](ncap_b4/RESULTS.md)、[table2_task3_colopt_d0.0.csv](ncap_b4/table2_task3_colopt_d0.0.csv)、[score_cv_task3_colopt_d0.0.log](ncap_b4/score_cv_task3_colopt_d0.0.log) |

- **比较的方法（任务 1、3）：** MC Dropout、GMM、RBD、LP-FCN、RiskMonitor-T、RiskMonitor-G。参考行 GMM-4、RBD-4 只用前 4 步。
- **指标：** AUROC、AP、Pr30、Pr50、Pr70、Pr100。95 % CI 和 p 值来自 test scene 的配对 bootstrap（2000 次）。
- **csv 列：** `method, metric, value, lo, hi, p`。
- **代码：** [ncap_b4/code/](ncap_b4/code/)
  - 标签：`ncap_labels.py`、`check_ncap_labels.py`
  - 碰撞率表：`rate_table.py`
  - 规划：`ncap_plan.py`
  - 数据转换：`ncap_to_cache.py`
  - 打分：`score_cv.py`、`score_lpfcn.py`
  - 评测：`ncap_eval.py`
  - 流水线：`run_pipeline.sbatch`
