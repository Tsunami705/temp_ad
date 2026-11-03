
## 2026-10-31 Gate 0 按步保存中间量（runner_v2）
- 改动（官方脚本、`runner.py`、`server.py` 都未改）：
  - 新增 `UniAD/inference/runner_v2.py`：继承 `UniADRunner`，用包装函数/forward hook 截取中间量，每次模型调用存一个文件；
    import 时替换 `inference.runner.UniADRunner`。
  - `03_run_gate0.sh`：新增可选 `LATENT_TAG`（设置后 `APPTAINER_BIND` 挂载 `latents/` 和 arrays，sitecustomize 导入 runner_v2）、
    `ARRAY`（只跑指定 array 行）；sbatch 加 `--export=ALL,NCAP_SCENARIO=$SCENARIO`。不设 `LATENT_TAG` 时行为不变。
  - 新增检查脚本 `UniAD/tools/latents/check_ncap_latents.py`。
- 命令：`LATENT_TAG=gate0_smoke RUNS=1 SCENARIOS=frontal ARRAY=1 TIME_LIMIT=00:30:00 bash 03_run_gate0.sh`
- job 4699806_1（frontal-0103，1 run，2:01），镜像 uniad.sif / neuro-ncap.sif / neurad-studio.sif，UniAD commit 4827b8b + 未提交新文件。
  NCAP 输出 `neuro-ncap/outputs/2026-10-31_21-14-29/frontal-0103`；碰撞，ncap_score 0.0，impact_speed 13.68。
- 输出 `latents/gate0_smoke/frontal-0103/frontal-0103_run0_step{0..10}.pt` + `meta.json`，每步 0.13–0.8 MB（不存 BEV）。
- 检查（job 4701439 等）：11 步 = 3 priming + 7 auto + 1 终止后未执行；auto 步的规划轨迹与 NCAP `trajectories.json` 完全一致；
  `metrics.json` 中目标车检测框与保存的 track 框 5/5 步重合（误差 0.000 m）；agent 对齐（vehicle_mask / traj_preds / track_boxes）全部通过。
- 与方案的偏差：没有存 BEVfeat（可用 `UNIAD_LATENT_SAVE_BEV=1` 打开，约 40 MB/步）；NCAP 碰撞标签仍需从 NCAP 输出（metrics.json）另行整理。

## 2026-11-01 §4 生成闭环数据（UniAD 关碰撞优化，按步保存中间量）

- 设置：合作者确认旧数据**没有开 U-opt** → `MODEL_CFG_PATH=/UniAD/projects/configs/stage2_e2e/inference_e2e_nocol.py`。
- runner_v2 `FORMAT_VERSION=2`：新增 `traj_query_all`（motion head 最后一层，全部 tracked agent + sdc 最后一行，
  UniAD 过滤成只剩车辆之前）。RiskMonitor-G（F3，`/nobackup/proj/disk/naiss2024-6-300/shared/riskmonitor_g/`）
  的 UniAD 版用所有 tracked agent（`traj_all`），旧字段只有车辆。检查脚本加了「车辆行 == traj_query」。
- `03_run_gate0.sh`：`MODEL_CFG_PATH`、`TIME_NOW` 可从命令行传（默认不变）。新增 `05_gen_latents.sh`。
- 冒烟：job 4709796（frontal-0103，1 run，2:02），`latents/gate0_nocol/`，检查 ALL CHECKS PASSED，`use_col_optim=False`。
- 正式：`bash 05_gen_latents.sh`，run 数按旧 `train_test_split_tmp.py`（19 个组合 750 run，无 side-0921），
  同场景同 run 数合并成 array 作业，共 12 个 sbatch：4709907–4709918，TIME_LIMIT 03:00:00。
  latents：`latents/ncap_uniad_nocol/`；NCAP 输出：`neuro-ncap/outputs/2026-11-01_10-50-48/`（标签从这里取）。
- **改小规模**（Shipeng：先跑通流程，不需要这么多 run）：上面 12 个作业已 scancel（只有 4709907_3 启动过，没存下任何步）。
  改为 `RUNS_ALL=5 LATENT_TAG=ncap_uniad_nocol_r5 TIME_LIMIT=01:00:00 bash 05_gen_latents.sh`：19 个组合 × 5 run = 95 run，
  3 个 sbatch：4709931（frontal）、4709932（side）、4709933（stationary）。
  latents：`latents/ncap_uniad_nocol_r5/`；NCAP 输出：`neuro-ncap/outputs/2026-11-01_10-52-45/`。
- 小规模数据：3 个作业全部完成（每个任务 3–7 分钟），19 个组合各 5 run、1259 个 step 文件，检查全部 ALL CHECKS PASSED。
  95 个 run 撞了 84 个：stationary 90 %、frontal 96 %、side 75 %。

## 2026-11-01 RiskMonitor-T / -G 在 NCAP 上跑通（临时规则，只验证流程）

- 代码：`ad/riskmonitor/ncap_to_cache.py`（转成 riskmonitor_g 的 CachedBEVDataset 缓存 + 配置）、`run_pipeline.sbatch`。
  临时规则（Shipeng 同意）：plan_org = plan_query_org；最后一层 motion token；标签 = 每个 run 最后 6 个 NCAP 记录的步 =
  该 run 的 any_collide；scene 0103/0106/0108/0110 训练，其余测试；T 用 mixup 1.0，G 不用；focal α 0.5；1 个 member，20 epoch。
- job 4710123（1:12）。train 270 样本（234 正），test 287 样本（257 正）。
  测试集 AUROC / AP：T 94.97 / 99.42（272 个有车辆的样本），G 85.20 / 97.78（278 个有 agent 的样本）。
  正样本占 89 %、负样本只来自 5 个没撞的测试 run，数字没有意义，只说明流程能跑通。
- **疑点**：旧数据（`ref_ad/neuro-ncap/UniAD/ncap/output_1step.csv`，1667 个 run）的碰撞率是
  stationary 28.3 % / frontal 78.1 % / side 67.7 %，和 U-opt（论文 34.8 / 92.4 / 78.8，我们的完整评测 36.6 / 85.2 / 70.0）接近，
  和关碰撞优化（论文 87.8 / 98.4 / 79.6，这次 90 / 96 / 75）差得远；旧 UniAD 副本里也只有 `use_col_optim=True` 的配置。
  旧数据很可能是开着碰撞优化生成的，和「没有开 U-opt」的说法矛盾 → 问 Ziliang。

## 2026-11-01 字段定义：Ziliang 答复「和当前版本对齐」

- 当前版本 = `riskmonitor_g/data_loader/data_loaders.py::_load_bev_minimal`（nuScenes）：`plan_org` = `plan_org_queryG_*.pth`
  = `plan_query_org`（attention 之前），motion token 取 `traj_query[-1]`（最后一层）。和 `ncap_to_cache.py` 的临时规则一致，不用改。
- 旧 NCAP gitlab 仓库（`ref_ad/neuro-ncap` = gitlab.liu.se/zilxi06/neuro-ncap）里的做法，供对照：
  - 导出（master `UniAD/inference/runner.py`）：`plan_query`、`plan_org_query`、`traj_query` 原样 (3,1,A,6,256) 都存了。
  - 旧 NCAP 训练 loader（`data_loaders.py::NCAPDataset`）：读 `plan_org_query`（attention 之前），`traj_query[0, -1]` = **第 0 层**。
  - 闭环刹车（`origin/shipeng` 分支，commit 44af66d / d52fd48，2025-11）：`CatTransformer(d_model=256, num_layers=3)`，
    `checkpoint-epoch19.pth`，输入 `plan_query`（**attention 之后**）和 `traj_query[0, -1]`（第 0 层），最多 128 个 agent。
    阈值那一行（`prob_crash > 0.5`）在最新提交里被注释掉了。同一分支还有 MC Dropout 的 `runner_dropout.py`。
  - 也就是说旧 NCAP 用的是第 0 层，而且训练（attention 之前）和刹车（attention 之后）用的 plan query 不一致；这次统一按当前版本。

## 2026-11-01 RiskMonitor 训练：随机 6:1:3 切分（等 Ziliang 回复期间的试跑）

- 代码拷到 `/nobackup/proj/disk/cvls/personal/shipengl/riskmonitor_g/`（和共享目录逐文件一致，未改动）。
- 切分：按 run 随机 6:1:3，按 run 是否碰撞分层，同一 run 的 6 个标注步在同一集合。train 57 run / 336 样本（294 正），
  val 9 run / 53（47 正），test 29 run / 168（150 正）。
- 两处坑：`BaseDataLoader` 固定 `drop_last=True`（val batch 64 > 53 样本 → 空）；训练时验证会调 `metric.get_AUSE`，
  cpu / cuda 张量混用报错。不改他的代码：训练不挂 val，每 5 epoch 存 checkpoint，用 `a1_test_export.py` 在 val 上打分选 epoch。
- job 4714049（1:59）。val AP 在 4 个 checkpoint 上全是 100（val 只有 1 个负样本 run），选到 epoch 5。
  test（167 个有 agent 的样本，149 正）：T AUROC / AP 96.31 / 99.52，G 95.82 / 99.45。
  训练 loss：T（mixup）停在 0.02 左右，G（无 mixup）降到 0.0005，过拟合明显。
- 结果不能当真：负样本太少（test 3 个负样本 run、18 个样本）；随机切分下同一个 scene-sequence 同时出现在训练和测试里，
  论文附录 D.3 也说过这种情况下测试接近满分。

## 2026-11-01  Ziliang 补充的代码：复制和字段对照
- 复制（只读源 → 我们的目录）：
  - `shared/uniad_rebuttal2/` → `/nobackup/proj/disk/cvls/personal/shipengl/uniad_rebuttal2/`（3.7 G，含 DE 5 个 member；diff 一致）。
  - `shared/riskmonitor_g/` 用 rsync 同步到本地副本，新增 `evalmode_data.py`、`v2fuse_baselines.py`、`lpfcn*`、`mcd_evalmode.py`、`bootstrap_*`、`table1_ci.py`、`configs/rebuttal2/{V2fuse_all/base,LPFCN}`；本地原有文件没有改过，同步后与共享目录一致。
- 字段对照（`uniad_rebuttal2/tools/arr/export_eval_tokens.py` 对比我们的 `UniAD/inference/runner_v2.py` + `riskmonitor/ncap_to_cache.py`）：
  - `plan_org` = `attn_module` 的输入（两边是同一个 hook）→ 我们的 `plan_query_org`。一致，"attention 之前"已确认。
  - `traj` = `traj_query[-1, 0]`，最后一层，只含车辆；`nq_track` = 车辆数。一致。
  - `traj_all` / `motion_all` / `motion_scores_all` = `motion_head.forward` 输出的最后一层，去掉 sdc 行。一致。
    - 他 `motion_all` 只存前 6 步，我们存 12 步；`geometry()` 内部会截成 `[:, :, :6]`，结果相同。
  - `tracked`：
    - x、y 他取 `gravity_center`，我们取 `tensor[:, :2]`，xy 相同。
    - w、l、yaw = `tensor[:, 3:5]` 和 `tensor[:, 6]`，class = `labels_3d`，score = `scores_3d`（就是 `track_bbox_results[1]`）。
    - `forward_test` 只把 sdc 行的 label 改成 0，agent 行不受影响。一致。
  - `pre`：
    - 开碰撞优化时是 `collision_optimization` 的输入，没开时是规划输出。
    - 我们的 `plan_preopt` 规则相同（没开时等于 `plan`）。一致。
  - UniAD 模型代码：`planning_head.py` 相同；`motion_head.py` 只差一处训练用的 `.cpu()` 索引修复，不影响推理。
  - 结论：已存的 `ncap_uniad_nocol_r5` 不用重新生成；`ncap_to_cache.py` 里的"临时规则"中字段这一条已确认。
- 仍未解决（`prompt_for_ziliang.md` 已更新，只保留这些）：
  - 旧数据是否开了碰撞优化；
  - NCAP 上标签怎么算（他说是 CollisionLoss，delta 1.0 m，碰撞优化前的规划；NCAP 上需要 actor 的未来框）；
  - 6:1:3 的划分；
  - NCAP 的训练设置；
  - 闭环 MC Dropout 的做法；
  - nuScenes 上训好的 F3 / T checkpoint；
  - 刹车实验和 VAD。
- 和我们临时设置的差别：nuScenes 的 RiskMonitor-T 用 `V2fuse_all/base`（mixup α 3，focal 0.75，4 个 bagging）；我们 NCAP 试跑用的是 mixup α 1.0、focal 0.5、1 个 member。

## 2026-11-01  全流程（Table 2 的 UniAD 部分）
- 用户决定：
  - 碰撞太多时开碰撞优化，重点是正负样本均衡；
  - 标签沿用"每个 run 最后 6 步"（碰撞率太高时可改为最后 4 步）；
  - 先跑通全流程，细节之后再定。
- `UniAD/inference/runner_v2.py` → FORMAT_VERSION 3，加入 MC Dropout（同 `uniad_rebuttal2/tools/arr/export_mcd.py`）：
  - 在 `mlp_fuser` 末尾加 Dropout(0.1)（这一层没有参数），正常前向时关闭，所以控制车辆的规划不变；
  - 每步额外跑 T = 10 次规划头，关闭碰撞优化，得到 `mcd_plans` (10, 6, 2)；
  - Dropout 关闭时再跑一次，得到 `mcd_det`，必须等于 `plan_preopt`；
  - 用独立的 RNG，种子 = run × 1000 + step；环境变量 `UNIAD_MCD_T` / `UNIAD_MCD_P`；
  - checker 加了 3 项检查。
- 冒烟测试（job 4744934，开碰撞优化，frontal-0103，1 run，latents `smoke_mcd`）：ALL CHECKS PASSED。
  - `|mcd_det - plan_preopt|` = 0；
  - `plan` 和 `plan_preopt` 最多差 2 m，说明碰撞优化生效。
- `05_gen_latents.sh` 加了 `COL_OPT=1`（`inference_e2e.py`）。正式数据：
  - 命令：`COL_OPT=1 LATENT_TAG=ncap_uniad_colopt_r10 RUNS_ALL=10 TIME_LIMIT=01:30:00 bash 05_gen_latents.sh`；
  - job 4745154–4745156；
  - NCAP 输出 `neuro-ncap/outputs/2026-11-01_21-27-23/`。
- 全流程脚本（`ad/riskmonitor/`）：
  - `ncap_to_cache.py`：
    - 字段按 `export_eval_tokens.py` 组织；
    - RBD / GMM 直接调 `v2fuse_baselines.one`，MCD 分数 = 前 4 次 pass 终点协方差的迹（`mcd_evalmode.endpoint_cov`）；
    - 输出 T / G / LP-FCN 的 cache；
    - 默认按 scene 划分（方案 §4.3：0103/0106/0108/0110 训练，其余 run 按标签分层，val : test = 1 : 2）。
  - `run_pipeline.sbatch`：
    - 并行训练 T、G、LP-FCN（各 1 个 member；T/G 20 epoch，LP-FCN 300 epoch、在 epoch 200 测试）；
    - 用 `a1_test_export` 和 `score_lpfcn.py` 给测试集打分；
    - 最后跑 `ncap_eval.py`（AUROC/AP/Pr30/50/70/100，按 scene 重采样 bootstrap 2000 次，seed 0；有 agent 打不了分的样本给最低分）。
- 验证（job 4745021，r5 关碰撞优化的数据，scene 划分）：全流程跑通。
  - 测试集 191 个样本，正样本 90.6 %；
  - AUROC：GMM 25.0，RBD 34.7，LP-FCN 97.2，T 89.3，G 80.2；
  - 正样本太多，2000 次 bootstrap 里有 1074 次重抽到只有一类、被重新抽样。
  - 疑点：RBD > 0 的比例很高（标签 0 和标签 1 都约 80 %，nuScenes 上约 6 %），而且没撞的 run 分数更高。
    - 可能是 NCAP 场景里停着的车多；
    - 要用 BEV 图核实坐标系，待查。
- 正式数据完成（4745154–4745156，19 个任务全部 COMPLETED，每个 6–12 分钟）：190 个 run，107 个碰撞（56 %）。
  - frontal 43 / 50，side 35 / 40，stationary 29 / 100。
- 全流程（job 4749005，3.5 分钟）：输出在 `latents/riskmonitor/colopt_r10_scene/`（`table2.csv`、`test_scores.csv`）。
  - 划分：train 90 run / 540 样本（正样本 318）；val 33 / 196（106）；test 67 / 392（206，52.6 %），共 9 个 test scene。
  - 测试集 AUROC / AP（括号里是 95 % scene bootstrap CI，没有重抽单类样本）：

    | 方法 | AUROC | AP |
    |---|---|---|
    | MC Dropout | 39.6 | 50.0 |
    | GMM | 74.2 [45.6, 92.8] | 64.9 |
    | RBD | 70.7 | 62.6 |
    | LP-FCN | 66.7 | 66.8 |
    | RiskMonitor-T | 61.4 [39.5, 79.8] | 64.1 |
    | RiskMonitor-G | 63.1 [42.3, 81.5] | 69.6 |

    G − T 的 AUROC 差 +1.7 [−22.0, 28.9]，CI 都很宽。
  - 注意：
    1. 标签和场景类型强相关：测试集里 side 全是正样本，frontal 90 % 是正样本，stationary 只有 42 % 是正样本。
    2. 测试集有 43 个样本没有 agent（主要是 stationary-0101、0796、0966 里标签为 0 的 run），T / G 给最低分。
    3. T / G 只训 1 个 member、20 epoch；G 的 loss 降到 0.008。
    4. LP-FCN 按他的 'plan' 学习率计划，实际 lr 约 1.5e-6。
    5. MC Dropout 的终点方差很小（约 1e-4），AUROC 低于 50。

## 2026-11-02 逐帧标签（和 nuScenes 的 planning.loss_collision_aa_2 同一定义）+ 关碰撞优化重新生成数据

- 决定（Shipeng）：
  - 标签从「run 是否碰撞 + 最后 6 步」改为逐帧的 CollisionLoss。
  - 关掉碰撞优化；如果正样本太多、无法均衡，就把 6 步改成 4 步。
  - 模型只在 NCAP 数据上训练，不用 nuScenes 上的 checkpoint。
- 标签（`ad/riskmonitor/ncap_labels.py`）：
  - 用 `uniad_rebuttal2` 拷贝里的 `CollisionLoss`（delta 0 / 0.5 / 1.0）；计算对象是碰撞优化前的规划 `plan_preopt`。
  - 对比对象是 NCAP 的真值车辆框：`actors.json` 里所有车辆的世界位姿和 w / l，在 t + 0.5 s … t + 3 s 线性插值后变换到 t 时刻的 UniAD lidar 坐标系（`lidar_pose`，旋转是转置存的）。
  - 只算当时存在的车辆，和 NCAP 自己的碰撞检测一致。
  - 自车框朝向：取规划轨迹方向（nuScenes 用真实朝向；CollisionLoss 只用轴对齐范围，朝向的正负号无影响）。
  - t + N × 0.5 s 超过 scene 结束时间的步没有标签，不用。
- 检查（`check_ncap_labels.py`，在开碰撞优化的 r10 数据上，srun）：
  1. 从 `actors.json` + `ego_poses.json` 算出的目标车位姿，和 `metrics.json` 的 `target_actor_in_ego` 一致：1750 个位姿，误差 0。
  2. lidar2ego 每步相同：(0.974, 0, 1.84) m，yaw −90°。
  3. 目标车到最近的 UniAD 跟踪框：中位 0.81 m，p90 5.3 m（未检测到时取到的是别的车）。
  - 自动驾驶步的正样本比例（delta 1.0）：N=6 为 64.3 %，N=4 为 48.2 %。这批数据开了碰撞优化，没撞的 run 里也有 55 % 的步是正样本（被优化纠正了的危险规划），所以比例要看关优化的数据。
  - 撞车 run 的最后一个自动步只有 48 % 是正样本：抽查发现这时目标车已经和自车重叠（碰撞发生在 t 时刻），而标签只检查 t + 0.5 s 起的未来，和 nuScenes 的定义一致，不算错误。撞车 run 里平均 83 % 的步是正样本。
- `ncap_to_cache.py`：
  - 样本改为每个 run 中 NCAP 记录到的所有自动驾驶步，每步用自己的标签（`--label-steps 6|4`）；
  - 划分仍按 run，按 run 结果分层；`samples.csv` 增加 `run_collided` 列。
  - `run_pipeline.sbatch` 增加 `LABEL_STEPS`。
- 关碰撞优化、带 MC Dropout 的 r10 数据：jobs 4810141–4810143，latents `latents/ncap_uniad_nocol_r10/`，NCAP 输出 `neuro-ncap/outputs/2026-11-02_09-20-09/`。
- 关优化数据完成（4810141–4810143，19 个任务全部 COMPLETED，5–13 分钟）：190 个 run，172 个碰撞（run 级）。
  - 坐标检查同样通过（误差 0；lidar2ego 不变；目标车到最近跟踪框中位 0.86 m）。
  - 逐帧标签正样本比例（delta 1.0）：N=6 为 71.9 %（frontal 76.2 / side 68.6 / stationary 71.4）；N=4 为 55.1 %（53.9 / 56.5 / 55.0）。
  - 按约定改用 **N=4**。
  - 没撞的 18 个 run 里也有 41 % 的步是正样本。
- 全流程（job 4812072，3.5 分钟，`LABEL_STEPS=4`，scene 划分）：输出 `latents/riskmonitor/nocol_r10_n4_scene/`。
  - 划分：train 90 run / 858 样本（454 正）；val 33 / 266（152）；test 67 / 578（332，57.4 %）；9 个 test scene。没有重抽单类样本。
  - 测试集 AUROC / AP（95 % scene bootstrap CI）：

    | 方法 | AUROC | AP |
    |---|---|---|
    | MC Dropout | 65.6 [49.2, 75.2] | 67.5 |
    | GMM | 74.3 [68.6, 85.2] | 77.8 |
    | RBD | 82.3 [73.7, 88.3] | 83.3 |
    | LP-FCN | 52.5 [44.4, 62.1] | 63.2 |
    | RiskMonitor-T | 62.8 [54.0, 75.5] | 68.8 |
    | RiskMonitor-G | 75.1 [66.1, 87.6] | 80.8 |

  - 配对差（AUROC）：G − T +12.3 [5.3, 22.3]；G − RBD −7.3 [−18.0, 8.4]；T − RBD −19.5 [−30.8, −3.4]。
  - 按场景类型拆分（AUROC）：

    | 场景 | 样本 | 正样本 | T | G | RBD | GMM |
    |---|---|---|---|---|---|---|
    | frontal | 88 | 49 | 82.4 | 88.6 | 77.6 | 82.0 |
    | side | 25 | 20 | 55.0 | 28.0 | 64.0 | 84.0 |
    | stationary | 465 | 263 | 59.5 | 72.6 | 85.0 | 74.5 |

  - 仍是临时设置：T / G 各 1 个 member、20 epoch；LP-FCN 的 lr 约 1.5e-6。

## 2026-11-02 按 scene 交叉验证 + 4 个 bagging 成员 + 用验证集选 epoch

- 决定（Shipeng）：照我提的方案做。如果表现还是差，就加 scene 和 run，重点防过拟合。
- 改动（`ad/riskmonitor/`）：
  - `ncap_to_cache.py --split cv`：13 个 scene 按样本数均衡分到 5 折。第 k 折用自己的 scene 做测试，用第 k+1 折的 scene 做验证，其余约 8 个 scene 训练。
  - bagging：每个成员去掉 1/4 的训练 run（按 run 结果分层）。数据本身已均衡，所以没有用 nuScenes 的「全部正样本 + 1/4 负样本」规则。
  - 训练设置和论文的 NCAP 设置一致：T / G / LP-FCN 都用 focal α 0.5、γ 2、lr 1e-3、20 epoch，每 2 个 epoch 存一次 checkpoint。
    - LP-FCN 改用 `lpfcn_train.py` 自带的 `riskmonitor` 学习率计划；
    - T 的 mixup α 仍是 1.0。
  - `score_cv.py`：每个保存的 epoch 用 4 个成员的集成给验证集打分，取验证 AUROC 最高的 epoch 给测试集打分。
  - 新增 RBD-4 / GMM-4（只看前 4 步，和标签一致），用原函数，只截断输入。
  - `ncap_eval.py`：把各折的测试分数合起来（每个样本测一次），在 13 个 scene 上做 bootstrap；另报各折的均值 ± 标准差（`folds.csv`）。
- 试跑（4813053，2 epoch、2 个成员）通过；汇总时 `n` 字段重名，已修。
- 正式运行（job 4813278，11 分钟）：输出 `latents/riskmonitor/nocol_r10_n4_cv/`。1702 个测试样本，正样本 55.1 %，13 个 scene，没有重抽单类样本。

  | 方法 | AUROC [95 % CI] | AP | 各折 AUROC |
  |---|---|---|---|
  | MC Dropout | 54.7 [45.1, 66.5] | 58.0 | 56.4 ± 13.4 |
  | GMM | 71.6 [68.1, 78.8] | 72.7 | 75.0 ± 5.1 |
  | GMM-4 | 78.6 [75.1, 84.3] | 79.0 | 80.4 ± 6.2 |
  | RBD | 74.3 [66.9, 82.2] | 69.9 | 76.0 ± 10.3 |
  | RBD-4 | 80.2 [74.6, 86.5] | 76.0 | 80.9 ± 5.7 |
  | LP-FCN | 58.7 [48.3, 70.0] | 62.0 | 61.3 ± 10.4 |
  | RiskMonitor-T | 68.6 [59.7, 77.3] | 70.1 | 69.4 ± 13.5 |
  | RiskMonitor-G | **85.1 [80.5, 90.0]** | **87.2** | 85.7 ± 3.8 |

  - 配对差（AUROC）：G − T +16.5 [9.1, 23.3]；G − RBD +10.8 [4.1, 17.5]；G − RBD-4 +4.9 [0.3, 9.5]；T − RBD −5.7 [−14.6, 3.1]。
  - 按场景类型拆分（AUROC）：

    | 场景 | G | T | RBD |
    |---|---|---|---|
    | frontal | 85.3 | 78.7 | 77.6 |
    | side | 81.0 | 60.9 | 59.1 |
    | stationary | 87.6 | 67.3 | 80.8 |

  - 选中的 epoch：T 为 16/18/12/18/2，G 为 18/16/12/20/16，LP-FCN 为 20/20/6/16/2。T 和 LP-FCN 的验证 AUROC 随 epoch 波动大。
  - 和旧论文 Table 2（UniAD）对比：RiskMonitor 70.6、LP-FCN 64.7、MC Dropout 50.1、GMM 49.4。T、LP-FCN、MC Dropout 和旧值接近；GMM / RBD 修正后明显更强。
- `ncap_eval.py` 增加配对差的双侧 bootstrap p 值：2 × 0 另一侧的重抽比例；2000 次重抽，分辨率 0.001，p < 0.0005 显示为 0。写入 `table2.csv` 的 `p` 列。
  - 结果（同一批 2000 次重抽）：
    - G − T：AUROC p < 0.0005，AP p < 0.0005；
    - G − RBD：AUROC 0.003，AP < 0.0005；
    - G − RBD-4：AUROC 0.041，AP < 0.0005，Pr70 0.539；
    - T − RBD：AUROC 0.249。
  - 4 组 AUROC 比较做 Holm 校正后，G − RBD-4 为 0.082，不再显著；其余结论不变。

## 2026-11-02 晚：按论文 B.4 规划数据集（任务 1、3）+ 安全距离 0 / 0.5 m 的逐帧碰撞率（任务 2）

- Shipeng 的要求（睡前，中间的问题让我自己定）：
  1. 用现有碰撞率表，按 B.4 的整数规划和论文的划分，把逐帧碰撞率（6 步）控制在约 50 %；20 个组合全部用上，训练和测试不重叠；重新生成数据，训练，报告 AUROC / AP / Pr30 / 50 / 70 / 100。
  2. 关碰撞优化，安全距离取 0.5 m 或 0 m，每个组合 10 个 run，统计同样的逐帧碰撞率表。
  3. 用任务 2 的结果，把碰撞率控制在约 10 %。不行就开碰撞优化重复第 2 步；再不行就在均衡条件下尽量低。
  - 结果表和日志推到 https://github.com/Tsunami705/temp_ad 。
- side-0921 补跑 10 个 run，关优化（4854507）、开优化（4854508）各一份，两种设置下都是 0 / 10 碰撞。数据检查：只缺雷达（NeuRAD 不用）。现在 20 个组合齐全。
- 逐帧标签在 20 个组合上重算（`check_ncap_labels.py`，srun 4855215）。坐标检查全部通过。新增「执行轨迹」列（碰撞优化后的规划），仅供参考，正式标签仍用优化前的规划。
- 任务 2：表格见 GitHub `ncap_b4/rates_*.md`。整体逐帧碰撞率（6 步，优化前规划）：
  - 关优化：δ 0 / 0.5 / 1.0 分别为 62.2 / 64.8 / 67.2 %；
  - 开优化：53.8 / 57.6 / 60.5 %。
  - 除 side-0921（0 %）外，没有哪个组合低于 14 %。
- 规划脚本 `ncap_plan.py`：
  - 每个组合的 t_i（每个 run 的有标签时间戳数）、c_i（其中碰撞的）取自 10 个 run 的试跑。
  - B.4 约束：r_i ≥ 45，Σ r_i ≤ 1200，整体碰撞比例在目标 ± 4 %，最小化 D = max r_i − min r_i（D 相同时取 run 多的解）。
  - 划分按论文：整 scene 分到 train / val / test，时间戳比例 6:1:3（± 3 %），每个集合都包含三种场景类型。候选划分在 3^14 种分配里按和 6:1:3 的接近程度取前 100 个，逐个求解，取 D 最小的。
  - 自己加的约束：每个集合的碰撞比例在目标 ± 8 / 12 / 16 / 20 % 内（取第一个可行的），让测试集和训练集的比例不要差太远。
  - 目标不可行时（`--fallback min`），目标每次加 1 %，直到可行。
  - 过程中的决定：
    - 第一版要求每个集合都在 ± 4 % 内：任务 1 不可行。
    - 第二版加了 r_max ≤ 2 r_min：任务 3 最低只到约 49 %，和任务 1 没有区别。
    - 最后按 B.4 原文的均衡条件（只要求 ≥ 45，并最小化 D），不加倍数上限（`--spread 100`）。
- 最终规划（`latents/labels/plans_v2/`）：
  - **任务 1** 选**开优化、δ = 1.0**：关优化在这些约束下最低是 64 %，不可行。期望整体 54.0 %（train 48.5 / val 70.0 / test 61.0 %）；1053 个 run，每个组合 45–84 个，D = 39。
    - train scene：0099 0101 0103 0108 0346 0783 0921 0923；val：0110 0796；test：0106 0278 0331 0966。
  - **任务 3** 选**开优化、δ = 0**：10 % 在所有设置下都不可行。各设置在均衡条件下能达到的最低期望整体比例：关优化 δ0 58.1 %、δ0.5 64.0 %；开优化 δ0 44.0 %、δ0.5 47.0 %。取最低的开优化 δ0：期望整体 44.0 %（train 42.5 / val 60.0 / test 40.7 %）；1200 个 run，每个组合 45–131 个（side-0921 和 stationary-0099 各 131 个），D = 86。
    - train scene：0099 0101 0106 0108 0278 0331 0783 0923；val：0110 0796；test：0103 0346 0921 0966。
  - 原因：标签基于优化前的规划，除 0921 外每个组合的逐帧碰撞率都在 26 % 以上，在 ≥ 45 run / 组合的条件下整体压不到 10 %。
- 生成（开优化，两个任务共用一批 run：每个组合跑两个规划中较大的 run 数，共 1239 个 run；side-0108 从 124 补到 131，合并成一组）：jobs 4856441–4856447（7 个 sbatch），latents `latents/ncap_uniad_colopt_b4/`，NCAP 输出 `neuro-ncap/outputs/2026-11-02_23-00-00/`。

## 2026-11-03：任务 1、3 的训练和评测（B.4 规划的数据集）

- 生成 jobs 4856441–4856447 全部 COMPLETED；1239 个 run，每个组合的 run 数和规划一致。
- 流水线（run_pipeline.sbatch，SPLIT=plan，标签步数和 δ 取自规划的 key，6 步）：
  - 任务 1：job 4859348，24 min，OUT `latents/riskmonitor/b4_task1_colopt_d1.0`。
  - 任务 3：job 4859349，25 min，OUT `latents/riskmonitor/b4_task3_colopt_d0.0`。
- 实际逐帧碰撞率（整体 / train / val / test）：
  - 任务 1：54.8 / 48.5 / 70.1 / 63.8 %，样本比例 62.5 / 9.6 / 27.9 %；
  - 任务 3：44.5 / 43.8 / 62.5 / 38.2 %，样本比例 61.0 / 11.9 / 27.0 %。
- AUROC：
  - 任务 1：G 89.1，GMM-4 84.7，GMM 83.0，RBD 82.2，RBD-4 80.7，T 73.5，LP-FCN 56.8，MCD 56.6。
    - G−T +15.6（p<0.001）；G−RBD +6.9（p=0.12）；G−RBD4 +8.4（p<0.001）。
  - 任务 3：G 92.8，GMM 90.4，GMM-4 90.3，T 90.0，RBD 83.5，LP-FCN 77.6，RBD-4 76.8，MCD 44.2。
    - AUROC 配对差全部不显著（只有 4 个 test scene）。
    - Pr30 / Pr50 上 G 和 RBD 都比 T 高（p<0.001）。
- 完整表格（含 95% CI、p 值、选中的 epoch）：temp_ad `ncap_b4/RESULTS.md`。
- 小问题：
  - 任务 3 中 T 选在 epoch 20（最后一个）。
  - T 不给无车辆样本打分（原版行为），按最低分计：任务 1 25 个、任务 3 78 个。
