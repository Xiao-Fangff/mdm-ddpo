# MDM-DDPO

本仓库在不修改外部参考仓库的前提下，将 HumanML3D MDM、DDPO/PPO 与
MotionRFT 的 retrieval + M2M 奖励组合起来，并针对长期训练中的奖励微差放大、
验证泄漏和原始 MDM 退化问题增加了完整的稳定化与统计评估流程。

外部仓库仅在运行时只读导入：

- `/home/zhiwei/projects/motion-diffusion-model`
- `/home/zhiwei/projects/MotionRFT`
- `/home/zhiwei/projects/ddpo-pytorch` 仅作为算法参考

## 稳定化实现

- PPO 在 sample level 生成随机 permutation，再组成等大的 minibatch。
- 强制 `rollout_batch_size * rollout_batches_per_epoch` 可被
  `train_batch_size` 整除，避免尾 batch 权重错误。
- 每个 rollout 的首次 optimizer update 前审计 old/new log-prob；超过阈值会
  立即停止，而不是带着错误 ratio 继续训练。
- 严格检查恢复 checkpoint 是否包含全部 trainable LoRA tensors。
- 记录 `log_ratio`、ratio dispersion、LoRA norm、parameter update norm 和非有限梯度。
- rollout 始终使用 HumanML3D `train`；checkpoint 选择始终使用 held-out `val`。
- 固定验证池持久化为 `fixed_eval_pool.pt`，包含 dataset indices、caption、length、
  随机裁剪后的 GT motion 和逐 prompt diffusion noise seed。
- 恢复训练或跨实验共享 pool 时校验 SHA-256 pool id；`test` split 禁止用于
  checkpoint 选择。
- 默认固定验证使用 128 prompts，每个 prompt 4 motions，并保存逐 prompt 的
  retrieval/M2M baseline、current 和 paired delta。
- 使用原始 MDM 的 1024 prompts × 4 motions 离线标定固定 reward scale 和
  shrinkage floor；训练中不动态更新这些尺度。
- 提供 `group_shrink` 和 `component_shrink`，避免 group whitening 对极小 reward
  差异产生数百倍放大。
- checkpoint 使用 calibration-normalized balanced score，而不是 raw total reward。
- 可选原生 MDM diffusion loss anchor，每个 optimizer update 只计算一次；支持把
  anchor 初始梯度自动标定到 PPO 梯度的 10% 或 20%。默认关闭。
- 可选 SwanLab 曲线记录；JSONL 始终写入本地输出目录。

## 核心目标

DDIM rollout 对每个随机 transition 保存：

```text
log p_old(x_{t-1} | x_t, text)
```

更新时在完全相同的 `x_t`、`x_{t-1}`、timestep 和 conditioning 上重算：

```text
log_ratio = log p_new - log p_old
ratio     = exp(log_ratio)
loss_ppo  = mean(max(-A * ratio,
                     -A * clip(ratio, 1-eps, 1+eps)))
```

terminal reward 为：

```text
reward = retrieval_weight * cosine(text, generated_motion)
       + m2m_weight       * cosine(gt_motion, generated_motion)
       + step_mask * step_weight * hard_step_reward
```

MDM 的 263-D motion 会先从 MDM normalization 反归一化，再转换到 MotionReward
normalization，避免两个项目 Mean/Std 不一致导致输入域偏移。

### Hard step-count reward 与混合数据

可选步数奖励完全不使用 `StepCountNet`，也不建立可微 reward path。生成 motion
先从 MDM normalization 反归一化，通过 `recover_from_ric` 恢复 HumanML22 xyz，
再由确定性 hard detector 计数：

- 默认 `progressive`：与 RFT_MLD 标签链路一致，使用
  `progressive_step(step_candidate_source=lead_offsets, lead_threshold=0.138)`；
- 可消融 `rgdno`：本仓库内置的左右 ankle motion-energy threshold-transition
  detector，不依赖 Motion-Rule 运行时。

RFT_MLD manifest 的原始 prompt 可能与 `detected_steps` 不一致，因此训练不会复用
原始 prompt，而是按 hard label 生成确定性的 digit/word、多模板 step prompt。
默认目标为 1–6；0-step 可显式加入，但因为 stand-still 文本与原始样本语义噪声更大，
不作为默认配置。

步数 reward 默认是有界、非稀疏的：

```text
error       = abs(hard_detected_steps - target_steps)
reward_step = exp(-error / temperature)
```

也支持 `linear`、`exact` 和 `negative_l1`。普通 HumanML 样本使用
`target_steps=-1` 和 `step_mask=false`，其 step reward 严格为 0；步数样本同时保留
retrieval、M2M 和 step reward。

默认 physical rollout batch 32、每 prompt 4 motions 时，共 8 个 prompt groups：
6 个 HumanML + 2 个 step，等价于 24:8 motion samples（3:1），接近 RFT_MLD 的
约 3.2:1 混合比例。比例由 `--step-data-ratio` 配置并记录实际整数 prompt 数。

## Advantage 模式

设同一 prompt 的 centered reward 为 `c = reward - group_mean`。

`group_whiten` 保留用于基线复现：

```text
A = c / (group_std + epsilon)
```

它可能把极小组内差异放大，因此会同时记录
`potential_group_whiten_scale_max`。

`group_centered` 保留用于消融：

```text
A = c / global_centered_std
```

`group_shrink` 使用 calibration 中固定的 total reward floor：

```text
A = c / sqrt(group_std^2 + std_floor^2)
```

`component_shrink` 分别处理 retrieval、M2M，以及被 mask 的可选 step component：

```text
A_ret = centered_ret / sqrt(group_std_ret^2 + floor_ret^2)
A_m2m = centered_m2m / sqrt(group_std_m2m^2 + floor_m2m^2)
A_step = centered_step / sqrt(group_std_step^2 + floor_step^2)
A      = w_ret*A_ret + w_m2m*A_m2m + step_mask*w_step*A_step
```

floor 只能来自固定 `reward_calibration.json` 的 p25 或 p50，不会在 minibatch
中动态估计。训练日志包含两个 component advantage 的 correlation、sign conflict
fraction、各自 mean-absolute contribution 和 effective maximum scale。
启用 step 时，step floor 来自独立的 `step_reward_calibration.json`。hard reward
离散且经常产生零方差 prompt group，所以 calibration 同时记录 raw quantile 和
zero-std fraction，实际 floor 使用正方差 group 的 p25/p50，避免得到 0 floor。

## Fixed validation 与 balanced checkpoint

验证对每个 prompt 先平均 4 个生成 motion，再做 paired baseline delta。每个
component 都记录：

- mean 与 median；
- improvement fraction；
- paired bootstrap standard error；
- baseline、current 和 delta。

使用 calibration 的全局标准差归一化：

```text
z_retrieval = retrieval_delta / retrieval_global_scale
z_m2m       = m2m_delta       / m2m_global_scale
balanced_score = 0.5 * z_retrieval + 0.5 * z_m2m
```

可行性阈值为：

```text
retrieval_delta >= -k * bootstrap_se_retrieval
m2m_delta       >= -k * bootstrap_se_m2m
```

默认 `k=1`。只有两个 component 都没有超出统计误差的退化时，当前 policy 才是
feasible，才允许更新 `best_balanced.pt`。retrieval 与 M2M 的单项最佳仍分别保存，
便于分析 trade-off。

`early_stop_min_delta_mode=auto` 时，有效最小改进为：

```text
max(early_stop_min_delta,
    early_stop_se_multiplier * balanced_score_bootstrap_se)
```

step held-out pool 独立持久化为 `fixed_step_eval_pool.pt`，默认每个目标 8 prompts、
每 prompt 4 motions。它记录 hard reward、exact、within-1、MAE、detected mean，以及
步数 prompt 上的 retrieval/M2M paired delta。为避免悄然改变已有模型选择语义，
`best_balanced.pt` 仍只由 HumanML val retrieval+M2M 决定；step 单项最佳另存为
`best_step.pt`。

## 原生 MDM diffusion anchor

anchor 使用真实 HumanML3D motion、当前带 LoRA 的基础 MDM 和原始 1000-step
diffusion 的 `training_losses`。它不经过 CFG sampling wrapper。

每个 optimizer update 的梯度等价于：

```text
loss = loss_ppo + lambda_anchor * loss_mdm
```

anchor 只在 accumulation group 即将 step 时计算一次，不会在每个 DDIM
transition 上重复计算。两种配置方式互斥：

- `--anchor-lambda X`：固定 lambda；
- `--anchor-auto-grad-ratio 0.1`：首次 update 自动令
  `||lambda * grad_anchor|| / ||grad_ppo|| ≈ 0.1`，之后固定该 lambda。

默认两者都是 0，完全关闭 anchor，保持向后兼容。
混合训练时 anchor 会跳过 step manifest 中的伪标注/生成 GT，只使用真实 HumanML3D
motion；即使某个随机 PPO accumulation group 恰好全是 step samples，也会从当前
trajectory 中确定性回退选择 HumanML motion。

## 环境和默认资源

当前环境：

```bash
/home/zhiwei/anaconda3/envs/motionrft/bin/python
```

默认模型资源：

```text
MDM checkpoint:
  /home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert/model000600000.pt

MDM args:
  /home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert/args.json

MotionReward backbone:
  /home/zhiwei/projects/MotionRFT/checkpoints/motionreward/stage1_retrieval_backbone_r128.pth

Sentence-T5:
  /home/zhiwei/projects/MotionRFT/deps/sentence-t5-large
```

## 1. 生成 reward calibration

正式训练前先在原始 MDM 上运行一次：

```bash
cd /home/zhiwei/projects/mdm-ddpo

CUDA_VISIBLE_DEVICES=7 \
python tools/calibrate_reward_stats.py \
  --output reward_calibration.json \
  --pool-path artifacts/reward_calibration_pool.pt \
  --samples-output artifacts/reward_calibration_samples.pt \
  --split train \
  --prompts 1024 \
  --samples-per-prompt 4 \
  --batch-size 32 \
  --sample-steps 50 \
  --device cuda:0 \
  --reward-device same \
  --precision bf16
```

生产 calibration 强制至少 1024×4。`--allow-small-run` 仅用于工具 smoke；其 JSON
会标记 `full_calibration=false`，训练端会拒绝加载。

输出统计包括：

- retrieval、M2M 和 total 的 `global_scale`；
- 每个 component 组内 std 的 p25/p50/mean/min/max；
- 组内 range 的 p25/p50/mean/min/max；
- raw/global Pearson correlation；
- group-centered correlation；
- prompt 内成对 ranking conflict fraction 和 tie fraction。

### 1b. 生成 hard-step reward calibration

正式 step 训练还需在原始 MDM、零 LoRA 上生成独立 calibration：

```bash
CUDA_VISIBLE_DEVICES=7 \
python tools/calibrate_step_reward_stats.py \
  --output step_reward_calibration.json \
  --pool-path artifacts/step_reward_calibration_pool.pt \
  --samples-output artifacts/step_reward_calibration_samples.pt \
  --prompts 384 \
  --samples-per-prompt 4 \
  --batch-size 32 \
  --sample-steps 50 \
  --step-targets 1,2,3,4,5,6 \
  --step-detector-backend progressive \
  --step-reward-mode exp \
  --step-reward-temperature 1.0 \
  --device cuda:0 \
  --precision bf16
```

生产 calibration 强制至少 256 prompts × 4 samples，并要求 prompt 数能被目标类别数
整除，以保持分层平衡。`--allow-small-run` 只用于流程 smoke；训练端会拒绝
`full_calibration=false`。恢复训练时还会校验 detector backend/threshold 和 reward
mode/temperature 是否与 calibration 完全一致。

## 2. 预检和 smoke test

不执行 rollout 的预检：

```bash
CUDA_VISIBLE_DEVICES=7 DEVICE=cuda:0 bash scripts/preflight.sh
```

带固定验证和一次 optimizer update 的 smoke：

```bash
CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
OUTPUT_DIR=/tmp/mdm-ddpo-smoke \
MDM_DDPO_REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
MDM_DDPO_FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
bash scripts/train_humanml.sh \
  --dry-run \
  --fixed-eval-every 1
```

若 calibration 尚未生成，只测试训练接口，可显式关闭 fixed validation，并使用
非 shrink advantage：

```bash
python train_ddpo.py \
  --dry-run \
  --device cuda:0 \
  --reward-device same \
  --fixed-eval-every 0 \
  --advantage-mode group_centered \
  --output-dir /tmp/mdm-ddpo-interface-smoke
```

## 3. 推荐训练命令

在完成 calibration 后，推荐先从 component shrink、无 anchor 开始：

```bash
cd /home/zhiwei/projects/mdm-ddpo

CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
OUTPUT_DIR=$PWD/outputs/humanml_component_shrink_p25 \
MDM_DDPO_USE_SWANLAB=1 \
MDM_DDPO_SWANLAB_MODE=online \
MDM_DDPO_SWANLAB_PROJECT=mdm-ddpo \
MDM_DDPO_SWANLAB_RUN_NAME=humanml-component-shrink-p25 \
MDM_DDPO_REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
MDM_DDPO_FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
bash scripts/train_humanml.sh \
  --epochs 100 \
  --sample-steps 50 \
  --rollout-batch-size 32 \
  --rollout-batches-per-epoch 4 \
  --samples-per-prompt 4 \
  --train-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --inner-epochs 1 \
  --timestep-fraction 0.5 \
  --learning-rate 1e-4 \
  --clip-range 1e-4 \
  --advantage-mode component_shrink \
  --advantage-std-floor-quantile p25 \
  --advantage-retrieval-weight 0.5 \
  --advantage-m2m-weight 0.5 \
  --fixed-eval-every 5 \
  --fixed-eval-prompts 128 \
  --fixed-eval-samples-per-prompt 4 \
  --early-stop-min-delta-mode auto \
  --early-stop-patience 8 \
  --anchor-auto-grad-ratio 0
```

该配置每个 epoch rollout 128 motions / 32 prompts；PPO physical batch 为 32，
gradient accumulation 为 2，因此每个 optimizer update 的有效样本数为 64，
每个 epoch 共 2 次 optimizer updates。

`rollout_batch_size * rollout_batches_per_epoch` 必须能被 `train_batch_size`
整除。增大 physical batch 时，若保持有效 update size 不变，应同步减小
`gradient_accumulation_steps`，不需要按 batch size 线性放大学习率。

### 3b. 推荐 hard-step 混合训练命令

先完成 retrieval/M2M 与 step 两份 calibration，然后运行：

```bash
cd /home/zhiwei/projects/mdm-ddpo

CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
OUTPUT_DIR=$PWD/outputs/humanml_step_component_shrink \
MDM_DDPO_ENABLE_STEP_REWARD=1 \
MDM_DDPO_REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
MDM_DDPO_STEP_REWARD_CALIBRATION_PATH=$PWD/step_reward_calibration.json \
MDM_DDPO_FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH=$PWD/artifacts/step_val_fixed_eval_pool.pt \
MDM_DDPO_USE_SWANLAB=1 \
MDM_DDPO_SWANLAB_PROJECT=mdm-ddpo-step \
MDM_DDPO_SWANLAB_RUN_NAME=humanml-step-p25 \
bash scripts/train_humanml_step.sh \
  --epochs 100 \
  --sample-steps 50 \
  --rollout-batch-size 32 \
  --rollout-batches-per-epoch 4 \
  --samples-per-prompt 4 \
  --train-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --timestep-fraction 0.5 \
  --learning-rate 1e-4 \
  --clip-range 1e-4 \
  --advantage-std-floor-quantile p25 \
  --step-data-ratio 0.25 \
  --step-reward-weight 0.5 \
  --advantage-retrieval-weight 0.375 \
  --advantage-m2m-weight 0.375 \
  --advantage-step-weight 0.25 \
  --fixed-eval-every 5 \
  --early-stop-patience 8
```

这里 `0.375/0.375/0.25` 是保守起点，不是已完成多 seed 搜索的最优权重。
`--step-reward-weight` 影响 raw total reward；在 `component_shrink` 下，真正控制
PPO step component 梯度占比的是 `--advantage-step-weight`。

步数奖励的最小消融脚本固定同一 HumanML/step pool、seed 和 30 epochs，比较
“只加入 step-labelled prompts”与两个 step advantage 权重：

```bash
CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
STEP_REWARD_CALIBRATION_PATH=$PWD/step_reward_calibration.json \
bash scripts/run_step_reward_ablations.sh
```

输出 `step_ablation_comparison.csv/.md`，同时包含 balanced retrieval/M2M 与
best-step reward/MAE/exact delta，避免只看 detector reward 而忽略原任务退化。

## 4. 固定消融 A0–A4

以下脚本保证：无 `--resume`、独立输出目录、原始 MDM、零 LoRA、同一 fixed val
pool、固定 30 epochs、关闭 early stopping：

```bash
CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
OUTPUT_ROOT=$PWD/outputs/stability_ablations \
bash scripts/run_stability_ablations.sh
```

矩阵为：

| ID | advantage | floor | learning rate | clip range |
| --- | --- | --- | ---: | ---: |
| A0 | `group_whiten` | — | `3e-4` | `1e-4` |
| A1 | `group_centered` | — | `1e-4` | `1e-4` |
| A2 | `group_shrink` | p25 | `1e-4` | `1e-4` |
| A3 | `group_shrink` | p50 | `1e-4` | `1e-4` |
| A4 | `component_shrink` | p25 | `1e-4` | `1e-4` |

脚本结束后生成：

```text
outputs/stability_ablations/ablation_comparison.csv
outputs/stability_ablations/ablation_comparison.md
```

也可手工汇总任意 runs：

```bash
python tools/summarize_experiments.py \
  outputs/stability_ablations/A* \
  --output-prefix outputs/stability_ablations/ablation_comparison
```

## 5. 选择最好两组后缩小 LR/clip 范围

不要直接运行 2×3×3 全排列。规划工具先按 feasible best balanced score 选前两组，
再依据训练期间的 `clip_fraction_mean` 与 `ratio_std_mean / clip_range` 生成单因素
优先的子集，候选值严格来自：

```text
learning rate = {3e-5, 1e-4, 3e-4}
clip range    = {1e-4, 3e-4, 1e-3}
```

```bash
python tools/plan_followup_sweeps.py \
  outputs/stability_ablations/A* \
  --output-json outputs/followup_plan/followup_plan.json \
  --output-script outputs/followup_plan/run_followups.sh \
  --run-output-root outputs/followup_sweeps

# 先检查 JSON 和 shell，再执行：
bash outputs/followup_plan/run_followups.sh
```

规划规则：高 clip fraction / 大 ratio dispersion 优先降低 LR 或放宽 clip；几乎没有
clipping 且 ratio dispersion 很小时优先提高 LR；中间区域一次只改变 LR 或 clip。

## 6. 最佳配置的 anchor × seed 复现

将最佳 run 目录传给脚本，会运行：

```text
anchor grad ratio = {0, 0.1, 0.2}
seed              = {42, 43, 44}
```

共 9 个独立、从原始 MDM 开始的 30-epoch runs：

```bash
CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
OUTPUT_ROOT=$PWD/outputs/anchor_seed_sweep \
bash scripts/run_anchor_seed_sweep.sh outputs/followup_sweeps/BEST_RUN
```

输出 `anchor_seed_comparison.csv/.md`，再按 seed 聚合 retrieval delta、M2M delta
和 balanced score 的均值与标准误。

## 参数速查

| 参数 | 说明 |
| --- | --- |
| `--reward-calibration-path` | 固定 calibration JSON；fixed checkpoint selection 和 shrink 模式必需 |
| `--step-reward-calibration-path` | hard-step 固定 calibration；step `component_shrink` 必需 |
| `--enable-step-reward` | 启用 HumanML + step 混合数据与 hard reward；默认关闭 |
| `--step-data-ratio` | 每个 rollout prompt batch 中 step prompt 的目标比例；默认 0.25 |
| `--step-targets` | hard label 类别，默认 `1,2,3,4,5,6` |
| `--step-data-manifest` | RFT_MLD/Motion-Rule step pseudo-label manifest |
| `--step-detector-backend` | `progressive`（默认）或本地 `rgdno` |
| `--step-reward-mode` | `exp`、`linear`、`exact` 或 `negative_l1` |
| `--step-reward-weight` | step component 在 raw total reward 中的权重 |
| `--advantage-step-weight` | step component-shrink advantage 权重；默认 0.25 |
| `--fixed-step-eval-pool-path` | 跨 run 共享的 held-out step fixed pool |
| `--split` | rollout split；只能为 `train` |
| `--eval-split` | fixed validation split；checkpoint selection 只能使用 `val` |
| `--fixed-eval-pool-path` | 跨 run 共享的精确 fixed pool；恢复时必须匹配 pool id |
| `--fixed-eval-prompts` | 默认 128 |
| `--fixed-eval-samples-per-prompt` | 默认 4，与 rollout K 独立 |
| `--fixed-eval-bootstrap-samples` | paired bootstrap 次数，默认 2000 |
| `--advantage-mode` | `group_whiten`、`group_centered`、`group_shrink`、`component_shrink` |
| `--advantage-std-floor-quantile` | calibration floor 的 `p25` 或 `p50` |
| `--advantage-retrieval-weight` | component advantage 固定权重，默认 0.5 |
| `--advantage-m2m-weight` | component advantage 固定权重，默认 0.5 |
| `--clip-range` | PPO ratio clipping，候选 `1e-4/3e-4/1e-3` |
| `--log-prob-audit-tolerance` | 首次 update old/new log-prob 最大允许差异，默认 `1e-4` |
| `--checkpoint-feasible-se-multiplier` | component 可行性容差的 bootstrap SE 倍数，默认 1 |
| `--early-stop-min-delta-mode` | `fixed` 或 `auto`；默认 `auto` |
| `--early-stop-se-multiplier` | auto min delta 的 balanced SE 倍数 |
| `--anchor-lambda` | 固定 diffusion anchor 系数；默认 0 |
| `--anchor-auto-grad-ratio` | 初始 anchor/PPO 梯度目标比；默认 0 |
| `--anchor-batch-size` | 每次 update 的 distinct real motions；0 使用最多一个 train batch |
| `--timestep-fraction` | 每个 motion 用于 PPO 的随机 transition 比例；默认 0.5 |
| `--train-batch-size` | PPO physical sample batch |
| `--gradient-accumulation-steps` | 组成一次 optimizer update 的 PPO minibatch 数 |
| `--reset-optimizer-on-resume` | 算法迁移时恢复 policy/RNG，但重置 AdamW/GradScaler |
| `--reward-device cpu` | 显存不足时把 MotionReward/T5 放到 CPU |

## 训练指标

重要 SwanLab/JSONL 指标：

- `audit/*`：首次 old/new log-prob 和 ratio 一致性；
- `ppo/log_ratio_{mean,std,abs_max}`、`ppo/ratio_std`、`ppo/clip_fraction`；
- `optimization/{grad_norm,lora_norm,update_norm,skipped_updates}`；
- `advantage/{std_floor,effective_shrink_scale_max,component_correlation,component_conflict_fraction}`；
- `advantage/{retrieval,m2m}_contribution_mean_abs`；
- `step/{exact_fraction,within_one_fraction,mae,detected_mean,target_mean}`；
- `advantage/step_*`、retrieval-step/M2M-step correlation 与 conflict fraction；
- `eval/reward_{retrieval,m2m}_delta`、paired bootstrap SE 和 improvement fraction；
- `eval/normalized_{retrieval,m2m}_delta`；
- `eval/balanced_score`、`eval/balanced_score_bootstrap_se`、`eval/feasible`；
- `anchor/{loss,weighted_loss,grad_norm,ppo_grad_norm,grad_ratio,lambda,calls}`。
- `step_eval/{reward,exact_fraction,within_one_fraction,mae}` 及 paired delta、SE；
- `step_eval/normalized_reward_delta` 与 `step_eval/is_best_step`。

随机 rollout 的 `reward/total` 会受到每轮 prompt 难度组成影响，不应作为主要模型
选择依据。优先观察 held-out paired component deltas、balanced score 和 feasibility。

## 输出文件

```text
OUTPUT_DIR/
├── config.json
├── metrics.jsonl
├── fixed_eval_pool.pt
├── fixed_eval.jsonl
├── fixed_eval_per_prompt.jsonl
├── fixed_step_eval_pool.pt
├── fixed_step_eval_per_prompt.jsonl
├── checkpoint_000004.pt
├── best_balanced.pt
├── best_retrieval.pt
├── best_m2m.pt
├── best_step.pt
├── latest.pt
└── swanlab/
```

`latest.pt` 每个 epoch 都更新；numbered checkpoint 按 `save_every`、best 更新、early
stop 或最终 epoch 保存。checkpoint 包含 optimizer、GradScaler、RNG、fixed baseline、
pool id、两份 calibration id、balanced/step best 状态和自动标定后的 anchor lambda。

恢复训练：

```bash
python train_ddpo.py \
  --resume outputs/run/latest.pt \
  --output-dir outputs/run \
  --reward-calibration-path reward_calibration.json \
  --epochs 200
```

恢复时会严格校验 train mode、全部 trainable tensors、calibration id 和 fixed pool id。
跨输出目录恢复会复制已有的四个 best checkpoint。旧版没有 balanced state 的
checkpoint 会以恢复后的 policy 建立新 baseline，并给出明确警告。

## 导出与标准 HumanML 快速评测

导出 LoRA 到标准 MDM checkpoint：

```bash
python export_ddpo.py \
  --checkpoint outputs/run/best_balanced.pt \
  --output outputs/exported/model_ddpo.pt
```

标准 HumanML `debug` 评测包含 5 次 replication。脚本先把 baseline checkpoint
复制到本仓库输出目录，因此不会向外部 MDM 仓库写文件：

```bash
CUDA_VISIBLE_DEVICES=7 EVAL_DEVICE=0 \
bash scripts/run_standard_humanml_eval.sh \
  outputs/run/best_balanced.pt \
  outputs/humanml_standard_eval/run
```

验收时比较 baseline 与 candidate 的 FID、Matching Score、R-precision 和 Diversity；
候选退化应不超过 baseline 自身 replication confidence interval。正式论文结果应使用
更完整的 `wo_mm` 或项目约定评测，而不是只报告 debug。

## 消融比较表模板

| Run | Seed | Advantage | Floor | LR | Clip | Anchor ratio | Best epoch | Retrieval Δ | M2M Δ | Balanced | Balanced SE | Feasible | Clip fraction | Ratio std |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| A0 | 42 | group_whiten | — | 3e-4 | 1e-4 | 0 |  |  |  |  |  |  |  |  |
| A1 | 42 | group_centered | — | 1e-4 | 1e-4 | 0 |  |  |  |  |  |  |  |  |
| A2 | 42 | group_shrink | p25 | 1e-4 | 1e-4 | 0 |  |  |  |  |  |  |  |  |
| A3 | 42 | group_shrink | p50 | 1e-4 | 1e-4 | 0 |  |  |  |  |  |  |  |  |
| A4 | 42 | component_shrink | p25 | 1e-4 | 1e-4 | 0 |  |  |  |  |  |  |  |  |

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q mdm_ddpo tools train_ddpo.py export_ddpo.py tests
bash -n scripts/*.sh
```

测试覆盖 DDIM log-prob、padding、sample-level shuffle、严格 checkpoint loading、
fixed pool 持久化、paired bootstrap、calibration checksum、shrinkage 数值稳定性、
balanced feasibility、hard step detector、step manifest 分层切分、masked step
advantage、step calibration、HumanML-only anchor、每 update 一次 anchor 以及实验汇总/规划。

## 已知限制

- 1024×4 calibration、A0–A4、follow-up 和 9 组 anchor/seed 是长时间 GPU 实验，
  仓库提供可复现脚本，但代码提交本身不能替代这些统计结果。
- validation 改善不保证标准 HumanML evaluator 的所有指标同步改善，因此最终配置
  必须通过标准评测门槛。
- retrieval 与 M2M 可能存在真实 ranking conflict；`component_shrink` 和 feasible
  balanced checkpoint 限制其破坏，但不能从理论上消除任务目标冲突。
- PPO clip 约束单次 update，不等价于对原始 MDM 的长期 KL 约束；diffusion anchor
  是当前的长期保护机制，仍需用 0/0.1/0.2 与三个 seed 验证收益。
- step manifest 标签本身来自 hard detector，不是人工步数标注；policy 可能学习
  detector-specific artifact，因此 exact/MAE 改善必须配合视频人工审计。
- hard step count 与 motion length 有明显混杂。实现不会根据 target 人为选择长度，
  fixed pool 会锁定真实长度，但仍需按 target/length 分层报告结果。
- `best_step.pt` 不保证 HumanML retrieval/M2M 可行；正式候选优先从
  `best_balanced.pt` 中检查 step delta，或在多目标实验中另行定义显式约束。
