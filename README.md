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
       + step_mask * step_weight * step_terminal_reward
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

`--no-step-use-m2m-reward` 可只移除 step-labelled samples 的 M2M raw reward 和
component-shrink advantage 贡献；HumanML 的 M2M 权重与训练逻辑完全不变。关闭后
仍计算并记录 step prompt 上的 M2M 指标，便于与启用 M2M 的实验做统计比较。

默认 step 配置使用两套独立的 group size：HumanML 每条文本 `K_human=4`，step
每条文本 `K_step=16`。`--step-data-ratio` 的语义是 **motion sample 占比**，不是
prompt 占比；默认 0.25 时，一个完整 physical rollout batch 为 64 motions：

```text
12 HumanML prompts x 4 motions = 48
 1 step prompt      x 16 motions = 16
-----------------------------------
total                              64
```

因此 HumanML:step 仍为 3:1，同时 K=16 提高同一 step 文本内的排序信息。配置会拒绝
无法组成完整 group 的组合，例如 `rollout_batch_size=32`、`K_step=16`、ratio=0.25，
不会静默改变 step 占比。

### Event-level soft count 与去混杂诊断

hard count 继续作为 exact、within-one 和 MAE 的最终评测口径，但不再是唯一可用的
训练信号。`progressive` backend 会同时读取 kept/filtered step candidate metadata，
用 lead、step-length、root/global-progress 的 signed margin 计算事件置信度。实现使用
相对阈值归一化：

```text
p_gate = sigmoid((measured - threshold) / (abs(threshold) * tau))
p_event = product(p_lead, p_length, p_progress, ...)
```

候选先按时间聚类，每个 cluster 只保留最大置信度，避免同一次落脚或脚踝抖动被重复
计数：

```text
soft_count = sum(max(candidate_confidence in temporal_cluster))
```

推荐 reward mode 为 `soft_huber_exact`：

```text
z = (soft_count - target) / per_target_error_scale
reward_step = -Huber(z; delta) + exact_bonus * 1[hard_count == target]
```

`per_target_error_scale` 在原始 MDM 上按 target 1--6 分别 calibration 后固定；训练中
禁止动态更新。默认 `tau=1.0`、cluster gap `0.15 s`、Huber delta `1.0`、exact bonus
`0.15`、scale floor `0.25`。rollout 会额外记录 unique reward levels、pairwise tie、
nonzero-advantage fraction、top-1 concentration、candidate count/spacing，以及 ankle
velocity 的高频能量占比，用来识别稀疏排序和 detector reward hacking。

`--step-rollout-source synthetic` 提供 count-learnability 隔离数据：不绑定某个
pseudo-labelled reference motion，target 与 length 独立；所有 target 共享同一套模板
顺序，length 只从各 target 共同支持区间采样。synthetic step motion 仅提供 MDM 所需
的 shape/length，不充当 GT，因此该模式强制 `--no-step-use-m2m-reward`。HumanML
rollout、retrieval/M2M 和 diffusion anchor 逻辑保持不变。

仓库还提供 counterfactual number probe：对每个 condition 固定初始 noise、每一步
DDIM noise、motion length 和 prompt template，只替换数字 1--6；它同时报告 hard/soft
count 的 target Spearman、target/length 回归效应、text embedding 可分性和成对 motion
距离。

已完成的 24 conditions × 2 noises × 6 targets（每个 policy 288 motions）正式 probe
表明：原始 MDM hard target-count Spearman 为 `0.4540`，target standardized effect
为 `0.5247`，length effect 为 `0.1810`；projected `embed_text` 的最小 RMS 距离为
`0.02306`。因此模型确实区分并使用数字 token，当前不触发 explicit count embedding
fallback。主要问题是 count response 在约 4 步处饱和，尤其 target 2 exact 为 0。
完整分析见 `reports/counterfactual_number_probe_analysis.md`。

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

Mixed training can override these weights only for step-labelled groups. For
example, HumanML `0.5/0.5` and step `0.2/0.0/0.8` are expressed with
`--step-advantage-retrieval-weight`, `--step-advantage-m2m-weight`, and
`--step-advantage-step-weight`. Unspecified step-specific weights inherit the
global HumanML values for backward compatibility.

floor 只能来自固定 `reward_calibration.json` 的 p25 或 p50，不会在 minibatch
中动态估计。训练日志包含两个 component advantage 的 correlation、sign conflict
fraction、各自 mean-absolute contribution 和 effective maximum scale。
启用 step 时，step floor 来自与当前 step K 对应的独立 calibration（默认
`step_reward_k16_calibration.json`）。hard reward
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
每 prompt 16 motions。它记录 hard reward、exact、within-1、MAE、detected mean，以及
soft count/error、candidate spacing 和 ankle 高频能量，以及步数 prompt 上的
retrieval/M2M paired delta。为避免悄然改变已有模型选择语义，
`best_balanced.pt` 仍只由 HumanML val retrieval+M2M 决定；step 单项最佳另存为
`best_step.pt`。另有更严格的 `best_step_acceptance.pt`：只有 HumanML feasible，且
hard MAE delta < 0、exact delta > 0、within-one delta > 0 三项同时满足时才可更新。
该文件仍只是单个验证点的候选，最终验收必须确认至少三个连续 counterfactual
验证点改善。

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

### 1b. 生成 step reward calibration

正式 step 训练还需在原始 MDM、零 LoRA 上生成独立 calibration：

```bash
CUDA_VISIBLE_DEVICES=7 \
python tools/calibrate_step_reward_stats.py \
  --output step_reward_k16_calibration.json \
  --pool-path artifacts/step_reward_k16_calibration_pool.pt \
  --samples-output artifacts/step_reward_k16_calibration_samples.pt \
  --prompts 384 \
  --samples-per-prompt 16 \
  --batch-size 64 \
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
mode/temperature 是否与 calibration 完全一致。calibration 还绑定 step K：已有的
`step_reward_calibration.json`（K=4）不能用于新的 K=16 训练，必须按上面的命令重新生成。

新的 K8 soft-count 隔离实验必须重新生成对应 calibration；不能复用 K8
`negative_l1` 或 K16 JSON：

```bash
CUDA_VISIBLE_DEVICES=7 \
python tools/calibrate_step_reward_stats.py \
  --output step_reward_k8_soft_huber_calibration.json \
  --pool-path artifacts/step_reward_k8_soft_huber_pool.pt \
  --samples-output artifacts/step_reward_k8_soft_huber_samples.pt \
  --prompts 384 \
  --samples-per-prompt 8 \
  --batch-size 64 \
  --sample-steps 50 \
  --step-pool-source synthetic \
  --step-reward-mode soft_huber_exact \
  --step-soft-lead-temperature 1.0 \
  --step-soft-length-temperature 1.0 \
  --step-soft-progress-temperature 1.0 \
  --device cuda:0 \
  --precision bf16
```

该 JSON 除固定 reward scale/floor 外，还保存 target 1--6 各自的 soft-count error
RMSE scale、reward ties/unique levels 和 detector candidate/jitter diagnostics。

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
  --train-batch-size 64 \
  --gradient-accumulation-steps 1 \
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
MDM_DDPO_STEP_REWARD_CALIBRATION_PATH=$PWD/step_reward_k16_calibration.json \
MDM_DDPO_FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH=$PWD/artifacts/step_val_fixed_eval_pool_k16.pt \
MDM_DDPO_USE_SWANLAB=1 \
MDM_DDPO_SWANLAB_PROJECT=mdm-ddpo-step \
MDM_DDPO_SWANLAB_RUN_NAME=humanml-step-p25 \
bash scripts/train_humanml_step.sh \
  --epochs 100 \
  --sample-steps 50 \
  --rollout-batch-size 64 \
  --rollout-batches-per-epoch 2 \
  --samples-per-prompt 4 \
  --step-samples-per-prompt 16 \
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
  --fixed-eval-samples-per-prompt 4 \
  --fixed-step-eval-samples-per-prompt 16 \
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
STEP_REWARD_CALIBRATION_PATH=$PWD/step_reward_k16_calibration.json \
bash scripts/run_step_reward_ablations.sh
```

输出 `step_ablation_comparison.csv/.md`，同时包含 balanced retrieval/M2M 与
best-step reward/MAE/exact delta，避免只看 detector reward 而忽略原任务退化。

### 3c. K8 count-learnability diagnostic

先生成与训练 K 和 reward mode 严格匹配的 calibration：

```bash
CUDA_VISIBLE_DEVICES=7 \
python tools/calibrate_step_reward_stats.py \
  --output step_reward_k8_negative_l1_calibration.json \
  --pool-path artifacts/step_reward_k8_negative_l1_calibration_pool.pt \
  --samples-output artifacts/step_reward_k8_negative_l1_calibration_samples.pt \
  --prompts 384 \
  --samples-per-prompt 8 \
  --batch-size 64 \
  --sample-steps 50 \
  --step-reward-mode negative_l1 \
  --device cuda:0 \
  --precision bf16
```

然后运行固定 30 epochs 的隔离诊断：

```bash
CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
OUTPUT_DIR=$PWD/outputs/step_k8_negative_l1_diagnostic \
MDM_DDPO_REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
MDM_DDPO_STEP_REWARD_CALIBRATION_PATH=$PWD/step_reward_k8_negative_l1_calibration.json \
MDM_DDPO_FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH=$PWD/artifacts/step_val_fixed_eval_pool_k8.pt \
MDM_DDPO_USE_SWANLAB=1 \
MDM_DDPO_SWANLAB_PROJECT=mdm-ddpo-step \
bash scripts/train_step_k8_diagnostic.sh
```

该脚本使用 50% step motions、K8、每 epoch 16 个均衡 pseudo-target prompt groups、
HumanML `0.5 retrieval + 0.5 M2M` 与 step
`0.2 retrieval + 0.0 M2M + 0.8 negative-L1 step`。优先检查 `best_step.pt`，并要求
MAE/exact/within-one 三项改善持续多个 fixed validation points。

训练前可运行原始 caption target 与 detector 的 reference-motion confusion：

```bash
python tools/validate_step_detector_gt.py \
  --output artifacts/step_gt_detector_validation.json \
  --step-targets 1,2,3,4,5,6 \
  --step-detector-backend progressive
```

### 3d. Counterfactual probe 与 K8 soft-count 隔离实验

先在不训练的情况下检查原始 MDM，以及可选的已有 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=7 \
python tools/probe_step_number_conditioning.py \
  --output artifacts/k8_negative_l1_counterfactual_probe.json \
  --pool-path artifacts/step_counterfactual_number_pool.pt \
  --samples-output artifacts/k8_negative_l1_counterfactual_probe_samples.pt \
  --conditions 24 \
  --samples-per-condition 2 \
  --batch-size 48 \
  --sample-steps 50 \
  --step-targets 1,2,3,4,5,6 \
  --number-style words \
  --device cuda:0 \
  --precision bf16 \
  --checkpoints \
    outputs/step_k8_negative_l1_diagnostic/checkpoint_000011.pt \
    outputs/step_k8_negative_l1_diagnostic/checkpoint_000029.pt
```

旧 K8 `negative_l1` run 在去混杂 pool 上的结果为：

| policy | hard MAE | exact | within-one | hard target Spearman |
| --- | ---: | ---: | ---: | ---: |
| original MDM | 1.14236 | 0.28125 | 0.65625 | 0.45403 |
| epoch 11 | 1.14583 | 0.28125 | 0.65278 | 0.44934 |
| epoch 29 | 1.13542 | 0.28472 | 0.65625 | 0.46389 |

epoch 29 的 MAE 仅改善 `-0.00694`、exact 仅 `+0.00347`；epoch 11 反而轻微退化。
因此不要继续延长或提高旧 hard-count run 的权重。`tau=0.25` 时 mean
`abs(soft-hard)=0.0051`，几乎退化为 hard count；`tau=1.0` 时提高到 `0.1742`，
所以生产 soft calibration 和训练必须使用新的默认 `1.0`。

完成上一节 K8 soft calibration 后，运行固定 30 epochs 的 synthetic 隔离实验：

```bash
cd /home/zhiwei/projects/mdm-ddpo

CUDA_VISIBLE_DEVICES=7 \
DEVICE=cuda:0 \
REWARD_DEVICE=same \
OUTPUT_DIR=$PWD/outputs/step_k8_soft_counterfactual \
MDM_DDPO_REWARD_CALIBRATION_PATH=$PWD/reward_calibration.json \
MDM_DDPO_STEP_REWARD_CALIBRATION_PATH=$PWD/step_reward_k8_soft_huber_calibration.json \
MDM_DDPO_FIXED_EVAL_POOL_PATH=$PWD/artifacts/humanml_val_fixed_eval_pool.pt \
MDM_DDPO_USE_SWANLAB=1 \
MDM_DDPO_SWANLAB_MODE=online \
MDM_DDPO_SWANLAB_PROJECT=mdm-ddpo-step \
bash scripts/train_step_soft_counterfactual.sh --seed 42
```

脚本固定 `step_data_ratio=0.5`、K8、每 epoch 16 个 balanced synthetic step
groups；HumanML 使用 `0.5 retrieval + 0.5 M2M`，step 使用
`0 retrieval + 0 M2M + 1.0 soft-step`，每 2 epochs 保存/验证且不 early stop。
训练完成后，用完全相同的 counterfactual pool 对所有 numbered checkpoints 做
post-hoc paired probe：

```bash
CUDA_VISIBLE_DEVICES=7 DEVICE=cuda:0 \
bash scripts/probe_step_counterfactual_checkpoints.sh \
  outputs/step_k8_soft_counterfactual
```

验收只看固定 counterfactual 结果与 HumanML feasibility：target-count correlation
明显为正并持续增强；hard MAE 至少三个连续点下降；exact/within-one 持续净增长；
soft reward tie fraction 显著低于 hard reward；candidate spacing 和 ankle 高频能量不
异常。只有 probe 显示 embedding 可区分、但生成 count 对数字不响应时，才进入
explicit count embedding + native diffusion SFT；当前正式 probe 尚未触发该条件。

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
| `--step-reward-calibration-path` | 固定 hard/soft step calibration；step `component_shrink` 与 soft mode 必需 |
| `--enable-step-reward` | 启用 HumanML + step 混合数据与 step reward；默认关闭 |
| `--step-data-ratio` | 每个 rollout 中 step **motion sample** 的比例；默认 0.25 |
| `--step-rollout-source` | `reference` 或去除 target-length/GT 混杂的 `synthetic` |
| `--step-synthetic-seed` | synthetic 共享 length/template 设计的固定 seed |
| `--samples-per-prompt` | HumanML 每条文本的 rollout K；默认 4 |
| `--step-samples-per-prompt` | step 每条文本的 rollout K；默认 16 |
| `--step-balanced-sampling` | 在短 rollout 窗口内轮转并均衡 target 类别；默认启用 |
| `--step-targets` | hard label 类别，默认 `1,2,3,4,5,6` |
| `--step-data-manifest` | RFT_MLD/Motion-Rule step pseudo-label manifest |
| `--step-detector-backend` | `progressive`（默认）或本地 `rgdno` |
| `--step-reward-mode` | `exp`、`linear`、`exact`、`negative_l1` 或 `soft_huber_exact` |
| `--step-soft-{lead,length,progress}-temperature` | progressive signed-margin sigmoid 温度；默认各 1.0 |
| `--step-soft-cluster-gap-seconds` | event candidate 时间聚类间隔；默认 0.15 s |
| `--step-soft-huber-delta` | normalized soft-count Huber delta；默认 1.0 |
| `--step-soft-exact-bonus` | hard exact 小额奖励；默认 0.15 |
| `--step-soft-target-scale-floor` | per-target calibration error scale 下限；默认 0.25 |
| `--step-ankle-high-frequency-cutoff-hz` | 脚踝速度高频审计 cutoff；默认 4 Hz |
| `--step-reward-weight` | step component 在 raw total reward 中的权重 |
| `--step-use-m2m-reward` / `--no-step-use-m2m-reward` | 是否让 M2M 参与 step 样本更新；不影响 HumanML，默认启用 |
| `--advantage-step-weight` | step component-shrink advantage 权重；默认 0.25 |
| `--step-advantage-retrieval-weight` | step groups 专用 retrieval advantage 权重；默认继承全局值 |
| `--step-advantage-m2m-weight` | step groups 专用 M2M advantage 权重；默认继承全局值 |
| `--step-advantage-step-weight` | step groups 专用 step advantage 权重；默认继承 `--advantage-step-weight` |
| `--fixed-step-eval-pool-path` | 跨 run 共享的 held-out step fixed pool |
| `--split` | rollout split；只能为 `train` |
| `--eval-split` | fixed validation split；checkpoint selection 只能使用 `val` |
| `--fixed-eval-pool-path` | 跨 run 共享的精确 fixed pool；恢复时必须匹配 pool id |
| `--fixed-eval-prompts` | 默认 128 |
| `--fixed-eval-samples-per-prompt` | 默认 4，与 rollout K 独立 |
| `--fixed-step-eval-samples-per-prompt` | step fixed eval 的 K；默认 16 |
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
- `step/{soft_count_mean,soft_count_mae,soft_hard_count_difference_mean}`；
- `step/{raw_candidate_count_mean,candidate_count_mean,candidate_spacing_mean,ankle_high_frequency_ratio}`；
- `advantage/step_*`、retrieval-step/M2M-step correlation 与 conflict fraction；
- `advantage/step_{unique_reward_levels,pairwise_reward_tie_fraction,nonzero_advantage_sample_fraction,top1_advantage_concentration}`；
- `eval/reward_{retrieval,m2m}_delta`、paired bootstrap SE 和 improvement fraction；
- `eval/normalized_{retrieval,m2m}_delta`；
- `eval/balanced_score`、`eval/balanced_score_bootstrap_se`、`eval/feasible`；
- `anchor/{loss,weighted_loss,grad_norm,ppo_grad_norm,grad_ratio,lambda,calls}`。
- `step_eval/{reward,exact_fraction,within_one_fraction,mae,soft_count,soft_mae}` 及 paired delta、SE；
- `step_eval/normalized_reward_delta`、`step_eval/is_best_step` 与
  `step_eval/is_best_acceptance`。

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
├── best_step_acceptance.pt
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
跨输出目录恢复会复制已有的五个 best checkpoint。旧版没有 balanced state 的
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
balanced feasibility、hard/event-level soft step detector、step manifest 分层切分、
synthetic target-length factorial sampler、counterfactual number probe、masked step
advantage、per-target soft calibration、step acceptance checkpoint、HumanML-only
anchor、每 update 一次 anchor 以及实验汇总/规划。

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
- reference step pool 的 hard count 与 motion length、模板仍可能混杂；synthetic
  rollout 和 counterfactual probe 已解除该混杂，但正式结果仍需按 target/length
  分层报告。
- event-level soft count 仍由 detector metadata 构造，filtered candidate 的 margin
  calibration 可能被 policy 利用；必须同时监控 candidate spacing、ankle 高频能量并
  人工检查高 reward motion。
- 当前 detector pseudo-label 尚未用人工 held-out motion 验证为“人类可见真实步数”；
  counterfactual improvement 只能证明 detector-count control。
- `best_step.pt` 不保证 HumanML retrieval/M2M 可行；正式候选优先从
  `best_step_acceptance.pt` 或 HumanML-feasible checkpoint 中检查 counterfactual
  step delta；单个尖峰仍不满足三个连续验证点的验收要求。
