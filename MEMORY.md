# MDM-DDPO 项目记忆

最后更新：2026-07-19

## 当前实现状态

retrieval + M2M DDPO 稳定化已按阶段完成，外部
`motion-diffusion-model` 与 `MotionRFT` 仓库未修改。

2026-07-18 已继续加入 hard step-count DDPO；仍然只读外部仓库，不使用
MotionRFT 的 `StepCountNet`。

2026-07-19 已加入 event-level soft count、counterfactual number probe 和
target-length 去混杂 synthetic step rollout。外部仓库仍未修改。

独立提交：

1. `773a2b2 Fix PPO training correctness`
2. `17e9d96 Add held-out fixed validation`
3. `6a98e66 Add fixed reward calibration`
4. `0bdc04c Add calibrated shrinkage advantages`
5. `94159eb Select balanced validation checkpoints`
6. `4f0a952 Add native MDM diffusion anchor`
7. 阶段 7 实验脚本、README 与最终验收提交见当前 HEAD。
8. `d9e5c00 Add hard step detector and mixed step data`
9. `0098d9e Integrate hard step rewards into DDPO`

## 已实现的关键约束

- PPO 使用 sample-level shuffle，不再只打乱连续 minibatch block。
- rollout 总样本数必须被 train batch size 整除。
- 每个 rollout 首次 optimizer update 前严格审计 old/new log-prob；真实 MDM smoke
  中 `max_abs_diff=0`、ratio=1。
- checkpoint 加载严格要求全部 trainable LoRA tensors 存在。
- rollout split 固定为 `train`；checkpoint selection 固定使用 held-out `val`；
  `test` 禁止用于选择 checkpoint。
- `fixed_eval_pool.pt` 保存 val dataset indices、text、length、GT motion、noise seeds
  和 pool checksum；默认 128 prompts × 4 motions。
- calibration 使用原始 MDM（无 LoRA）至少 1024 prompts × 4 motions；训练加载时
  要求 full calibration 和 checksum 一致。
- advantage 支持 `group_whiten`、`group_centered`、`group_shrink`、
  `component_shrink`。shrink floor 只能来自固定 calibration p25/p50。
- balanced checkpoint 使用固定 global scale：
  `0.5*z_retrieval + 0.5*z_m2m`；两项 delta 都必须在 paired bootstrap 容差内。
- 命名 checkpoint 为 `best_balanced.pt`、`best_retrieval.pt`、`best_m2m.pt`、
  `latest.pt`，不再用 raw total reward 维护 `best.pt`。
- 原生 MDM diffusion anchor 默认关闭；启用时每个 optimizer update 只计算一次，
  可自动标定到 PPO gradient 的 0.1/0.2。

## Hard step reward 当前实现

- 默认 detector 与 RFT_MLD pseudo-label 链路一致：
  `progressive_step + lead_offsets + lead_threshold=0.138`。
- 内置 `rgdno` ankle-energy transition hard detector 作为无 Motion-Rule runtime
  的消融后端。
- 真实 RFT_MLD manifest 样本检查中 targets `[1,2,3]` 均被本适配器精确复现。
- step manifest 的原 prompt 不复用；prompt 按 `detected_steps` 重新生成，多模板且
  digit/word 交替，避免“请求 6 步但 hard label 为 0”之类的标签/文本冲突。
- 默认只用 target 1–6。普通 HumanML 的 `target_steps=-1`、`step_mask=false`，
  step reward 严格为 0。
- 默认 reward：`exp(-abs(detected-target)/temperature)`；另支持 linear/exact/
  negative_l1。
- `--no-step-use-m2m-reward` 只屏蔽 step-labelled samples 的 M2M raw reward 与
  component advantage；HumanML M2M 不变，step M2M 指标仍保留用于比较。默认
  `--step-use-m2m-reward`。
- 支持按数据类型设置 component advantage：全局 retrieval/M2M 权重用于 HumanML，
  `step_advantage_retrieval/m2m/step_weight` 只覆盖 step groups；K8 count 隔离实验
  使用 HumanML `0.5/0.5` 与 step `0.2/0.0/0.8`。
- step sampler 默认按 target 轮转均衡；K8、50% step、4 rollouts/epoch 时，每个
  rollout 有 4 个 step groups，每 epoch 16 个，target 1–6 各 2–3 groups。
- mixed K 已解耦：HumanML `K=4`，step `K=16`。`step_data_ratio` 现在按 motion
  sample 计，而非 prompt 计；默认 physical batch 64 严格组装为 12 HumanML prompts
  × 4 = 48 与 1 step prompt × 16 = 16（仍为 3:1）。
- step K=16 需要独立 `step_reward_k16_calibration.json`（以及建议独立
  `step_val_fixed_eval_pool_k16.pt`）；旧 K=4 calibration 会在加载时明确拒绝。
- step train/held-out val 按 target 分层、seed 固定且 source sample 不重叠；
  `fixed_step_eval_pool.pt` 保存 exact motion/text/target/length/noise seed/checksum。
- fixed step eval 记录 reward、exact、within-1、MAE、detected mean，以及 step prompt
  上 retrieval/M2M delta。
- 原 balanced checkpoint 定义保持 retrieval+M2M 不变；step 单项最佳保存
  `best_step.pt`。
- `component_shrink` 可使用第三个 masked step component；step floor 只能来自与当前
  step K 匹配的 calibration。离散 reward 的 raw p25/p50 可能为 0，因此实际
  floor 使用正方差 prompt groups 的 quantile，并保留 zero-std fraction 日志。
- diffusion anchor 明确跳过 step pseudo-GT，仅锚定真实 HumanML3D。

## Soft count 与 counterfactual 实现

- `progressive_step` 的 kept/filtered candidates 会按时间聚类；每个 cluster 只取
  lead/length/progress signed-margin confidence 的最大值，避免重复奖励同一脚踝事件。
- soft gate 使用相对阈值 margin：
  `sigmoid((measured-threshold)/(abs(threshold)*temperature))`；lead、length、
  progress 默认 temperature 均为 1.0，cluster gap 为 0.15 秒。
- 新 reward `soft_huber_exact`：
  `-Huber((soft_count-target)/per_target_scale) + 0.15*hard_exact`。
  per-target scale 来自原始 MDM calibration 的 soft error RMSE，训练中固定不更新。
- 新增 reward 信息量日志：unique levels、pairwise tie fraction、nonzero advantage
  fraction、top-1 concentration；新增 candidate count/spacing、soft-hard difference、
  ankle velocity 高频能量及其与 reward 的相关性。
- `--step-rollout-source synthetic` 不绑定 target-specific GT；target 与 length 独立，
  所有 target 使用相同模板顺序和共同支持区间中的同一 length distribution。该模式
  强制关闭 step M2M。
- `tools/probe_step_number_conditioning.py` 固定 initial/transition noise、length、
  template，只替换数字 1--6；报告 hard/soft target effect、length effect、text
  embedding separability 和生成 motion 成对距离。
- `best_step_acceptance.pt` 只有在 HumanML feasible 且 hard MAE/exact/within-one
  三项同时改善时才更新；仍需人工确认至少三个连续 counterfactual 验证点。

## Counterfactual number probe 结论

正式 pool 为 24 conditions × 2 paired noises × 6 targets，共 288 motions/policy；
pool id 为 `52127be993e53464a44b96d2524c4e382d3765b1eacbc5539f0b640b77e0bf69`。

原始 MDM：

```text
hard target-count Spearman = 0.4540
target standardized effect = 0.5247
length standardized effect = 0.1810
projected embed_text min RMS distance = 0.02306
adjacent target motion RMS = 0.31483
```

因此数字 embedding 与生成结果都可区分，explicit count embedding fallback 暂不触发。
当前根因是 count response 非线性并在约 4 步处饱和，而不是模型完全不看数字；target 2
在去混杂 pool 上 exact 仍为 0。

旧 K8 negative-L1：

```text
original: MAE=1.14236 exact=.28125 within-one=.65625 Spearman=.45403
epoch 11: MAE=1.14583 exact=.28125 within-one=.65278 Spearman=.44934
epoch 29: MAE=1.13542 exact=.28472 within-one=.65625 Spearman=.46389
```

epoch 29 仅 MAE `-0.00694`、exact `+0.00347`，不应继续延长旧 run 或增加 hard
step 权重。soft temperature 0.25 时 mean `abs(soft-hard)=0.0051`，几乎退化为 hard；
temperature 1.0 时为 0.1742，因此生产 calibration 必须按新默认重跑。

## Step reward 已完成验证

- 110 个单元测试全部通过（最终提交前需再次运行并以实际计数为准）。
- GT/reference audit（1,842 motions）显示：manifest pseudo label 可 100% 复现，
  但相对原 caption 请求步数 exact 仅 14.0%、MAE 2.61，且没有原始 target-1
  caption。step 结果必须视为 detector pseudo-count，而非已验证的真实数字控制。
- `compileall`、`bash -n scripts/*.sh`、`git diff --check` 通过。
- 早期 3-sample 检查仅证明 263-D 恢复路径能复现 manifest pseudo label；不能作为
  原 caption 请求步数的准确性证据。
- 6 prompts × 2 motions × 4 diffusion steps 的 calibration GPU smoke 成功；输出
  `full_calibration=false`，仅验证流程。
- 1 HumanML prompt + 1 step prompt、每 prompt 2 motions、4 diffusion steps 的
  混合 DDPO GPU smoke 成功：
  - 首次 old/new log-prob max diff = 0；ratio = 1；
  - `global_step=1`；无 skipped update；
  - hard step reward/target/detected/MAE 指标进入 epoch JSON；
  - checkpoint 成功保存。
- 258 prompts × 4 motions 的临时 4-step calibration 达到
  `full_calibration=true`；zero-std prompt fraction 为 0.1589，正方差 p25 floor
  为 0.01363。
- 使用该 step calibration 和临时 retrieval/M2M calibration 的三分量
  `component_shrink` GPU smoke 成功；step group 恰为零方差时贡献保持 0，未出现
  非有限梯度，首次 ratio 仍为 1，checkpoint 正常保存。

新的 soft-count 诊断正式使用前需要运行：

```text
tools/calibrate_step_reward_stats.py --prompts 384 --samples-per-prompt 8 \
  --step-pool-source synthetic --step-reward-mode soft_huber_exact --batch-size 64
scripts/train_step_soft_counterfactual.sh
scripts/probe_step_counterfactual_checkpoints.sh outputs/step_k8_soft_counterfactual
```

## 当前推荐起点

在完成正式 calibration 后：

```text
advantage_mode=component_shrink
advantage_std_floor_quantile=p25
advantage component weights=0.5/0.5
learning_rate=1e-4
clip_range=1e-4
rollout_batch_size=32
rollout_batches_per_epoch=4
train_batch_size=32
gradient_accumulation_steps=2
timestep_fraction=0.5
anchor_auto_grad_ratio=0
```

当前 count-learnability 隔离实验：

```text
step_data_ratio=0.5
step_samples_per_prompt=8
step_rollout_source=synthetic
step_targets=1,2,3,4,5,6
step_detector_backend=progressive
step_reward_mode=soft_huber_exact
HumanML advantage weights=0.5 retrieval / 0.5 M2M
Step advantage weights=0 retrieval / 0 M2M / 1.0 step
rollout_batches_per_epoch=4
fixed_eval_every=2
epochs=30, early_stop=off
```

该配置只回答 detector count 是否可学习，不是最终多任务最优配置。先完成单 seed
30-epoch 与 counterfactual checkpoint sequence；通过后才进行多 seed/权重回调。

## 已完成验证

- 全部单元测试、compileall、Shell syntax 和 `git diff --check`。
- held-out val pool 创建与跨输出目录 resume smoke。
- calibration 工具原始 MDM、小规模 GPU smoke。
- component shrink GPU smoke，无非有限梯度。
- balanced checkpoint GPU smoke，负 component delta 不会覆盖 best balanced。
- 原生 MDM anchor GPU smoke：
  - `ppo_grad_norm=0.00328896`
  - `anchor_grad_norm=0.0106896`
  - 自动 `lambda=0.0307678`
  - 实际 grad ratio=`0.1000`
  - anchor calls=optimizer updates=1

## 尚未完成的统计实验

- 正式 1024×4 reward calibration。
- A0–A4 各 30 epochs。
- 最好两组的非全排列 LR/clip follow-up。
- 最佳配置的 anchor ratio `{0,0.1,0.2}` × seed `{42,43,44}`。
- 三 seed 平均 retrieval、M2M、balanced score 验收。
- 标准 HumanML baseline/candidate replication 对比。
- K8 synthetic soft-Huber calibration（384 prompts × 8，50 diffusion steps）。
- K8/50% step、HumanML `0.5/0.5`、step `0/0/1.0` 的 30-epoch soft-count
  隔离实验及所有 checkpoint 的 counterfactual probe。
- step advantage weight 最小消融 S0/S1/S2 与三个 seed 复现。
- step exact/MAE 改善后的可视化人工审计和按 target/length 分层统计。

这些任务已有脚本，不能用短 smoke 结果代替。

## 仍需关注的风险

- retrieval 与 M2M 存在真实 ranking conflict，balanced feasibility 只能限制退化，
  不能消除目标冲突。
- PPO clip 仍是单 update 约束；长期保护依赖 anchor 的实证效果。
- MotionReward 是 surrogate，held-out reward 上升不保证 FID/R-precision 同步改善。
- 旧版 `best.pt`/`latest.pt` 的历史结论不可直接与新 held-out pool、calibration 和
  balanced selection 数值比较。
- step pseudo labels 来自 detector 而非人工标注，存在 reward hacking 风险；
  `best_step.pt` 不能替代 `best_balanced.pt` 的 HumanML 可行性约束。
- step count 与 clip length 有显著混杂；实现不使用 target-conditioned length oracle，
  synthetic rollout/counterfactual probe 已做 target-length 解耦，但 reference fixed
  pool 仍有混杂，正式结论必须按 length 分层并配合视频审计。
- soft count 仍来自 detector metadata；filtered candidate margin 可能带来新的 reward
  hacking 路径，必须审计 temporal spacing、ankle 高频能量和生成视频。
- detector pseudo-count 尚未由人工 held-out motion 标注验证，不能直接对外称为真实
  人类可见步数控制。
