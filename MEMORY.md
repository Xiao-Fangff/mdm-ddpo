# MDM-DDPO 项目记忆

最后更新：2026-07-18

## 当前实现状态

retrieval + M2M DDPO 稳定化已按阶段完成，外部
`motion-diffusion-model` 与 `MotionRFT` 仓库未修改。

独立提交：

1. `773a2b2 Fix PPO training correctness`
2. `17e9d96 Add held-out fixed validation`
3. `6a98e66 Add fixed reward calibration`
4. `0bdc04c Add calibrated shrinkage advantages`
5. `94159eb Select balanced validation checkpoints`
6. `4f0a952 Add native MDM diffusion anchor`
7. 阶段 7 实验脚本、README 与最终验收提交见当前 HEAD。

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

该配置是理论稳定化起点，不是已经由三 seed 长实验确认的最终最优配置。必须先运行
A0–A4、follow-up 和 anchor × seed 实验。

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

这些任务已有脚本，不能用短 smoke 结果代替。

## 仍需关注的风险

- retrieval 与 M2M 存在真实 ranking conflict，balanced feasibility 只能限制退化，
  不能消除目标冲突。
- PPO clip 仍是单 update 约束；长期保护依赖 anchor 的实证效果。
- MotionReward 是 surrogate，held-out reward 上升不保证 FID/R-precision 同步改善。
- 旧版 `best.pt`/`latest.pt` 的历史结论不可直接与新 held-out pool、calibration 和
  balanced selection 数值比较。
