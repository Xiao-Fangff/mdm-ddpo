# MDM-DDPO 项目记忆

最后更新：2026-07-18

## 当前状态

- 当前代码提交：`31bdcaa Fix long-run DDPO reward drift`。
- 该提交已完成单元测试、编译、Shell 语法、GPU 冒烟和 50-step DDPO 回归。
- 当前没有应继续使用的训练进程；迁移训练在 epoch 59 因 early stopping 正常结束。

## 应保留与应避免的权重

- 当前最佳 DDPO 权重：
  `outputs/humanml_ddpo_ep9_migration_validation/best.pt`
  - checkpoint epoch：19
  - global step：40
  - 固定池 reward：1.696524
  - 相对迁移 baseline（旧 epoch 9）：`+0.001466`
  - 相对预训练固定池：约 `+0.003309`
- 同目录的 `latest.pt` 是 epoch 59，固定池 reward 相对 epoch 9 为
  `-0.000580`，不能作为最终模型。
- 旧运行 `outputs/humanml_ddpo_group4_lr3e4_mean` 的统一 32-prompt 固定池复评：
  - epoch 9：`+0.001843`（旧运行最佳）
  - epoch 69：`-0.003734`
  - epoch 99：`-0.005231`

## 已确认的实现修复

- MotionReward 的 mean embedding 路径会调用内部 `rsample()`；现已隔离 CPU/CUDA
  RNG，固定评估不会改变后续 rollout 随机数序列。
- 固定评估改为 32 prompts / 128 motions，并用独立 collate，避免此前被训练物理
  batch 截断为 8 prompts。
- 固定评估记录完整签名（采样步数、物理 eval batch、guidance、eta、精度、奖励权重等）；
  签名变化时自动以恢复 policy 建立新 baseline。
- `best.pt` 保存初始 baseline 或后续固定评估最佳权重；不能用 `latest.pt` 代替。
- 新增 `--reset-optimizer-on-resume`，适用于从旧算法设置迁移。
- 默认配置：`group_whiten`、`timestep_fraction=0.5`、
  `fixed_eval_prompts=32`、`early_stop_patience=8`。

## 已验证的训练结果

相同 32 prompts / 128 motions 固定池、20 epochs：

| advantage / timestep fraction | 最佳 delta | 最终 delta |
| --- | ---: | ---: |
| `group_whiten / 0.5` | `+0.002656` | `+0.002656` |
| `group_whiten / 1.0` | `+0.001586` | `+0.000150` |
| `group_centered / 0.5` | `+0.000938` | `+0.000676` |

从旧 epoch 9 出发、重置 Adam、使用新默认设置继续 10 epochs 后，相对 epoch 9
再提升 `+0.001466`，因此迁移路径是可行的。

## 当前主要瓶颈（尚未解决）

训练循环能正常更新，但 MotionReward 的标量优化目标没有稳定转化为真实 HumanML
质量提升：

- `group_whiten` 在每个 prompt 仅有 4 个样本时，会把非常小的组内 reward 差异
  强制归一化为单位 advantage。训练日志中的潜在放大倍数常为 `100-700x`；最佳
  checkpoint 审计的中位数为 `16.7x`、最大值为 `163x`。
- PPO clip 只限制单次旧/新策略更新；当前没有相对预训练/reference MDM 的长期 KL
  正则，因此噪声驱动的微小更新会累计漂移。
- retrieval 与单一配对 GT 的 M2M 都是局部 surrogate：前者不衡量自然度，后者不适合
  文本到动作的一对多性。继续训练时出现 retrieval 上升、M2M 下降的漂移现象。
- 现有固定池来自 train split 的固定 32 条样本，不能代替 held-out HumanML test 指标。

快速标准 HumanML test 预检（256 generated samples、单次 replication，仅作诊断）显示
DDPO best 并无一致收益：

| 指标 | 预训练 MDM | DDPO best |
| --- | ---: | ---: |
| FID（低更好） | 0.8736 | 0.8705 |
| Matching Score（低更好） | 3.0070 | 3.0077 |
| R-precision Top-1（高更好） | 0.5234 | 0.5195 |
| R-precision Top-2（高更好） | 0.7383 | 0.7227 |
| Diversity | 9.9565 | 9.9558 |

## 后续工作优先级（未实施）

1. 为 DDPO 加入冻结 reference MDM 的显式 KL penalty，抑制累计漂移。
2. 用 variance floor / shrinkage 替换纯 `group_whiten`，或在 reward spread 太小时跳过
   该 prompt 的更新；不要把数值噪声放大。
3. 将 checkpoint 选择切换为 held-out test/validation 的标准 HumanML 指标，并以多随机
   种子重复评测。
4. 使用约束式或多目标奖励，至少防止 M2M 低于 reference；若追求可见动作质量，加入
   自然度、脚滑或物理一致性奖励。
