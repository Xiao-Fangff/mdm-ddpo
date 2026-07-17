# MDM-DDPO

本项目在不修改参考仓库的前提下，将
[motion-diffusion-model](/home/zhiwei/projects/motion-diffusion-model)
的 HumanML3D MDM 与
[ddpo-pytorch](/home/zhiwei/projects/ddpo-pytorch)
的 DDPO/PPO 训练方式结合，并接入
[RFT_MLD](/home/zhiwei/projects/MotionRFT/RFT_MLD)
所使用的 MotionReward retrieval 与 M2M 奖励。

## 已实现内容

- 从 MDM checkpoint 和配套 `args.json` 构建模型、数据集与扩散过程。
- 支持对原始 50 步扩散过程做可变步数 respacing。
- 实现随机 DDIM transition 的采样及可重算 log-probability。
- 保存 rollout 中的 `x_t`、`x_{t-1}`、timestep 和旧策略 log-probability。
- 使用标准 PPO clipped objective 执行 DDPO 更新。
- 每个文本生成多个独立 motion，并只在同一文本组内标准化 advantage，
  避免不同文本/GT 难度差异主导策略梯度。
- 对 HumanML3D padding 帧做掩码，padding 不参与策略似然。
- 排除确定性的 `t=0` transition，只使用随机 transition 做策略梯度。
- 支持全参数训练和无额外依赖的参数化 LoRA；默认使用 LoRA。
- 直接加载 MotionReward Stage-1 backbone，计算：

  - retrieval：文本 embedding 与生成 motion embedding 的余弦相似度；
  - M2M：配对 GT motion 与生成 motion embedding 的余弦相似度。

  与 RFT_MLD 训练路径一致，生成 motion 以 `timestep=0` 编码，文本与
  GT motion 不带 timestep。

- 奖励前先把 MDM z-normalized 263-D 特征反归一化，再用 MotionReward
  数据统计量重新归一化，避免两个参考项目的 Mean/Std 文件差异造成输入域偏移。

- 支持断点保存与恢复，LoRA 模式只保存可训练参数。
- 可选接入 SwanLab，按 epoch 记录 reward、PPO、梯度与耗时训练曲线。
- 数据缓存写入本项目共享的 `.cache/mdm`，不会写入参考仓库。

## 算法对应关系

对每个随机 DDIM transition：

```text
x_t --MDM--> predicted x_0 --DDIM(eta > 0)--> Normal(mean_theta, sigma_t)
```

rollout 时保存旧策略的：

```text
log p_old(x_{t-1} | x_t, text)
```

更新时在相同 transition 上重算：

```text
ratio = exp(log p_theta - log p_old)
loss  = mean(max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps)))
```

reward 是 terminal motion 上的：

```text
reward = retrieval_weight * cosine(text, generated)
       + m2m_weight       * cosine(gt_motion, generated)
```

对每个 prompt 生成 `K` 个 motion，并计算组内 advantage：

```text
A[i, k] = (reward[i, k] - mean_k(reward[i, k]))
          / (std_k(reward[i, k]) + eps)
```

此外，每隔若干 epoch 使用固定 prompt、固定扩散噪声和 MotionReward mean
embedding 做验证，得到可跨 checkpoint 比较的 `eval/reward_*` 曲线。

## 环境

当前机器可直接使用：

```bash
/home/zhiwei/anaconda3/envs/motionrft/bin/python
```

如需另建环境，可参考 `requirements.txt`。MDM 和 MotionReward 源码通过运行时路径引入，不会复制或修改原仓库。

默认资源路径为：

```text
MDM source:
  /home/zhiwei/projects/motion-diffusion-model

MDM checkpoint:
  /home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert/model000600000.pt

MDM args:
  /home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert/args.json

MotionReward backbone:
  /home/zhiwei/projects/MotionRFT/checkpoints/motionreward/stage1_retrieval_backbone_r128.pth

Sentence-T5:
  /home/zhiwei/projects/MotionRFT/deps/sentence-t5-large
```

## 预检

预检会真实加载 HumanML3D、MDM、LoRA、MotionReward 与 T5，但不执行 rollout：

```bash
python train_ddpo.py \
  --preflight \
  --device cuda:0 \
  --reward-device same \
  --output-dir outputs/preflight
```

如果只想验证 CPU 接口：

```bash
python train_ddpo.py \
  --preflight \
  --device cpu \
  --reward-device cpu \
  --precision no \
  --data-workers 0 \
  --sample-steps 4 \
  --rollout-batch-size 2 \
  --samples-per-prompt 2 \
  --output-dir /tmp/mdm-ddpo-preflight
```

## 正式训练

```bash
python train_ddpo.py \
  --device cuda:0 \
  --reward-device same \
  --precision bf16 \
  --output-dir outputs/humanml_retrieval_m2m \
  --epochs 100 \
  --sample-steps 50 \
  --rollout-batch-size 32 \
  --rollout-batches-per-epoch 4 \
  --samples-per-prompt 4 \
  --train-batch-size 32 \
  --inner-epochs 1 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-4 \
  --guidance-scale 2.5 \
  --ddim-eta 1.0 \
  --train-mode lora \
  --lora-rank 8 \
  --lora-alpha 8 \
  --retrieval-weight 1.0 \
  --m2m-weight 1.0 \
  --reward-embedding-mode mean
```

上述配置每个 rollout batch 包含 8 个不同 prompt，每个 prompt 生成 4 个
motion；每个 epoch 共 128 个 motion、32 个不同 prompt、2 次 optimizer
更新，有效优化 batch 为 64。

也可以使用启动脚本：

```bash
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 bash scripts/train_humanml.sh
```

启用 SwanLab 在线记录：

```bash
MDM_DDPO_USE_SWANLAB=1 \
MDM_DDPO_SWANLAB_PROJECT=mdm-ddpo \
MDM_DDPO_SWANLAB_RUN_NAME=humanml-retrieval-m2m \
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 \
bash scripts/train_humanml.sh
```

启动脚本仍兼容旧的 `USE_SWANLAB`、`SWANLAB_PROJECT` 等变量，但推荐使用
`MDM_DDPO_SWANLAB_*` 前缀，避免与 SwanLab SDK 自身的环境配置名称冲突。

也可以直接向 Python 入口传参：

```bash
python train_ddpo.py \
  --use-swanlab \
  --swanlab-project mdm-ddpo \
  --swanlab-run-name humanml-retrieval-m2m \
  --swanlab-mode online \
  --output-dir outputs/humanml_retrieval_m2m
```

SwanLab 默认关闭，不影响原有训练和 `metrics.jsonl`。`--swanlab-mode`
支持 `online`、`offline`、`local` 和 `disabled`；本地日志默认写入
`OUTPUT_DIR/swanlab`。`--preflight` 不会创建 SwanLab run。

额外参数会原样传给 Python 入口：

```bash
bash scripts/train_humanml.sh --epochs 20 --rollout-batch-size 8
```

## 端到端冒烟训练

`--dry-run` 会执行一个 batch、4 个采样步和一个训练 epoch：

```bash
python train_ddpo.py \
  --dry-run \
  --device cuda:0 \
  --reward-device same \
  --precision bf16 \
  --output-dir /tmp/mdm-ddpo-dryrun
```

## 关键参数

| 参数 | 含义 |
| --- | --- |
| `--sample-steps` | 采样步数；0 使用 checkpoint 的 diffusion step 数 |
| `--ddim-eta` | transition 随机性；DDPO 要求大于 0 |
| `--guidance-scale` | MDM classifier-free guidance scale |
| `--samples-per-prompt` | 每个 prompt 的独立生成数；默认 4，组内计算 advantage |
| `--timestep-fraction` | 每个样本用于 PPO 的 transition 比例 |
| `--train-batch-size` | 每次 PPO forward 使用的 rollout 样本数 |
| `--gradient-accumulation-steps` | 累积多少个样本 minibatch；每个 minibatch 的所选 timesteps 会先全部累积 |
| `--clip-range` | PPO ratio clipping 范围，默认与 ddpo-pytorch 一致为 `1e-4` |
| `--train-mode` | `lora` 或 `full` |
| `--reward-embedding-mode mean` | 默认；使用 distribution mean，降低策略梯度中的奖励噪声 |
| `--reward-embedding-mode sample` | 与 RFT_MLD 一样从 embedding distribution 随机采样 |
| `--fixed-eval-every` | 固定 prompt/noise 验证间隔；默认每 5 epochs，0 表示关闭 |
| `--reward-device cpu` | GPU 显存不足时将 MotionReward/T5 放在 CPU |
| `--data-cache-dir` | 可写共享数据缓存；默认是项目下的 `.cache/mdm` |
| `--use-swanlab` | 启用 SwanLab epoch 级训练曲线记录；默认关闭 |
| `--swanlab-mode` | SwanLab 运行模式：`online`、`offline`、`local` 或 `disabled` |

默认精度为 BF16。RTX 3090 的真实冒烟测试中 BF16 梯度有限；FP16 个别 minibatch 可能溢出，因此训练循环会检测非有限梯度、跳过对应更新并给出告警。

## 输出

```text
OUTPUT_DIR/
├── config.json
├── metrics.jsonl
├── fixed_eval.jsonl        # 固定 prompt/noise 验证及相对 baseline 增量
├── swanlab/                 # 启用 SwanLab 时生成
├── checkpoint_000000.pt
└── latest.pt
```

SwanLab 中记录的曲线包括 `reward/{total,retrieval,m2m,std}`、组内/组间
reward 方差、`eval/{reward_total,reward_retrieval,reward_m2m}`、
`ppo/{loss,approx_kl,clip_fraction,ratio}`、
`optimization/{grad_norm,skipped_updates,learning_rate}`、训练进度和每轮耗时。

`reward/total` 是每轮随机 prompt 和随机 motion 的 rollout 均值，不同 epoch
之间的 prompt 难度组成不同，因此不应把它当作主要收敛曲线。判断训练是否真正
改善时，请优先观察固定 prompt、固定噪声的 `eval/reward_total_delta`，并同时
确认 `eval/reward_retrieval_delta` 和 `eval/reward_m2m_delta` 的方向。

首次加载 train split 时还会生成约 3.5 GB 的共享缓存：

```text
.cache/mdm/
├── dataset/t2m_train.npy
└── glove -> MotionRFT/deps/glove
```

checkpoint 包含 epoch、global step、可训练 policy 参数、optimizer、GradScaler 和随机数状态。恢复训练：

```bash
python train_ddpo.py \
  --resume outputs/humanml_retrieval_m2m/latest.pt \
  --output-dir outputs/humanml_retrieval_m2m \
  --epochs 200
```

恢复时 `--train-mode`、LoRA rank/target 和基础 MDM checkpoint 必须与原训练一致。

## 导出为标准 MDM checkpoint

DDPO checkpoint 默认只保存 LoRA 与 optimizer 状态。导出工具会把 LoRA 合并到
基础 MDM，并在目标目录写入标准 checkpoint 和 `args.json`：

```bash
python export_ddpo.py \
  --checkpoint outputs/humanml_retrieval_m2m/latest.pt \
  --output outputs/exported_mdm/model_ddpo.pt
```

导出后可直接使用参考 MDM 的生成入口：

```bash
cd /home/zhiwei/projects/motion-diffusion-model
python sample/generate.py \
  --model_path /home/zhiwei/projects/mdm-ddpo/outputs/exported_mdm/model_ddpo.pt \
  --text_prompt "a person walks forward" \
  --motion_length 6 \
  --guidance_param 2.5
```

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q mdm_ddpo train_ddpo.py export_ddpo.py tests
```

当前测试覆盖 DDIM log-prob 重算及梯度、padding mask、确定性末步、LoRA 初始等价性和奖励组合。

## 本机验证结果

- CPU 真实预检成功：HumanML3D 训练样本 24,546，MDM 与 MotionReward checkpoint 均成功加载。
- 注入 50 个 LoRA adapter，可训练参数 606,264，占完整模型参数约 0.649%。
- RTX 3090 BF16 默认奖励冒烟成功：旧/新策略首次重算
  `ratio=1.0`、`approx_kl=0`、无非有限梯度。
- 原生 50 步、全部 49 个随机 transition 回归成功。
- LoRA checkpoint 约 7.1 MB，所有保存参数有限，50 个 LoRA-B 均得到更新。
- checkpoint 恢复后成功从 epoch 1、global step 1 继续训练并保存下一 checkpoint。
- LoRA 合并导出得到 156 个标准 MDM state tensors，并通过原 MDM loader 重新加载。
