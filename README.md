# MiniMax H3 DMD 与 Crossres 核心说明

本仓库复用 Bidirectional 的训练框架，为 MiniMax H3 实现原生音视频 DMD、可选 PCM warmup，以及跨分辨率 rollout/score。student（G）生成，冻结 teacher 提供真实分布估计，critic（fake score）拟合当前 G 的生成分布。本文按当前源码说明，不修改算法。

## 代码入口

| 功能 | 入口与实现 |
| --- | --- |
| 主训练 | `scripts/train_h3.sh` → `scripts/train.py` → `rcm/models/h3_distill.py` |
| 训练循环与保存恢复 | `rcm/trainers/trainer_distillation.py`、`rcm/checkpointers/`，复用上游 closure、交替 optimizer、DCP |
| H3 接入 | `rcm/networks/h3.py`、`h3dmd/h3/{adapter,layout,scheduler}.py`，负责加载、FSDP2、packed AV 布局与原生 RF 时间 |
| DMD 与 rollout | `h3dmd/methods/{t2va_distill,rollout}.py`、`h3dmd/methods/losses/{dmd,rf,pcm}.py` |
| Crossres 与 multires | `h3dmd/methods/{crossres_dmd,multires_dmd,staged_crossres_dmd}.py` |
| Upsampler | `scripts/train_h3_upsampler.py`、`h3dmd/{upsampler_training,upsampler_data,upsampling,ltx_upsampling}.py` |
| 实验注册 | `rcm/configs/registry_h3.py`、`rcm/configs/experiments/h3/` |

`h3dmd.cli.train` 是旧配置兼容入口，仍转到上游 trainer。复用训练框架不表示已经实现 H3 rCM/JVP。

## Crossres 原理

Upsampler 是视频 latent 的空间上采样算子，代码中称 `mapper` / `score_mapper`。输入 clean latent，输出更高分辨率的 clean latent：

```text
video_480p = upsampler(video_320p)
[B,24,T,20,32] → [B,24,T,30,48]  # 对应像素 512×320 → 768×480
```

只扩大空间尺寸，保留通道、帧数和时间位置；音频不做上采样。具体算子由 `crossres.mapper_kind` 选择：

| 配置值 | 本仓库的上采样操作 |
| --- | --- |
| `dysample` | 学习采样偏移，用 bilinear `grid_sample` 重采样特征 |
| `carafe` | 学习每个位置的 5×5 邻域权重，加权重组特征 |
| `liif` | 将局部特征、相对坐标和网格尺寸送入小网络，预测目标位置 |
| `rcan` | 通道注意力提取特征，再做 bicubic 放大和卷积 |
| `ltx` | Conv3d 残差块 → PixelShuffle ×3 → 模糊滤波降采样 ÷2 → Conv3d 残差块 |
| `bicubic` | 固定 bicubic 插值，无可训练参数 |

前四种逐帧处理，输出为 bicubic 基线加学习残差。LTX 利用时间上下文，在 H3 latent 上从头训练，不加载 LTX 权重。

Upsampler 单独训练：LR/HR latent 来自同一源帧和裁剪，训练集与验证集按源视频隔离。目标为：

```text
prediction = upsampler(low_res_clean_latent)
loss = MSE(prediction, high_res_clean_latent)
     + 0.1 * spatial_difference_error_MSE
     + 0.1 * temporal_difference_error_MSE
```

差分项匹配 HR 的空间边缘和帧间变化。

神经 upsampler 验证通过后冻结：默认 MSE/bicubic ≤ 0.995、temporal-MSE/bicubic ≤ 1.0。DMD 加载时还校验训练步数（默认至少 2000）、架构、尺寸、标准化和 SHA256。更新 G 时，梯度仍经过 upsampler 回到输入；upsampler 参数不更新。

## Upsampler 训练入口

安装项目依赖后，创建不变的配对缓存索引，再运行独立 DDP 训练；索引引用现有归档，不包含 latent 本体。缓存准备可参考 `scripts/prepare_h3_upsampler_pairs.py`、`scripts/cache_h3_crossres_pairs.py` 及 `h3dmd/stride_pairs.py`。

```bash
export PYTHONPATH="$PWD:$PWD/.venv/extra-site-packages${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" -m scripts.train_h3_upsampler snapshot \
  --cache-root /path/to/committed_pair_cache --output /path/to/pairs.sqlite

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m scripts.train_h3_upsampler train \
  --config configs/h3_upsampler_train_320to480.json \
  --index /path/to/pairs.sqlite --output outputs/upsampler-dysample
```

默认训练随机对齐的 5-latent-frame crop，支持 BF16、梯度累积及按时长分桶的完整 clip 训练。LTX 的完整时长实验应使用对应 mixed 配置；crop 与完整 clip 的时间上下文不同。`latest.pt` 用于恢复，`frozen.pt` 是通过验证后导出的 DMD artifact；仅有 `COMPLETE.json` 不代表验证通过。训练不加载 H3 transformer；开启可视化时另加载 video VAE。

也可用 `PAIR_INDEX=... bash scripts/train_h3_upsampler.sh`。**该 shell 会覆盖 JSON 中的 upsampler 类型、宽度、batch、训练预算等字段**；选非默认配置时必须同步设置环境变量，或直接使用上面的 Python CLI，避免 LTX/RCAN 被覆盖成 DySample。旧的小规模自动预训练入口为 `scripts/pretrain_h3_score_upsampler.py`，可由 `PRETRAIN_MAPPER=1` 触发。

## DMD 训练入口

依赖在 `pyproject.toml`（`pip install -e '.[h3,train,test]'`）。H3 需要对应的 Diffusers 实现。H3 权重、prompt/AV cache 与 upsampler 权重单独准备，不纳入代码仓库。多节点在每台节点执行同一命令，使用相同 rendezvous；常用全参数配置为 FSDP32、每卡 batch 1。

```bash
# 原生分辨率 DMD；train_h3.sh 默认是 synthetic_audio 实验。
PYTHON_BIN=python PRETRAINED_PATH=/path/to/H3/FL2VA \
MANIFEST=/path/to/prompt_av/manifest.jsonl \
NNODES=4 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=32 \
RDZV_ENDPOINT=<master-ip>:29503 OUTPUT_DIR=outputs/native-dmd \
  bash scripts/train_h3.sh

# 全程低分辨率 rollout，高分辨率 score。
PYTHON_BIN=python PRETRAINED_PATH=/path/to/H3/FL2VA \
MANIFEST=/path/to/320p/manifest.jsonl \
MAPPER_KIND=dysample MAPPER_CHECKPOINT=/path/to/frozen.pt \
NNODES=4 NPROC_PER_NODE=8 RDZV_ENDPOINT=<master-ip>:29503 \
OUTPUT_DIR=outputs/crossres-dmd bash scripts/train_h3_crossres.sh
```

| 路径 | 启动脚本 | rollout / score |
| --- | --- | --- |
| `h3_crossres_dmd` | `train_h3_crossres.sh` | 最多 4 次 320p forward；clean 输出上采样到 480p 后计算 score |
| `h3_multires_dmd` | `train_h3_multires_rcan.sh` / `train_h3_multires_bicubic.sh` | 320p×3 → 480p×1；始终在 480p score |
| `h3_pcm_warmup_crossres_dmd` | `train_h3_pcm_crossres.sh` | 先原生分辨率真实 AV PCM，再切到 crossres DMD |

RCAN multires 需指定 `MAPPER_KIND=rcan` 对应的 `MAPPER_CHECKPOINT`；bicubic 无需 artifact。multires wrapper 默认检查 manifest 同目录的 `CACHE_COMPLETE.json`。神经 upsampler 的宽度、groups 等必须与预训练 spec 一致，可通过尾部 `--set crossres.mapper_width=128` 等参数设置。`DRY_RUN=1` 可检查最终命令。使用新的 `OUTPUT_DIR` 开始新实验，已有输出目录可能自动恢复 DCP。

## Crossres 下的 rollout 与 score

H3 使用 dataward rectified flow，音视频共享基础坐标 `q`，但各自应用 shift `s`：

```text
sigma(q,s) = s*q / (1 + (s-1)*q)
t_model = 1-sigma
x_sigma = (1-sigma)*x0 + sigma*epsilon
v_target = x0-epsilon
x0_hat = x_sigma + sigma*v_model
```

**Rollout。** 从音视频高斯噪声开始，4 次 forward 的基础 q 为 `[1,.75,.5,.25]`，末端为 0。训练可循环选择退出点 1…4：保留原 q 前缀，最后一次直接预测 clean x0；不会为较短轨迹重新等分 q。默认 `renoise` 在每次预测 clean 后按下一个 sigma 重新采样噪声，只对最后一次 student forward 反传，前缀 detach。

普通 crossres 全程在 320p rollout，然后上采样 clean 视频：

```text
y_v = upsampler(student_clean_video)  # 480p，供 teacher/fake 计算 score
y_a = student_clean_audio            # 音频保持原尺寸
```

multires 退出点 1…3 使用相同流程；退出点 4 的流程为：

```text
第 3 次 forward 的 320p clean 视频
→ upsampler → 480p clean 视频
→ 按下一时刻重新加噪，重建 packed layout
→ 第 4 次 forward 在 480p 运行 → 直接计算 score
```

第四次输出已是 480p，无需再次上采样。默认只对最后一次 student forward 反传，因此这条路径的前三次 forward 和上采样步骤均 detach。

**Score 与 G loss。** 在生成结果所在的 score 网格采样新的 IID 噪声，并独立于 rollout 采样 score q。teacher/fake 使用完全相同的 noisy AV 输入、q 和 prompt；扩大视频后重建 packed indices、空间 RoPE、序列长度与 timestep 布局。音频长度保持不变，但联合 AV attention 会使音频预测受视频网格影响。条件 teacher 不额外做 CFG。

对视频和音频分别计算（下式为默认梯度归一化）：

```text
real_x0 = noisy + sigma*v_teacher
fake_x0 = noisy + sigma*v_fake
d = clamp(mean_per_sample(abs(y-real_x0)), min=normalization_eps)
g = nan_to_num((fake_x0-real_x0)/d)
target = stop_gradient(y-g)
L_G = dmd_weight * (w_video*mean((y_v-target_v)^2)
                  + w_audio*mean((y_a-target_a)^2))
```

teacher/fake 的估计在 `no_grad` 中。视频 loss 和归一化分母均在 480p 网格计算，音频单独归一化；不做面积补偿，loss 不乘 `0.5`。低分辨率出口的反传路径为 `loss → upsampler → student`，使用上采样算子对输入的真实梯度。

**Fake 更新。** 在 `no_grad` 中 rollout，将 clean 视频上采样到 480p（已是 480p 则跳过）。detach 后重新采样 score q 与噪声，用音视频加权 MSE 训练 fake 预测 `v=y-epsilon`。只更新 fake；teacher、G、upsampler 均不更新。默认每 6 个 optimizer iteration 更新 G 一次，其余 5 次更新 fake。

crossres wrapper 默认 rollout shift 视频/音频为 6/3，score 为 2/2；multires wrapper 为 12/3 和 6/3。二者的时间分布与输出分辨率不同，比较实验时应同时核对这些配置。
