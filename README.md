# MiniMax H3 DMD Crossres

核心说明见 [H3 DMD 与 Crossres](docs/h3_dmd_crossres_core.md)：训练入口、upsampler、rollout、score 与 review。

包含 H3 DMD/PCM/crossres/multires 核心源码、Bidirectional 训练框架及许可证、必要启动脚本、配置与自包含测试。源码逐字复制当前工作区（包含已有 LTX 改动），未修改核心逻辑。

不包含模型权重、数据、虚拟环境、实验输出、集群任务记录、凭据或原仓库 Git 历史。缓存、权重和配置中的本机路径需按环境准备。可视化默认本地输出；显式配置 OSS 发布需要独立安装 oss2 并提供自己的配置。

安装：`pip install -e '.[h3,train,test]'`。PyTorch >= 2.7；完整 H3 GPU 验证需匹配 CUDA、H3 Diffusers 与外部权重。

`CORE_MANIFEST.json` 记录源文件 SHA256，用于核对快照。上游来源及许可见 `third_party/bidirectional/`。
