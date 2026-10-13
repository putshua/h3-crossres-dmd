# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Imaginaire 训练器模块

本模块提供了 Imaginaire 框架的基础训练器类，支持大规模分布式训练。
主要特性：
1. 分布式训练支持：DDP（数据并行）和 FSDP（全分片数据并行）
2. 模型并行支持：通过 Megatron-Core 实现 pipeline/tensor/context parallelism
3. 混合精度训练：支持 fp16/bf16
4. 梯度累积：支持大 batch 训练
5. EMA（指数移动平均）：模型权重平均
6. 检查点管理：自动保存/加载模型、优化器状态
7. 回调机制：灵活扩展训练流程
"""

import functools
import inspect
import os
import signal

import torch
import torch.distributed as dist
import torch.utils.data

# 性能分析工具：可选启用 PyTorch Profiler 和内存快照
from imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

# Megatron-Core 导入：用于大规模模型并行训练
# 如果未安装 Megatron-Core，则禁用模型并行功能
try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")


from imaginaire.lazy_config import LazyConfig, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import callback, distributed, log, misc
from imaginaire.utils.checkpointer import Checkpointer


class ImaginaireTrainer:
    """Imaginaire 基础训练器类
    
    所有 Imaginaire 训练器都应该继承此类。它提供了大规模训练所需的基本功能：
    
    核心功能：
    - 数据并行（DDP/FSDP）：支持多 GPU 训练
    - 模型权重平均（EMA）：提升模型稳定性和性能
    - 混合精度训练（fp16/bf16）：加速训练并减少显存占用
    - 检查点管理：自动保存/加载训练状态
    - 回调系统：灵活扩展训练流程
    
    训练流程：
    1. 初始化分布式环境和模型并行配置
    2. 加载模型检查点（如果存在）
    3. 循环执行：数据加载 → 前向传播 → 反向传播 → 优化器更新
    4. 定期保存检查点和验证模型
    
    The base trainer class of Imaginaire.

    All trainers in Imaginaire should inherit ImaginaireTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): 检查点管理器，负责保存/加载模型权重和优化器状态
        training_timer (misc.Timer): 计时器，用于性能分析和 benchmark
        callbacks (CallBackGroup): 回调函数组，用于扩展训练流程
    """

    def __init__(self, config):
        """训练器构造函数
        
        初始化训练器的各个组件，包括分布式环境、回调、检查点器等。
        
        Args:
            config (Config): Imaginaire 配置对象，包含所有训练相关参数
            
        初始化流程：
        1. 初始化分布式环境（NCCL/Gloo）
        2. 配置 Megatron 模型并行（如果可用）
        3. 创建工作目录并保存配置
        4. 设置随机种子、cuDNN、TF32
        5. 初始化回调和检查点器
        6. 设置训练超时保护
        
        Constructor of the trainer.

        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        super().__init__()
        self.config = config
        # ==================== 步骤 1: 初始化分布式环境 ====================
        # 初始化 NCCL/Gloo 进程组，建立进程间通信
        # Set up the distributed computing environment.
        with misc.timer("init_distributed"):
            distributed.init()
            # ==================== 步骤 2: 配置 Megatron 模型并行 ====================
            # Megatron-Core 支持三种并行策略：
            # - Pipeline Parallelism: 按层切分模型，不同层在不同 GPU 上
            # - Tensor Parallelism: 按张量切分，单个算子在不同 GPU 上并行计算
            # - Context Parallelism: 用于长序列训练，将序列长度维度切分
            # Set up parallel states.
            # 处理废弃的配置项：config.model.context_parallel_size 已废弃
            # 应使用 config.model_parallel.context_parallel_size
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            # 如果 Megatron-Core 可用，初始化模型并行状态
            if USE_MEGATRON:
                # 检查 parallel_state.initialize_model_parallel 的签名
                # 不同版本的 Megatron-Core API 可能有差异
                if (
                    "create_gloo_process_groups"
                    in inspect.signature(parallel_state.initialize_model_parallel).parameters
                ):
                    # 新版本 Megatron-Core API
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                        create_gloo_process_groups=False,
                    )
                else:
                    # 旧版本 Megatron-Core API
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                    )
                # 设置序列并行（Sequence Parallelism）
                # 这是 Megatron-Core 的扩展功能，不包含在原始 API 中
                # 序列并行：将 Transformer 的序列长度维度切分到多个 GPU 上
                # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
                # It is not part of the original `parallel_state` API, so we need to set it manually.
                parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
                if parallel_state.sequence_parallel:
                    # 限制 CUDA 操作队列深度，避免序列并行时出现死锁
                    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # ==================== 步骤 3: 创建工作目录并保存配置 ====================
        # 仅在 rank 0 上创建目录和保存配置，避免多进程冲突
        # Create the local job directory, save the config file, and pipe to a local log.
        if distributed.is_rank0():
            # 创建本地工作目录
            os.makedirs(config.job.path_local, exist_ok=True)
            # 保存配置为 .pkl 文件，用于复现实验
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            # 保存配置为 .yaml 文件，便于阅读和解析超参数
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        # 等待所有进程完成目录创建（同步点）
        dist.barrier()
        # 初始化日志文件
        log.init_loguru_file(f"{config.job.path_local}/stdout.log")
        if distributed.is_rank0():
            # 在 rank 0 上打印配置信息
            # Print important environment variables and the effective config.
            log.info("Config:\n" + config.pretty_print(use_color=True))
        # 打印重要环境变量
        misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        # ==================== 步骤 4: 设置随机种子和 cuDNN ====================
        # 设置随机种子，确保可复现性
        # 如果是多 GPU 训练，每个 rank 使用不同的种子（基于 rank id）
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # 配置 cuDNN（CUDA 深度神经网络库）
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # 启用 TF32（TensorFloat-32），提升 A100/RTX3090 等新 GPU 的计算性能
        # TF32 使用 10 位尾数（而非 fp32 的 23 位），牺牲部分精度换取速度
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # ==================== 步骤 5: 初始化回调和检查点器 ====================
        # 初始化回调函数组
        # 回调系统允许在不修改训练器代码的情况下扩展训练流程
        # 例如：EMA、进度条、日志记录、模型保存等
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # 初始化检查点管理器
        # 检查点器负责保存/加载模型权重、优化器状态、学习率调度器等
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            # 使用默认检查点器
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            # 使用自定义检查点器
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # ==================== 步骤 6: 初始化计时器和超时保护 ====================
        # 初始化训练计时器，用于性能分析和 benchmark
        # Initialize the timer for speed benchmarking.
        self.training_timer = misc.TrainingTimer()
        # 设置训练超时保护
        # 如果一个训练步骤耗时超过 timeout_period 秒，发送 TimeoutError
        # 这可以防止训练卡死（例如数据加载阻塞、NCCL 通信死锁等）
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """主训练函数
        
        执行完整的训练流程，包括模型初始化、训练循环、验证和检查点保存。
        
        Args:
            model (ImaginaireModel): PyTorch 模型，必须继承 ImaginaireModel
            dataloader_train (torch.utils.data.DataLoader): 训练数据加载器
            dataloader_val (torch.utils.data.DataLoader): 验证数据加载器
            
        训练流程：
        1. 模型移动到 GPU 并调用初始化钩子
        2. 初始化优化器、学习率调度器、梯度缩放器
        3. 加载检查点（恢复训练）
        4. 包裹 DDP/FSDP 分布式模型
        5. 训练主循环：
           - 数据加载
           - 前向传播 → 反向传播 → 梯度累积/优化器更新
           - 定期保存检查点和验证
        6. 训练结束：保存最终检查点
        
        The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # ==================== 步骤 1: 模型初始化 ====================
        # 将模型移动到 GPU，设置内存格式（channels_last 或 contiguous）
        # channels_last 格式在某些场景下可以提升性能（如卷积网络）
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        # 调用模型的训练开始钩子
        # 子类可以在此方法中实现自定义初始化逻辑（如加载预训练权重）
        model.on_train_start(self.config.trainer.memory_format)

        # ==================== 步骤 2: 初始化优化器、调度器和梯度缩放器 ====================
        # 触发优化器初始化开始回调
        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        # 由模型初始化优化器和学习率调度器
        # 这样设计是因为不同模型可能需要不同的优化器配置
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        # 初始化梯度缩放器（用于混合精度训练）
        # GradScaler 会自动缩放损失，防止 fp16 梯度下溢
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        # 触发优化器初始化结束回调
        self.callbacks.on_optimizer_init_end()
        # ==================== 步骤 3: 加载检查点 ====================
        # 从检查点加载模型权重、优化器状态、学习率调度器状态
        # 返回当前迭代次数（用于恢复训练）
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        # 梯度累积计数器
        # 用于实现大 batch 训练：累积多个小 batch 的梯度后再更新优化器
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        # ==================== 步骤 4: 分布式模型包裹 ====================
        # 根据配置选择分布式策略
        if self.config.trainer.distributed_parallelism == "ddp":
            # DDP（DistributedDataParallel）：数据并行
            # 每个进程持有完整的模型副本，梯度在反向传播时自动同步
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            # FSDP（Fully Sharded Data Parallel）：全分片数据并行
            # 模型参数、梯度、优化器状态都被切分到多个 GPU 上
            # 可以训练更大的模型，但通信开销更大
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        log.info("Starting training...")
        # 触发训练开始回调
        self.callbacks.on_train_start(model, iteration=iteration)
        # ==================== 步骤 5: 初始验证 ====================
        # 如果是首次训练（iteration == 0）且启用了验证，执行初始验证
        # 这有助于在训练开始前评估模型的初始性能
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
            log.info("Initial validation done.")
        # 训练结束标志
        _end_training = False
        # ==================== 步骤 6: 训练主循环 ====================
        # 使用上下文管理器启用性能分析工具（可选）
        # - torch_profiler: PyTorch Profiler，用于性能分析
        # - memory_profiler: 内存快照，用于内存分析
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            # 外层循环：遍历整个数据集多次（多个 epoch）
            while True:
                # 创建数据加载器迭代器
                dataloader_train_iter = iter(dataloader_train)
                # 内层循环：遍历一个 epoch 的所有 batch
                while True:
                    # 触发数据加载开始回调
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        # 加载一个 batch 的数据，并计时
                        with self.training_timer("dataloader_train"):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        # 数据集遍历完毕，跳出内层循环
                        break
                    finally:
                        # 触发数据加载结束回调
                        self.callbacks.on_after_dataloading(iteration)
                    # ==================== 检查是否达到最大迭代次数 ====================
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # ==================== 数据移动到 GPU ====================
                    # 将数据 batch 中的所有张量移动到 GPU
                    # misc.to 会递归处理字典、列表等嵌套结构
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # ==================== 执行训练步骤 ====================
                    # 触发训练步骤开始回调
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    # 确保模型处于训练模式
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    # 执行单步训练（前向传播 + 反向传播 + 梯度累积）
                    # 返回：输出、损失、更新后的梯度累积计数器
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    # 触发训练步骤 batch 结束回调
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # ==================== 梯度累积检查 ====================
                    # 如果还在梯度累积中（grad_accum_iter != 0），继续加载下一个 batch
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # ==================== 优化器更新后执行 ====================
                    # 只有当梯度累积完成时（grad_accum_iter == 0）才执行以下逻辑
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    # 定期保存检查点
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    # 触发训练步骤结束回调
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # ==================== 定期验证 ====================
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # ==================== 重置超时信号 ====================
                    # 训练步骤成功完成，重置超时计时器
                    # 如果下一个步骤超时，会触发 SIGALRM
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    # 更新性能分析器
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                # 检查是否结束训练
                # 检查是否结束训练
                if _end_training:
                    break
        # ==================== 步骤 7: 训练结束 ====================
        log.success("Done with training.")
        # 如果最后一次迭代没有保存检查点，在训练结束时保存
        # 确保训练进度不会丢失
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        # 触发训练结束回调
        self.callbacks.on_train_end(model, iteration=iteration)
        # 关闭检查点器（如关闭文件流、上传到云存储等）
        self.checkpointer.finalize()
        # 等待所有进程完成（同步点）
        distributed.barrier()
        # 触发应用结束回调
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """单步训练函数
        
        执行一个训练步骤：前向传播 → 反向传播 → 梯度累积 → 优化器更新
        
        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): 
                分布式模型（DDP 包裹后的模型或原始模型）
            optimizer (torch.optim.Optimizer): 模型优化器
            scheduler (torch.optim.lr_scheduler.LRScheduler): 学习率调度器
            grad_scaler (torch.amp.GradScaler): 梯度缩放器（用于混合精度训练）
            data (dict[str, torch.Tensor]): 数据 batch（张量字典）
            iteration (int): 当前迭代次数
            grad_accum_iter (int): 梯度累积计数器
            
        Returns:
            output_batch (dict[str, torch.Tensor]): 模型输出（辅助信息）
            loss (torch.Tensor): 总损失（用于反向传播）
            grad_accum_iter (int): 更新后的梯度累积计数器
            
        核心流程：
        1. DDP 梯度同步控制（只在梯度累积的最后一步同步）
        2. 前向传播：计算模型输出和损失
        3. 反向传播：计算梯度（带梯度缩放）
        4. 梯度累积检查：
           - 如果累积未完成：只更新计数器
           - 如果累积完成：执行优化器更新、学习率调度、梯度清零
           
        梯度累积机制详解：
        假设 grad_accum_iter = 4（累积 4 个 batch 的梯度）
        - 第 1-3 次调用：grad_accum_iter 从 0 → 1 → 2 → 3，只累积梯度
        - 第 4 次调用：grad_accumiter 从 3 → 4，触发优化器更新，然后归零
        
        The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # ==================== 步骤 1: DDP 梯度同步控制 ====================
        # 关键机制：只在梯度累积的最后一步同步 DDP 梯度
        # 
        # 为什么这样设计？
        # - DDP 默认在每次 backward() 后同步梯度
        # - 如果使用梯度累积（如累积 4 个 batch），每次同步会浪费通信带宽
        # - 正确做法：前 3 次 backward() 不同步，第 4 次 backward() 时同步
        #
        # distributed.ddp_syncgrad 的实现原理：
        # - 当 enable_sync=False 时，设置 model_ddp.require_backward_grad_sync = False
        # - DDP 在 backward() 结束时会检查这个标志，决定是否同步
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            # ==================== 步骤 2: 前向传播 ====================
            # 触发前向传播开始回调
            self.callbacks.on_before_forward(iteration=iteration)
            # 执行前向传播，计算模型输出和损失
            with self.training_timer("forward"):
                output_batch, loss = model_ddp.training_step(data, iteration)
            # 触发前向传播结束回调
            self.callbacks.on_after_forward(iteration=iteration)
            # ==================== 步骤 3: 反向传播 ====================
            # 触发反向传播开始回调
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            # 执行反向传播
            with self.training_timer("backward"):
                # 梯度缩放：将损失除以累积步数，防止梯度爆炸
                # 例如：如果累积 4 个 batch，每个 batch 的梯度会累加，需要除以 4
                loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                # 执行反向传播，计算梯度
                loss_scaled.backward()
                # 调用模型的后向钩子
                # 子类可以在此方法中实现自定义逻辑（如梯度裁剪、参数调整）
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_after_backward()
                else:
                    model_ddp.on_after_backward()
            # 触发反向传播结束回调
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        # ==================== 步骤 4: 梯度累积检查 ====================
        # 更新梯度累积计数器
        grad_accum_iter += 1
        # 检查是否完成梯度累积
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            # ==================== 步骤 5: 优化器更新 ====================
            with self.training_timer("optimizer_step"):
                # 触发优化器更新开始回调
                self.callbacks.on_before_optimizer_step(
                    model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                )
                # 执行优化器更新
                # grad_scaler.step() 会检查梯度是否包含 inf/nan
                # 如果梯度正常，执行 optimizer.step()；否则跳过本次更新
                grad_scaler.step(optimizer)
                # 更新梯度缩放器
                # 如果出现梯度下溢，会自动调整缩放因子
                grad_scaler.update()
                # 更新学习率
                scheduler.step()
                # 触发梯度清零开始回调
                self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                # 调用模型的梯度清零前钩子
                # 子类可以在此方法中实现自定义逻辑（如保存梯度统计）
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                else:
                    model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                # 清零梯度
                # set_to_none=True 可以释放梯度占用的显存
                optimizer.zero_grad(set_to_none=True)
            # 重置梯度累积计数器
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """验证函数
        
        在整个验证数据集上评估模型性能。
        
        Args:
            model (ImaginaireModel): PyTorch 模型
            dataloader_val (torch.utils.data.DataLoader): 验证数据加载器
            iteration (int): 当前迭代次数
            
        验证流程：
        1. 设置模型为评估模式（model.eval()）
        2. 启用 EMA（如果配置了）
        3. 遍历验证数据集，计算损失
        4. 触发验证回调
        
        Validate on the full validation dataset.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        log.info(f"Validating at iteration {iteration}...")
        # 触发验证开始回调
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        # 设置模型为评估模式
        model.eval()
        # ==================== 遍历验证数据集 ====================
        # 使用 EMA 模型进行验证（如果配置了）
        # Evaluate on the full validation set.
        with model.pipe.ema_scope(context="Validation", is_cpu=False):
            for val_iter, data_batch in enumerate(dataloader_val):
                # 检查是否达到最大验证迭代次数
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                # 将数据移动到 GPU
                data_batch = misc.to(data_batch, device="cuda")
                # 触发验证步骤开始回调
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                # 执行验证步骤
                output_batch, loss = model.validation_step(data_batch, iteration)
                # 触发验证步骤结束回调
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        # 触发验证结束回调
        self.callbacks.on_validation_end(model, iteration=iteration)
