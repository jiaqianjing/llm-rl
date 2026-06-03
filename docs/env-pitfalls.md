# 环境踩坑记录

> 在 AWS EC2 单节点 8x A100-SXM4-80GB（driver 580.95.05 / CUDA 12.9）上运行 DPO/GRPO 训练时遇到的问题及解决方案。

---

## 0. 快速结论：可用的环境配置

```
Python:   /opt/pytorch/bin/python3
PyTorch:  2.8.0+cu129
NCCL:     2.27.3（/usr/local/cuda/lib/libnccl.so.2.27.3，AWS 预装）
PYTHONPATH: /mnt/nvme6/ken/opt-pytorch-pkgs:/opt/pytorch/lib/python3.12/site-packages
启动命令:
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=... \
  /opt/pytorch/bin/python3 -m accelerate.commands.accelerate_cli launch \
    --config_file configs/fsdp_4gpu.yaml train_dpo.py --config configs/dpo_gsm8k.yaml
```

不要用 pip 安装的任何 PyTorch 版本（cu126/cu128/cu129 的 wheel 都带坏的 NCCL）。

---

## 1. cuBLAS 批量 GEMM 崩溃

**症状：** `RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmStridedBatchedEx`

**复现条件：** PyTorch 2.10.0+cu128（miniforge3 全局环境），batch_size >= 2 的任意 bfloat16 / float16 / float32 矩阵乘法。

```python
a = torch.randn(4, 64, 128, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4, 128, 64, dtype=torch.bfloat16, device='cuda')
c = a @ b  # CUBLAS_STATUS_INVALID_VALUE，batch_size=1 时正常
```

**根因：** PyTorch 2.10.0 在 CUDA 12.8 环境下 `cublasGemmStridedBatchedEx` 有 bug，非批量（2D）可以，批量（3D）必崩。

**临时 workaround：**
```python
torch.backends.cuda.preferred_blas_library("cublaslt")
```

**根治：** 升级到 PyTorch 2.11.0+cu128，bug 不再出现。

---

## 2. pip 安装的 NCCL 与本机不兼容（SIGSEGV）

**症状：** `torchrun --nproc_per_node=4` 跑任意分布式脚本，`dist.init_process_group('nccl')` 成功，但第一个 GPU collective（`all_gather`、`broadcast`）立刻 SIGSEGV，4 个 rank 同时崩。

**复现：**
```python
dist.init_process_group(backend='nccl')  # OK
dist.all_gather_into_tensor(out, t)       # SIGSEGV
```

**版本组合一览（均不可用）：**

| PyTorch | NCCL（pip 安装） | 结果 |
|---------|-----------------|------|
| 2.11.0+cu126 | 2.29.3+cuda12.9 | SIGSEGV |
| 2.11.0+cu128 | 2.28.9+cuda12.9 | SIGSEGV |
| 系统 2.27.3（LD_PRELOAD）| — | ImportError: undefined symbol `ncclDevCommCreate` |

**根因：**
- PyPI 的 `nvidia-nccl-cu12` 不管你装的是 cu126 还是 cu128，打包进去的 NCCL 全是针对 CUDA 12.9 编译的（版本字符串带 `+cuda12.9`）。
- 该版本的 NCCL 在本机 A100 + driver 580.95.05 上 communicator 初始化（`ncclCommInitRank`，延迟到第一个 collective 触发）就 crash。
- 系统 NCCL 2.27.3 工作正常，但 PyTorch >= 2.9 编译时依赖了 `ncclDevCommCreate`（2.28 才引入），所以无法用旧版替换。

**解决方案：** 用 `/opt/pytorch`（AWS DLAMI 预装的 PyTorch 2.8.0+cu129），它链接的是系统 NCCL 2.27.3，该版本由 AWS 针对本机 A100 优化过，实测 all_gather 正常。

验证命令：
```bash
# 确认 NCCL 版本
strings /usr/local/cuda/lib/libnccl.so.2 | grep "NCCL version"
# → NCCL version 2.27.3 compiled with CUDA 12.9

# 确认 nccl-tests 可用
CUDA_VISIBLE_DEVICES=4,5 /usr/local/cuda-12.9/efa/test-cuda-12.9/all_gather_perf -b 8 -e 128M -f 2 -g 2
```

---

## 3. FSDP + fsdp_sync_module_states SIGSEGV

**症状：** FSDP 初始化时 SIGSEGV，堆栈指向 `_sync_params_and_buffers` → `_sync_module_params_and_buffers`。

**配置：**
```yaml
fsdp_cpu_ram_efficient_loading: true
fsdp_sync_module_states: true
```

**根因：** `cpu_ram_efficient_loading: true` 让 rank 0 独自加载模型，再通过 NCCL broadcast 同步给其他 rank。这个 broadcast 触发了问题 #2 的 NCCL crash。

**修复：**
```yaml
fsdp_cpu_ram_efficient_loading: false
fsdp_sync_module_states: false
```
每个 rank 独立加载模型，绕过 broadcast。代价：4 个 rank 分别读磁盘，加载稍慢，CPU 内存占用翻 4 倍（约 56 GB）。

---

## 4. Accelerate DataLoader RNG 同步用 NCCL 广播 CPU tensor

**症状：** 解决 FSDP init crash 后，training loop 第一步崩溃，堆栈：
```
accelerate/data_loader.py:570 in __iter__
accelerate/utils/random.py:165 in synchronize_rng_states
torch/distributed/distributed_c10d.py:3074 in broadcast
RuntimeError: No backend type associated with device type cpu
```

**根因：** Accelerate DataLoader 在迭代时调用 `synchronize_rng_states`，将 CPU 上的 RNG state tensor 通过 NCCL broadcast。NCCL 不支持 CPU tensor。

**修复：** 在 prepare 完成后禁用 RNG 同步：
```python
self.dataloader = self.accelerator.prepare(DataLoader(...))
if hasattr(self.dataloader, 'rng_types'):
    self.dataloader.rng_types = None
```

---

## 5. 根磁盘满，conda/pip 安装失败

**症状：** `pip install` 报 `OSError: [Errno 28] No space left on device`。

**根因：** `/`（150GB）100% 占满，但各 NVMe 盘有大量空闲。

**修复：** 把 conda env 和 pip cache 建在 nvme6：
```bash
mkdir -p /mnt/nvme6/ken/tmp /mnt/nvme6/ken/pip-cache
TMPDIR=/mnt/nvme6/ken/tmp PIP_CACHE_DIR=/mnt/nvme6/ken/pip-cache pip install ...
conda create --prefix /mnt/nvme6/ken/conda/llm-rl python=3.12
```

---

## 6. 显存 OOM（DPO 双倍激活 + ref model 全量）

**症状：** forward pass 在 MLP `down_proj` 处 OOM：
```
torch.OutOfMemoryError: GPU 3 has 79.25 GiB total, 153.88 MiB free, 79.09 GiB in use
```

**分析：**

| 组件 | 内存/卡 |
|------|---------|
| Policy model（FSDP 分片）| 14GB / 4 = 3.5 GB |
| Ref model（全量，不走 FSDP）| 14 GB |
| 优化器状态（FSDP 分片）| 28GB / 4 = 7 GB |
| DPO 激活（batch_size=16，chosen+rejected 各一次 forward）| ~50 GB |

**修复：**
1. `batch_size: 16 → 2`
2. `model.gradient_checkpointing_enable()` 以算力换显存
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片

---

## 7. loss NaN（小 batch_size 梯度爆炸）

**症状：** step 210 起 loss 变 NaN，之前约 0.7 是正常值。

**根因：** batch_size=2 时梯度方差极大，没有梯度裁剪导致权重爆炸。

**修复：** 在 optimizer.step() 前加梯度裁剪：
```python
self.accelerator.backward(loss)
self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.optimizer.step()
```

---

## 8. FSDP FULL_STATE_DICT 保存触发 DistBackendError

**症状：** 训练 step 200 触发 `save_checkpoint`，随即 SIGABRT：
```
terminate called after throwing an instance of 'c10::DistBackendError'
Signal 6 (SIGABRT) received
```

**根因：** `fsdp_state_dict_type: FULL_STATE_DICT` 需要在保存前做一次全量 all_gather（把各卡的 1/4 分片聚回完整 14GB 模型）。这个大规模 all_gather 在 NCCL 2.27.3 环境下不稳定，可能触发超时或通信失败。

**临时绕过：** 把 `save_steps` 设大于训练总步数（如 9999），先跑通训练，后处理保存。

**根治方向：**
- 改用 `fsdp_state_dict_type: LOCAL_STATE_DICT`，每卡保存自己的分片，规避大 all_gather
- 或在训练结束后单独做一次权重合并

---

## 附：诊断工具

```bash
# 验证 NCCL 可用性（不走 Python）
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/cuda-12.9/efa/test-cuda-12.9/all_gather_perf -g 4

# 最小 NCCL collective 测试
torchrun --nproc_per_node=4 nccl_test.py

# 找实际加载的 NCCL 库（注意 libtorch_cuda.so 有 RPATH，优先级高于 LD_LIBRARY_PATH）
ldd /opt/pytorch/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so | grep nccl

# GPU ECC 错误检查
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv,noheader

# GPU 拓扑（验证 NVLink 连接）
nvidia-smi topo -m
```
