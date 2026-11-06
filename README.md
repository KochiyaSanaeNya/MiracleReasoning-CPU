# MiracleReasoning

Miracle 系列大语言模型 的通用 OpenCL CPU 推理后端,调用 OpenCL 1.2

## 架构覆盖

| 模型 | family 参数 | 保留的架构规则 |
|---|---|---|
| Miracle 1.5 FlashNano，含现有 TPU 转换权重 | `flashnano-1.5` | GQA、Q/K RMSNorm、交错 RoPE、局部与全局层 |
| Miracle 1.6 FlashNano | `flashnano-1.6` | 局部 GQA、CBR 学习查询池化、摘要位置锚点、跨块摘要读取 |
| Miracle 1.7 FlashNano | `flashnano-1.7` | split-half RoPE、逐层 KV 头数、局部环形与全局 KV cache |
| Miracle 2.0 FlashLite | `flashlite-2.0` | GQA、可选 Q/K Norm、交错 RoPE、串行残差缩放 |
| Miracle 2.0 FlashLighting | `flashlighting-2.0` | latent KV、带权记忆压缩、YaRN/linear/dynamic NTK、并行门控残差 |
| Miracle 2.0 Heavy Medium | `heavy-medium-2.0` | latent 稀疏注意力、全局块 top-k、分组自适应 MoE、共享专家、独立内循环 cache |

## 安装及 CPU runtime

```bash
python -m pip install -e .
```

操作系统的 OpenCL loader 和 **CPU 设备实现**都必须安装。仅安装显卡驱动通常只会提供 GPU 设备。

- Debian/Ubuntu 可使用 `sudo apt install pocl-opencl-icd ocl-icd-libopencl1`。
- Windows 安装适用于本机 CPU 的 OpenCL runtime，例如 Intel CPU Runtime，或支持 Windows 的 PoCL 发行版。`OpenCL.dll` 本身不代表已经具备 CPU 设备。
- macOS 的系统 OpenCL 已废弃；只有枚举出可用 CPU 设备且支持 OpenCL C 1.2 编译时才可运行。
- 特殊安装可设置 `MIRACLE_OPENCL_LIBRARY`，指定 OpenCL loader 的绝对路径。

```bash
python -m miracle_reasoning devices
```

输出包含平台/设备索引、`kind`、内存和单次分配上限。生产 CLI **只选择 CPU**；没有匹配的 CPU 时明确报错，不自动切换到 GPU。`--device-index` 是平台内设备索引，建议同时传入 `--platform-index`。

## 加载和推理

模型目录须包含 `config.json`，以及以下任一种权重文件：

- `model.safetensors`：支持 F32/F16/BF16，以内存映射读取并在使用时转换为 FP32。
- `model.safetensors.index.json` 和对应分片：每个逻辑张量必须已经合并完整，支持按文件切分的权重，不会自动拼接 TP 切片。
- `model.npz`：无 pickle 的 NumPy 浮点数组，适合小模型和测试。

支持原项目的参数命名，也支持本项目导出的统一命名。`model_type` 等字段足够明确时自动识别架构；否则通过 `--family` 指定。逐张量检查名称和形状，缺失权重不会用随机值填充。已有 1.5 `pretrain/converted_model` 可直接加载。

```bash
python -m miracle_reasoning inspect "../Miracle 1.5 FlashNano/pretrain/converted_model"

python -m miracle_reasoning generate "../Miracle 1.5 FlashNano/pretrain/converted_model" \
  --ids "1,5,73,106" --context-length 2048 --max-new-tokens 32
```

文本输入使用原模型的分词器。按需安装 `python -m pip install -e '.[text]'`：

```bash
python -m miracle_reasoning generate "../Miracle 1.5 FlashNano/pretrain/converted_model" \
  --prompt "你好" \
  --tokenizer "../Miracle 1.5 FlashNano/pretrain/converted_model/tokenizer.model" \
  --max-new-tokens 32 --temperature 0.8 --top-k 40 --top-p 0.95
```

支持 SentencePiece `.model` 和 `tokenizers` 格式 `.json`。后端不自动添加 BOS 或聊天模板；应按模型训练方式编码提示词，需要完整控制时使用 `--ids`。

Python API：

```python
from miracle_reasoning import MiracleModel, OpenCLBackend

backend = OpenCLBackend()  # 严格选择 CPU；可指定 platform_index/device_index
model = MiracleModel.from_pretrained(
    "path/to/model", backend=backend, context_length=2048,
    weight_cache_bytes=256 * 1024 * 1024,
)
logits = model.prefill([1, 5, 73])       # [vocab_size]
logits = model.decode(106)              # 追加单个 token
model.reset()                          # 新序列，释放 KV / 摘要 / 内循环状态
ids = model.generate([1, 5], max_new_tokens=32, temperature=0.0)
```

`generate` 重置请求状态，返回新生成的 token；遇到 EOS 停止。`prefill(..., return_all=True)` 返回 `[tokens, vocab]`，用于评分和数值检查。输入只接受单条、非空的一维整数序列；拼接文档、padding、attention mask 和 batch 暂未开放。一个模型实例不能并发用于多条请求。

`context_length` 默认是 `min(2048, 模型上下文上限)`。生成前检查提示词与请求的生成长度，越界直接报错，不截断历史。局部层使用环形 cache，全局层保留本请求完整历史；CBR 摘要和每个 Heavy Medium 内循环状态相互独立。设备执行失败后必须先 `reset()` 再复用实例。

## 导出 PyTorch 权重

```bash
python -m pip install -e '.[export]'
python -m miracle_reasoning export \
  --checkpoint path/to/model.pt \
  --config path/to/config.json \
  --family flashnano-1.7 \
  --output models/flashnano17
```

支持直接 `state_dict`，以及 `state_dict`、`model_state_dict`、`parameters` 或 `model` 包装字段。配置可为 JSON 或 YAML。仅用 `torch.load(weights_only=True)` 读取；不会启用任意对象反序列化。JAX/Flax checkpoint 应先用原项目转换为 PyTorch/safetensors。

输出为统一配置和 FP32 `model.safetensors`，导出后推理无需安装 PyTorch。MTP 辅助训练头不参与下一 token 推理。分词器需单独保留并通过 CLI 指定。

## 内存与实现边界

权重使用有预算的 LRU 设备缓存，较大线性矩阵和词表投影按输出行切块，单块受设备分配上限约束；`--weight-cache-mib` 可调常驻权重预算。该预算不包含 KV、临时激活和当前不可拆分的单行。大模型仍需足够的主机地址空间、存储和 KV 内存；922B 的 FP32 权重约 3.7 TB，架构支持不代表普通电脑能实用地运行完整模型。

线性投影、RMSNorm、RoPE、SwiGLU、注意力分数/softmax/值归约、CBR 池化和 latent 合并均使用 OpenCL。MoE 自适应路由决策、全局块 top-k、采样及缓存元数据由 CPU 控制。Heavy Medium 按单进程完整专家集合执行，没有远程专家传输；输入 shard 缺少专家时会报错。

FlashLighting 和 Heavy Medium 的压缩/稀疏记忆按原实现的**逐 token 缓存推理语义**执行。它们原生大块 prefill 的记忆构建时机可能不同，因此这里不承诺与任意 prefill 分块方案相同。FP16/BF16 权重在此执行 FP32 运算，与原生低精度执行会有舍入差异。

## 验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

常规测试严格使用 CPU，CPU runtime 缺失时相关执行测试标记为 skip。可设置 `MIRACLE_SOURCE_ROOT` 为包含上述原项目的目录，运行与原 PyTorch 实现的六种架构对照；开发环境默认尝试本项目的上级目录。此项另需 PyTorch。测试子进程禁止在原目录写入字节码。

仅用于交叉设备验证，可以显式设置 `MIRACLE_TEST_DEVICE_TYPE=gpu` 和可选的 `MIRACLE_TEST_PLATFORM`；这不改变生产 CLI 的 CPU 选择规则。

本机验证结果：NVIDIA 和 AMD 的 OpenCL 设备上各 22 项通过；六种微型模型覆盖窗口回绕、CBR 跨块、latent memory 合并、全局层 KV 头数和双内循环，与参考 logits 对比容差为 `atol=2e-5, rtol=2e-4`。现成的 19 层 FlashNano TPU 转换模型另用 4 个 token 与原 PyTorch 权重检查，最大 logits 绝对误差为 `8.11e-6`。本机只枚举出 GPU，CPU 默认测试为 8 项通过、14 项因缺少 CPU ICD 跳过，尚未验证真实 CPU ICD 执行性能。提供的 GitHub Actions 工作流安装 PoCL 并要求 CPU 设备存在后再运行测试；该工作流尚未在远端执行。

Heavy Medium 原项目的 `_local_window` 返回 `[1,1,T,W]` 掩码，而 `_global_blocks` 返回 `[B,H,T,G]`，直接 `torch.cat` 会在多头情况下失败。测试夹具仅将局部掩码展开到分数张量形状，以验证其既定数学规则；没有修改原项目文件。
