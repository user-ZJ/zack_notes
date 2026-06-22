# SGLang 目录说明

本文档以sglang v0.5.13代码为例


## 1 仓库顶层目录

| 目录 | 说明 |
| --- | --- |
| `python/` | **核心 Python 代码**（运行时 srt、前端 lang、CLI、测试等），日常阅读主要在这里 |
| `sgl-kernel/` | 重量级 AOT 编译的 CUDA / C++ 高性能算子库（attention、moe、量化等底层 kernel） |
| `rust/` | Rust 实现的组件（如路由器 router、gRPC 服务等高性能基础设施） |
| `proto/` | gRPC / protobuf 协议定义 |
| `sgl-model-gateway/` | 模型网关服务 |
| `3rdparty/` | 第三方依赖与 vendored 代码 |
| `benchmark/` | 各类性能基准测试脚本（按模型/场景组织） |
| `test/` | 集成测试、CI 测试用例 |
| `examples/` | 使用示例（离线推理、各类 API 调用、RL 集成等） |
| `docs/` | 旧版文档站点源码 |
| `docs_new/` | 新版文档站点（Mintlify，含 Cookbook） |
| `docker/` | Dockerfile 与容器构建配置 |
| `scripts/` | 开发 / 发布 / CI 辅助脚本 |
| `assets/` | README 等用到的图片资源 |
| `experimental/` | 实验性、未稳定的功能 |

## 2 `python/sglang/` 顶层

| 路径 | 说明 |
| --- | --- |
| `srt/` | **SGLang RunTime，推理服务引擎核心**（下一节展开） |
| `lang/` | 前端 DSL（SGLang 编程语言）：`api.py`、`interpreter.py`、`ir.py`、`tracer.py`、`backend/` 等，用于编写结构化 LLM 程序 |
| `cli/` | 命令行入口（`sglang serve` 等子命令） |
| `eval/` | 评测工具（如 MMLU 等基准评估） |
| `test/` | 单元测试工具与测试基类（`CustomTestCase` 等） |
| `jit_kernel/` | 轻量级 JIT 编译的 CUDA kernel |
| `benchmark/` | 库内基准测试辅助 |
| `multimodal_gen/` | 多模态生成（图像/视频）相关 |
| `launch_server.py` | 服务启动入口 |
| `bench_*.py` | 各种 benchmark 脚本（offline throughput、one batch、serving 等） |
| `server_args` 等顶层模块 | 全局配置、版本、环境检查工具 |

## 3 `python/sglang/srt/` 核心子目录（重点）

按"请求主干 → 内存 → 计算 → 进阶特性 → 基础设施"分组：

**请求主干 / 入口**

| 目录 | 说明 |
| --- | --- |
| `entrypoints/` | 服务入口：`engine.py`、`http_server.py`、`grpc_server.py`、`openai/`、`anthropic/`、`ollama/` 等各协议适配 |
| `managers/` | **调度核心**：`scheduler.py`、`tokenizer_manager.py`、`detokenizer_manager.py`、`schedule_batch.py`、`schedule_policy.py`、`io_struct.py`、`tp_worker.py` 等 |
| `tokenizer/` | 分词器封装 |
| `sampling/` | 采样逻辑（温度、top-p、惩罚项等） |
| `constrained/` | 结构化输出 / 约束解码（JSON、正则、压缩 FSM） |
| `function_call/` | 工具调用 / function calling 解析 |

**内存与缓存**

| 目录 | 说明 |
| --- | --- |
| `mem_cache/` | **RadixAttention 前缀缓存** + KV cache 分页分配器、分层缓存存储 |

**模型与计算**

| 目录 | 说明 |
| --- | --- |
| `model_executor/` | `ModelRunner`：连接调度器与模型，管理 forward 与 CUDA Graph |
| `models/` | 各具体模型实现（llama、qwen、deepseek、glm、gemma 等，每个一个文件） |
| `model_loader/` | 权重加载（多种格式、量化权重校验） |
| `layers/` | 算子层：`attention/`（各 backend）、`moe/`、`quantization/`、linear 等 |
| `configs/` | 各模型 / 组件的配置定义 |
| `multimodal/` | 多模态输入处理（图像/音频/视频 processor） |
| `compilation/` | torch.compile 相关编译逻辑 |

**进阶 / 分布式特性**

| 目录 | 说明 |
| --- | --- |
| `speculative/` | 投机解码（DFlash / Spec V2 等） |
| `disaggregation/` | PD 分离（prefill / decode 拆到不同实例） |
| `distributed/` | 分布式通信原语（TP/PP 通信、device communicators） |
| `eplb/` | Expert Parallelism Load Balancer（专家并行负载均衡） |
| `elastic_ep/` | 弹性专家并行 |
| `lora/` | 多 LoRA 批处理 |
| `batch_overlap/` | 批次计算/通信重叠优化 |
| `dllm/` | Diffusion LLM 相关 |

**基础设施 / 工具**

| 目录 | 说明 |
| --- | --- |
| `server_args.py` / `arg_groups/` / `environ.py` | 命令行参数、参数分组、环境变量（**特性索引表**） |
| `platforms/` / `hardware_backend/` | 多硬件平台抽象（NVIDIA/AMD/XPU/TPU/NPU 等） |
| `connector/` | KV / 权重的外部存储连接器（如 Mooncake） |
| `weight_sync/` / `checkpoint_engine/` | 权重同步、checkpoint（RL 训练场景常用） |
| `observability/` | 监控、指标、日志 |
| `debug_utils/` | 调试工具（comparator、coredump、schedule simulator 等） |
| `grpc/` / `ray/` | gRPC 服务、Ray 部署支持 |
| `session/` / `multiplex/` | 会话管理、多路复用 |
| `parser/` / `plugins/` | 解析器、插件机制 |
| `utils/` | 通用工具函数 |




