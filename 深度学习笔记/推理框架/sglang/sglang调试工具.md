# SGLang 调试工具详解

> 本文档基于 `python/sglang/srt/debug_utils/`、`python/sglang/srt/utils/`、`python/sglang/srt/environ.py`、`server_args.py` 等源码，系统梳理 SGLang 内置的**调试工具**：从精度对齐（tensor dump/对比）、崩溃与挂起诊断，到性能/调度分析和快速调试辅助工具。

---

## 一、调试工具全景

SGLang 的调试工具大致分为五类，覆盖"结果不对 / 进程崩溃 / 卡住不动 / 太慢 / 想快速复现"这几种典型问题：

| 类别 | 解决的问题 | 主要工具 |
| --- | --- | --- |
| **张量 Dump 与对比** | 输出精度不对、与参考实现有 diff | `dumper` + `comparator` / `dump_comparator` + forward hook |
| **精度回归定位** | 换了实现/框架后精度回退，定位到哪一层/哪个 token | `comparator`（unshard/对齐）、`text_comparator`、`grafter` |
| **崩溃 / 挂起诊断** | CUDA 崩溃、分布式 hang、进程卡死 | `watchdog` + py-spy、`cuda_coredump`、崩溃诊断环境变量 |
| **性能 / 调度分析** | 太慢、GPU 利用率低、调度不均衡 | torch profiler、NVTX、`schedule_simulator` |
| **快速调试辅助** | 大模型太慢难以调试、非侵入式打点 | `model_truncator`、`source_patcher`、`pr_fix_toggle` |

代码主要集中在 `python/sglang/srt/debug_utils/` 目录，配合 `environ.py` 中的一批 `SGLANG_*` 调试开关和 `server_args.py` 中的 `--debug-*` 参数。

---

## 二、张量 Dump：`dumper`

`python/sglang/srt/debug_utils/dumper.py` 提供了一个全局单例 `dumper`，用于把前向过程中的任意张量落盘（`.pt` 文件），是精度调试的核心工具。它设计为**可以脱离 SGLang 独立使用**（也支持 Megatron），并且开销可控（不启用时几乎零成本）。

### 2.1 最简用法

在代码里打点：

```python
from sglang.srt.debug_utils.dumper import dumper

dumper.dump("layer_start__hidden_states", hidden_states, layer_id=self.layer_id)
...
dumper.step()   # 每次迭代结束时在所有 rank 上调用
```

然后用环境变量启用（前缀是 `DUMPER_`，**不是** `SGLANG_`）：

```bash
DUMPER_ENABLE=1 python -m sglang.launch_server ...
# 自动清理上次的 dump：
DUMPER_CLEANUP_PREVIOUS=1 DUMPER_ENABLE=1 python ...
```

dump 文件默认写到 `/tmp/dumper/<exp_name>/`，文件名由 tags 拼成，如：

```
name=xxx___step=3___rank=0___dump_index=12___layer_id=2.pt
```

每个 `.pt` 里存 `{"value": tensor, "meta": {...}}`，meta 自动带上并行信息（tp/pp/ep rank、dp attention 等，见 `_SGLangPlugin.collect_parallel_info`）。

### 2.2 配置项（`DumperConfig`）

`DumperConfig` 是一个 frozen dataclass，所有字段都能通过 `DUMPER_<FIELD大写>` 环境变量设置。常用的：

| 环境变量 | 作用 |
| --- | --- |
| `DUMPER_ENABLE` | 总开关 |
| `DUMPER_DIR` | 输出目录（默认 `/tmp/dumper`） |
| `DUMPER_FILTER` | Python 过滤表达式，只 dump 命中的（见下） |
| `DUMPER_ENABLE_VALUE` / `DUMPER_ENABLE_GRAD` | 是否 dump 值 / 梯度 |
| `DUMPER_ENABLE_MODEL_VALUE` / `DUMPER_ENABLE_MODEL_GRAD` | `dump_model` 时是否 dump 权重值/梯度 |
| `DUMPER_CLEANUP_PREVIOUS` | 首次写入前清理旧 dump |
| `DUMPER_SERVER_PORT` | 启动 HTTP 控制端口，运行时动态开关 |
| `DUMPER_NON_INTRUSIVE_MODE` | 非侵入式 hook 模式：`off`/`core`/`all` |
| `DUMPER_INCLUDE_PARALLEL_RANK_IN_FILENAME` | 文件名带上并行 rank，避免多 rank 落到同目录冲突 |

### 2.3 过滤表达式

`DUMPER_FILTER` 是一段 Python 表达式，对每次 dump 的 tags 求值，返回 True 才落盘。未知的 tag 键解析为 `None`，还能用 `search()`/`match()` 正则：

```bash
DUMPER_FILTER='layer_id in [0,1,2,3]'
DUMPER_FILTER='search("hidden", name) and layer_id == 0'
```

### 2.4 上下文（ctx）与装饰器

用 `set_ctx` / `ctx` 给一段代码内的所有 dump 自动附加 tag（例如 layer_id），避免每次手写：

```python
dumper.configure_default(filter='layer_id=[0-3]')
dumper.set_ctx(layer_id=self.layer_id)
...
dumper.set_ctx(layer_id=None)   # 清除

# 或者用装饰器
@dumper.ctx(lambda self: dict(layer_id=self.layer_id))
def forward(self, x): ...

@dumper.ctx(phase="decode")
def decode_step(self, x): ...
```

### 2.5 非侵入式 Dump（forward hook）

不想改模型代码时，用 `register_non_intrusive_dumper(model)` 给模型每个子模块自动挂 forward hook，捕获所有子模块的输入/输出：

- `DUMPER_NON_INTRUSIVE_MODE=core`：只 dump 核心字段（`input_ids`/`positions`/`seq_lens`/`req_pool_indices`/`rids` 等）；
- `=all`：dump 所有中间张量；
- 会自动识别 `layers.<N>` 模块并把 `layer_id` 注入 ctx。

它对 root 模型用 monkey-patch `forward` 的方式（`replace_fn`），保证即使代码直接调用 `.forward()`（SGLang 就是这样）hook 也能触发。

### 2.6 运行时 HTTP 控制

启动时不开、运行中再动态配置：

```bash
# 1. 启动时设 DUMPER_SERVER_PORT（或复用 SGLang 端口）
# 2. 通过 HTTP 打开、设过滤器
curl -X POST http://localhost:30000/dumper/configure -d '{"enable": true, "filter": "layer_id=[0-3]"}'
curl -X POST http://localhost:30000/dumper/reset
```

内部用一个 ZMQ RPC 把配置广播到所有 rank（`_ZmqRpcBroadcast`），保证多卡一致。

### 2.7 `dump_model`：dump 权重

```python
dumper.dump_model(model, name_prefix="param")   # dump 所有 named_parameters
```

配合 `DUMPER_ENABLE_MODEL_VALUE=1`。在 PP 场景下会自动把流水线本地的 `layers.N` 映射成全局层号（`transform_model_param_name`），使不同 stage 的权重名可比较。

---

## 三、张量对比：`dump_comparator` 与 `comparator`

Dump 出 baseline 和 target 两套后，用对比工具找 diff。SGLang 提供**简版**和**完整版**两个工具。

### 3.1 简版：`dump_comparator`

`python/sglang/srt/debug_utils/dump_comparator.py` 是单文件脚本，逐个张量对比：

```bash
python -m sglang.srt.debug_utils.dump_comparator \
    --baseline-path /tmp/dumper/dump_A \
    --target-path   /tmp/dumper/dump_B \
    --diff-threshold 1e-3 \
    --filter "layer_id=0"
```

它按 meta（rank/step/name/layer_id...）在 baseline 里 `find_row` 找到对应张量，然后打印：
- shape / dtype 对比；
- mean/std/min/max/分位数对比；
- **相对误差 `rel_diff`**（来自 DeepGEMM 的公式）、`max_abs_diff`、`mean_abs_diff`；
- 最大误差发生的坐标及两边的值；
- dtype 不一致时会额外降精度再比一次。

相对误差公式：

$$
\text{rel\_diff} = 1 - \frac{2 \sum x_i y_i}{\sum x_i^2 + \sum y_i^2}
$$

### 3.2 完整版：`comparator` 包

`python/sglang/srt/debug_utils/comparator/` 是一个功能完整的对比流水线，处理简版搞不定的复杂情况：

```bash
python -m sglang.srt.debug_utils.comparator \
    --baseline-path <dir> --target-path <dir> \
    --preset sglang_dev \
    --token-aligner smart \
    --override-dims "hidden_states:b s h[tp] d" \
    --viz-bundle-details --visualize-per-token /tmp/per_token.png
```

相比简版，它多了：

| 能力 | 说明 |
| --- | --- |
| **Bundle 匹配** | 按 meta 分组匹配（`bundle_matcher.py`），而非逐文件 |
| **Unshard（反分片）** | 用 `meta["dims"]` 标注 + 并行信息把多卡分片重建成逻辑张量：`Pick`（复制轴，校验各分片相等）、`Concat`（沿 `[tp]` 拼接）、`ReduceSum`（`[tp:partial]` 部分和）、CP-THD 拼接 |
| **Token 对齐** | `concat_steps`（BS=1，跨 step 沿 token 维拼接）/ `smart`（BS>1，用 `input_ids` 匹配序列）对齐后再比 |
| **Reorder** | 还原 context-parallel 的 zigzag 排序（`s[cp:zigzag]`） |
| **Dims 标注** | dump 时给 `meta["dims"]` 写维度语义，如 `"b s[cp:zigzag] h[tp] d # dp:=moe_dp"`；对比时可用 `--override-dims` 免重跑 |
| **可视化** | 逐张量 6×2 面板热图（`--viz-bundle-details`）、逐 token 误差热图（`--visualize-per-token`） |
| **结构化报告** | JSONL 报告、可配置详细级别、按 diff 决定退出码 |

内置 preset（`preset.py`）：`raw`、`sglang_dev`（跳过 rank）、`sglang_megatron`（跳过 rank+step，用 concat_steps 对齐）——后者专门用于 SGLang vs Megatron 的跨框架精度对齐。

### 3.3 dump 索引：`dump_loader`

`dump_loader.py` 负责把一个 dump 目录读成可查询的表：
- `read_meta(dir)`：glob 所有 `.pt`，从文件名解析 `key=value` 段（`___` 分隔）建成 Polars DataFrame，加 `duplicate_index`；
- `find_row(df, conditions)`：按条件精确匹配唯一一行；
- `ValueWithMeta.load(path)`：加载并解包 `{value, meta}`。

### 3.4 服务端 forward hook dump：`--debug-tensor-dump-*`

除了 `dumper`，SGLang server 还内置了一套 forward hook dump（`tensor_dump_forward_hook.py`），通过 server args 启用：

| 参数 | 作用 |
| --- | --- |
| `--debug-tensor-dump-output-folder` | 输出目录（EAGLE 模式下 draft/target 分子目录） |
| `--debug-tensor-dump-layers` | 只 dump 指定层 |
| `--debug-tensor-dump-input-file` | dump 时使用的输入文件 |
| `--debug-tensor-dump-inject` | 注入外部（如 jax）输出作为每层输入 |

输出按 `TP{tp}_PP{pp}_Rank{rank}_pid{pid}/Pass{N}.pt` 组织，适合整模型逐算子抓取。

---

## 四、跨系统精度定位：`grafter`、`text_comparator`、`source_patcher`

### 4.1 `grafter`：跨系统张量移植

`grafter` 内置在 `dumper.py` 里，用于**在两个系统之间移植张量**（例如把 baseline 系统某层的输出直接搬到 target 系统对应位置），从而二分定位"精度从哪一层开始跑偏"。

- 两侧设置相同的 `DUMPER_GRAFTER_B2T_FILTER`（baseline→target 的张量名）和 `DUMPER_GRAFTER_T2B_FILTER`，只有 `DUMPER_GRAFTER_ROLE`（`baseline`/`target`）不同；
- baseline 占全局 rank `0..N-1`，target 占 `N..N+M-1`，通过一个独立 NCCL 进程组通信；
- 命中过滤器时，发送侧 all-gather 张量，接收侧用可自定义的 transform（`DUMPER_GRAFTER_TRANSFORM_PATH` 指向 `transform(graft_input) -> Tensor`）处理后 `copy_` 覆盖本地张量；
- transform 出错不会让整个训练/推理崩溃（捕获并跳过）。

这是排查"两套实现逐层等价性"的强力手段。

### 4.2 `text_comparator`：benchmark 输出对比

`text_comparator.py` 面向**端到端正确率回归**——对比两次 benchmark 的输出，找出"从对变错 / 从错变对"的样本：

```bash
python -m sglang.srt.debug_utils.text_comparator \
    --baseline-path baseline_samples.jsonl \
    --target-path target_samples.jsonl
```

支持 `lm_eval --log_samples` 的输出、`gsm8k/mmlu bench` 的 raw result、simple_evals。它按 prompt 聚合，算 `correctness_delta`，并打印 Good→Bad / Bad→Good 的具体样本和它们输出的最长公共前缀长度（定位从第几个 token 开始分叉）。

### 4.3 `source_patcher` 与 `pr_fix_toggle`

`source_patcher/` 提供**运行时源码补丁**——不改仓库代码，就能在任意函数里注入 `dumper.dump(...)`（或其他编辑）。原理是 `inspect.getsource` → 文本 match/replace → 重新 `compile`/`exec` → 替换 `__code__`。

用 YAML 配置：

```yaml
patches:
  - target: sglang.srt.layers.foo.Bar.forward   # 全限定名
    edits:
      - match: |
          return output
        prepend: |
          dumper.dump("bar_out", output, dims="b s h d")
```

通过 `DUMPER_SOURCE_PATCHER_CONFIG=/path/to/patch.yaml` 启用，`dumper.apply_source_patches()` 会自动注入 `from ...dumper import dumper`，无需在 YAML 里写 import。

`pr_fix_toggle.py` 是它的一个应用：用 `SGLANG_DEBUG_REVERT_PR=<pr_number>` 在运行时**回滚某个已合入的 bugfix PR**，用于回归/金丝雀测试（内置了若干注册 PR 号）。

---

## 五、崩溃与挂起诊断

### 5.1 Watchdog + py-spy

SGLang 的 Scheduler 有一个 watchdog（`python/sglang/srt/utils/watchdog.py`），当某步耗时超过阈值判定为卡死时，会自动对所有 scheduler 进程做 **py-spy dump**（`cudacore_pyspy_dump_utils.pyspy_dump_schedulers`），打印每个线程的栈——通常能直接看到卡在哪个 NCCL 集合通信或 CUDA 同步上。

- **硬 watchdog**（默认）：dump py-spy 后 `SIGQUIT` 杀掉进程；
- **软 watchdog**：只记录超时不杀进程，留时间手动 attach 调试器。

手动 dump：

```bash
py-spy dump --pid <scheduler_pid>
py-spy dump --native --pid <scheduler_pid>   # 带 C 栈
```

（对应的系统性排查流程见仓库内 `debug-distributed-hang` skill。）

### 5.2 CUDA Coredump

`debug_utils/cuda_coredump.py`：设 `SGLANG_CUDA_COREDUMP=1` 后，import 时自动注入 `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` 等环境变量，GPU 异常（如非法内存访问）时生成轻量 coredump，供 `cuda-gdb` 事后分析：

```bash
SGLANG_CUDA_COREDUMP=1 python -m sglang.launch_server ...
# 崩溃后：
cuda-gdb -c <coredump_file>
```

dump 目录由 `SGLANG_CUDA_COREDUMP_DIR` 控制（CI 下自动隔离到 `RUNNER_TEMP`）。

### 5.3 崩溃前诊断（Crash diagnostics）

`environ.py` 里的一组开关，控制进程崩溃前自动收集诊断信息：

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `SGLANG_PYSPY_DUMP_BEFORE_CRASH` | `True` | 崩溃前做 py-spy dump |
| `SGLANG_CUDA_COREDUMP_BEFORE_CRASH` | `True` | 崩溃前生成 CUDA coredump |
| `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS` | `60` | 等待 coredump 完成的秒数 |
| `SGLANG_DETECT_SLOW_RANK` | `False` | 检测慢 rank（定位掉队的卡） |

### 5.4 故障注入（测试用）

一组 `SGLANG_TEST_STUCK_*` / `SGLANG_TEST_CRASH_*` 环境变量用于**主动注入故障**，测试容错和诊断路径：

- `SGLANG_TEST_STUCK_TOKENIZER` / `_SCHEDULER_INIT` / `_DETOKENIZER` / `_DP_CONTROLLER`：让对应组件卡住 N 秒；
- `SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS`：输出若干次后崩溃；
- `SGLANG_TEST_DISAGG_FAILURE_PROB`：PD 分离随机失败概率。

---

## 六、性能与调度分析

### 6.1 Torch Profiler 与 NVTX

| 环境变量 / 端点 | 作用 |
| --- | --- |
| `/start_profile`、`/stop_profile` HTTP 端点 | 抓取 torch profiler trace |
| `SGLANG_TORCH_PROFILER_DIR` | trace 输出目录（默认 `/tmp`） |
| `SGLANG_PROFILE_WITH_STACK` / `SGLANG_PROFILE_RECORD_SHAPES` | profiler 是否记录调用栈 / 张量 shape |
| `SGLANG_ENABLE_NVTX_SCHEDULER` / `SGLANG_ENABLE_NVTX_OPERATIONS` | 打 NVTX 标记，配合 Nsight Systems |
| `SGLANG_RECORD_STEP_TIME` | 记录每步耗时 |

（端到端 profiling 的封装流程见仓库内 `generate-profile` skill。）

### 6.2 调度模拟器：`schedule_simulator`

`debug_utils/schedule_simulator/` 是一个**离散事件调度模拟器**，用于在不跑真实模型的情况下评估路由/调度策略下的 batch 均衡性和 attention 计算均衡性：

```bash
python -m sglang.srt.debug_utils.schedule_simulator \
    --input request_logger.json \
    --num-gpus-per-engine 8 --num-engines 1 \
    --router sticky --scheduler fifo \
    --max-total-tokens 100000 \
    --output summary.json
```

- **数据源**：真实的 `request_logger` JSON / 合成负载（`--synthetic`）/ 共享前缀合成（`--synth-gsp`）；
- **Router**：`round_robin` / `random` / `sticky`（相同前缀组亲和到同一 GPU）；
- **Scheduler**：`fifo`（超预算时从运行 batch 尾部回退，再 FIFO 准入）；
- **指标**：batch size 均衡度、attention 计算均衡度、平均 batch size。

用于回答"换个路由策略会不会更均衡"这类问题，成本远低于真实压测。

---

## 七、快速调试辅助：`model_truncator`

调试大模型（如 DeepSeek-V3）时，完整加载太慢。`model_truncator.py` 把一个 HF 模型**截断成只保留前 N 层**的小模型，便于快速复现结构性 bug：

```bash
python -m sglang.srt.debug_utils.model_truncator \
    --input deepseek-ai/DeepSeek-V3-0324 \
    --output /tmp/DeepSeek-V3-0324-5layer \
    --keep-num-layers 5
```

它会改写 `config.json` 的 `num_hidden_layers`、过滤 `safetensors.index.json` 的 weight_map、删掉多余层的权重，并拷贝 tokenizer/generation_config 等。

> 也可以用 `--json-model-override-args '{"num_hidden_layers": 5}'` 在线截断，但对测试 RL 框架不总是可靠。

其他辅助开关：
- `SGLANG_DEBUG_MEMORY_POOL`：调试 KV 内存池；
- `SGLANG_PHASE_CHECKER_DEBUG`：阶段检查器调试（`utils/phase_checker.py`）；
- `SGLANG_SIMULATE_ACC_LEN` / `SGLANG_SIMULATE_UNIFORM_EXPERTS` 等：模拟投机解码接受长度、均匀专家分布，用于隔离变量。

---

## 八、环境变量与参数速查

### 8.1 Dumper（前缀 `DUMPER_`）

| 变量 | 作用 |
| --- | --- |
| `DUMPER_ENABLE` | 启用 dump |
| `DUMPER_DIR` | 输出目录 |
| `DUMPER_FILTER` | Python 过滤表达式 |
| `DUMPER_CLEANUP_PREVIOUS` | 首写前清理旧 dump |
| `DUMPER_ENABLE_GRAD` | dump 梯度 |
| `DUMPER_SERVER_PORT` | HTTP 动态控制端口 |
| `DUMPER_NON_INTRUSIVE_MODE` | `off`/`core`/`all` 非侵入式 hook |
| `DUMPER_SOURCE_PATCHER_CONFIG` | 源码补丁 YAML 路径 |
| `DUMPER_GRAFTER_*` | 跨系统张量移植配置 |

### 8.2 调试相关 `SGLANG_*`（前缀 `SGLANG_`）

| 变量 | 作用 |
| --- | --- |
| `SGLANG_CUDA_COREDUMP` / `_DIR` | 启用 CUDA coredump / 目录 |
| `SGLANG_PYSPY_DUMP_BEFORE_CRASH` | 崩溃前 py-spy dump |
| `SGLANG_CUDA_COREDUMP_BEFORE_CRASH[_WAIT_SECS]` | 崩溃前 coredump |
| `SGLANG_DETECT_SLOW_RANK` | 检测慢 rank |
| `SGLANG_DEBUG_MEMORY_POOL` | 调试内存池 |
| `SGLANG_DEBUG_REVERT_PR` | 回滚指定 PR 的 fix |
| `SGLANG_PHASE_CHECKER_DEBUG` | 阶段检查器调试 |
| `SGLANG_RECORD_STEP_TIME` | 记录每步耗时 |
| `SGLANG_TORCH_PROFILER_DIR` | profiler 输出目录 |
| `SGLANG_ENABLE_NVTX_SCHEDULER` / `_OPERATIONS` | NVTX 标记 |
| `SGLANG_TEST_STUCK_*` / `SGLANG_TEST_CRASH_*` | 故障注入（测试用） |

### 8.3 服务器参数

| 参数 | 作用 |
| --- | --- |
| `--debug-tensor-dump-output-folder` | forward hook dump 目录 |
| `--debug-tensor-dump-layers` | 只 dump 指定层 |
| `--debug-tensor-dump-input-file` | dump 输入文件 |
| `--debug-tensor-dump-inject` | 注入外部输出作为每层输入 |

---

## 九、典型调试工作流

1. **精度不对（与参考实现有 diff）**
   - 两侧用 `dumper` 打点（或 `source_patcher` 非侵入注入 / `--debug-tensor-dump-*`）；
   - `DUMPER_ENABLE=1` 分别跑出 baseline / target 两套 dump；
   - `python -m sglang.srt.debug_utils.comparator` 对比，用 unshard/token 对齐处理分片和多 step；
   - 用 `--visualize-per-token` 看误差在哪个 token / 哪层开始放大；
   - 必要时用 `grafter` 逐层移植，二分定位第一处分叉。

2. **端到端正确率回退**
   - 跑 lm_eval / gsm8k benchmark 存 samples；
   - `python -m sglang.srt.debug_utils.text_comparator` 找 Good→Bad 样本和分叉位置。

3. **进程崩溃 / CUDA 报错**
   - `SGLANG_CUDA_COREDUMP=1` 复现 → `cuda-gdb -c <coredump>`；
   - 崩溃诊断开关默认已开（py-spy + coredump before crash）。

4. **分布式卡住 / hang**
   - 看 watchdog 自动打的 py-spy 栈；或手动 `py-spy dump --pid`；
   - `NCCL_DEBUG=INFO` 看是哪个集合通信、size 是否不匹配。

5. **太慢 / 想调策略**
   - torch profiler / NVTX 抓 trace 找瓶颈；
   - `schedule_simulator` 评估路由/调度策略的均衡性。

6. **大模型难调试**
   - `model_truncator` 截成 5 层小模型快速复现。

---

## 十、关键文件速查

| 文件 | 职责 |
| --- | --- |
| `srt/debug_utils/dumper.py` | 张量 dump 单例、过滤、ctx、非侵入 hook、HTTP 控制、grafter |
| `srt/debug_utils/dump_loader.py` | dump 目录索引（`read_meta`/`find_row`） |
| `srt/debug_utils/dump_comparator.py` | 简版逐张量对比 |
| `srt/debug_utils/comparator/` | 完整对比流水线（unshard / token 对齐 / dims 标注 / 可视化） |
| `srt/debug_utils/text_comparator.py` | benchmark 正确率回归对比 |
| `srt/debug_utils/source_patcher/` | 运行时源码补丁（非侵入打点） |
| `srt/debug_utils/pr_fix_toggle.py` | 运行时回滚指定 PR 的 fix |
| `srt/debug_utils/schedule_simulator/` | 调度/路由离散事件模拟器 |
| `srt/debug_utils/model_truncator.py` | 模型截断为小模型 |
| `srt/debug_utils/cuda_coredump.py` | CUDA coredump 注入/收集 |
| `srt/debug_utils/tensor_dump_forward_hook.py` | server 端 forward hook dump（`--debug-tensor-dump-*`） |
| `srt/utils/watchdog.py` | 卡死检测 + py-spy dump |
| `srt/utils/cudacore_pyspy_dump_utils.py` | py-spy / coredump 工具 |
| `srt/environ.py` | 全部 `SGLANG_*` 调试开关 |
