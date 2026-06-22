# sglang整体架构

SGLang 的运行时由**三类独立进程**组成，彼此通过 ZMQ 进行进程间通信（IPC）：

```
HTTP 请求
  → TokenizerManager（分词、组装请求）
  → Scheduler（核心调度器：批处理、KV 缓存管理、跑模型）
  → DetokenizerManager（反分词、流式返回）
  → HTTP 响应
```

进程之间传递的消息结构统一定义在 `srt/managers/io_struct.py`，这是整个系统的"通用语言"

## 请求数据流

数据流

```{mermaid}
%%{init: { 'flowchart': { 'nodeSpacing': 80, 'rankSpacing': 120, 'padding': 60 } } }%%
flowchart TD
    Client([客户端 HTTP / OpenAI API])

    subgraph P3["进程 N+1：DetokenizerManager"]
        DETOK["detokenizer_manager.py<br/>token id → 文本"]
    end

    subgraph P2["进程 2..N：Scheduler (每个 TP rank 一个，核心)"]
        EL["event_loop_normal / event_loop_overlap<br/>主循环"]
        SCHED["get_next_batch_to_run<br/>连续批处理 + 抢占调度"]
        MEM["mem_cache: RadixAttention 前缀缓存<br/>+ KV 分页分配器"]
        RUN["run_batch → TpModelWorker"]
        MR["model_executor/ModelRunner<br/>forward + CUDA Graph"]
        MODEL["models/*.py + layers/*<br/>attention / moe / quant"]
        RESULT["process_batch_result<br/>采样 → 释放 KV"]
    end

    subgraph P1["进程 1：前端 / TokenizerManager (asyncio)"]
        HTTP["http_server.py<br/>路由 /generate, /v1/chat/completions"]
        TM["tokenizer_manager.py<br/>分词 → 组装请求对象"]
        DETOK_RECV["接收 detokenized 输出<br/>流式返回客户端"]
    end

    Client -->|HTTP 请求| HTTP
    HTTP --> TM
    TM -.->|ZMQ: recv_from_tokenizer<br/>io_struct 消息| EL

    EL --> SCHED
    SCHED <-->|查询/复用前缀<br/>分配 KV| MEM
    SCHED --> RUN
    RUN --> MR
    MR --> MODEL
    MODEL --> RESULT
    RESULT <-->|释放/更新| MEM
    RESULT -->|未完成: 下一轮| EL

    RESULT -.->|ZMQ: send_to_detokenizer| DETOK
    DETOK -.->|ZMQ: send_to_tokenizer| DETOK_RECV
    DETOK_RECV -->|SSE 流式 / 完整响应| Client

    style Client fill:#6366f1,color:#fff,stroke:#4f46e5,stroke-width:2px
    style P1 fill:#f0f9ff,color:#1e3a5f,stroke:#3b82f6,stroke-width:2px
    style P2 fill:#ecfeff,color:#0c4a6e,stroke:#06b6d4,stroke-width:3px
    style P3 fill:#faf5ff,color:#4c1d95,stroke:#8b5cf6,stroke-width:2px
    style HTTP fill:#dbeafe,color:#1e40af,stroke:#3b82f6,stroke-width:1px
    style TM fill:#eff6ff,color:#1e3a5f,stroke:#60a5fa,stroke-width:1px
    style DETOK_RECV fill:#f0f9ff,color:#1e3a5f,stroke:#93c5fd,stroke-width:1px
    style SCHED fill:#cffafe,color:#0891b2,stroke:#0891b2,stroke-width:2px
    style MEM fill:#d1fae5,color:#059669,stroke:#059669,stroke-width:2px
    style RUN fill:#ecfeff,color:#0f766e,stroke:#0d9488,stroke-width:1px
    style MR fill:#f0fdfa,color:#0f766e,stroke:#14b8a6,stroke-width:1px
    style MODEL fill:#f0fdf4,color:#14532d,stroke:#2dd4bf,stroke-width:1px
    style RESULT fill:#e0f2fe,color:#0369a1,stroke:#06b6d4,stroke-width:2px
    style DETOK fill:#f3e8ff,color:#6d28d9,stroke:#8b5cf6,stroke-width:2px
```

**图的关键点**

1. **三类进程，ZMQ 通信**：前端（TokenizerManager）、Scheduler、DetokenizerManager 是独立进程，靠 ZMQ 传递消息（图中虚线箭头）。消息结构都定义在 `srt/managers/io_struct.py`。
2. **Scheduler 是闭环**：`event_loop` → 调度组批 → 跑模型 → 处理结果，未完成的请求（还在 decode 阶段）会回到循环里参与下一个 batch，这就是**连续批处理（continuous batching）**。
3. **RadixAttention 贯穿调度与执行**：组批时查前缀缓存决定复用哪些 KV，出结果后释放/更新 KV。这是 SGLang 性能的核心。
4. **TP 多 rank**：实际部署中 Scheduler 按 tensor parallel 大小有多个副本（每个 GPU 一个），图里画一个代表。`run_batch` 通过 `TpModelWorker` → `ModelRunner` 调到具体模型。
5. **输出回流路径**：正常生成结果走 `Scheduler → send_to_detokenizer → Detokenizer → send_to_tokenizer → 客户端`，反分词后回到前端进程再返回客户端，不是直连。