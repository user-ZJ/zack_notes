# Qwen-TTS 完整介绍

> 本文介绍阿里巴巴 Qwen 团队语音合成产品线的技术定位、版本演进、模型架构、性能表现、应用场景、竞品对比、部署方式与选型要点。技术架构、时间线与评测数字以 Qwen3-TTS（arXiv:2601.15621）、Qwen-Audio-3.0-TTS（arXiv:2607.23938）、Qwen3-Omni（arXiv:2509.17765）及官方仓库、阿里云百炼文档为准；Artificial Analysis 榜单数据截至 2026-09-04。

---

## 1. 概述

Qwen-TTS 是阿里巴巴 **Qwen 主品牌**下的语音合成（Text-to-Speech, TTS）产品线，不是单一模型权重。它从通义千问多模态模型的语音生成侧独立出来，做成专用合成服务，并与 FunAudioLLM / CosyVoice 并列存在：Qwen 主线先补齐音频理解，再补语音生成；CosyVoice 线从 2024 年 7 月起持续做合成与声音克隆。

开源一代把 TTS 写成全因果的离散多码本语言模型：文本经大语言模型（Large Language Model, LLM）对齐后，由轻量卷积网络把码本还原为波形。闭源一代在此基础上重新引入流匹配（Flow Matching）和 48 kHz 超分声码器，用更高音质换取更高延迟。

产品线按时间可分成三代，名称并不对应同一套权重：

| 代际 | 时间 | 交付方式 | 核心能力 | 语言覆盖 |
|------|------|----------|----------|----------|
| **`qwen-tts`** | 2025-04 上架阿里云百炼 | 仅 API | 独立 TTS 接口；中英双语、7 个预设音色 | 中英；2025-07 起增加北京话、上海话、四川话 |
| **Qwen3-TTS** | 2026-01-21/22 开源 | 1.7B / 0.6B，Apache-2.0 | 12.5 Hz 多码本 + 多 token 预测（Multi-Token Prediction, MTP）+ 因果 Code2Wav；约 3 秒零样本克隆 | 10 语种 + 9 种方言 |
| **Qwen-Audio-3.0-TTS** Flash / Plus | 2026-07 世界人工智能大会（WAIC）发布 | 仅 API | LLM + 流匹配 + 48 kHz 超分；细粒度行内标签；单次合成最长约 3 分钟 | 16 语种 + 约 20–21 个中文方言区 |

理解这条产品线，需要同时看清五件事实：

1. **生成能力晚于理解能力。** 2023–2024 年的 Qwen-Audio、Qwen2-Audio 只做音频理解，不含 TTS。语音生成始于 2025 年 3 月的 Qwen2.5-Omni（Thinker-Talker）；独立商业接口 `qwen-tts` 的最早公开快照是 2025-04-10。2025 年 7 月起的方言更新、11–12 月的 Qwen3-TTS API 上架，都是后续迭代，不是产品线起点。
2. **两套命名不是并列开源。** `Qwen3-TTS`（2026-01 开源）与 `Qwen-Audio-3.0-TTS`（2026-07，仅 API）是上下代关系。2026 年 8 月，自动语音识别（Automatic Speech Recognition, ASR）、TTS 与 Realtime 被统一到 `Qwen-Audio-3.0` 家族品牌下。
3. **开源主线是一次去扩散化。** 系统从 25 Hz 单码本 + 分块扩散变换器（Diffusion Transformer, DiT）+ BigVGAN，收敛到 12.5 Hz 多码本 + MTP + 轻量因果卷积网络（Convolutional Network, ConvNet）。DiT 需要块级上下文，BigVGAN 带约 130 ms 右向前瞻（look-ahead），二者把延迟锁在约 190 ms 以上；12.5 Hz 全因果路径把开源 0.6B、并发 1 的首包压到约 97 ms。
4. **开源权重与闭源 API 不是同一模型。** Artificial Analysis 盲测中，开源 `Qwen3-TTS` 的 Elo 为 926，闭源 `Qwen-Audio-3.0-TTS-Plus` 为 1241，相差约 315 分，对应两套技术栈。
5. **榜单位置随时间变化。** 2026 年 7 月榜上 Plus 的 Elo 为 1237，位列第一，并与当时第二名 Simba 3.2（1234）在 95% 置信区间上重叠。截至 2026-09-04，第一名是 Cartesia Sonic 3.6（1282），Plus 以 1241 列第三。

开源 Qwen3-TTS 的 0.6B 变体在论文 Table 2、并发为 1 时，首包约 **97 ms**，1.7B 约 101 ms。闭源 Flash 的发布口径约为 300 ms，百炼模型页写“200 ms 以内”。两代延迟对应不同解码器，不宜直接纵向比较。

---

## 2. 技术背景与模型定位

### 2.1 从音频理解到专用合成

2023 年前后，TTS 从“音素对齐 + 梅尔频谱预测”转向“离散语音 token + 语言模型”。零样本克隆和跨语种迁移因此更容易，同时带来文本—语音对齐、流式首包延迟、方言与指令控制等工程问题。

Qwen 主线没有从合成缺口直接起步，而是先做音频理解：

- **Qwen-Audio（2023-12）**：覆盖 30 余项音频理解任务，不具备合成能力。
- **Qwen2-Audio（2024-08）**：支持语音聊天与音频分析，仍不具备 TTS。

语音生成来自 **Qwen2.5-Omni（2025-03）** 的 Thinker-Talker：Thinker 负责理解，Talker 负责出声。2025 年 9 月的 **Qwen3-Omni** 将 Talker 定为后来开源 TTS 的四项基础：多码本表示、MTP、轻量 ConvNet 形式的 Code2Wav，以及 12.5 Hz 帧率。Qwen3-TTS 是把该语音侧技术栈抽出，换成稠密（dense）Qwen3 主干后单独训练的专用合成模型。

因此，Qwen-TTS 的定位是：**在 Omni 已经验证的语音 token 接口上，提供可独立部署、可进入百炼的专用合成能力；开源代优先降低首包延迟，闭源代再提高音质与可控性。**

### 2.2 与 CosyVoice 的并行产品线

阿里同时维护两条语音路线。Qwen 主线优先多模态理解；FunAudioLLM / CosyVoice 线从 2024 年 7 月起持续开源合成与克隆。这一结构延续到 2026 年。

| 维度 | Qwen-TTS 线（Qwen 主品牌） | CosyVoice 线（FunAudioLLM / 通义百聆） |
|------|---------------------------|----------------------------------------|
| 归属 | Qwen 团队 | 通义 / 达摩语音组（仓库现属 QwenAudio 组织） |
| 开源 | 2026-01 首次开源 Qwen3-TTS（Apache-2.0） | 自 2024-07 持续开源（Apache-2.0） |
| GitHub 热度（2026-09） | 约 13.1k star / 1.7k fork | 约 23.3k star / 2.66k fork |
| 工程化 | vLLM-Omni 首日支持，目前仅离线推理 | TensorRT-LLM（约 4 倍加速）、vLLM、Docker + gRPC / FastAPI |
| 生态 | 官方 `qwen-tts` pip、Gradio、微调脚本；ComfyUI / GGUF 为社区项目 | 被 LiveKit Agents、Pipecat 用作 TTS 组件；CV3-Eval 为常用学术基准 |
| 百炼定位 | `Qwen-Audio-3.0-TTS-Plus`：内置音色 / 标准合成 | `CosyVoice-v3.5`：声音设计 / 复刻 |
| 实时协议 | Qwen3-TTS 系列不支持 AOQ 客户端直连 | 支持 AOQ 客户端直连 |

百炼上两条线是并列选型。官方迁移说明将 OpenAI gpt-4o-tts、Google Chirp 3 指向 `qwen-audio-3.0-tts-plus`，将 ElevenLabs Multilingual v3 的复刻场景指向 `qwen-audio-3.0-tts-flash`，将声音设计场景指向 `cosyvoice-v3.5-plus`。

### 2.3 能力边界

Qwen-TTS 覆盖的能力包括：

- 开源 12.5 Hz 全因果流式合成；0.6B、并发 1 时首包约 97 ms；
- 约 3 秒参考音频的零样本音色克隆，以及自然语言音色设计（Voice Design）；
- 中文及中文方言、中英混说、跨语种条件下尽量保持同一说话人；
- 闭源一代的 48 kHz 输出、细粒度标签与 Free-style 自然语言指令。

它不覆盖：

- 端到端语音对话或实时打断。Speech Agent Arena 中，Qwen Audio 3.0 Realtime Plus 的首次音频时间（Time To First Audio, TTFA）为 1.54 秒、Elo 为 699，低于 Gemini 3.1 Flash Live Preview（0.96 秒 / 1046）与 GPT-Realtime-1.5（Elo 1000）；
- 自带识别的全栈语音系统。完整“听—理解—说”需要另接 ASR、对话 LLM 或 Omni Realtime；
- 开源权重即盲测最优。开源 Qwen3-TTS 的主要价值是可本地部署与可微调。

---

## 3. 发展过程

发展过程可分为四段：音频理解、Omni 打通生成、抽出开源 TTS、收拢为 Qwen-Audio-3.0 品牌。

### 3.1 音频理解阶段（2023-12 — 2024-12）

| 时间 | 事件 | 开源 | 说明 |
|------|------|------|------|
| 2023-12-01 | Qwen-Audio | 是 | 音频理解大模型，无 TTS |
| 2024-07-07 | FunAudioLLM（CosyVoice + SenseVoice） | 是，Apache-2.0 | 第二条语音线起步 |
| 2024-08-13 | Qwen2-Audio 7B / 7B-Instruct | 是 | 语音聊天与音频分析，仍无 TTS |
| 2024-12 | CosyVoice 2（0.5B） | 是 | 双流式，首包约 150 ms |

此阶段 Qwen 主线只提供理解，合成能力在 CosyVoice 线上迭代。

### 3.2 Omni 打通生成（2025-03 — 2025-09）

| 时间 | 事件 | 开源 | 说明 |
|------|------|------|------|
| 2025-03-26/27 | Qwen2.5-Omni | 是 | Thinker-Talker，模型首次具备语音生成 |
| **2025-04-10** | `qwen-tts` 上架百炼 | 否，API | 独立 TTS 产品线。中英、7 音色（Cherry / Ethan / Chelsie / Serena / Dylan / Jada / Sunny）；1 秒音频对应 50 token |
| 2025-07-01 | 新增北京话、上海话、四川话 | 否，API | 方言能力首次进入该 API |
| 2025-07-15 | `qwen-tts-realtime` | 否，API | 双向流式 |
| **2025-09-24** | Qwen3-Omni（云栖大会） | 是 | Thinker / Talker 改为混合专家（Mixture of Experts, MoE）；引入多码本、MTP、ConvNet Code2Wav 与 12.5 Hz |

Qwen3-Omni 论文写明：相对 Qwen2.5-Omni，Talker 从单轨 codec 转为 MTP 多轨，Code2Wav 用轻量 ConvNet 替换分块 DiT，音频码率降至 12.5 Hz。这四项构成半年后 Qwen3-TTS 的技术栈。

### 3.3 Qwen3-TTS 开源（2025-11 — 2026-01）

| 时间 | 事件 | 开源 | 说明 |
|------|------|------|------|
| 2025-11-27 | `qwen3-tts-flash`、`qwen3-tts-flash-realtime`、`qwen3-tts-vc-realtime`、`qwen-voice-enrollment` 上架百炼 | 否，API | Qwen3-TTS 商用上架，早于开源约两个月 |
| 2025-12-12/16 | `qwen-voice-design`、`qwen3-tts-vd-realtime` | 否，API | 自然语言音色设计 |
| 2025-12-23 | 通义百聆：Fun-ASR / Fun-CosyVoice3 / Fun-Audio-Chat | 是 + 托管 | CosyVoice 线同步发布 |
| **2026-01-21/22** | Qwen3-TTS 1.7B / 0.6B 开源 | 是，Apache-2.0 | 官方博客日期为 01-21，GitHub 为 01-22 |

开源覆盖中、英、日、韩、德、法、俄、葡、西、意 10 种语言，以及普通话、闽南语、吴语、粤语等 9 种方言。同期还提供音色克隆（Voice Clone, VC）与音色设计（Voice Design, VD）接口。

### 3.4 收拢为 Qwen-Audio-3.0（2026-02 — 2026-08）

| 时间 | 事件 | 开源 | 说明 |
|------|------|------|------|
| 2026-02-09/10 | `cosyvoice-v3-flash` / `cosyvoice-v3-plus` 上架百炼 | 否 | CosyVoice 继续承担复刻与设计档 |
| 2026-03 | CosyVoice-V3.5 plus / flash | 否 | 百炼将其标为声音设计 / 复刻 |
| 2026-04-15 | `voice-enrollment` 统一音色管理 | 否 | 跨模型复用音色资产 |
| 2026-05-28（榜期） | Fun-Realtime-TTS-Preview，Elo 1190 | 否，预览 | 全球第 5、国内第 1 |
| **2026-07-20** | Qwen-Audio-3.0-TTS Flash / Plus | 否，API | 16 语种、细粒度标签、约 3 分钟一次合成 |
| 2026-07-24 | 48 kHz 输出上架百炼 | 否 | 相对开源侧 24 kHz 提升采样率 |
| 2026-07-27 | arXiv 2607.23938 | 论文 | 闭源一代技术报告 |
| **2026-08-17** | ASR / TTS / Realtime 上线千问 AI 与百炼 | 否 | 品牌统一；进入千问 App、千问办公、Qoder |

### 3.5 开源范围

Qwen3-TTS 开源内容包括 12.5 Hz 推理权重、推理代码、官方 pip / Gradio 与微调脚本。论文另描述过 25 Hz 单码本 + 分块 DiT + BigVGAN 路线，开源说明曾写“近期发布”，截至 2026 年 9 月该权重仍未放出。

Qwen-Audio-3.0-TTS 仅通过 API 提供，未见开源计划。论文署名为 Qwen Audio、Token Foundry、Alibaba Group。论文指标、开源权重与云端档位需要分开阅读。

---

## 4. 技术架构

### 4.1 开源主路径

Qwen3-TTS 采用“离散多码本语言模型 + 轻量声码器”的两阶段结构，工程上拆为三个模块。Code2Wav 不依赖说话人向量提取，也不使用 DiT、流匹配或 HiFi-GAN，而是一条纯卷积路径。

```mermaid
flowchart LR
    textOrRef["文本或参考音频"] --> talker["Talker Qwen3 dense"]
    talker --> mtp["MTP Code Predictor"]
    mtp --> code2wav["Code2Wav 因果 ConvNet"]
    code2wav --> wave["24kHz 波形"]
```

模块规格来自开源 `config.json`。论文表述为 “Qwen3-TTS leverages the Qwen3 LM family”。Qwen3-1.7B 文本模型与 Qwen3-TTS-1.7B Talker 在层数、隐维和注意力头上对齐（28 层 / 2048 / 16 头 / 8 KV / 6144 / head_dim 128），与逐权重插值结果一致，1.7B 可由 Qwen3-1.7B 初始化，0.6B 可由 Qwen3-0.6B 初始化。

| 模块 | 作用 | 规格 |
|------|------|------|
| **Talker** | 稠密解码器，每帧预测第 0 层码本 | 1.7B：28 层 / hidden 2048 / 16 头 8 KV（分组查询注意力, Grouped-Query Attention, GQA）；0.6B：28 层 / hidden 1024；多分辨率旋转位置编码（Multi-Resolution Rotary Position Embedding, MRoPE）`[24,20,20]` + QK-norm；双 embedding：文本词表 151936 + codec 词表 3072 |
| **Code Predictor（MTP）** | 逐帧补齐其余 15 层残差码本 | 5 层稠密 Transformer，hidden 1024 |
| **Code2Wav** | 将码本还原为波形 | 残差矢量量化（Residual Vector Quantization, RVQ）反量化 → 8 层滑动窗口 Transformer（窗口 72）→ ConvNeXt 上采样 → SnakeBeta SEANet × `[8,5,4,3]` → 每帧 1920 样本 @ 24 kHz |

### 4.2 Tokenizer-12Hz

开源一代的离散化组件命名为 12 Hz，实际帧率为 **12.5 Hz**。

| 参数 | 数值 |
|------|------|
| 帧率 | 12.5 Hz |
| 码本层数 | 16 层 |
| 每层码本大小 | 2048 |
| 比特率 | 12.5 × 16 × 11 ≈ **2.2 kbps** |
| 分层 | codebook 0 为语义层，由 WavLM 作教师监督；codebook 1–15 为 15 层 RVQ 声学层 |
| 因果性 | 编码器与解码器全因果，无 look-ahead |
| 解码器 | 轻量因果 ConvNet |

在 LibriSpeech test-clean 上的重建指标：宽带感知语音质量评价（Perceptual Evaluation of Speech Quality, PESQ）3.21、窄带 PESQ 3.68、短时客观可懂度（Short-Time Objective Intelligibility, STOI）0.96、自动听感预测 UTMOS 4.16、说话人相似度（Speaker Similarity, SIM）0.95。

语义层与声学层的划分发生在 tokenizer 内部，与下一节的 Dual-Track 不是同一件事。

### 4.3 Dual-Track 与首包延迟

Dual-Track 的官方定义是：将**文本 token 与声学 codec token 沿 channel 维拼接**。模型每接收一个文本 token，即预测对应的声学 token，再由 Code2Wav 转为波形。25 Hz 与 12.5 Hz 是两条波形还原路线，分别对应未开源的分块 DiT 与开源的因果 ConvNet，并不是 Dual-Track 的两条轨。

论文 Table 2（并发 = 1）给出的首包延迟如下：

| 模型 | LM 首包时间（Time To First Packet, TTFP） | Tokenizer 解码 | 首包延迟 | 实时因子（Real-Time Factor, RTF） |
|------|-------------------------------------------|----------------|----------|-----------------------------------|
| 12Hz-**0.6B** | 93 ms | 4 ms | **97 ms** | 0.434 |
| 12Hz-1.7B | 97 ms | 4 ms | **101 ms** | 0.463 |
| 12Hz-0.6B（并发 6） | — | — | 299 ms | — |
| 12Hz-1.7B（并发 6） | — | — | 333 ms | — |

官方博客将 97 ms 表述为端到端合成延迟，未绑定模型尺寸。按论文表格，该数字对应 **0.6B、并发 1**。一个语音包（speech packet）包含 4 个 token，对应 320 ms 音频；12 Hz 解码器只使用左上下文，token 到齐即可出波形；MTP 一次给出 16 层码本，因此第一个 codec 帧即可发声。

### 4.4 三条波形还原路线

| 维度 | 25 Hz（未开源） | 12.5 Hz（Qwen3-TTS 开源） | 流匹配 + 超分（Qwen-Audio-3.0-TTS） |
|------|-----------------|---------------------------|-------------------------------------|
| Tokenizer | 25 Hz 单码本，码本大小 32768，基于 Qwen2-Audio 继续预训练 | 12.5 Hz、16 层多码本，每层 2048 | 12.5 Hz 低帧率 tokenizer |
| 波形还原 | 分块 DiT（流匹配）+ 改进 BigVGAN | 轻量因果 ConvNet（Code2Wav） | LLM + 流匹配 + 48 kHz 超分声码器 |
| 首包延迟 | 140–500 ms；chunk=8 时需等待 16 token（320 ms），叠加 BigVGAN 约 130 ms look-ahead 后约 190 ms | **97 ms**（0.6B / 并发 1） | Flash 发布口径约 300 ms，百炼文档写 200 ms 以内 |
| 采样率 | — | 24 kHz | **48 kHz** |
| 取舍 | 重建质量较高，延迟受块上下文限制 | 首包更低，音质上限低于扩散路线 | 音质优先，吞吐较低 |

闭源 Flash 的云端延迟高于开源 97 ms，是因为 Qwen-Audio-3.0-TTS 采用自回归 LLM + 分块流匹配（需多步常微分方程（Ordinary Differential Equation, ODE）采样）+ 48 kHz 超分 BigVGAN。12.5 Hz tokenizer 只降低自回归段成本，不消除流匹配与超分声码器的计算，云端测量还包含网络往返。

### 4.5 指令控制与细粒度标签

Qwen3-TTS 将声音控制作为语言建模任务：训练数据统一为 ChatML，并按概率启用思维模式（thinking pattern），以加强对复杂描述的遵循，无需预设 speaker ID。Voice Design 允许用自然语言指定音色，例如“语调平和、适合有声书朗读的成年女声”，而不必先选择音色编号。

Qwen-Audio-3.0-TTS 论文给出 **86 个细粒度行内标签**，支持短语级与词级控制，分为两类：

- **控制类**（作用于其后文本，直至下一个标签）：`[excited]`、`[sad]`、`[whispers]`、`[asmr]`、`[angry]`、`[crying]`、`[very slowly]`
- **富语言类**（在当前位置插入一次性发声，不改变周围风格）：`[gasp]`（吸气）、`[giggles]`（轻笑）、`[laughing]`、`[sighing]`、`[cough]`、`[snorts]`、`[clears throat]`

论文写 86 个标签，官方英文文档列出 23 个控制类与 7 个富语言类。标签是否由标注数据驱动，论文未说明。上述标签仅在单向流式模式下可用。

### 4.6 语音克隆

克隆时长存在两档公开说明：

- **3 秒**：论文摘要与 GitHub README 给出的模型能力下限（"3-second voice cloning"）。
- **10–20 秒**：百炼 `qwen-voice-enrollment` 的录入建议；最长 60 秒，采样率不低于 24 kHz，单声道，至少 3 秒连续清晰语音。

实现路径有两条：

1. **说话人嵌入（speaker embedding / x-vector）**：参考音频经编码器得到条件，适合实时克隆。
2. **文本—语音对的上下文学习（in-context learning）**：同时提供参考音频及其转写，韵律通常更稳。公开社区测试中，补全准确转写后 SIM 约从 0.75 升至 0.89。

跨语言克隆在 CV3-Eval 上，中文到韩语错误率为 4.82，CosyVoice3 为 14.4；12 个方向均值为 4.418。Seed-TTS 的 test-zh 上，CosyVoice3 的字符错误率（Character Error Rate, CER）为 0.71，Qwen3-TTS 为 0.77。官方对开源一代在 test-en 上给出当时最优（State of the Art, SOTA）表述。

闭源论文标题强调可控性与鲁棒性（Freely Controllable and Highly Robust），包括噪声、混响参考音频下的克隆。实际效果仍取决于参考音质量。

### 4.7 训练

开源一代的训练数据规模为 **超过 500 万小时**，覆盖 10 种语言。训练分三阶段：

1. **S1 通用预训练**
2. **S2 高质量持续预训练（Continued Pre-Training, CPT）**，用于降低幻觉
3. **S3 长上下文**：序列长度从 8,192 扩展到 32,768，并对长语音做上采样

数据统一为 ChatML。说话人编码器与主干联合训练。Qwen-Audio-3.0-TTS 论文另述五阶段训练，细节见 arXiv:2607.23938，开源仓库不能复现该过程。

---

## 5. 核心能力与性能

下列数字来自不同协议与测试集，不宜排成单一进步曲线：

1. Qwen3-TTS 论文：LibriSpeech 重建、CV3-Eval 跨语、长语音词错误率（Word Error Rate, WER）
2. Qwen-Audio-3.0-TTS 论文与发布说明：16 语种、说话人相似度、3 分钟、48 kHz
3. Artificial Analysis 盲测榜（2026-09-04）
4. 公开社区在消费级显卡上的 RTF

### 5.1 零样本克隆与跨语

开源一代以约 3 秒参考音频做零样本换嗓，覆盖 10 语种，并尽量在跨语条件下保持音色。评价通常看两项：内容是否念对（中文 CER，英文 WER），以及是否像目标说话人（SIM / SS）。

跨语克隆是开源 Qwen3-TTS 相对 CosyVoice3 的优势项（见 4.6）。中文干净朗读上，CosyVoice3 的 test-zh CER 更低。Qwen-Audio-3.0-TTS 的发布说明称，在 CV3-Eval 的 16 种语言中有 10 种取得最佳 WER / CER，Plus 在说话人相似度上居 16 语第一，平均 SS 为 82.75。该组数字与开源论文表不是同一代模型。

### 5.2 流式与实时

| 版本 | 流式形态 | 延迟 |
|------|----------|------|
| Qwen3-TTS 0.6B 开源 | 全因果，首帧可出声 | 首包 **97 ms**（论文 Table 2，并发 1） |
| Qwen3-TTS 1.7B 开源 | 同上 | 首包 101 ms（并发 1）；并发 6 时 333 ms |
| Qwen-Audio-3.0-TTS Flash | 云端 API | 发布口径约 **300 ms**；百炼模型页为 **200 ms 以内** |
| 25 Hz 未开源路线 | 分块 DiT + BigVGAN | 约 190 ms 起 |

97 ms 是论文给定条件下的自测值。消费级显卡上的 RTF 见 5.5 节。

### 5.3 多语种与方言

- **`qwen-tts`（2025-04）**：中英；2025-07 起增加北京话、上海话、四川话。
- **Qwen3-TTS 开源**：10 语种 + 9 种方言。
- **Qwen-Audio-3.0-TTS**：16 语种，包括阿拉伯语、越南语、马来语、菲律宾语等；中文方言在媒体报道中写为 20 种，百炼语言清单列为 **21 个方言区**。

官方建议每个说话人使用其母语文本；跨语种合成质量通常下降。公开讨论中还出现过英语口音偏移、0.6B 在非英语上降质、长文本偶发表情发声等现象，尚未形成统一评测集。

多方言覆盖是该产品线相对 ElevenLabs、OpenAI、Google 等国际通用接口的主要差异之一。

### 5.4 长语音内容一致性

公开材料给出三组长语音 WER，测试协议不同：

| 来源 | 中文 | 英文 | 条件 |
|------|------|------|------|
| 官方博客 | 2.36% | 2.81% | 连续 10 分钟合成 |
| 论文 Table 10，25Hz-1.7B | 1.517% | 1.225% | 与博客不是同一协议 |
| 论文正文 | 1.533% | 1.571% | 与同论文表格数值不同 |

闭源一代的产品说明为单次合成最长约 3 分钟、一次前向完成（one-pass），不宜与上表直接比较。

### 5.5 Artificial Analysis 榜与本地 RTF

2026-09-04 实时榜（共 98 个模型）节选：

| 排名 | 模型 | Elo | 榜面价格（$/1M 字符） |
|------|------|-----|----------------------|
| 1 | Cartesia Sonic 3.6 | **1282** | $49 |
| 2 | Inworld Realtime TTS-2 | 1252 | $20.8 |
| 3 | **Qwen-Audio-3.0-TTS-Plus** | **1241** | $27.6 |
| 4 | Simba 3.2 | 1240 | $10 |
| 9 | Gemini 3.1 Flash TTS | 1208 | $18.3 |
| — | ElevenLabs Eleven v3 | 1175 | $100 |
| — | MiniMax Speech 2.8 HD | 1171 | $100 |
| **85** | Qwen3 TTS Flash | **945** | $10 |
| **86** | Qwen3 TTS | **926** | $20 |

7 月榜 Plus 为 1237，当时列第一，与 Simba 3.2 的 1234 在 95% 置信区间重叠。9 月实时榜降至第三。开源 Qwen3-TTS 与闭源 Plus 相差约 315 Elo。同分段位上，Simba 3.2（1240）价格约为 Plus 榜面价的 1/2.8，Gemini 3.1 Flash TTS 低 33 分、价格约为其 2/3。

第三方测得 Plus 吞吐约 16 字符/秒；官方未公布吞吐。按中文约 4–5 字/秒估算，该速度仍高于实时语速，但低于 Sonic 3.5 一类速度档（约 120 字符/秒）。可能原因是 ODE 采样步数与 48 kHz 超分。

公开社区给出的开源 RTF：

| 硬件 | 模型 | RTF |
|------|------|-----|
| RTX 3090 | 1.7B（35 秒音频用时 44 秒） | **≈ 1.26**（慢于实时） |
| RTX 3090 | 0.6B（用时 30 秒） | ≈ 0.86 |
| GTX 1080 | 0.6B | **2.11** |

在消费级显卡上要接近实时，0.6B 是更现实的起点。1.7B 更适合离线质量优先的任务。

---

## 6. 竞品对比

Qwen-TTS 的位置可以概括为：主品牌标准合成、与 Omni 协同、多语多方言；开源可私有化，闭源再提高听感。它与 CosyVoice 是并行产品，不是互相替代。

### 6.1 与 CosyVoice

| 维度 | Qwen3-TTS 开源 | Qwen-Audio-3.0-TTS | CosyVoice 3 / 3.5 |
|------|----------------|--------------------|-------------------|
| 骨架 | 多码本 LLM + 因果 ConvNet，无流匹配 | LLM + 流匹配 + 48 kHz 超分 | 监督语义 token + LLM + 条件流匹配 + HiFT |
| 开源规模 | 1.7B / 0.6B | 无开源权重 | 主力 0.5B，另有 1.5B / 3.5 |
| 流式 | 全因果；97 ms 为 0.6B / 并发 1 自测 | Flash 公开口径 200–300 ms | 2.0 起同一权重兼离线与流式，首包约 150 ms；3.5 约为 97.5 ms |
| 控制 | Voice Design 与指令 | 86 标签 + Free-style | 情感、口音、发音修补、FreeStyle |
| 产品档位 | 3 秒克隆；百炼标准合成 | 标准音色、音质优先 | 百炼标为声音设计 / 复刻 |
| 评测 | 跨语 CV3-Eval 优于 CosyVoice3 | 闭源盲测 Elo 进入第一梯队 | test-zh CER 0.71，低于 Qwen3-TTS 的 0.77 |

需要高相似度复刻或开放域声音设计时，百炼将 CosyVoice-v3.5 放在更前。需要标准播报、与千问 / Omni 共用账号和音色资产、或本地部署 12.5 Hz 开源权重时，Qwen-TTS 更直接。

### 6.2 与国际闭源接口

在 2026-09 盲测中，Plus 仍处于第一梯队，但不再是该榜第一。价格高于 Simba 3.2 与 Gemini 3.1 Flash TTS；若吞吐按 16 字符/秒计，也不属于速度优化档。

相对 Seed-TTS：开源 Qwen3-TTS 在跨语克隆上有公开 CV3-Eval 数字；中文干净集的内容一致性仍常见 CosyVoice RL、Seed-TTS 一类模型领先。闭源旗舰（包括 Plus）通常是听感上限；需要可审计、可私有化时，应比较开源 Qwen3-TTS 与 CosyVoice。

### 6.3 与识别、语音智能体的边界

Whisper、Qwen3-ASR、Fun-ASR 解决识别。Qwen-TTS 解决合成。TTS 表中的 CER / WER 是“合成后再识别”的内容一致性，不是 ASR 转写误差。

Speech Agent Arena 衡量端到端对话延迟与交互能力，不是 TTS 自然度。合成 Elo 高并不等于实时语音智能体强。

### 6.4 选型对照

| 需求 | 更合适的选择 | 依据 |
|------|----------------|------|
| 闭源听感优先、可接受 API | Qwen-Audio-3.0-TTS-Plus 或当时 AA 榜前列 | 2026-09 实时榜 Plus 第三（1241） |
| 标准音色、少运维、与千问同一套账号 | 百炼 `qwen-audio-3.0-tts-plus` / `flash` | 官方标准合成档 |
| 声音设计 / 复刻 | CosyVoice-v3.5 | 百炼产品分工 |
| 中文方言、开源流式、指令控制 | CosyVoice 3 / 3.5 | 工程栈与 CV3-Eval |
| 私有化、可微调、需要亚 100 ms 首包自测 | Qwen3-TTS 0.6B | 97 ms 对应 0.6B / 并发 1 |
| 本地质量优先 | Qwen3-TTS 1.7B | 听感通常优于 0.6B；RTX 3090 上 RTF 约 1.26 |
| 端到端语音对话、实时打断 | Qwen Omni / Audio Realtime，或 Gemini Live / GPT-Realtime | TTS 只是链路中的一环 |
| 偶发配音、无推理运维 | 云 API | 不必自建权重 |

---

## 7. 应用场景

百炼将标准音色用于智能客服、有声阅读、内容播报和电商直播，将自定义音色用于品牌声、虚拟主播和游戏角色。按能力可以对应到下列场景。

| 应用场景 | 主要能力 | 典型形态 |
|----------|----------|----------|
| 实时语音交互 | 低首包、Flash / 开源 12.5 Hz | 智能客服、语音助手、交互式语音应答（Interactive Voice Response, IVR） |
| 标准播报 | 预设音色、多语多方言 | 资讯播报、直播口播、有声阅读 |
| 零样本声音克隆 | 3 秒下限、10–20 秒录入、跨语保音色 | 品牌音色、个人声音包、虚拟主播 |
| 音色设计 | Voice Design，无需参考音频 | 游戏非玩家角色（Non-Player Character, NPC）、多角色内容、广告配音 |
| 有声内容生产 | 长文本、标签 / 指令、Plus 48 kHz | 有声书、播客、宣传片 |
| 数字人与直播 | 流式输出与语气切换 | 虚拟主播、讲解 |
| 游戏与影视 | 细粒度标签、中英混说 | NPC、技能语音、后期补录 |
| 无障碍与教育 | 稳定声线、方言、可变语速 | 朗读辅助、方言教材 |

### 7.1 语音到语音与级联流水线

百炼提供两条组装方式。

**路线 A：语音到语音（Speech-to-Speech, S2S）单模型**

- 代表接口：`qwen3.5-omni-plus/flash-realtime`、`qwen-audio-3.0-realtime-plus/flash`、`qwen3.5-livetranslate-flash-realtime`（约 60 语种同传）
- 延迟较低，能够利用语调与情绪信息
- 音色限于预设

**路线 B：级联（ASR → LLM → TTS）**

- ASR：`Qwen-Audio-3.0-ASR`（流式 / 非流式 / 最长约 12 小时文件转写，含说话人分离）或 `Fun-ASR-Realtime`
- TTS：`qwen-audio-3.0-tts-plus/flash`、`cosyvoice-v3.5-plus` 等
- 自定义音色走此路线

共用组件包括 `voice-enrollment`，以及 AOQ / WebRTC / WebSocket。CosyVoice 与 Qwen-Audio-TTS 支持 AOQ 直连，Qwen3-TTS 系列不支持。

官方示例库 `aliyun/alibabacloud-bailian-speech-demo` 提供呼叫中心、播报配音、会议分析等接入代码，并包含 Qwen-Audio-3.0-TTS 的指令控制与克隆示例。2026-08-17 起，该能力进入千问 App、千问办公和 Qoder。

### 7.2 智能客服与品牌音色

Flash 与开源 12.5 Hz 适合坐席辅助、售后问答和多语 IVR：同一套模型切换语种和方言，不必为每种语言单独采购合成引擎。品牌声通过授权录音写入 `voice-enrollment`。Voice Design 可以用自然语言描述客服语气（例如温和、带歉意、语速适中），从而减少为每条新话术重新棚录的成本。

跨境场景通常将“母语坐席 + 机器翻译 + 多语 TTS”串联。闭源一代对东南亚语种的覆盖宽于开源 10 语名单，目标语种仍需单独听测。接通率、排队时长和满意度还取决于路由、ASR 和对话策略，不能单独由 TTS 决定。

### 7.3 有声书、播客与数字人

Plus 提供 48 kHz 输出和约 3 分钟一次合成，适合离线内容和品牌片子。云端 API 与本地开源权重都可以用于有声生产：按章节设置语气、用标签插入停顿或笑声、在中英混排文本上保持读法。长篇多角色建议在应用层做角色标记和分轨，而不是依赖单次前向自动完成整本书。数字、商标和多音字需要预置读法，并用 CER 复核。

数字人场景需要稳定的流式音频和可切换语气。TTS 只负责波形；唇形、表情和渲染由外部模块完成。声、唇、画面是否同步是系统问题，不是 Code2Wav 单独保证的指标。

按百炼国内站 Plus 价 ¥1.4 / 万计费字符、中文按 2 计，10 万汉字约对应 20 万计费字符，合成费用约 ¥28（不含音色注册）。具体以当时计价规则为准。

### 7.4 游戏、影视与二创

标签可以控制呼吸、情绪转折和非语言发声；Voice Design 可以按角色描述生成 NPC 声线；流式接口支持边生成边播放。中英混说台词（例如中文句子中夹英文任务名）是开源与闭源都明确覆盖的能力。游戏 NPC 适合批量生产可用声线，不替代主演级表演。

影视和二创更常使用 Plus 的 48 kHz 与噪声参考音克隆。历史声线、公众人物声音和未授权角色音受版权与人格权约束，不属于模型许可范围。

### 7.5 合规边界

约 3 秒即可克隆，降低了未授权复刻的成本。官方禁止侵权复刻、冒充、诈骗和违规外呼。本地部署减少音频出域，使用责任仍由部署方承担。

业务层通常需要：权利人书面授权、声纹与文本黑名单、合成内容水印或审计日志，以及医疗、金融、政务外呼等场景的人工复核。

---

## 8. 生态、部署与工程实践

### 8.1 开源组件

官方提供：

- Qwen3-TTS 1.7B / 0.6B 推理权重（Hugging Face / GitHub，Apache-2.0）；
- `qwen-tts` pip 包、Gradio 演示与微调脚本；
- Tokenizer-12Hz 模型卡；
- vLLM-Omni 支持，当前仅离线推理。

ComfyUI 节点与 GGUF 量化来自社区。25 Hz 权重截至 2026 年 9 月未开源。常见路径是先用官方脚本验证克隆和 Voice Design，再决定继续本地服务或把流量放到百炼。

### 8.2 百炼定价

华北 2（北京），2026-09-04：

| 模型 | 国内站 | 国际站 | 免费额度（90 天） |
|------|--------|--------|-------------------|
| `qwen-audio-3.0-tts-plus` | ¥1.4 / 万字符 | ¥1.49884 / 万字符 | 1 万字符 |
| `qwen-audio-3.0-tts-flash` | ¥1.0 / 万字符 | ¥1.12413 / 万字符 | 1 万字符 |
| `qwen3-tts-flash` | ¥0.8 / 万字符 | ¥0.733924 / 万字符 | 11 万字符 |
| `qwen3-tts-instruct-flash` | ¥0.8 / 万字符 | ¥0.8 / 万字符 | 11 万字符 |
| `qwen3-tts-vd-2026-01-26` | ¥0.8 / 万字符 | ¥0.8 / 万字符 | 11 万字符 |
| `qwen3-tts-vc-2026-01-22` | ¥0.8 / 万字符 | ¥0.8 / 万字符 | 11 万字符 |
| `qwen3-tts-instruct-flash-realtime` | ¥1.0 / 万字符 | — | 11 万字符 |
| `qwen-tts`（旧版，按 token） | 输入 ¥1.6 / 百万 token、输出 ¥10 / 百万 token | $0.23 / $1.434 每百万 token | 100 万 token |

计费规则：

- 输出音频不计费，只计输入字符；
- 免费额度仅中国大陆（北京）提供；
- 每分钟请求数（Requests Per Minute, RPM）上限为 180；
- Plus / Flash 免费额度为 1 万字符，Qwen3-TTS 系列为 11 万字符；
- 公开渠道另列出音色注册价：克隆约 $0.01 / 个、设计约 $0.20 / 个；
- AA 榜 Plus 为 $27.6 / 1M 字符，国际站报价约合 $20.4 / 1M。差异来自计费规则中**中文字符按 2 计**（官方示例：「你好」计 4 字符）。成本对比需统一口径。

### 8.3 部署方式

| 方式 | 适用对象 | 说明 |
|------|----------|------|
| 百炼云 API | 需要服务等级协议（Service Level Agreement, SLA）、少运维 | 数据出域；受 RPM 与免费额度约束 |
| 开源私有化（GPU） | 内网、研究、微调 | vLLM-Omni 目前承诺离线推理 |
| WebUI / CLI | 试用与演示 | 不适合高并发 |
| 社区量化（GGUF 等） | 小流量试验 | 需听测保真度 |
| CosyVoice 云档 / Studio | 复刻与声音设计 | 与 Qwen-TTS 并列 |

RTX 3090 上，0.6B 的 RTF 约 0.86，1.7B 约 1.26；GTX 1080 上 0.6B 的 RTF 为 2.11。显存随精度、是否同时加载 tokenizer / Code2Wav 以及批大小变化。

### 8.4 落地顺序

1. 用目标语种、方言和中英混说文本测量 CER 与听感。
2. 分别测量短句、长句、并发 1 与并发 6 的首包和 RTF。97 ms 只对应 0.6B、并发 1，不能直接写进 1.7B 或云端 Flash 的时延承诺。
3. 品牌音色使用授权录音和 `voice-enrollment`；API 侧按 10–20 秒、不低于 24 kHz 录入。需要更稳韵律时，同时提供准确转写。
4. 数字、多音字、商标名在业务层预置读法；词级情绪或笑声、呼吸需确认当前档位是否支持单向流式标签。
5. 对话任务另接 Qwen-Audio-3.0-ASR / Fun-ASR 和业务 LLM，或改用 Omni Realtime。
6. 无参考造声使用 Voice Design；高相似度复刻同时对比 CosyVoice-v3.5。

---

## 9. 优势、局限与选型建议

### 9.1 优势

1. **技术来源连续**：Qwen3-TTS 是 Qwen3-Omni 语音侧的专用化，token 接口与 Omni 对齐。
2. **开源全因果低首包**：12.5 Hz + MTP + 因果 ConvNet，0.6B / 并发 1 时首包 97 ms。
3. **多语多方言覆盖**：开源 10 语 + 9 方言，闭源 16 语 + 约 20–21 个方言区。
4. **控制接口完整**：Voice Design、行内标签、克隆和预设音色同属一条产品线。
5. **产品配套齐全**：开源权重、百炼 API、`voice-enrollment`、示例库，以及与 ASR / Realtime / 千问 App 的统一入口。

### 9.2 局限

1. **开源与闭源差距大**：约 315 Elo；最新流匹配 + 48 kHz 方案不开放。
2. **25 Hz 权重未发布**：高音质路线截至 2026 年 9 月仍不可下载。
3. **榜单位置随时间变化**：2026 年 7 月第一，9 月第三；7 月相对第二名仅为统计并列。
4. **实时语音智能体偏弱**：TTFA 1.54 秒、Elo 699，低于 Gemini Live 与 GPT-Realtime。
5. **消费卡上不一定实时**：RTX 3090 上 1.7B 的 RTF 约 1.26。
6. **名称对应多套系统**：`qwen-tts`、Qwen3-TTS、Qwen-Audio-3.0-TTS、Fun-Realtime-TTS-Preview 容易被当成同一模型。
7. **部分公开数字存在多口径**：长语音 WER、Flash 延迟、方言数量、吞吐。
8. **克隆门槛低带来滥用风险**。

### 9.3 适合采用的情况

- 中文及方言、中英混说或东南亚语种播报，需要标准合成或可授权克隆；
- 需要私有化开源权重，并能够接受 12.5 Hz 一代的音质、自行测量 RTF；
- 已使用或计划使用千问、百炼、Qwen-Audio-3.0 ASR，希望听、说和音色资产共用账号；
- 需要 Voice Design 或标签控制语气，但不想为每个声音单独微调；
- 需要一个可复现的开源多码本流式基线。

### 9.4 不宜作为默认选择的情况

- 只要求单一闭源音色的最高相似度，且可以接受当时 AA 榜更高或更便宜的接口；
- 主要需求是声音设计或高相似度复刻——百炼将该档指向 CosyVoice-v3.5；
- 必须做端到端语音对话和实时打断——应使用 Omni Realtime 或同类 Live 模型；
- 没有 GPU 推理经验，且只是偶发配音——使用云 API 即可；
- 需要把 97 ms 或不带日期的“榜首”写成对客时延或排名承诺。

### 9.5 版本选择

| 需求 | 选择 |
|------|------|
| 复现 12.5 Hz 全因果管线 | Qwen3-TTS 论文与 0.6B / 1.7B 权重 |
| 本地实时、显存受限 | Qwen3-TTS 0.6B（先测 RTF） |
| 本地质量优先、可离线 | Qwen3-TTS 1.7B |
| 云端标准合成、48 kHz | `qwen-audio-3.0-tts-plus` |
| 云端更低延迟 | `qwen-audio-3.0-tts-flash`（按 200–300 ms 口径实测） |
| 复刻 / 自然语言造声 | CosyVoice-v3.5 |
| 端到端对话 | Omni / Audio Realtime |
| 只调用、少运维 | 百炼对应档位，按 8.2 节计算字符成本 |

---

## 10. 总结

Qwen-TTS 是通义千问将 Omni 语音侧独立成专用合成线的结果。2025 年 3 月模型首次具备语音生成，4 月出现独立 API，2026 年 1 月开源 12.5 Hz 多码本一代，7 月以闭源流匹配 + 48 kHz 提高音质，8 月将 ASR / TTS / Realtime 归入 Qwen-Audio-3.0。它与 CosyVoice 并行：一条对接主品牌与 Omni，一条承担开源生态与复刻档位。

阅读评测时需要分开四组数字：开源论文中的重建、跨语和有条件的 97 ms；闭源论文与发布说明中的 16 语种、标签和 48 kHz；Artificial Analysis 榜上带日期的排名；以及本地硬件上的 RTF。开源与闭源相差约 315 Elo。实时对话智能体是另一项指标，不能用 TTS Elo 代替。

选型上应区分 `Qwen3-TTS` 与 `Qwen-Audio-3.0-TTS`，将 97 ms 限制在 0.6B、并发 1 的条件内，并在获得授权后再做声音克隆。用业务文本测量 CER、相似度、首包和禁用声线之后，再决定自建开源、调用百炼 Plus / Flash，或改用 CosyVoice 与 Omni Realtime。

---

## 参考资料

1. Qwen Team. [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621). arXiv:2601.15621, 2026.
2. Qwen Audio et al. [Qwen-Audio-3.0-TTS Technical Report](https://arxiv.org/abs/2607.23938). arXiv:2607.23938, 2026.
3. Qwen Team. [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765). arXiv:2509.17765, 2025.
4. [Qwen3-TTS 官方博客（2026-01-21）](https://qwen.ai/blog?id=qwen3tts-0115)
5. [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
6. [Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)
7. [阿里云百炼：语音合成模型](https://help.aliyun.com/zh/model-studio/tts-model)
8. [阿里云百炼：模型定价](https://help.aliyun.com/en/model-studio/model-pricing)
9. [阿里云百炼：S2S 模型选型](https://help.aliyun.com/zh/model-studio/s2s-model)
10. [阿里云百炼：声音复刻](https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning)
11. [alibabacloud-bailian-speech-demo](https://github.com/aliyun/alibabacloud-bailian-speech-demo)
12. [Artificial Analysis Text-to-Speech Leaderboard](https://artificialanalysis.ai/text-to-speech)（榜面截至 2026-09-04）
13. [vLLM-Omni TTS 支持说明](https://docs.vllm.com.cn/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/text_to_speech/)
