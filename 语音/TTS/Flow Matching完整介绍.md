# Flow Matching 完整介绍


## 1. Flow Matching 是什么

**Flow Matching（FM，流匹配）是一种通过回归速度场来训练连续时间生成模型的方法**。它学习怎样让一个简单分布中的样本随时间移动，最终形成目标数据分布。

例如，文本转语音可以从高斯噪声开始，在文本和参考音频条件下，逐渐把噪声变成目标 mel 频谱或连续音频 latent，再通过声码器或解码器还原波形。

```text
简单源分布                    学习到的速度场                    目标分布
高斯噪声样本  ───────────→  沿 ODE 轨迹逐步移动  ───────────→  语音声学表示
                                      ↑
                            文本、参考音色等条件
```

Lipman 等人的 *Flow Matching for Generative Modeling* 于 2022 年 10 月提交，发表于 ICLR 2023。该工作提出以预先构造的条件概率路径为监督，直接回归向量场，从而避免传统连续归一化流训练时反复穿过 ODE 求解器。[Flow Matching 原论文](https://arxiv.org/abs/2210.02747)

这里的核心变化发生在**训练方法**。普通 FM 模型推理时通常仍需数值求解 ODE；少步或一步生成则依赖路径、模型、采样器和专门的训练设计。

### 1.1 FM、DiT 与 AR/NAR 的关系

| 层面 | 回答的问题 | 例子 |
|---|---|---|
| 训练与生成框架 | 如何学习分布变换 | Flow Matching、去噪扩散 |
| 网络骨干 | 用什么网络预测 | U-Net、DiT、Zipformer |
| 生成顺序 | 沿输出时间轴如何生成 | AR、整段 NAR、分块 AR |
| 生成空间 | 在什么表示上生成 | mel、波形 latent、其他连续特征 |

FM 不等于 DiT，也不要求非自回归。F5-TTS 是整段 NAR 的 DiT 流匹配系统；CosyVoice 2 将 AR 语言模型与 U-Net 类 CFM 声学模块组合；连续自回归系统则可以在每个音频块上运行一个 FM 生成头。

### 1.2 四个容易误读的结论

- **确定性 ODE 仍可产生多样样本**：固定初始噪声与条件后轨迹确定，不同噪声仍会得到不同输出。
- **Simulation-free 主要描述训练**：标准 CFM 构造训练样本时不必积分生成 ODE，推理仍通常要积分。
- **直线监督不保证直线推理**：逐样本插值路径与模型学习的边际速度场不是同一对象。
- **少步不是普遍保证**：不能把 FM 固定等同于 10 步，也不能把扩散固定等同于 1000 步。

## 2. 数学基础

### 2.1 统一记号

除 MeanFlow 专节明确说明外，本文统一使用**从源到目标**的时间方向：$t=0$ 为源，$t=1$ 为数据。

| 符号 | 含义 |
|---|---|
| $p_0$ | 源分布，常取标准高斯，也可选择其他分布 |
| $p_1$ | 目标数据分布 |
| $p_t$ | 中间时刻的概率分布 |
| $X_0,X_1$ | 源样本与目标样本 |
| $X_t$ | 中间状态随机变量 |
| $u_t(x)$ | 理想边际速度场 |
| $v_\theta(x,t,c)$ | 神经网络预测的速度场 |
| $c$ | 文本、参考音频或观测等生成条件 |
| $\pi(x_0,x_1)$ | 源与目标样本的联合配对分布，即耦合 |

“耦合”规定哪些源样本与哪些数据样本配成一对。同样的源分布和目标分布，可以有许多不同配对方式。

### 2.2 ODE 与流映射

给定时间相关的速度场，样本按照常微分方程演化：

$$
\frac{dX_t}{dt}=v_\theta(X_t,t,c),
\qquad X_0\sim p_0.
$$

把从初始状态到时刻 $t$ 的映射记为 $\phi_t$，则：

$$
X_t=\phi_t(X_0),\qquad
p_t=(\phi_t)_\#p_0.
$$

符号 $\# $ 表示推前分布：把一批服从 $p_0$ 的样本经过同一个映射，得到新的分布。理想情况下，$p_{t=1}$ 与目标数据分布一致。

在速度场满足适当正则性条件时，ODE 存在唯一解并定义可逆流。真实模型只近似这些理想条件，有限精度求解也会引入误差。

### 2.3 连续性方程

样本的移动对应概率质量的移动。若 $p_t$ 有足够光滑的密度，它满足：

$$
\frac{\partial p_t(x)}{\partial t}
+\nabla_x\cdot\big(p_t(x)u_t(x)\big)=0.
$$

这是连续性方程，描述概率质量守恒。FM 要学习的并不是某一对噪声和数据之间唯一的“正确路线”，而是能够产生所需分布演化的速度场。

### 2.4 连续归一化流与似然

Continuous Normalizing Flow（CNF）通过连续时间流建立分布变换。沿模型轨迹，密度满足瞬时变量替换公式：

$$
\frac{d}{dt}\log p_t(X_t)
=-\nabla_x\cdot v_\theta(X_t,t).
$$

因此在条件适用时，可以结合 ODE 与散度积分计算密度。但**FM 回归训练不需要每次计算这个似然**，也不等同于直接最大化似然。生成模型可计算似然，不意味着似然评估没有额外成本。

理论与实现的系统说明可参考 [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)。

## 3. 从 Flow Matching 到 Conditional Flow Matching

### 3.1 理想 FM 目标

如果知道中间分布和对应速度场，可以最小化：

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t\sim U[0,1],\,X_t\sim p_t}
\left[\|v_\theta(X_t,t)-u_t(X_t)\|_2^2\right].
$$

困难在于：真实数据分布通常未知，直接获得 $u_t$ 并不容易。

### 3.2 用可采样路径构造监督

最简单的办法是采样一对端点并做线性插值：

$$
(X_0,X_1)\sim\pi,
\qquad X_t=(1-t)X_0+tX_1.
$$

对这条逐样本路径求导：

$$
\dot X_t=X_1-X_0.
$$

于是可以训练：

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{t,(X_0,X_1)\sim\pi}
\left[\|v_\theta(X_t,t)-(X_1-X_0)\|_2^2\right].
$$

实现时只需随机采样时间、构造插值和计算均方误差。网络接收 $X_t$ 与 $t$，无需知道这一训练样本的全部端点信息。

### 3.3 为什么逐样本回归能学到边际速度

在平方损失下，最优预测是目标的条件期望：

$$
u_t(x)
=\mathbb E[X_1-X_0\mid X_t=x].
$$

令 $Y=X_1-X_0$，利用条件期望的正交分解：

$$
\begin{aligned}
\mathbb E\|v_\theta(X_t,t)-Y\|^2
={}&\mathbb E\|v_\theta(X_t,t)-u_t(X_t)\|^2\\
&+\mathbb E\|Y-u_t(X_t)\|^2.
\end{aligned}
$$

第二项不依赖模型参数，所以两种目标在相应正则条件下具有相同的参数梯度与最优解。这是 CFM 能绕过难算边际速度的关键。

CFM 并不限于线性端点插值，还可以先选择条件路径 $p_t(x\mid Z)$ 及其速度，再通过条件平均得到边际速度。[FM 原论文](https://arxiv.org/abs/2210.02747)、[广义 CFM 与 minibatch OT](https://arxiv.org/abs/2302.00482)

### 3.4 “条件”有两种含义

**数学构造条件**是为了获得容易采样的路径和监督，例如目标端点或端点对。**任务生成条件**是用户希望控制输出的文本、说话人、参考音频等。

两者可以同时出现，但不能等同。CFM 已在最初 FM 工作中出现，并不是后来才增加文本输入的一种接口。

带任务条件时，线性路径目标可写成：

$$
\mathbb E\left[
\left\|v_\theta(X_t,t,c)-(X_1-X_0)\right\|^2
\right],
$$

对应的理想预测是 $\mathbb E[X_1-X_0\mid X_t=x,c]$。训练构造中使用目标样本，不意味着推理时把目标样本提供给模型。

## 4. 概率路径、耦合与 Rectified Flow

### 4.1 一般仿射路径

线性插值是以下路径的特例：

$$
X_t=a(t)X_1+b(t)X_0,
\qquad
\dot X_t=a'(t)X_1+b'(t)X_0.
$$

要连接指定端点，通常要求：

$$
a(0)=0,\ b(0)=1,\qquad
a(1)=1,\ b(1)=0.
$$

选择 $a,b$ 就是在选择信号与噪声如何随时间混合。实际模型可能保留少量终端噪声，因此端点也可能只是接近数据分布。

FM **允许选择不同路径**，并不表示完全不需要噪声日程或时间表。训练时间采样分布、路径系数和推理时间网格是相关但不同的设计。

### 4.2 独立配对与最优传输配对

最简单的耦合为：

$$
\pi(x_0,x_1)=p_0(x_0)p_1(x_1).
$$

也可以求解最优传输耦合，例如平方欧氏代价：

$$
\pi^*\in\operatorname*{arg\,min}_{\pi\in\Pi(p_0,p_1)}
\mathbb E_\pi\|X_1-X_0\|^2.
$$

minibatch OT 在一个 batch 内寻找较低代价配对，可能简化流并降低求解难度，但它是有限样本近似，还增加配对计算开销。[Tong 等人的 OT-CFM 工作](https://arxiv.org/abs/2302.00482)

需要分清：一些论文所说的“OT 条件路径”强调每个条件高斯路径的位移插值；这**不自动意味着**整个噪声分布与数据分布的配对就是全局最优传输。

### 4.3 直线训练路径为什么不等于直线生成轨迹

许多端点对的插值会经过邻近位置，却具有不同方向。网络只能根据当前状态与时间输出一个速度，因此学到的是条件平均，而不是保留每条端点直线。

ODE 按这个边际速度场前进，得到新的确定性耦合。其轨迹可能弯曲，也不必复现训练时随机抽到的原始端点配对。

一个可计算的一维例子是：

$$
X_0\sim\mathcal N(0,1),\qquad
X_1\sim\mathcal N(2,1),\qquad X_0\perp X_1.
$$

线性插值的边际分布为：

$$
p_t=\mathcal N\big(2t,q(t)\big),\qquad
q(t)=(1-t)^2+t^2.
$$

利用高斯条件期望，可得：

$$
u_t(x)=2+\frac{2t-1}{q(t)}(x-2t).
$$

其 ODE 轨迹为：

$$
X_t=2t+\sqrt{q(t)}X_0.
$$

这通常不是关于时间的直线，但确实满足所需边际分布，并在 $t=1$ 得到 $\mathcal N(2,1)$。该例是本文根据线性高斯假设推导的说明性例子，不是论文性能实验。

### 4.4 Rectified Flow 与 Reflow

Rectified Flow 于 2022 年 9 月提出，与 FM 和 Stochastic Interpolants 是同期的相关路线。其典型做法使用端点直线插值训练速度场，再以模型生成的端点配对重新训练。

```mermaid
flowchart LR
    A["源与目标端点配对"] --> B["插值与速度回归"]
    B --> C["得到第一轮 ODE 模型"]
    C --> D["由模型生成源到目标端点对"]
    D --> E["重新插值与训练 Reflow"]
    E --> F["改善轨迹与少步采样能力"]
```

Reflow 改变监督配对，目标是使后续模型的轨迹更适合粗时间离散。它不是把采样步数直接调小，也不保证任意模型一次 Reflow 后就能无损一步生成。[Rectified Flow 原论文](https://arxiv.org/abs/2209.03003)

## 5. 与扩散模型和其他 Flow 的关系

### 5.1 扩散也有确定性 ODE

对于扩散 SDE：

$$
dX_t=f(X_t,t)dt+g(t)dW_t,
$$

在适用条件下，对应的 probability-flow ODE 为：

$$
\frac{dX_t}{dt}
=f(X_t,t)-\frac12g(t)^2\nabla_x\log p_t(X_t).
$$

这里的扩散时间通常从数据指向噪声，生成时反向积分；与本文默认从噪声到数据的方向不同，直接复制公式时需要调整方向。

因此，“扩散只能随机生成，FM 才能确定性生成”不成立。两者可以描述相关的概率路径，并通过不同预测参数化、损失权重和采样方法建立联系。[Flow Matching Guide](https://arxiv.org/abs/2412.06264)

### 5.2 噪声、score 与速度预测

对高斯路径 $X_t=a(t)X_1+b(t)\epsilon$，在内部时刻 $b(t)>0$ 时：

$$
\nabla_x\log p_t(x)
=-\frac{\mathbb E[\epsilon\mid X_t=x]}{b(t)}.
$$

速度场则为：

$$
u_t(x)=a'(t)\mathbb E[X_1\mid X_t=x]
+b'(t)\mathbb E[\epsilon\mid X_t=x].
$$

这表明相关参数化可以在给定路径下互相转换。但端点奇异性、损失权重和数值条件不同，不能仅把网络输出名字改成 velocity 就完成模型转换。

| 维度 | 典型去噪扩散训练 | 典型 FM 训练 |
|---|---|---|
| 常见预测 | 噪声、score、干净样本或扩散 $v$ | 瞬时流速度，也可重参数化 |
| 路径 | 加噪过程定义的概率路径 | 设计者选定的条件概率路径 |
| 训练是否必需积分 ODE | 常见去噪训练不需要 | 常见 CFM 训练不需要 |
| 生成 | 反向 SDE 或 ODE 等 | 以 ODE 积分为典型方式 |
| 少步能力 | 取决于模型、求解器和蒸馏 | 同样取决于具体设计 |

### 5.3 Normalizing Flow 不等于 Flow Matching

NICE、RealNVP、Glow 等通过可逆层和变量替换公式建模密度。CNF 用 ODE 定义连续变换；FM 是训练这类连续生成模型的一种方法。

语音中的 Glow-TTS、WaveGlow，以及 VITS 中的可逆流组件，不能仅因为名字包含 flow 就归为 Flow Matching。光流估计、网络流量建模等领域中的 flow 也与这里的生成方法不同。

### 5.4 Flow Matching 不等于 Schrödinger Bridge

Schrödinger Bridge 研究在给定参考随机过程和端点分布下的随机路径问题。部分 CFM、随机插值和桥方法存在联系，但“源是带噪语音、目标是干净语音”本身不足以证明模型求解了 Schrödinger Bridge。

## 6. 采样、引导与计算成本

### 6.1 Euler 与高阶求解器

Euler 更新为：

$$
x_{k+1}=x_k+h_kv_\theta(x_k,t_k,c),
\qquad h_k=t_{k+1}-t_k.
$$

显式 midpoint 方法先预测中点，再用中点速度更新：

$$
x_{\mathrm{mid}}=x_k+\frac{h_k}{2}v_\theta(x_k,t_k,c),
$$

$$
x_{k+1}=x_k+h_kv_\theta
\left(x_{\mathrm{mid}},t_k+\frac{h_k}{2},c\right).
$$

midpoint 通常每个积分步需要两次函数评估，因此比较“10 步”并不一定比较了相同算力。求解器更高阶，也不能自动修复模型预测本身的错误。

### 6.2 NFE 与积分步数

NFE（Number of Function Evaluations）表示函数评估次数。固定步长 Euler 中，积分步数通常等于 NFE；高阶和自适应求解器则可能不同。

单次网络评估还可能包含多个引导分支、不同 batch 或不同上下文长度。因此，NFE 有助于描述采样预算，但不是端到端耗时的完整替代指标。

### 6.3 CFG

Classifier-Free Guidance 将条件与无条件预测组合：

$$
v_{\mathrm{cfg}}
=v_{\mathrm{uncond}}
+s\big(v_{\mathrm{cond}}-v_{\mathrm{uncond}}\big).
$$

按此约定，$s=1$ 为普通条件预测。某些实现以 $1+\beta$ 表示引导倍率，读取配置前应检查实际公式。

训练一般通过随机丢弃条件，让网络同时学会条件与无条件预测。引导可以提高条件遵循，但强度过高可能损伤自然度和多样性。两个分支会增加计算，即使在同一次调用中拼接运行也不等于免费。

### 6.4 时间表与缓存

均匀时间步不是唯一选择。可以把更多计算分配给对齐、整体结构或细节变化更敏感的区域，但哪个阶段更重要应由模型实验确定。

Sway Sampling 和 EPSS 等方法优化推理时的时间点或步骤选择；它们与通过训练得到少步模型的蒸馏不是同一件事。[EPSS 论文](https://arxiv.org/abs/2505.19931)

跨流时间的状态会变化，不能直接假设所有注意力缓存都像 AR 语言模型一样可复用。静态文本特征、固定条件投影和近似中间激活应分别评估缓存的正确性与误差。

## 7. 从多步 FM 到 MeanFlow

### 7.1 瞬时速度与区间平均速度

标准 FM 预测当前位置的瞬时速度，多步求解器不断查询速度并积分。一步模型则需要学习大时间跨度的位移或等价映射，不能只依靠未经专门训练的局部速度。

MeanFlow 建模沿真实流轨迹的**区间平均速度**，利用它与瞬时速度之间的恒等式构造训练目标。原始 MeanFlow 可从零训练，不要求预训练教师；后续 MeanFlow 蒸馏方法则可能依赖已有教师，两者应分开描述。[MeanFlow 原论文](https://arxiv.org/abs/2505.13447)

### 7.2 平均速度恒等式

为贴近原论文，本小节使用它的常见约定：$z_0$ 为数据、$z_1$ 为噪声，生成方向是从 $t=1$ 到 $t=0$。这与前文 FM 示例的时间方向相反。

沿轨迹，对 $r<t$ 定义：

$$
u(z_t,r,t)
=\frac{z_t-z_r}{t-r}
=\frac1{t-r}\int_r^t v(z_\tau,\tau)d\tau.
$$

固定 $r$ 并沿轨迹对 $t$ 求导，可得：

$$
u(z_t,r,t)
=v(z_t,t)-(t-r)\frac{d}{dt}u(z_t,r,t),
$$

其中总导数为：

$$
\frac{d}{dt}u
=\partial_tu+(\nabla_zu)v.
$$

模型通过时间条件、方向导数等构造训练目标；实现中涉及 Jacobian-vector product 和停止梯度，而不是在训练时先精确积分整条轨迹。[MeanFlow 方法部分](https://arxiv.org/html/2505.13447v1)

若平均速度已准确学到，从噪声端进行一步更新可写为：

$$
\hat z_0=z_1-u_\theta(z_1,0,1).
$$

这与普通 Euler 在起点查询一次瞬时速度不同：网络学习的对象已经改变。实际模型存在近似误差，一步结果不能称为任意 ODE 的精确解。

### 7.3 几类加速路线

| 路线 | 改变什么 | 是否通常需要再训练 |
|---|---|---|
| 时间表与步骤选择 | 何时调用现有速度网络 | 不一定 |
| 更高阶求解 | 如何组合局部速度 | 不一定 |
| Reflow | 训练端点耦合与轨迹 | 是 |
| 教师蒸馏 | 学习教师的少步转移 | 是 |
| 一致性方法 | 学习跨时间的一致映射 | 是，具体训练路线不同 |
| MeanFlow | 学习区间平均速度 | 是，可从零训练或用于后续蒸馏 |

少步生成、输出流式化和低首包延迟分别解决不同问题。即便流生成头只需一次调用，整句文本等待、LM 计算和波形解码仍可能带来延迟。

## 8. 技术发展时间线

| 时期 | 代表方向 | 主要变化 |
|---|---|---|
| 2014—2018 | 可逆归一化流 | 用可逆结构与变量替换公式建模密度 |
| 2018 年起 | Neural ODE、CNF、FFJORD | 连续时间生成，但传统似然训练要处理 ODE 与散度 |
| 2020—2021 | 去噪扩散、Score SDE 与 PF-ODE | 建立高质量生成与连续时间采样视角 |
| 2022 年 9 月 | Rectified Flow、Stochastic Interpolants | 从端点插值或概率流构造可回归目标 |
| 2022 年 10 月 | Flow Matching | 系统提出 FM/CFM 训练框架 |
| 2023 年 | OT-CFM、Voicebox、Matcha-TTS、VoiceFlow | 研究耦合，落地语音填充与快速声学生成 |
| 2024 年 | SD3、E2 TTS、F5-TTS、CosyVoice 2 | 扩展规模，简化对齐，发展分块流生成 |
| 2025 年 | ZipVoice、MeanFlow、MeanFlowSE | 紧凑主干、蒸馏与平均速度一步生成 |
| 2026 年的已核实例 | dots.tts、FlashTTS、AudioCALM | 连续 AR、少步声学头和统一音频生成 |

这不是单线替代史。归一化流、扩散、FM、Rectified Flow 与 Stochastic Interpolants 之间既有联系，也有不同的问题设置与训练构造。

Rectified Flow 和 Stochastic Interpolants 分别在 2022 年 9 月公开，而不是等到 2024 年才出现；后者使用随机插值与概率流建立了相关的连续生成视角。[RF 原论文](https://arxiv.org/abs/2209.03003)、[Stochastic Interpolants 原论文](https://arxiv.org/abs/2209.15571)

## 9. 语音中的表示、对齐与条件

### 9.1 生成表示

| 表示 | 完整输出路径 | 主要权衡 |
|---|---|---|
| mel 频谱 | FM → mel → 声码器 | 管线成熟，但有声码器误差与较长序列 |
| 连续音频 latent | FM → latent → 音频解码器 | 降低帧率，但依赖压缩器的质量与可生成性 |
| 时频或时域表示 | FM → 频谱或波形还原 | 需处理相位、高频与长度成本 |
| 由离散 token 提供条件 | LM → token → FM 声学表示 → 波形 | 内容规划与声学渲染分工，但有多阶段误差 |

“连续生成”不等于“没有编码器”。不使用离散 codec token 的模型仍可能使用连续 AudioVAE；省去 mel 也不代表直接生成原始波形。

### 9.2 文本和语音长度不一致

NAR 系统要先确定输出张量长度，再学习字符或音素与声学时间轴之间的对应。常见方式包括外部音素对齐、MAS、时长预测、filler 补齐、平均上采样和跨注意力。

MAS（Monotonic Alignment Search）可在训练中学习单调对应，因此“无需外部对齐”不等于“没有对齐机制”。F5-TTS 不依赖音素级时长预测器，但推理仍需估计或指定总声学长度。

### 9.3 条件注入

| 条件 | 常见接口 |
|---|---|
| 噪声或流时间 | 时间嵌入、归一化调制 |
| 文本序列 | 编码后拼接、跨注意力或联合注意力 |
| 参考音频 | 已知帧、参考 latent、说话人嵌入 |
| 混合或带噪音频 | 独立观测条件，部分方法也作为初始化 |
| 情感、任务和风格 | 标签、指令、全局向量或时间变化条件 |

条件设计决定模型如何使用信息，不保证模型必然学会正确的音色、情感和内容解耦。

## 10. 语音系统的结构分类

### 10.1 整段 NAR 声学生成

```text
文本 + 参考音频 + 对齐或总长度
                  ↓
        FM 生成整段 mel / latent
                  ↓
             解码得到波形
```

代表包括 Voicebox、Matcha-TTS、E2 TTS、F5-TTS 和 ZipVoice。它们采用的对齐、骨干和条件不同；整段非因果生成通常需要完整上下文，不能仅因 NAR 就视为实时流式模型。

### 10.2 AR 语言模型与 FM 渲染器

```text
文本 → AR 语言模型 → 语音 token 或隐藏表示
                              ↓
                         条件 FM 渲染
                              ↓
                         mel / latent → 波形
```

CosyVoice 系列与 Qwen2.5-Omni 提供了这种组合的实例。语言模型承担顺序规划，FM 还原连续声学细节。能否低延迟输出，取决于两侧分块、前瞻和解码器缓冲。

### 10.3 连续自回归与 FM 头

系统沿音频块自回归推进，但每一步不是对离散码本分类，而是根据历史上下文，通过局部 FM 生成下一块连续 latent。dots.tts 与 AudioCALM 是已核实的近期实例。

这同时包含两个时间轴：外层是音频播放时间，内层是每块从噪声到目标的流时间。减少内层 NFE 不会消除外层自回归依赖。

### 10.4 语音增强、分离与目标说话人提取

这些任务需要保留观测中的真实内容。带噪语音、混合语音、参考说话人或视频可以成为条件；生成状态则可能从高斯噪声或经过初步估计的表示开始，必须按具体方法检查。

基础资料把“无离散 tokenizer”单列为一种结构，但它与 AR/NAR 分类交叉。本文把它作为**表示选择**，避免误以为连续表示模型天然非自回归或天然支持流式。

## 11. 代表性 TTS 系统

### 11.1 Voicebox

Voicebox 用非自回归 Flow Matching 训练文本引导的语音填充：给定文本与部分音频，补全缺失区域。它展示了零样本 TTS、内容编辑、去噪和风格转换等任务，是大规模 FM 语音生成的早期代表。[Voicebox 论文](https://arxiv.org/abs/2306.15687)

填充范式支持使用左右两侧声学上下文，但并不等于无需文本对齐或时长处理，也不意味着自动具备低首包流式能力。

### 11.2 Matcha-TTS

Matcha-TTS 使用 OT-CFM 训练 ODE 声学解码器，并结合文本编码、单调对齐和时长相关建模。它无需预先提供外部对齐，却仍有内部对齐机制，不能归为“完全不做对齐”。[Matcha-TTS 论文](https://arxiv.org/abs/2309.03199)

### 11.3 VoiceFlow

VoiceFlow 于 2023 年 9 月提交，发表于 ICASSP 2024。它把 mel 生成写成文本条件 ODE，并通过 Rectified Flow 改善采样轨迹，在单说话人与多说话人设置中研究质量和少步效率。[VoiceFlow 论文](https://arxiv.org/abs/2309.05027)

论文中的多说话人实验不能直接推导为任意未见说话人的零样本克隆。CPU RTF、采样步数和声音适应能力需要分别查验实验设置。

### 11.4 E2 TTS 与 F5-TTS

E2 TTS 用字符加 filler token 补齐声学长度，让网络学习文本和声学对应。其原始骨干是带 U-Net 式长跳连的 Transformer，不能直接等同于原始平坦 DiT。[E2 TTS 论文](https://arxiv.org/html/2406.18009v1)

F5-TTS 在此基础上采用 DiT 骨干和 ConvNeXt 文本细化，并提出无需重训的 Sway Sampling。它展示了 FM 目标、条件入口和推理时间表可以共同影响对齐、自然度和效率。[F5-TTS 论文](https://arxiv.org/abs/2410.06885)

### 11.5 CosyVoice 2 与 CosyVoice 3

CosyVoice 2 将语音 token 语言模型与 chunk-aware 因果流匹配结合，以不同注意力掩码兼容离线和流式需求。其声学生成骨干是因果 Conv-Transformer U-Net。[CosyVoice 2 论文](https://arxiv.org/html/2412.10117v3)

CosyVoice 3 改用 DiT 作为 CFM 骨干，并扩大数据和模型规模。两者都采用 FM，但骨干不同，说明 FM 与 DiT 并非一一对应。[CosyVoice 3 论文](https://arxiv.org/html/2505.17589v2)

### 11.6 Qwen2.5-Omni

Qwen2.5-Omni 在 Thinker–Talker 结构中使用 Flow-Matching DiT 把语音 code 转成 mel，再由修改后的 BigVGAN 还原波形。滑动窗口的四块感受野包括**回看两块、当前一块、前瞻一块**；前瞻提高上下文信息，也增加等待。[Qwen2.5-Omni 技术报告](https://arxiv.org/html/2503.20215)

### 11.7 ZipVoice

ZipVoice 采用 Zipformer 速度场估计器、文本编码与平均上采样初始化对齐，并通过流蒸馏减少步骤和 CFG 开销。它说明改善 FM 效率不仅可以减少 NFE，还可以优化单次调用的网络结构。[ZipVoice 论文](https://arxiv.org/abs/2506.13053)

### 11.8 dots.tts、FlashTTS 与 AudioCALM

| 模型 | 已核实的主要方向 | 不宜过度推断的结论 |
|---|---|---|
| dots.tts | 连续 AR、全历史条件 FM 头、自校正后训练与 CFG-aware MeanFlow 蒸馏 | 单个模型的低延迟数字不能当作行业统一水平 |
| FlashTTS | 流式输入、MTP 与两次函数评估的 X-pred Mean Flow 声学解码 | 两次声学评估不等于整套系统只有两次网络调用 |
| AudioCALM | 薄 FM 头替代离散分类输出，统一语音、声音和音乐的连续 AR | 统一生成不表示所有任务拥有相同对齐和评价协议 |

三份报告均可回溯一手 arXiv，基础资料中的“编号待核”可相应更新；不过预印本结果仍需按版本与配置阅读。[dots.tts](https://arxiv.org/abs/2606.07080)、[FlashTTS](https://arxiv.org/abs/2606.09141)、[AudioCALM](https://arxiv.org/abs/2606.23080)

## 12. 语音增强、分离与统一音频生成

### 12.1 FlowSE

FlowSE 使用带噪 mel 和可选转写作为条件，由 DiT 类流模型产生增强后的 mel，再交给声码器还原波形。具体方法部分描述了高斯噪声初始化、插值训练和 ODE 求解，不能根据摘要中的“single pass”一词就断言严格为 1-NFE。[FlowSE 论文方法部分](https://arxiv.org/html/2505.19476v2)

这也说明“从带噪语音恢复干净语音”是任务描述，**带噪语音是条件还是 ODE 初始状态**则是模型实现问题，二者需要分开。

### 12.2 MeanFlowSE

MeanFlowSE 于 2025 年 9 月提交，使用平均速度场进行一步 latent 生成式增强，并以 SSL 表示提供声学与语义条件。其论文主要报告 Interspeech 2020 DNS Challenge 盲测集和模拟集结果，不能沿用基础资料中未经对应验证的 VoiceBank-DEMAND 与 SI-SNR 数字。[MeanFlowSE 论文](https://arxiv.org/abs/2509.23299)

一步流生成模块之外，SSL 编码和 VAE 解码仍有计算成本。“1-NFE”不等于整个增强系统只执行一个网络。

### 12.3 FlowTSE

FlowTSE 接收目标说话人的注册语音与混合语音，以 mel 表示作为条件，生成目标说话人的干净语音。针对相位重建，论文提出使用混合信号复数 STFT 作为条件的声码器。[FlowTSE 论文](https://arxiv.org/abs/2505.14465)

它关注的是目标说话人提取，不是只生成相似音色。评价必须同时检查干扰抑制、内容保留、音色与相位相关的重建质量。

### 12.4 FlowSep 与 FlowAVSE

FlowSep 研究语言查询音源分离，在潜空间采用 Rectified Flow，按文本描述提取目标声音。FlowAVSE 则融合音频和视觉条件进行语音增强，其首次 arXiv 提交为 **2024 年 6 月**，不是基础资料中的 2023 年。[FlowSep](https://arxiv.org/abs/2409.07614)、[FlowAVSE](https://arxiv.org/abs/2406.09286)

### 12.5 Audiobox 与统一音频生成

Audiobox 通过自然语言提示等条件研究统一音频生成，覆盖语音与其他声音；AudioCALM 则提供连续自回归的后续实例。它们支持“共享连续生成框架”的研究方向，但具体表示、条件和任务能力仍需分别核对。[Audiobox 论文](https://arxiv.org/abs/2312.15821)、[AudioCALM 论文](https://arxiv.org/abs/2606.23080)

## 13. 一个最小实现思路

### 13.1 线性 CFM 训练示意

下面是教学代码，省略数据处理、网络定义和变长掩码，不是可直接部署的 TTS：

```python
# x1: 目标连续表示，形状可为 [B, T, D]
# cond: 推理时同样可以获得的文本、参考音频等条件
x0 = torch.randn_like(x1)
t = torch.rand(x1.shape[0], 1, 1, device=x1.device)

xt = (1.0 - t) * x0 + t * x1
target_velocity = x1 - x0
pred_velocity = model(xt, t.flatten(), cond)

loss = (pred_velocity - target_velocity).square().mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这里采用独立高斯源和目标配对，不包含 minibatch OT、CFG 条件丢弃、语音填充、蒸馏或额外感知损失。变长训练必须屏蔽 padding，不能把补齐区域当作真实声学帧计入损失。

### 13.2 Euler 采样示意

```python
@torch.no_grad()
def sample(model, shape, cond, steps, device):
    assert steps > 0
    model.eval()
    x = torch.randn(shape, device=device)
    grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t = grid[i].expand(shape[0])
        dt = grid[i + 1] - grid[i]
        x = x + dt * model(x, t, cond)
    return x
```

采样长度由 `shape` 预先决定。真实 TTS 还需要总时长或块长度、条件构造、提示区域处理、声码器和边界拼接。此例只表达源到目标方向，不能直接用于前文采用反向时间约定的 MeanFlow 模型。

### 13.3 语音填充与条件泄漏

训练填充模型时，从真实音频保留部分上下文，遮盖需要生成的区域。模型输入应明确区分带噪生成状态、已知参考音频和生成掩码。

条件分支不能泄露目标区域的干净音频；训练中可见的参考上下文也应尽量匹配推理形态。否则训练损失可能很好，真实生成却明显退化。

### 13.4 工程核对项

1. 训练与推理采用相同的时间方向、路径系数和输出参数化。
2. mel 或 latent 的归一化、采样率和帧率一致。
3. 条件输入在推理时真实可得，没有目标信息泄漏。
4. padding、参考区域与目标区域的掩码语义明确。
5. 记录求解器、时间网格、NFE 和引导分支成本。
6. 将声码器、VAE、LM、缓冲和拼接计入端到端耗时。

官方工具可参考 [Meta Flow Matching 文档](https://facebookresearch.github.io/flow_matching/) 与 [代码仓库](https://github.com/facebookresearch/flow_matching)，耦合变体可参考 [TorchCFM](https://github.com/atong01/conditional-flow-matching)。

## 14. 流式化与效率

### 14.1 首包延迟来自整条链路

首包延迟可能由文本缓冲、语言模型预填充、语音 token 积累、FM 多步评估、声码器前瞻和输出缓冲共同决定。这些阶段可以部分重叠，不能只把各模块平均时间简单相加。

把一整段生成完成后切成小块发送，不等于模型已经支持低首包流式生成。

### 14.2 分块与因果约束

块因果或滑动窗口限制声学模型能看到的未来上下文，使其能在完整句子结束前输出。因果 AudioVAE 或声码器有帮助，但如果上游 FM 仍需完整句子，也不能单靠因果解码器实现端到端流式。

较小块可减少等待，较大块可提供更多上下文；还需处理块边界的音高、能量、音色和相位连续性。

### 14.3 网络与表示优化

降低声学帧率、缩短条件序列、使用更紧凑的速度估计器，以及混合精度、融合算子、注意力优化，都可能减少开销。收益依赖硬件、长度、batch 和模型结构，不应给出没有配置支持的固定倍数。

更高压缩率也可能损伤重建或增加生成难度；连续表示不存在一个适用于所有模型的固定显存倍率。

### 14.4 后训练

语音 FM 可通过感知损失、教师蒸馏或基于奖励的方法改善发音、音色和表达。确定性 ODE 的策略梯度训练需要具体的概率建模或随机轨迹设计，但并非所有后训练都必须使用策略梯度。

FM 的初始噪声本来就提供随机性；“为了后训练引入额外噪声”通常服务于可计算策略概率等方法需求，不代表原模型完全没有采样随机性。

## 15. 性能评价

### 15.1 不同指标回答不同问题

| 维度 | 常见指标 | 必须说明的条件 |
|---|---|---|
| 内容正确性 | WER、CER、漏字重复率 | ASR 模型、语言、文本规范化 |
| 音色一致性 | SIM、说话人嵌入相似度 | 验证模型、参考音频与切分协议 |
| 主观听感 | MOS、CMOS、SMOS、偏好测试 | 听测题目、听者数量与置信区间 |
| 自动音质 | DNSMOS、UTMOS 等 | 适用分布及与真实听感的差异 |
| 增强与分离 | SI-SDR、SI-SNR、PESQ、STOI 等 | 对齐、采样率、相位及参考信号 |
| 性能 | NFE、RTF、首包延迟、显存、吞吐 | 硬件、精度、时长、batch 和完整链路 |
| 稳定性 | 长句失败率、漂移、跨语言一致性 | 测试长度和困难样本分布 |

MOS 是 **Mean Opinion Score，平均意见分**，不是“音乐感知相似度”。预测 MOS 与真实人工 MOS 也不是同一个测量。

### 15.2 RTF 不等于首包延迟

$$
\mathrm{RTF}
=\frac{\text{生成所需时间}}{\text{输出音频时长}}.
$$

例如生成 10 秒语音耗时 1.5 秒，RTF 为 0.15。但如果必须等整段生成后才开始播放，用户仍要等待约 1.5 秒，因此低 RTF 不自动等于低首包。

### 15.3 同一论文中的采样预算例子

F5-TTS 在 LibriSpeech-PC test-clean 上报告以下结果；其 RTF 使用 RTX 3090 生成 10 秒语音测量：

| 配置 | WER | SIM-o | RTF |
|---|---:|---:|---:|
| 16 NFE | 2.53% | 0.66 | 0.15 |
| 32 NFE | 2.42% | 0.66 | 0.31 |

该表适合说明同一协议下计算预算与效果的权衡，不宜同其他论文、版本或硬件的结果直接排名。[F5-TTS 论文表 1](https://arxiv.org/html/2410.06885v3)

### 15.4 可复现比较需要控制什么

比较 FM 与扩散，至少需要尽量匹配数据、输出表示、骨干规模、训练预算、引导、求解器与端到端解码范围。否则观察到的速度或音质差异，可能来自主干、声码器或数据规模，而非训练目标本身。

## 16. 局限与发展方向

### 16.1 速度场误差与数值误差

误差既来自模型没学准，也来自有限步积分。增加步数主要改善后者，不能保证修复错误发音、条件忽略或目标分布欠拟合。一步模型则把大跨度映射的难度转移到训练中。

### 16.2 长句、对齐与误差累积

整段生成面临长度预算与隐式对齐问题，分块 AR 则有跨块漂移和历史误差传播。长文本稳定性需要独立测量，不能由几秒样例的听感推断。

### 16.3 感知质量与内容保真

在增强、分离和提取任务中，生成更自然的声音不一定更忠实于原内容。模型可能补出合理但错误的发音，或者保留音色却损失弱语音。应联合评估内容、干扰抑制和声学质量。

### 16.4 耦合与几何设计

独立配对简单但可能产生复杂边际流；OT 配对更有结构，却增加成本并受 batch 和距离度量影响。声学空间的欧氏距离也未必完全对应感知距离。

### 16.5 统一模型与多条件控制

语音转写需要紧密的时间对齐，音乐或环境声描述往往只有全局语义。共享速度场需要解决这种条件不对称，不能只把不同数据拼接训练就假设任务能互相促进。

### 16.6 几个持续发展的方向

- 更适合生成的音频 latent，以及表示质量与生成难度的联合优化。
- 从多步局部速度转向平均速度、流映射和少步蒸馏。
- 连续自回归、分块流式和对齐机制的协同设计。
- 保留观测内容的生成式增强与分离。
- 在共享骨干中统一语音、音乐与环境声音的条件建模。

这些方向来自代表性研究的归纳，不表示 FM 已在所有语音应用中优于其他方法，也不证明闭源产品采用某种未公开的声学结构。

## 17. 其他扩展

**流形 FM** 将欧氏空间的路径推广到非欧几何，用于具有旋转、球面或其他结构约束的数据。相关论文 *Flow Matching on General Geometries* 发表于 ICLR 2024，基础资料中的 ICML 归属需要修正。[流形 FM 论文](https://arxiv.org/abs/2302.03660)

**离散 FM** 为文本等离散状态构造概率路径及转移动态，不能直接对 token ID 执行欧氏插值。连续速度回归与离散状态转移的数学对象不同。[Discrete Flow Matching](https://arxiv.org/abs/2407.15595)

更广的 FM 方法还研究图像、视频、分子、生物结构及其他分布变换。本文重点是连续 FM 与语音，不将这些扩展全部展开为领域综述。

## 附录 A 基础资料的主要修订

| 基础资料中的问题 | 本文采用的口径 |
|---|---|
| CFM 被解释为后来才出现的文本条件生成 | 区分路径构造条件与任务生成条件，CFM 已在原始 FM 中出现 |
| Rectified Flow 被放在 2024 年起源 | 首次提交于 2022 年 9 月 |
| 直线插值被等同于直线推理轨迹 | 区分逐样本监督路径与边际速度场的 ODE 轨迹 |
| FM 完全无需日程设计 | 仍需选择概率路径、时间采样和求解时间表 |
| 扩散训练必须模拟动态，FM 才无需模拟 | 典型扩散去噪训练也不必积分生成过程 |
| 确定性被解释为没有生成随机性 | 固定初值后确定，不同初始噪声仍产生多样输出 |
| MeanFlow 只是普通 ODE 改成一步求解 | 模型学习对象变成区间平均速度，需相应训练 |
| E2 TTS 直接写成原始 DiT | 原始实现为带 U-Net 式长跳连的 Transformer |
| Qwen2.5-Omni 前后块方向写反 | 回看两块、当前一块、前瞻一块 |
| 因果 VAE 被视为充分的流式条件 | 上游 FM、LM、文本缓冲也必须满足流式约束 |
| FlowSE 被一概写成严格单次评估 | 根据方法中实际 ODE 求解和配置判断 NFE |
| MeanFlowSE 的数据集和指标混用 | 按其 DNS Challenge 实验，不沿用未核实数字 |
| FlowAVSE 被写为 2023 年工作 | 首次 arXiv 提交为 2024 年 6 月 |
| “无 tokenizer”与 NAR 被等同 | 离散或连续表示与 AR/NAR 是独立设计维度 |
| MOS 被展开成音乐感知相似度 | 正确为平均意见分 Mean Opinion Score |
| 流形 FM 会议归属写为 ICML 2024 | 原论文记录为 ICLR 2024 |

Word 基础资料中的部分公式在普通文本提取后为空，本篇以统一记号重新写出核心推导。其参考文献也混入了光流、图像校正和其他无关主题，本文仅保留与正文相关、可回溯的资料。

## 附录 B 术语速查

| 术语 | 含义 |
|---|---|
| FM | Flow Matching，流匹配 |
| CFM | Conditional Flow Matching，条件流匹配 |
| CNF | Continuous Normalizing Flow，连续归一化流 |
| RF / Reflow | Rectified Flow 及重新构造端点耦合的训练过程 |
| OT | Optimal Transport，最优传输 |
| ODE / SDE | 常微分方程 / 随机微分方程 |
| PF-ODE | Probability-Flow ODE，概率流 ODE |
| CFG | Classifier-Free Guidance，无分类器引导 |
| DiT | Diffusion Transformer，扩散 Transformer 骨干 |
| NFE | 函数评估次数 |
| RTF | 实时因子，即耗时与输出时长之比 |
| AR / NAR | 自回归 / 非自回归 |
| SSL | 自监督学习 |
| VAE | 变分自编码器 |
| TSE | 目标说话人提取 |

## 附录 C 主要参考资料

### 理论与工具

1. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。
2. [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)，附 [官方库](https://github.com/facebookresearch/flow_matching)。
3. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)。
4. [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571)。
5. [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482)。
6. [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)。
7. [Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660)。
8. [Discrete Flow Matching](https://arxiv.org/abs/2407.15595)。

### 语音生成与加速

9. [Voicebox](https://arxiv.org/abs/2306.15687)。
10. [Matcha-TTS](https://arxiv.org/abs/2309.03199)。
11. [VoiceFlow](https://arxiv.org/abs/2309.05027)。
12. [E2 TTS](https://arxiv.org/abs/2406.18009)。
13. [F5-TTS](https://arxiv.org/abs/2410.06885)。
14. [CosyVoice 2](https://arxiv.org/abs/2412.10117) 与 [CosyVoice 3](https://arxiv.org/abs/2505.17589)。
15. [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)。
16. [ZipVoice](https://arxiv.org/abs/2506.13053)。
17. [Accelerating Flow-Matching-Based Text-to-Speech via Empirically Pruned Step Sampling](https://arxiv.org/abs/2505.19931)。
18. [dots.tts Technical Report](https://arxiv.org/abs/2606.07080)。
19. [FlashTTS](https://arxiv.org/abs/2606.09141)。
20. [AudioCALM](https://arxiv.org/abs/2606.23080)。

### 增强、分离与统一音频

21. [FlowSE](https://arxiv.org/abs/2505.19476)。
22. [MeanFlowSE](https://arxiv.org/abs/2509.23299)。
23. [FlowTSE](https://arxiv.org/abs/2505.14465)。
24. [FlowSep](https://arxiv.org/abs/2409.07614)。
25. [FlowAVSE](https://arxiv.org/abs/2406.09286)。
26. [Audiobox](https://arxiv.org/abs/2312.15821)。
