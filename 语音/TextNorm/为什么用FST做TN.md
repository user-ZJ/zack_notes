# 为什么用 FST 做文本正则化（TN）

> 本文解释 WeTextProcessing 为什么用 **有限状态转换器（FST, Finite-State Transducer）** 来做 TN/ITN，而不是用「普通 JSON 规则表 + 代码 if/else」的方案。会先讲清楚「JSON 方案哪里不好」，再讲「FST 好在哪」，最后结合本仓库给出可运行的对照示例。

---

## 0. 一句话结论

TN 的本质是**字符串到字符串的映射**，而且这些映射之间会**互相组合、互相竞争、依赖上下文**。

- 「普通 JSON + 手写代码」适合**孤立的、静态的替换**，一旦规则要**组合/嵌套/带上下文/比优先级**，代码会迅速膨胀成难以维护的意大利面。
- FST 天生就是「带权重的字符串映射 + 可组合的代数系统」，把「规则怎么写」和「规则怎么高效执行」彻底分离，正好命中 TN 的需求。

---

## 1. 先看 TN 到底难在哪

TN 不是简单查表替换。看几个真实难点（都来自本仓库 `tn/README.md` 的示例）：

| 输入 | 输出 | 难点 |
|------|------|------|
| `共465篇` | `共四百六十五篇` | 数字要按**进位规则**展开（百/十/个），不是逐字替换 |
| `2002/01/28` | `二零零二年一月二十八日` | 需要**结构化解析**（年/月/日）再重排 |
| `8:00 a.m.准时开会` | `早上八点准时开会` | **上下文相关**：`a.m.` 决定了「早上」 |
| `价格是$13.5` | `价格是十三点五美元` | 数字规则**嵌套**在货币规则里 |
| `比分定格在78:96` | `比分定格在七十八比九十六` | 同样是 `:`，在时间里读「点」，在比分里读「比」——**歧义，要比优先级** |

核心特征：
1. **组合**：金额里套着数字，数字里套着「个/十/百」。
2. **竞争与歧义**：同一段文本，多条规则都能匹配，要选一个「最合理」的。
3. **上下文相关**：同一个符号在不同语境读法不同。
4. **顺序无关的可维护性**：加一条新规则，不该把已有规则全改一遍。

---

## 2. 用「普通 JSON」方案会怎样

假设我们用一个 JSON 规则表 + 一段处理代码：

```json
{
  "rules": [
    { "pattern": "100", "replace": "一百" },
    { "pattern": "465", "replace": "四百六十五" },
    { "pattern": "%",   "replace": "百分之", "position": "prefix" }
  ]
}
```

```python
def normalize(text, rules):
    for r in rules:
        text = text.replace(r["pattern"], r["replace"])
    return text
```

看起来能用，但很快就崩：

### 问题 1：数字是无穷的，没法枚举

`100 → 一百`、`465 → 四百六十五`…… 数字有无限种组合，你不可能把每个数字都列进 JSON。于是你被迫在**代码里**写一个「数字转中文」的函数。规则一旦需要「计算」，JSON 就退化成了配置壳子，真正的逻辑全在代码里 —— **规则和代码耦合了**。

### 问题 2：上下文相关要写一堆 if

`:` 在 `5:02`（时间）里读「点」，在 `78:96`（比分）里读「比」。用 replace 根本区分不了，只能：

```python
if is_time_context(text, pos):
    ...
elif is_score_context(text, pos):
    ...
```

每多一种上下文，就多一层判断。`a.m.`/`km/h`/`°C`/货币符号…… 判断分支爆炸。

### 问题 3：规则顺序 = 隐藏的 bug

`replace` 是顺序敏感的。先替换 `1` → `一`，再想匹配 `100` 就废了。你得小心翼翼地排列 JSON 里规则的先后，**顺序本身变成了不可见的逻辑**。加一条新规则可能悄悄破坏另一条。

### 问题 4：无法表达「优先级/最优选择」

`2002` 既能被「数字规则」读成「两千零二」，也能被「日期规则」读成「二零零二年」。你想让日期优先。用 JSON+replace，你只能靠调顺序硬凑，没有一个统一的「谁更优」机制。

### 问题 5：组合爆炸，无法复用

金额 `$13.5` = 货币符号 + 数字 + 小数。数字规则你想在「金额、度量、分数、百分比」里都复用。JSON+replace 没有「子规则」概念，你要么到处复制粘贴，要么在代码里手动拼字符串，越写越乱。

### 问题 6：难以反向（ITN）

你写了一套 TN 的 replace 规则，现在要做 ITN（反过来）。JSON 方案得**从头再写一套**。而 TN 和 ITN 在数学上只是「方向相反」而已。

**小结**：JSON+replace 适合「有限、孤立、无上下文」的替换（比如 emoji 表、固定缩写表）。一旦规则要计算、组合、看上下文、比优先级，它就从「配置」退化成「一堆散落在代码里的特判」。

---

## 3. FST 是什么，为什么正好合适

### 3.1 直观理解

**转换器（Transducer）** = 一个「读入一个字符串、吐出另一个字符串」的状态机。每条边上写着 `输入:输出`。

把「个位数字」映射成中文，可以画成这样一个极简 FST：

```
        1:一
   ┌──────────────►
 (start)  2:二   (accept)
   ├──────────────►
   │      3:三
   └──────────────►
```

读入 `2`，走 `2:二` 这条边，输出 `二`。就这么简单。

**加权（Weighted）** = 每条边/每条路径还带一个权重（成本）。当多条路径都能匹配同一输入时，选**总权重最小**的那条 —— 这就天然解决了「歧义/优先级」。

### 3.2 FST 的四种「代数运算」——TN 需求的完美映射

FST 的威力在于它是一个**可组合的代数系统**。TN 的每个难点，都对应一种 FST 运算：

| TN 需求 | FST 运算 | 在 Pynini 里 |
|---------|----------|-------------|
| 拼接（年+月+日） | 连接（concatenation） | `a + b` |
| 多规则竞争（数字\|日期\|金额） | 并集（union） | `a \| b` |
| 重复（一句话多个数字） | 闭包（closure） | `a.star` / `a.plus` |
| 规则串联（先转简体再打标签） | 复合（composition） | `a @ b` |
| 优先级/歧义消解 | 加权 + 最短路径 | `add_weight` + `shortestpath` |
| 上下文相关替换 | 上下文重写 | `cdrewrite(fst, l, r, sigma)` |
| ITN（反向） | 求逆 | `invert(fst)` |

**关键点**：这些运算的结果**还是一个 FST**，可以继续参与运算。于是你能像搭积木一样，把「个位数字」拼成「两位数」，拼成「金额」，拼成「整句规则」——每一层都可复用、可独立测试、可独立加权。

### 3.3 它怎么一次性解决了 JSON 方案的 6 个问题

- **数字无穷** → 用「个/十/百」的 FST 递归拼接表达进位，一个有限的 FST 就能匹配无限个数字。
- **上下文相关** → `cdrewrite` 内建左/右上下文，不用写 if。
- **顺序敏感** → FST 编译后是一个整体图，匹配是「全局最优路径」，不依赖规则书写顺序。
- **优先级** → `add_weight` 给每类规则设权重，`shortestpath` 自动选最优，机制统一。
- **组合复用** → 子 FST 直接被上层引用（`Cardinal` 被 `Money`/`Measure` 复用）。
- **反向 ITN** → `invert` 或反向写 `cross`，同一套框架搞定。

---

## 4. 结合本仓库看真实实现

WeTextProcessing 用 [Pynini](https://github.com/kylebgorman/pynini)（OpenFst 的 Python 封装）来「描述」FST 规则，编译成 `.fst` 文件后由引擎执行。

### 4.1 数字进位：一个有限 FST 匹配无限数字

看 `tn/chinese/rules/cardinal.py`，它没有枚举任何具体数字，而是用 FST 拼接表达进位规则：

```python
        # 11 => 十一, 10 => 十
        ten = teen + insert("十") + (digit | rmzero)
        # 21 => 二十一, 30 => 三十
        tens = digit + insert("十") + (digit | rmzero)
        # 111 => 一百一十一, 101 => 一百零一, 100 => 一百
        hundred = digit + insert("百") + (tens | (zero + digit) | rmzero**2)
        ...
        number = digits | ten | hundred | thousand | ten_thousand | hundred_million
        number = sign.ques + number + (dot + digits.plus).ques
```

- `insert("十")`：在合适位置插入「十」。
- `digit | rmzero`：个位是数字，或者是 0 就删掉（`30 → 三十` 而不是 `三十零`）。
- `hundred` 引用 `tens`，`thousand` 引用 `hundred`…… **层层复用**。

这就是 JSON 方案做不到的：**用有限规则表达无限映射**。

### 4.2 优先级：加权解决歧义

看 `tn/chinese/normalizer.py` 如何用 `RuleSpec(规则, 权重)` 声明清单，再由 `tagger_union` 做「带权并集」：

```python
        rules = (
            RuleSpec(date, 1.02),
            RuleSpec(whitelist, 1.03),
            RuleSpec(sport, 1.04),
            RuleSpec(fraction, 1.05),
            RuleSpec(measure, 1.05),
            RuleSpec(money, 1.05),
            RuleSpec(time, 1.05),
            RuleSpec(cardinal, 1.06),
            RuleSpec(math, 90),
            RuleSpec(char, 100),
            RuleSpec(range_rule),
        )
        tagger = self.tagger_union(rules).star
```

`tagger_union` 内部对每个 `RuleSpec` 做 `add_weight(rule.tagger, 权重)` 再 `union`，所以本质仍是「带权并集」：

- 所有规则一起竞争匹配同一段输入。
- 权重 `date`(1.02) < `cardinal`(1.06)：**越小越优先**，所以 `2002` 优先当日期而非普通数字。
- `char`(100) 是兜底：谁都不匹配才逐字保留。
- `.star`：整句反复应用。

对比 JSON 方案里「靠调 replace 顺序硬凑优先级」，这里是**统一、可解释、可微调**的权重机制。

### 4.3 上下文相关：cdrewrite

`tn/processor.py` 把上下文重写封装成 `build_rule`：

```python
    def build_rule(self, fst, l="", r=""):
        rule = cdrewrite(fst, l, r, self.VSIGMA)
        return rule
```

`cdrewrite(fst, l, r, sigma)` 表示「只有当左边是 `l`、右边是 `r` 时才应用 `fst`」。例如 `cardinal.py` 里：

```python
        number @= self.build_rule(cross("二百", "两百") | cross("二千", "两千") | cross("二万", "两万") | cross("二亿", "两亿"),
                                  "[BOS]").optimize()
```

只有在句首（`[BOS]`）才把「二百」改成「两百」。JSON+replace 表达这种「仅在某上下文生效」要写一堆特判，这里一行搞定。

### 4.4 组合复用：数字被金额/度量复用

`normalizer.py` 里 `Money`/`Measure`/`Fraction` 都接收同一个 `cardinal` 实例：

```python
        cardinal = Cardinal()
        ...
        fraction = Fraction(cardinal=cardinal)
        measure = Measure(cardinal=cardinal)
        money = Money(cardinal=cardinal)
```

数字规则写一次，到处复用。JSON 方案只能复制粘贴或在代码里手拼。

### 4.5 反向 ITN：同一套框架

TN 在 `tn/`，ITN 在 `itn/`，共用同一个引擎 `Processor`。ITN 很多规则就是 TN 规则的方向反转（`invert` 或反向 `cross`）。JSON 方案得从头重写一套。

---

## 5. 端到端对照示例

### 5.1 JSON+replace 方案（会失败的写法）

```python
rules = [
    {"pattern": "465", "replace": "四百六十五"},
    {"pattern": ":",   "replace": "点"},   # 想处理时间 5:02
]

def normalize(text):
    for r in rules:
        text = text.replace(r["pattern"], r["replace"])
    return text

normalize("共465篇")        # "共四百六十五篇"  ✅ 但 466、467… 全要手动加
normalize("比分定格在78:96") # "比分定格在78点96" ❌ 应该是「比」，无法区分上下文
normalize("我是5:02开始的")  # "我是5点02开始的"  ❌ 02 没转成中文，也没有「分」
```

每个 case 都得打补丁，越补越乱。

### 5.2 FST 方案（本仓库）

```bash
# --overwrite_cache 首次会现场编译规则成 .fst
python -m tn --text "共465篇"        # => 共四百六十五篇
python -m tn --text "比分定格在78:96" # => 比分定格在七十八比九十六
python -m tn --text "我是5:02开始的"  # => 我是五点零二分开始的
```

同一套规则，数字自动进位、比分和时间的 `:` 自动按上下文和权重区分，无需任何 case 补丁。

你还能打印中间串，看到 tagger 如何结构化地打标签：

```bash
python -m tn --text "2002/01/28"
# tag 输出（中间串，结构化；注意 value 存的是原始书面值，尚未转汉字）:
#   date { year: "2002" month: "01" day: "28" }
# verbalize/normalize 输出（口语化转换在 verbalizer 完成）:
#   二零零二年一月二十八日
```

> 当前实现里，tagger 用 `Processor.tag_field()` 对规则图做 `project("input")`，只保留**书面侧**，所以中间串存的是原始值（`2002`/`01`/`28`）；真正的「2002 → 二零零二年」发生在 verbalizer。这不影响本文「为什么用 FST」的结论，只是说明一下中间串的样子。

---

## 6. 什么时候「不」需要 FST

FST 不是银弹，它有成本：学习曲线陡（要懂 Pynini/OpenFst 的代数），编译 `.fst` 慢，调试不如普通代码直观。

如果你的需求是：
- 少量、固定、无上下文的替换（如敏感词表、固定缩写表）；
- 规则之间不组合、不嵌套、不比优先级；

那 JSON+replace（甚至一个 `dict`）完全够用，上 FST 是杀鸡用牛刀。

**判断标准**：一旦你的规则开始出现「数字要计算」「同一符号多种读法」「规则要嵌套复用」「要区分优先级」——就该上 FST 了。TN/ITN 恰好全中，所以 WeTextProcessing、NeMo、Kaldi 等主流方案都用 FST。

---

## 7. 一句话总结

| 维度 | JSON + replace | FST |
|------|----------------|-----|
| 无限映射（数字进位） | ❌ 无法枚举 | ✅ 有限规则表达无限 |
| 上下文相关 | ❌ 靠 if 爆炸 | ✅ `cdrewrite` 内建 |
| 优先级/歧义 | ❌ 靠调顺序硬凑 | ✅ 加权 + 最短路径 |
| 组合复用 | ❌ 复制粘贴 | ✅ 子 FST 引用 |
| 反向（ITN） | ❌ 重写一套 | ✅ `invert` |
| 执行效率 | 逐条 replace，慢 | ✅ 编译成图，一次匹配 |
| 简单孤立替换 | ✅ 够用且直观 | ⚠️ 杀鸡用牛刀 |

TN 的本质是「可组合、带上下文、有歧义的字符串映射」，而 FST 正是为这类问题量身定制的数学工具——这就是 WeTextProcessing 选择 FST 的根本原因。
