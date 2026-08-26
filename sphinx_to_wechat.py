#!/usr/bin/env python3
"""
后处理脚本：修复 Sphinx 生成的 HTML 中代码块换行符和公式，使其兼容微信公众号编辑器。

原理：
  1. 代码块：浏览器渲染 <pre> 时靠 CSS white-space:pre 保留换行，但复制到微信公众号
     编辑器后 <pre> 和 white-space 会被丢掉，整段挤成一行。解决方法是把 <pre>
     改成微信可识别的 <section>，每一行单独用 <p> 包住。
  2. 公式：保留原始 LaTeX，由 MathJax 渲染为自包含 SVG，再在浏览器中栅格化为
     PNG 图片。复制到微信公众号编辑器时，复杂公式的排版不会丢失。

用法：
    make html && python sphinx_to_wechat.py

或指定目录：
    python sphinx_to_wechat.py --build-dir _build/html
"""

import argparse
import re
from pathlib import Path

# WaveNet 文档中有代表性的公式，用来确保后处理不会再破坏原始 LaTeX。
TEST_FORMULAS = [
    r"\(\mathbf{x} = (x_1, x_2, \ldots, x_T)\)",
    r"\[p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})\]",
    r"\[F(x)=\operatorname{sgn}(x)\cdot\frac{\ln(1+\mu |x|)}{\ln(1+\mu)}\]",
    r"\[\mathbf{z}=\tanh(W_f * \mathbf{x})\odot\sigma(W_g * \mathbf{x})\]",
    r"\(\mathcal{L}_{\text{NLL}}\)",
]


def test_formula_processing():
    """测试公式处理功能。"""
    formula_html = "\n".join(
        f'<div class="math notranslate nohighlight">{formula}</div>'
        for formula in TEST_FORMULAS
    )
    original = (
        "<html><head>"
        '<script>window.MathJax = {"options": {"processHtmlClass": "math"}}</script>'
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
        'tex-mml-chtml.js"></script>'
        f"</head><body>{formula_html}</body></html>"
    )

    result = fix_math_blocks(original)
    repeated = fix_math_blocks(result)

    checks = {
        "保留所有原始 LaTeX": all(formula in result for formula in TEST_FORMULAS),
        "切换到 MathJax SVG": "tex-mml-svg.js" in result,
        "启用本地字体缓存": 'fontCache: "local"' in result,
        "注入 PNG 转换器": MATH_IMAGE_CONVERTER_MARKER in result,
        "设置微信公式图片标记": "data-wechat-math-image" in result,
        "处理过程幂等": repeated == result,
        "不再生成 Unicode 近似公式": "Times New Roman" not in result,
    }

    for description, passed in checks.items():
        print(f"{'✓' if passed else '✗'} {description}")

    failed = [description for description, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("公式处理测试失败：" + "、".join(failed))
    print(f"\n全部 {len(checks)} 项公式处理测试通过")


# WaveGlow 2.2 节这类纯文本流程图，复制到微信后最容易丢换行。
TEST_PRE_HTML = """
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>文本
  ↓
文本前端（规范化、音素、韵律）
  ↓
WaveGlow 声码器
  ↓
原始音频波形
</pre></div>
</div>
"""


def test_pre_block_processing():
    """测试代码块换行在微信公众号复制场景下是否被显式保留。"""
    result = fix_pre_blocks(TEST_PRE_HTML)
    repeated = fix_pre_blocks(result)

    checks = {
        "不再使用 pre 标签": "<pre" not in result,
        "改用 section 作为代码块容器": 'data-wechat-code="true"' in result,
        "每一行都是独立段落": result.count("<p ") >= 7,
        "保留箭头行": "↓" in result,
        "行首缩进转为不间断空格": "&nbsp;&nbsp;↓" in result,
        "处理过程幂等": repeated == result,
        "没有把整段挤进一行": "</p><p " in result.replace("\n", ""),
    }

    for description, passed in checks.items():
        print(f"{'✓' if passed else '✗'} {description}")

    failed = [description for description, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("代码块处理测试失败：" + "、".join(failed))
    print(f"\n全部 {len(checks)} 项代码块处理测试通过")


# LaTeX符号到Unicode数学字符的映射
LATEX_TO_UNICODE = {
    # 希腊字母
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
    r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
    r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
    r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
    r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    # 大写希腊字母
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Chi': 'Χ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
    # 运算符
    r'\cdot': '·', r'\times': '×', r'\div': '÷', r'\pm': '±',
    r'\mp': '∓', r'\cdot': '·', r'\ast': '∗', r'\star': '⋆',
    # 关系符
    r'=': '=', r'\neq': '≠', r'<': '<', r'>': '>',
    r'\leq': '≤', r'\geq': '≥', r'\approx': '≈', r'\equiv': '≡',
    r'\sim': '∼', r'\propto': '∝',
    # 箭头
    r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
    r'\Leftarrow': '⇐', r'\leftrightarrow': '↔', r'\Leftrightarrow': '⇔',
    r'\to': '→',
    # 符号
    r'\infty': '∞', r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
    r'\sqrt': '√', r'\nabla': '∇', r'\partial': '∂', r'\cdot': '·',
    r'\dots': '…', r'\ldots': '…', r'\cdots': '⋯', r'\in': '∈',
    # 尖括号
    r'\langle': '⟨', r'\rangle': '⟩',
    # 三角函数
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
    r'\arcsin': 'arcsin', r'\arccos': 'arccos', r'\arctan': 'arctan',
    # 对数
    r'\log': 'log', r'\ln': 'ln',
    # 数学函数
    r'\exp': 'exp', r'\max': 'max', r'\min': 'min',
    # 括号
    r'\{': '{', r'\}': '}', r'\(': '(', r'\)': ')',
    r'\[': '[', r'\]': ']',
    # 其他
    r'\text': '',  # 移除\text命令
    r'\mathrm': '',  # 移除\mathrm命令
}


def latex_to_unicode(formula_text):
    """将LaTeX公式转换为Unicode字符。"""
    if not formula_text:
        return formula_text
    
    result = formula_text
    
    # 保护 \text{} 内容，避免在下标处理中被错误转换
    text_contents = []
    def protect_text(match):
        content = match.group(1)
        placeholder = f'__TEXT_PLACEHOLDER_{len(text_contents)}__'
        text_contents.append(content)
        return placeholder
    
    result = re.sub(r'\\text\{([^}]+)\}', protect_text, result)
    
    # 在处理任何下标之前，先将占位符转换为 HTML 实体
    # 这样占位符中的下划线不会被当作下标标记处理
    result = result.replace('__LT__', '&lt;').replace('__GT__', '&gt;')
    
    # 先替换大型运算符（\sum, \prod, \int），这样它们后面的下标不会被错误转换
    for latex, unicode_char in LATEX_TO_UNICODE.items():
        result = result.replace(latex, unicode_char)
    
    # 处理上标（^后接花括号或单个字符）
    # 只在非字母数字后面的 ^ 才转换
    result = re.sub(r'\^\{(.+?)\}', lambda m: superscript_text(m.group(1)), result)
    result = re.sub(r'(?<![a-zA-Z0-9])\^([a-zA-Z0-9])', lambda m: superscript_char(m.group(1)), result)
    
    # 处理下标（_后接花括号或单个字符）
    # 只在以下情况下转换：
    # - 前面是希腊字母（如 α_i）
    # - 前面是大写字母（如 X_i）
    # - 前面是运算符（如 ∑_i）
    # - 前面是小写字母（如 e_i）
    # - 不转换小写字母之间的下划线（如 total_prob）
    # 对于花括号形式的下标（如 y_{<i}），总是转换
    # 对于单字符下标，使用负向后瞻来排除小写字母后的下划线（但允许其他情况）
    # 对于 _< 或 _> 的形式（如 y_<i），也需要处理（没有负向后瞻限制）
    
    # 处理花括号下标
    result = re.sub(r'_\{([^}]+)\}', lambda m: subscript_text(m.group(1)), result)
    
    # 处理 _< 或 _> 的形式（如 y_<i）- 这些肯定是下标，不需要负向后瞻限制
    # 先处理 _< 后面跟字符的情况（如 _<i），保持后面的字符原样
    result = re.sub(r'_<([a-zA-Z0-9])', '&lt;\\1', result)
    result = re.sub(r'_>([a-zA-Z0-9])', '&gt;\\1', result)
    
    # 然后处理单独的 _< 或 _>
    result = re.sub(r'_<', '&lt;', result)
    result = re.sub(r'_>', '&gt;', result)
    
    # 处理 _<...> 的形式（如 _<i+1>）
    result = re.sub(r'_<([^>]+)>', lambda m: '&lt;' + m.group(1) + '&gt;', result)
    
    # 处理单字符下标（需要负向后瞻限制，避免转换变量名）
    result = re.sub(r'(?<![a-z])_([a-zA-Z0-9])', lambda m: subscript_char(m.group(1)), result)
    
    # 清理多余的花括号
    result = result.replace('{', '').replace('}', '')
    
    # 恢复 \text{} 内容
    for i, content in enumerate(text_contents):
        result = result.replace(f'__TEXT_PLACEHOLDER_{i}__', content)
    
    return result


def superscript_char(c):
    """将字符转换为上标形式。"""
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
        'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
        'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ',
        'p': 'ᵖ', 'q': 'ᵠ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ',
        'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
        'A': 'ᴬ', 'B': 'ᴮ', 'C': 'ᶜ', 'D': 'ᴰ', 'E': 'ᴱ',
        'F': 'ᶠ', 'G': 'ᴳ', 'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ',
        'K': 'ᴷ', 'L': 'ᴸ', 'M': 'ᴹ', 'N': 'ᴺ', 'O': 'ᴼ',
        'P': 'ᴾ', 'Q': 'ᵠ', 'R': 'ᴿ', 'S': 'ˢ', 'T': 'ᵀ',
        'U': 'ᵁ', 'V': 'ⱽ', 'W': 'ᵂ', 'X': 'ˣ', 'Y': 'ʸ', 'Z': 'ᶻ',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
    }
    return superscript_map.get(c, c)


def subscript_char(c):
    """将字符转换为下标形式。"""
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
        'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
        'v': 'ᵥ', 'x': 'ₓ',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎'
    }
    return subscript_map.get(c, c)


def superscript_text(text):
    """将文本转换为上标形式（简化版）。"""
    return ''.join(superscript_char(c) for c in text)


def subscript_text(text):
    """将文本转换为下标形式（简化版）。"""
    # 在转换的同时，将 < 和 > 转义为 HTML 实体，避免被当作标签处理
    result = []
    i = 0
    while i < len(text):
        # 检查是否是占位符（__LT__ 和 __GT__）
        if i + 6 <= len(text) and text[i:i+6] == '__LT__':
            result.append('&lt;')
            i += 6
            # 继续转换剩余文本为下标形式
            while i < len(text):
                result.append(subscript_char(text[i]))
                i += 1
            break
        elif i + 6 <= len(text) and text[i:i+6] == '__GT__':
            result.append('&gt;')
            i += 6
            # 继续转换剩余文本为下标形式
            while i < len(text):
                result.append(subscript_char(text[i]))
                i += 1
            break
        # 检查是否是 HTML 实体（&lt; 和 &gt;）- &lt; 和 &gt; 都是4个字符
        elif i + 4 <= len(text) and text[i:i+4] == '&lt;':
            result.append('&lt;')
            i += 4
            # 继续转换剩余文本为下标形式
            while i < len(text):
                result.append(subscript_char(text[i]))
                i += 1
            break
        elif i + 4 <= len(text) and text[i:i+4] == '&gt;':
            result.append('&gt;')
            i += 4
            # 继续转换剩余文本为下标形式
            while i < len(text):
                result.append(subscript_char(text[i]))
                i += 1
            break
        elif text[i] == '<':
            result.append('&lt;')
            i += 1
            # 继续转换剩余文本为下标形式
            while i < len(text):
                result.append(subscript_char(text[i]))
                i += 1
            break
        elif text[i] == '>':
            result.append('&gt;')
            i += 1
            # 继续转换剩余文本为下标形式
            while i < len(text):
                result.append(subscript_char(text[i]))
                i += 1
            break
        else:
            result.append(subscript_char(text[i]))
            i += 1
    return ''.join(result)


def _legacy_unicode_math_conversion(html_content: str) -> str:
    """旧版 LaTeX 到 Unicode 转换，保留仅供历史行为参考。"""
    
    def process_formula_content(formula_text):
        """处理公式内容：移除环境声明、转换LaTeX到Unicode。"""
        if not formula_text:
            return formula_text
        
        # 注意：占位符转换已经在调用之前完成（在 process_math_span/process_math_div 等函数中）
        # 这里不再进行占位符转换，直接处理公式内容
        
        # 去除前后的 \( \) 或 \[ \] 标记
        formula_text = formula_text.strip()
        if formula_text.startswith(r'\(') and formula_text.endswith(r'\)'):
            formula_text = formula_text[2:-2].strip()
        elif formula_text.startswith(r'\[') and formula_text.endswith(r'\]'):
            formula_text = formula_text[2:-2].strip()
        elif formula_text.startswith(r'\('):
            formula_text = formula_text[2:].strip()
        elif formula_text.endswith(r'\)'):
            formula_text = formula_text[:-2].strip()
        elif formula_text.startswith(r'\['):
            formula_text = formula_text[2:].strip()
        elif formula_text.endswith(r'\]'):
            formula_text = formula_text[:-2].strip()
        
        # 去除环境声明（使用字符串替换，更可靠）
        formula_text = formula_text.replace('\\begin{split}', '')
        formula_text = formula_text.replace('\\end{split}', '')
        formula_text = formula_text.replace('\\begin{align*}', '')
        formula_text = formula_text.replace('\\end{align*}', '')
        formula_text = formula_text.replace('\\begin{align}', '')
        formula_text = formula_text.replace('\\end{align}', '')
        formula_text = formula_text.replace('\\begin{equation*}', '')
        formula_text = formula_text.replace('\\end{equation*}', '')
        formula_text = formula_text.replace('\\begin{equation}', '')
        formula_text = formula_text.replace('\\end{equation}', '')
        
        # 处理 \text{} 命令（包括后面可能紧跟的括号，保护内容不被错误转换）
        text_contents = []
        def protect_text(match):
            content = match.group(1)
            # 也保护后面的括号内容（如 \text{foll}(i)）
            paren_content = match.group(2) if match.group(2) else ''
            full_content = content + paren_content
            # 使用字母作为占位符索引（避免数字被转换为下标）
            placeholder = f'__TEXTCONTENT{chr(ord("A") + len(text_contents))}__'
            text_contents.append(full_content)
            return placeholder
        
        # 匹配 \text{内容}(参数) 或 \text{内容}
        formula_text = re.sub(r'\\text\{([^}]+)\}(\([^)]+\))?', protect_text, formula_text)
        # 处理可能残留的 \text 命令
        formula_text = re.sub(r'\\text\s*', '', formula_text)
        
        # 处理 align 环境中的 & 符号
        formula_text = formula_text.replace('&', '')
        
        # 处理 \frac{a}{b} -> a/b（支持嵌套花括号）
        def parse_frac(text):
            # 找到 \frac 命令的位置
            idx = text.find(r'\frac')
            if idx == -1:
                return text
            
            # 找到分子和分母的花括号
            start = idx + 5  # 跳过 \frac
            depth = 0
            brace_positions = []
            
            # 跳过花括号前的空白
            i = 0
            while i < len(text[start:]):
                c = text[start + i]
                if c.isspace():
                    i += 1
                    continue
                if c == '{':
                    depth += 1
                    if depth == 1:
                        brace_positions.append(start + i)
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        brace_positions.append(start + i)
                i += 1
            
            # 应该有4个位置：分子开始、分子结束、分母开始、分母结束
            if len(brace_positions) >= 4:
                numerator_start = brace_positions[0] + 1
                numerator_end = brace_positions[1]
                denominator_start = brace_positions[2] + 1
                denominator_end = brace_positions[3]
                
                numerator = text[numerator_start:numerator_end]
                denominator = text[denominator_start:denominator_end]
                
                # 替换 \frac{...}{...} 为 (...) / (...)
                result = text[:idx] + f'({numerator})/({denominator})' + text[denominator_end+1:]
                return result
            return text
        
        # 处理 \frac 命令（最多处理10次，防止无限循环）
        for _ in range(10):
            old_text = formula_text
            formula_text = parse_frac(formula_text)
            if formula_text == old_text:
                break
        
        # 将 LaTeX 换行符转换为 HTML 换行
        formula_text = formula_text.replace(r'\\', '\n')
        # 保留实际换行符
        # 清理多余的空白（但保留换行）
        lines = formula_text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        formula_text = '\n'.join(cleaned_lines)
        
        # 将LaTeX转换为Unicode
        formula_text = latex_to_unicode(formula_text)
        
        # 恢复 \text{} 内容
        for i, content in enumerate(text_contents):
            placeholder = f'__TEXTCONTENT{chr(ord("A") + i)}__'
            formula_text = formula_text.replace(placeholder, content)
        
        return formula_text
    
    def wrap_formula_with_content(formula_text):
        """将公式内容用span包裹。"""
        result = process_formula_content(formula_text)
        if result:
            # 如果是多行公式，用 <br> 分隔
            if '\n' in result:
                lines = result.split('\n')
                wrapped_lines = [f'<span style="font-family: \'Times New Roman\', serif; font-style: italic;">{line}</span>' for line in lines]
                return '<br>'.join(wrapped_lines)
            else:
                return f'<span style="font-family: \'Times New Roman\', serif; font-style: italic; white-space: nowrap;">{result}</span>'
        return None
    
    # 处理传统的 math 类 span 标签（inline公式）
    def process_math_span(match):
        full_match = match.group(0)
        # 提取标签内的文本内容（即原始LaTeX）
        text_content = re.sub(r'<[^>]+>', '', full_match)
        # 将原始的 < 和 > 替换为占位符（处理未转义的情况）
        text_content = text_content.replace('<', '__LT__').replace('>', '__GT__')
        # 将 HTML 实体也转换为占位符
        text_content = text_content.replace('&lt;', '__LT__').replace('&gt;', '__GT__')
        # 将 &amp; 解码为 &
        text_content = text_content.replace('&amp;', '&')
        # 调用处理函数
        result = wrap_formula_with_content(text_content)
        return result if result else full_match
    
    html_content = re.sub(r'<span\s+class="math[^"]*"[^>]*>.*?</span>', process_math_span, html_content, flags=re.DOTALL)
    
    # 处理传统的 math 类 div 标签（display公式）
    def process_math_div(match):
        full_match = match.group(0)
        # 提取标签内的文本内容（即原始LaTeX）
        text_content = re.sub(r'<[^>]+>', '', full_match)
        # 将原始的 < 和 > 替换为占位符（处理未转义的情况）
        text_content = text_content.replace('<', '__LT__').replace('>', '__GT__')
        # 将 HTML 实体也转换为占位符
        text_content = text_content.replace('&lt;', '__LT__').replace('&gt;', '__GT__')
        # 将 &amp; 解码为 &
        text_content = text_content.replace('&amp;', '&')
        # 调用处理函数
        result = wrap_formula_with_content(text_content)
        # 在每个公式后面添加换行
        if result:
            return result + '\n'
        return full_match
    
    html_content = re.sub(r'<div\s+class="math[^"]*"[^>]*>.*?</div>', process_math_div, html_content, flags=re.DOTALL)
    
    # 将公式之间的换行转换为 HTML 换行
    html_content = re.sub(r'</span>\s*\n\s*<span style="font-family:', '</span><br><span style="font-family:', html_content)
    
    # 处理 MathJax 3+ 渲染后的 mjx-container 标签
    def process_mjx_container(match):
        full_match = match.group(0)
        # 尝试从 data-latex 属性获取原始 LaTeX
        latex_match = re.search(r'data-latex="([^"]+)"', full_match)
        if latex_match:
            latex = latex_match.group(1)
            # 解码 HTML 实体，但将 &lt; 和 &gt; 转换为占位符而不是原始字符
            latex = latex.replace('&amp;', '&')
            latex = latex.replace('&lt;', '__LT__').replace('&gt;', '__GT__')
            # 也处理可能存在的原始 < 和 >
            latex = latex.replace('<', '__LT__').replace('>', '__GT__')
            result = wrap_formula_with_content(latex)
            return result if result else full_match
        # 提取文本内容
        text_content = re.sub(r'<[^>]+>', '', full_match)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        # 将 < 和 > 转换为占位符
        text_content = text_content.replace('<', '__LT__').replace('>', '__GT__')
        result = wrap_formula_with_content(text_content)
        return result if result else full_match
    
    html_content = re.sub(r'<mjx-container[^>]*>.*?</mjx-container>', process_mjx_container, html_content, flags=re.DOTALL)
    
    # 处理 mjx-math 标签
    html_content = re.sub(r'<mjx-math[^>]*>.*?</mjx-math>', process_mjx_container, html_content, flags=re.DOTALL)
    
    # 移除隐藏的 mjx-assistive-mml 标签
    html_content = re.sub(r'<mjx-assistive-mml[^>]*>.*?</mjx-assistive-mml>', '', html_content, flags=re.DOTALL)
    
    # 处理 tex2jax_process 类
    html_content = re.sub(r'<(span|div)\s+class="[^"]*tex2jax_process[^"]*"[^>]*>.*?</\1>', process_math_span, html_content, flags=re.DOTALL)
    
    # 处理 equation 类
    html_content = re.sub(r'<(span|div)\s+class="[^"]*equation[^"]*"[^>]*>.*?</\1>', process_math_span, html_content, flags=re.DOTALL)
    
    # 处理已经被包裹在 font-family: 'Times New Roman' span 中的公式
    # 这些公式可能仍然包含 LaTeX 环境声明或未转义的 < > 字符
    def process_existing_math_span(match):
        full_match = match.group(0)
        # 使用正则表达式提取标签内容（不包括标签本身）
        # 匹配 <span ...>content</span> 并提取 content
        content_match = re.search(r'<span[^>]*>(.*?)</span>', full_match, flags=re.DOTALL)
        if content_match:
            text_content = content_match.group(1).strip()
        else:
            # 如果匹配失败，使用简单方法提取文本
            text_content = re.sub(r'<[^>]+>', '', full_match).strip()
        
        # 检查是否已经处理过（防止重复处理）
        # 已经处理过的标志是：包含 &lt; 或 &gt;，但不包含原始的 < 或 >
        has_escaped = '&lt;' in text_content or '&gt;' in text_content
        has_unescaped = '<' in text_content or '>' in text_content
        if has_escaped and not has_unescaped:
            # 如果只有转义的 &lt; &gt;，没有原始的 < >，说明已经处理过了
            return full_match
        
        # 检查是否包含 LaTeX 环境声明或 < > 字符（包括转义形式或占位符）
        if '\\begin{' in text_content or '\\end{' in text_content or '<' in text_content or '>' in text_content or '&lt;' in text_content or '&gt;' in text_content or '__LT__' in text_content or '__GT__' in text_content:
            # 保存原始的 &lt; 和 &gt;
            text_content = text_content.replace('&lt;', '__LT__').replace('&gt;', '__GT__')
            # 同时替换原始的 < 和 > 为占位符（处理未转义的情况）
            text_content = text_content.replace('<', '__LT__').replace('>', '__GT__')
            processed = process_formula_content(text_content)
            if processed:
                return f'<span style="font-family: \'Times New Roman\', serif; font-style: italic; white-space: nowrap;">{processed}</span>'
        return full_match
    
    html_content = re.sub(r'<span\s+style="[^"]*Times New Roman[^"]*">.*?</span>', process_existing_math_span, html_content, flags=re.DOTALL)
    
    return html_content


MATH_IMAGE_CONVERTER_MARKER = 'id="wechat-math-image-converter"'

MATHJAX_SVG_CONFIG = """<script id="wechat-mathjax-svg-config">
window.MathJax = window.MathJax || {};
window.MathJax.svg = Object.assign({}, window.MathJax.svg, {fontCache: "local"});
</script>
"""

MATH_IMAGE_CONVERTER = r"""<script id="wechat-math-image-converter">
(function () {
  "use strict";

  const SCALE = 3;

  function svgToPng(svg, scale) {
    return new Promise((resolve, reject) => {
      const rect = svg.getBoundingClientRect();
      const width = Math.max(rect.width, 1);
      const height = Math.max(rect.height, 1);
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", width + "px");
      clone.setAttribute("height", height + "px");

      const source = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([source], {type: "image/svg+xml;charset=utf-8"});
      const objectUrl = URL.createObjectURL(blob);
      const sourceImage = new Image();

      sourceImage.onload = function () {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = Math.ceil(width * scale);
          canvas.height = Math.ceil(height * scale);
          const context = canvas.getContext("2d");
          context.scale(scale, scale);
          context.drawImage(sourceImage, 0, 0, width, height);
          resolve({
            dataUrl: canvas.toDataURL("image/png"),
            width: width,
            height: height
          });
        } catch (error) {
          reject(error);
        } finally {
          URL.revokeObjectURL(objectUrl);
        }
      };
      sourceImage.onerror = function () {
        URL.revokeObjectURL(objectUrl);
        reject(new Error("无法将 MathJax SVG 转换为图片"));
      };
      sourceImage.src = objectUrl;
    });
  }

  async function convertMathToImages() {
    if (!window.MathJax || !window.MathJax.startup) {
      console.warn("MathJax 未加载，公式图片转换已跳过");
      return;
    }

    await window.MathJax.startup.promise;
    const containers = Array.from(document.querySelectorAll("mjx-container"));

    await Promise.all(containers.map(async (container) => {
      const svg = container.querySelector("svg");
      if (!svg) {
        return;
      }

      const isDisplay = container.getAttribute("display") === "true";
      const rendered = await svgToPng(svg, SCALE);
      const image = document.createElement("img");
      image.src = rendered.dataUrl;
      image.alt = container.getAttribute("aria-label") || "数学公式";
      image.setAttribute("data-wechat-math-image", "true");
      image.style.width = rendered.width + "px";
      image.style.height = rendered.height + "px";
      image.style.maxWidth = "100%";
      image.style.objectFit = "contain";

      if (isDisplay) {
        image.style.display = "block";
        image.style.margin = "0.6em auto";
      } else {
        image.style.display = "inline-block";
        image.style.margin = "0 0.08em";
        image.style.verticalAlign = "-0.2em";
      }

      container.replaceWith(image);
    }));

    document.documentElement.setAttribute("data-wechat-math-ready", "true");
  }

  window.addEventListener("load", function () {
    convertMathToImages().catch((error) => {
      console.error("微信公众号公式转换失败", error);
    });
  });
})();
</script>
"""


def fix_math_blocks(html_content: str) -> str:
    """保留 LaTeX，并注入 MathJax SVG 转 PNG 的浏览器端转换逻辑。

    微信公众号编辑器会移除 MathJax 依赖的脚本和 CSS，但可以接收从网页复制的
    PNG 图片。公式先由 MathJax 精确排版，再转成内嵌 PNG，因此分式、上下标、
    粗体向量和大型运算符都能保持原样。
    """
    if MATH_IMAGE_CONVERTER_MARKER in html_content:
        return html_content

    # SVG 输出能完整保留公式结构，并可直接绘制到 Canvas。local 字体缓存使每个
    # SVG 都自包含，避免栅格化时引用页面外部的 MathJax 字形定义。
    html_content = html_content.replace(
        "tex-mml-chtml.js",
        "tex-mml-svg.js",
    )

    mathjax_loader = re.search(
        r'<script\b[^>]*src="[^"]*mathjax[^"]*tex-mml-svg\.js"[^>]*></script>',
        html_content,
        flags=re.IGNORECASE,
    )
    if mathjax_loader:
        html_content = (
            html_content[:mathjax_loader.start()]
            + MATHJAX_SVG_CONFIG
            + html_content[mathjax_loader.start():]
        )

    if "</body>" in html_content:
        return html_content.replace(
            "</body>",
            MATH_IMAGE_CONVERTER + "\n</body>",
            1,
        )
    return html_content + "\n" + MATH_IMAGE_CONVERTER


WECHAT_CODE_BLOCK_STYLE = (
    "margin: 12px 0; padding: 16px; background-color: #f6f8fa; "
    "border-radius: 6px; font-family: Consolas, Monaco, 'Courier New', monospace; "
    "font-size: 14px; line-height: 1.6; overflow-wrap: break-word; word-break: break-all;"
)
WECHAT_CODE_LINE_STYLE = "margin: 0; padding: 0; line-height: 1.6;"


def _nbsp_leading_spaces(line: str) -> str:
    """把行首空格换成 &nbsp;，避免微信折叠缩进。"""
    leading_spaces = ""
    i = 0
    while i < len(line):
        if line[i] == " ":
            leading_spaces += " "
            i += 1
        elif line[i:i + 6] == "&nbsp;":
            i += 6
        else:
            break
    if leading_spaces:
        return "&nbsp;" * len(leading_spaces) + line[len(leading_spaces):]
    return line


def _normalize_pre_line(line: str) -> str:
    """清洗 Pygments / 旧版后处理留下的标签，并保留缩进。"""
    line = re.sub(
        r'<span class="w">([^<]*)</span>',
        lambda m: m.group(1).replace(" ", "&nbsp;"),
        line,
    )
    line = re.sub(r"^<span></span>", "", line)
    line = re.sub(r"^<span>(.*)</span>$", r"\1", line, flags=re.DOTALL)
    return _nbsp_leading_spaces(line)


def fix_pre_blocks(html_content: str) -> str:
    """把 <pre> 转成微信公众号可复制的代码块。

    微信编辑器会丢掉 <pre> 和 white-space:pre，只保留纯文本，于是换行消失。
    改成 <section> 包一层，每一行单独一个 <p>，复制后仍按行显示。
    """

    def fix_pre_content(match):
        full_match = match.group(0)
        if 'data-wechat-code="true"' in full_match:
            return full_match

        opening_tag_match = re.match(r"<pre[^>]*>", full_match)
        inner = full_match[len(opening_tag_match.group(0)):-len("</pre>")]

        if re.search(r"<br\s*/?>", inner):
            raw_lines = re.split(r"<br\s*/?>", inner)
        else:
            raw_lines = inner.split("\n")

        fixed_lines = [_normalize_pre_line(line) for line in raw_lines]
        while fixed_lines and not fixed_lines[-1].strip():
            fixed_lines.pop()
        while fixed_lines and not fixed_lines[0].strip():
            fixed_lines.pop(0)

        if not fixed_lines:
            return full_match

        line_html = "".join(
            f'<p style="{WECHAT_CODE_LINE_STYLE}">{line if line else "&nbsp;"}</p>'
            for line in fixed_lines
        )
        return (
            f'<section data-wechat-code="true" style="{WECHAT_CODE_BLOCK_STYLE}">'
            f"{line_html}</section>"
        )

    return re.sub(r"<pre[^>]*>.*?</pre>", fix_pre_content, html_content, flags=re.DOTALL)


def process_directory(build_dir: Path):
    html_files = list(build_dir.rglob('*.html'))
    modified_count = 0

    for html_file in html_files:
        original = html_file.read_text(encoding='utf-8')
        # 先处理公式，再处理代码块
        fixed = fix_math_blocks(original)
        fixed = fix_pre_blocks(fixed)
        if fixed != original:
            html_file.write_text(fixed, encoding='utf-8')
            modified_count += 1

    print(f"处理完成：共扫描 {len(html_files)} 个文件，修改了 {modified_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description='修复 Sphinx HTML 代码块换行和公式，兼容微信公众号')
    parser.add_argument('--build-dir', default='_build/html', help='HTML 构建输出目录 (默认: _build/html)')
    parser.add_argument('--test', action='store_true', help='测试公式处理功能')
    args = parser.parse_args()

    if args.test:
        test_formula_processing()
        test_pre_block_processing()
        return

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        print(f"错误：目录 {build_dir} 不存在，请先运行 make html")
        return

    process_directory(build_dir)


if __name__ == '__main__':
    main()