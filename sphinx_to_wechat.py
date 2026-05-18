#!/usr/bin/env python3
"""
后处理脚本：修复 Sphinx 生成的 HTML 中代码块换行符，使其兼容微信公众号编辑器。

原理：
  浏览器渲染 <pre> 时靠 CSS white-space:pre 保留换行，但复制粘贴到微信公众号
  编辑器后 CSS 丢失，\n 被当作普通空白折叠。解决方法是在每行末尾补 <br>。
  由于浏览器在 white-space:pre 下会忽略 <br>，所以本地预览效果不受影响。

用法：
    make html && python fix_wechat_code_block.py

或指定目录：
    python fix_wechat_code_block.py --build-dir _build/html
"""

import argparse
import re
from pathlib import Path


def fix_pre_blocks(html_content: str) -> str:
    """在 <pre> 块内插入换行和缩进，使其兼容微信公众号编辑器。"""

    def fix_pre_content(match):
        full_match = match.group(0)
        opening_tag_match = re.match(r'<pre[^>]*>', full_match)
        opening_tag = opening_tag_match.group(0)
        closing_tag = '</pre>'
        inner = full_match[len(opening_tag):-len(closing_tag)]

        lines = inner.split('\n')

        fixed_lines = []
        for line in lines:
            # 处理 <span class="w"> 标签 - 将其内容提取出来（去掉标签）
            # 微信公众号编辑器无法正确处理带标签的缩进
            def replace_span_w(m):
                span_content = m.group(1)
                # 将普通空格替换为 &nbsp;，然后移除 <span class="w"> 标签
                span_content = span_content.replace(' ', '&nbsp;')
                return span_content
            
            line = re.sub(
                r'<span class="w">([^<]*)</span>',
                replace_span_w,
                line
            )
            
            # 处理行首裸空格（不在任何标签内的空格）
            # 需要找到第一个非空格字符的位置
            leading_spaces = ''
            i = 0
            while i < len(line):
                if line[i] == ' ':
                    leading_spaces += ' '
                    i += 1
                elif line[i:i+6] == '&nbsp;':
                    # 已经是 &nbsp;，跳过
                    i += 6
                elif line[i] == '<':
                    # 遇到标签，停止
                    break
                else:
                    # 遇到其他字符，停止
                    break
            
            if leading_spaces:
                line = '&nbsp;' * len(leading_spaces) + line[len(leading_spaces):]
            
            fixed_lines.append(line)

        # 过滤掉末尾的空行
        while fixed_lines and fixed_lines[-1] == '':
            fixed_lines.pop()
        
        # 使用 <span> 标签配合 <br> 换行，避免 <p> 标签产生的额外段落间距
        # 每行用 <span> 包裹，末尾加 <br>
        inner_fixed = '<br>'.join(f'<span>{line}</span>' for line in fixed_lines)

        return opening_tag + inner_fixed + closing_tag

    return re.sub(r'<pre[^>]*>.*?</pre>', fix_pre_content, html_content, flags=re.DOTALL)


def process_directory(build_dir: Path):
    html_files = list(build_dir.rglob('*.html'))
    modified_count = 0

    for html_file in html_files:
        original = html_file.read_text(encoding='utf-8')
        fixed = fix_pre_blocks(original)
        if fixed != original:
            html_file.write_text(fixed, encoding='utf-8')
            modified_count += 1

    print(f"处理完成：共扫描 {len(html_files)} 个文件，修改了 {modified_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description='修复 Sphinx HTML 代码块换行，兼容微信公众号')
    parser.add_argument('--build-dir', default='_build/html', help='HTML 构建输出目录 (默认: _build/html)')
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        print(f"错误：目录 {build_dir} 不存在，请先运行 make html")
        return

    process_directory(build_dir)


if __name__ == '__main__':
    main()