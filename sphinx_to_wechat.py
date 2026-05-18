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
    """在 <pre> 块内插入 <br> 保留换行，并将空格替换为 &nbsp; 保留缩进。"""

    def fix_pre_content(match):
        full_match = match.group(0)
        opening_tag_match = re.match(r'<pre[^>]*>', full_match)
        opening_tag = opening_tag_match.group(0)
        closing_tag = '</pre>'
        inner = full_match[len(opening_tag):-len(closing_tag)]

        lines = inner.split('\n')

        fixed_lines = []
        for line in lines:
            # 处理所有 <span class="w"> 中的空格（不仅是行首的，中间的也要保留）
            line = re.sub(
                r'<span class="w">( +)</span>',
                lambda m: '<span class="w">' + '&nbsp;' * len(m.group(1)) + '</span>',
                line
            )
            # 处理行首裸空格
            leading_match = re.match(r'^( +)', line)
            if leading_match:
                leading_spaces = leading_match.group(1)
                line = '&nbsp;' * len(leading_spaces) + line[len(leading_spaces):]
            fixed_lines.append(line)

        # 最后一个元素如果是空字符串（即原文以 \n 结尾），不加 <br>
        if fixed_lines and fixed_lines[-1] == '':
            inner_fixed = '<br>\n'.join(fixed_lines[:-1]) + '\n'
        else:
            inner_fixed = '<br>\n'.join(fixed_lines)

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