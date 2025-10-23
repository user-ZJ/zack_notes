# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'docs'
copyright = '2022, zack'
author = 'zack'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser','sphinx.ext.githubpages','sphinx.ext.todo',
    'sphinxcontrib.mermaid',
    # 'sphinxcontrib.images'
]

# # 配置 Mermaid 生成图片（而非依赖浏览器渲染）
# mermaid_output_format = 'png'  # 可选 'png' 或 'svg'
# # mermaid_png_output_dir = '_static/mermaid_png'  # 图片输出目录（相对 source 目录）
# mermaid_cmd = 'mmdc'  # 本地 mermaid-cli 的命令（需确保已安装）

# images_config = {
#     # 启用点击放大功能（点击图片弹出全屏预览）
#     'override_image_directive': True,  # 覆盖默认的 .. image:: 指令
#     'zoom_image': True,                # 允许图片被点击放大
#     'default_image_width': '100%',      # 图片在网页中的默认显示宽度
#     'default_image_height': 'auto',    # 高度自动适应
# }

todo_include_todos = True

suppress_warnings = ["myst.header"]

source_suffix = {
    '.rst': 'restructuredtext',
    # '.txt': 'markdown',
    '.md': 'markdown',
}

highlight_langeuage="c,cpp,python,shell,xml,yaml,json,protobuf"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
# html_search_fields = ['title', 'subtitle', 'content','keywords','tags']

