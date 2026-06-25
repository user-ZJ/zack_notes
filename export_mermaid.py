#!/usr/bin/env python3
"""
从 Markdown 中提取 ```mermaid 代码块，导出为 PNG 图片。

依赖：@mermaid-js/mermaid-cli（mmdc 命令）

用法：
    python export_mermaid.py 深度学习笔记/推理框架/sglang/sglang一条请求的旅程.md --no-keep-mmd -o exported
    python export_mermaid.py --recursive 深度学习笔记/
    python export_mermaid.py file.md --scale 2 --background white

每个 mermaid 块会生成：
    {markdown 同目录}/{stem}_mermaid_{序号}.mmd
    {markdown 同目录}/{stem}_mermaid_{序号}.png
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MERMAID_BLOCK_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL,
)


def find_mermaid_blocks(text: str) -> list[str]:
    return [block.strip() for block in MERMAID_BLOCK_RE.findall(text)]


def default_output_paths(md_path: Path, index: int, output_dir: Path | None) -> tuple[Path, Path]:
    out_dir = output_dir if output_dir is not None else md_path.parent
    stem = md_path.stem
    base = out_dir / f"{stem}_mermaid_{index:02d}"
    return base.with_suffix(".mmd"), base.with_suffix(".png")


def export_one_block(
    content: str,
    mmd_path: Path,
    png_path: Path,
    *,
    background: str,
    scale: float,
    mmdc: str,
    dry_run: bool,
) -> None:
    mmd_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] write {mmd_path}")
        print(f"[dry-run] mmdc -i {mmd_path} -o {png_path} -b {background} -s {scale}")
        return

    mmd_path.write_text(content + "\n", encoding="utf-8")

    cmd = [
        mmdc,
        "-i",
        str(mmd_path),
        "-o",
        str(png_path),
        "-b",
        background,
        "-s",
        str(scale),
    ]
    print(f"→ {png_path}")
    subprocess.run(cmd, check=True)


def export_markdown(
    md_path: Path,
    *,
    output_dir: Path | None,
    background: str,
    scale: float,
    mmdc: str,
    dry_run: bool,
    keep_mmd: bool,
) -> int:
    text = md_path.read_text(encoding="utf-8")
    blocks = find_mermaid_blocks(text)
    if not blocks:
        print(f"skip (no mermaid): {md_path}")
        return 0

    print(f"export {len(blocks)} diagram(s) from {md_path}")
    for i, block in enumerate(blocks, start=1):
        mmd_path, png_path = default_output_paths(md_path, i, output_dir)
        export_one_block(
            block,
            mmd_path,
            png_path,
            background=background,
            scale=scale,
            mmdc=mmdc,
            dry_run=dry_run,
        )
        if not dry_run and not keep_mmd and mmd_path.exists():
            mmd_path.unlink()

    return len(blocks)


def collect_markdown_files(paths: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_file():
            if path.suffix.lower() in {".md", ".markdown"}:
                files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)

        pattern = "**/*" if recursive else "*"
        for candidate in path.glob(pattern):
            if candidate.is_file() and candidate.suffix.lower() in {".md", ".markdown"}:
                files.append(candidate)

    return sorted(set(files))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mermaid code blocks in Markdown to PNG via mmdc.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Markdown file(s) or directory(ies) to process",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="When path is a directory, scan subdirectories",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as each markdown file)",
    )
    parser.add_argument(
        "-b",
        "--background",
        default="transparent",
        help="Background color for mmdc (default: transparent)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=3.0,
        help="Scale factor for mmdc (default: 3)",
    )
    parser.add_argument(
        "--mmdc",
        default="mmdc",
        help="Path to mmdc executable (default: mmdc)",
    )
    parser.add_argument(
        "--keep-mmd",
        action="store_true",
        default=True,
        help="Keep intermediate .mmd files (default: keep)",
    )
    parser.add_argument(
        "--no-keep-mmd",
        action="store_false",
        dest="keep_mmd",
        help="Delete intermediate .mmd files after export",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    md_files = collect_markdown_files(args.paths, args.recursive)
    if not md_files:
        print("No markdown files found.", file=sys.stderr)
        return 1

    total = 0
    for md_path in md_files:
        total += export_markdown(
            md_path,
            output_dir=args.output_dir,
            background=args.background,
            scale=args.scale,
            mmdc=args.mmdc,
            dry_run=args.dry_run,
            keep_mmd=args.keep_mmd,
        )

    print(f"done: {total} diagram(s) from {len(md_files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
