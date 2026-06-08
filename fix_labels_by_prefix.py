"""fix_labels_by_prefix.py

将 data/images/{train,val} 下以 s2 开头的标签改为类 1（漏胶），
以 s3 开头的标签改为类 2（凸边）。

用法示例:
    # 先预览，不会改动文件
    python fix_labels_by_prefix.py --root data

    # 实际修改（会在修改前备份每个被修改的 txt 为 .bak）
    python fix_labels_by_prefix.py --root data --apply

可选参数:
    --split train 或 val 或 both (默认 both)
"""
from pathlib import Path
import argparse
from collections import defaultdict
import shutil


def process_file(path: Path, new_cls: str, apply: bool = False) -> int:
    """Process a single .txt label file. Replace first token with new_cls.
    Returns number of lines modified.
    """
    changed = 0
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip('\n')
            if not s.strip():
                lines.append(line)
                continue
            parts = s.split()
            if parts[0] != new_cls:
                parts[0] = new_cls
                changed += 1
            lines.append(" ".join(parts) + "\n")

    if changed and apply:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(str(path), str(bak))
        # atomic write
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        tmp.replace(path)
    return changed


def main():
    parser = argparse.ArgumentParser(description="按文件名前缀修正 YOLO txt 标签的类别 ID")
    parser.add_argument("--root", type=Path, default=Path("data"), help="数据根目录，默认 data")
    parser.add_argument("--split", choices=["train","val","both"], default="both")
    parser.add_argument("--apply", action="store_true", help="实际写入修改（默认只预览）")
    args = parser.parse_args()

    splits = ["train","val"] if args.split == "both" else [args.split]
    summary = defaultdict(int)
    file_changes = 0
    file_total = 0

    for sp in splits:
        img_dir = args.root / "labels" / sp
        if not img_dir.exists():
            print(f"跳过不存在目录: {img_dir}")
            continue
        for txt in sorted(img_dir.glob("*.txt")):
            file_total += 1
            stem = txt.stem
            if stem.startswith("s2"):
                new_cls = "1"
            elif stem.startswith("s3"):
                new_cls = "2"
            else:
                continue
            changed = process_file(txt, new_cls, apply=args.apply)
            if changed:
                file_changes += 1
                summary[new_cls] += changed
                print(f"{('MODIFIED' if args.apply else 'PREVIEW')} {txt} -> set class {new_cls} (lines changed: {changed})")

    print("\n处理完毕。")
    print(f"扫描文件总数: {file_total}, 修改过的文件: {file_changes}")
    for cls, cnt in summary.items():
        print(f"类 {cls} 被修改的行数: {cnt}")
    if args.apply:
        print("已在每个被修改的文件旁生成 .bak 备份")


if __name__ == '__main__':
    main()
