#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def read_mem_words(path: Path) -> list[int]:
    words: list[int] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("//", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue

        if line.startswith("@"):
            raise ValueError(f"{path}:{line_no}: address directive '@' is not supported")

        token = line.split()[0]
        token = token.removeprefix("0x").removeprefix("0X")

        if not re.fullmatch(r"[0-9a-fA-F]{1,8}", token):
            raise ValueError(f"{path}:{line_no}: invalid 32-bit hex word: {token!r}")

        words.append(int(token, 16) & 0xFFFFFFFF)

    return words


def emit_array(name: str, words: list[int], expected: int | None) -> str:
    if expected is not None and len(words) != expected:
        raise ValueError(f"{name}: expected {expected} words, got {len(words)}")

    lines: list[str] = []
    lines.append(f"static const u32 {name}[{len(words)}] = {{")

    for i in range(0, len(words), 4):
        chunk = words[i:i + 4]
        values = ", ".join(f"0x{word:08x}U" for word in chunk)
        comma = "," if i + 4 < len(words) else ""
        lines.append(f"    {values}{comma}")

    lines.append("};")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Vivado/SystemVerilog .mem hex files to a C header."
    )
    parser.add_argument("--k", required=True, type=Path, help="k_ram.mem path")
    parser.add_argument("--q", required=True, type=Path, help="q_ram.mem path")
    parser.add_argument("--golden", required=True, type=Path, help="golden_score.mem path")
    parser.add_argument("--out", required=True, type=Path, help="output .h path")
    parser.add_argument("--k-words", type=int, default=1024)
    parser.add_argument("--q-words", type=int, default=1024)
    parser.add_argument("--golden-words", type=int, default=256)
    args = parser.parse_args()

    k_words = read_mem_words(args.k)
    q_words = read_mem_words(args.q)
    golden_words = read_mem_words(args.golden)

    text = "\n\n".join([
        "#pragma once",
        '#include "xil_types.h"',
        "",
        emit_array("k_data", k_words, args.k_words),
        emit_array("q_data", q_words, args.q_words),
        emit_array("golden_score", golden_words, args.golden_words),
        "",
    ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"k_data: {len(k_words)} words")
    print(f"q_data: {len(q_words)} words")
    print(f"golden_score: {len(golden_words)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
