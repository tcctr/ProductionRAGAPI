#!/usr/bin/env python3
"""Split parsed PostgreSQL docs into embedding-sized chunks.

Usage:
    pip install tokenizers
    python chunk_docs.py

Reads data/parsed/docs.jsonl, writes data/chunks/chunks.jsonl.

Strategy:
  1. A page that fits in one chunk stays whole (most sql_command pages).
  2. Otherwise split on markdown headers, then pack sections greedily up to
     --target tokens, preferring to break at section boundaries.
  3. Blocks that are still too big are split by lines; code fences are
     re-opened/closed and table header rows are repeated in each piece.

Every chunk starts with a breadcrumb ("PostgreSQL 18 > CREATE INDEX > Parameters")
so a chunk from the middle of a page still says what it is about. Token counts
use the nomic-embed-text-v1.5 tokenizer and include the "search_document: "
prefix and [CLS]/[SEP], so --max-tokens is the true size the embedding server
sees. llama.cpp's default --ubatch-size is 512; raise it if you raise this.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ANCHOR_RE = re.compile(r"\s*\[#\]\(#([^)]+)\)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
CAPTION_RE = re.compile(r"^\*\*(Table|Example|Figure) [^*]+\*\*$")


@dataclass
class Block:
    """A paragraph, code fence, list or table, tagged with its heading path."""
    text: str
    path: tuple[str, ...]
    anchor: str | None
    starts_section: bool = False


class TokenCounter:
    """Token counter for the embedding model's tokenizer."""

    def __init__(self) -> None:
        self.tok = Tokenizer.from_pretrained(TOKENIZER)
        # [CLS] + [SEP] + prefix
        self.overhead = 2 + len(self.tok.encode(DOC_PREFIX, add_special_tokens=False).ids)

    def __call__(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def clean_heading(raw: str) -> tuple[str, str | None]:
    """'9.3. Math Functions [#](#FUNCTIONS-MATH)' -> ('9.3. Math Functions', 'FUNCTIONS-MATH')."""
    m = ANCHOR_RE.search(raw)
    if m:
        return raw[: m.start()].strip().replace("\xa0", " "), m.group(1)
    return raw.strip().replace("\xa0", " "), None


def to_blocks(text: str, section_title: str) -> list[Block]:
    """Split page markdown into blocks, tracking the heading hierarchy.

    Blank lines separate blocks except inside code fences. Headings become
    their own block so they stay attached to the text that follows.
    """
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    anchor: str | None = None
    buf: list[str] = []
    in_fence = False
    new_section = True

    def path() -> tuple[str, ...]:
        return tuple(t for _, t in stack if t != section_title)

    def flush() -> None:
        nonlocal new_section
        body = "\n".join(buf).strip("\n")
        buf.clear()
        if body.strip():
            blocks.append(Block(body, path(), anchor, new_section))
            new_section = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title, a = clean_heading(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            anchor = a or anchor
            new_section = True
            buf.append(f"{m.group(1)} {title}")
            flush()
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush()

    # Glue headings and table/example captions to the block that follows.
    merged: list[Block] = []
    for b in blocks:
        prev = merged[-1] if merged else None
        if prev and (HEADING_RE.match(prev.text) or CAPTION_RE.match(prev.text)) \
                and "\n" not in prev.text and prev.path == b.path:
            prev.text = f"{prev.text}\n\n{b.text}"
            continue
        merged.append(b)
    return merged


def split_block(block: Block, budget: int, count) -> list[Block]:
    """Split an oversized block by lines, keeping fences and table headers valid."""
    lines = block.text.split("\n")
    header: list[str] = []
    footer: list[str] = []

    # Leading heading / caption lines are repeated in every piece.
    i = 0
    while i < len(lines) and (HEADING_RE.match(lines[i]) or CAPTION_RE.match(lines[i]) or not lines[i].strip()):
        i += 1
    lead = lines[:i]
    body = lines[i:]

    if body and FENCE_RE.match(body[0]):
        header = [body[0]]
        footer = ["```"]
        body = body[1:-1] if len(body) > 1 and FENCE_RE.match(body[-1]) else body[1:]
    elif len(body) >= 2 and body[0].startswith("|") and re.match(r"^\|\s*:?-{3}", body[1]):
        header = body[:2]
        body = body[2:]

    frame = lead + header
    frame_tokens = count("\n".join(frame + footer)) if frame or footer else 0
    room = budget - frame_tokens
    if room < budget // 4:  # giant heading/table header: don't repeat it
        frame, footer, room = [], [], budget
        body = lines

    pieces: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for line in body:
        t = count(line) + 1
        if t > room:  # single enormous line: hard-split by words
            if cur:
                pieces.append(cur)
                cur, cur_tokens = [], 0
            words, part = line.split(" "), []
            for w in words:
                if part and count(" ".join(part + [w])) > room:
                    pieces.append([" ".join(part)])
                    part = []
                part.append(w)
            if part:
                pieces.append([" ".join(part)])
            continue
        if cur and cur_tokens + t > room:
            pieces.append(cur)
            cur, cur_tokens = [], 0
        cur.append(line)
        cur_tokens += t
    if cur:
        pieces.append(cur)

    return [
        Block("\n".join(frame + p + footer), block.path, block.anchor, block.starts_section and n == 0)
        for n, p in enumerate(pieces)
    ]


def breadcrumb(version: str, section_title: str, path: tuple[str, ...]) -> str:
    return " > ".join([f"PostgreSQL {version}", section_title, *path])


def chunk_record(rec: dict, count, target: int, max_tokens: int) -> list[dict]:
    version = rec["version"]
    title = rec["section_title"]
    blocks = to_blocks(rec["text"], title)

    def budget_for(path: tuple[str, ...]) -> int:
        # breadcrumb line + blank line separator
        return max_tokens - count.overhead - count(breadcrumb(version, title, path)) - 2

    # 1. Whole page fits: single chunk.
    whole = "\n\n".join(b.text for b in blocks)
    if count(whole) <= budget_for(()):
        groups = [blocks]
    else:
        # 2. Oversized blocks get split first.
        sized: list[Block] = []
        for b in blocks:
            limit = min(target, budget_for(b.path))
            sized.extend(split_block(b, limit, count) if count(b.text) > limit else [b])

        # 3. Greedy packing, preferring section boundaries.
        groups, cur, cur_tokens = [], [], 0
        for n, b in enumerate(sized):
            t = count(b.text) + 2
            limit = min(target, budget_for(cur[0].path if cur else b.path))
            if cur and b.starts_section:
                # tokens of the whole upcoming section
                section_tokens = t
                for nb in sized[n + 1:]:
                    if nb.starts_section:
                        break
                    section_tokens += count(nb.text) + 2
                if cur_tokens + section_tokens > limit and cur_tokens >= target // 3:
                    groups.append(cur)
                    cur, cur_tokens = [], 0
            if cur and cur_tokens + t > limit:
                groups.append(cur)
                cur, cur_tokens = [], 0
            cur.append(b)
            cur_tokens += t
        if cur:
            groups.append(cur)
        # Fold a tiny trailing piece (e.g. a lone "See Also") into its predecessor.
        if len(groups) > 1:
            tail = sum(count(b.text) + 2 for b in groups[-1])
            prev = sum(count(b.text) + 2 for b in groups[-2])
            if tail < target // 4 and prev + tail <= budget_for(groups[-2][0].path):
                groups[-2].extend(groups.pop())

    out = []
    for idx, group in enumerate(groups):
        path = group[0].path if len(groups) > 1 else ()
        content = breadcrumb(version, title, path) + "\n\n" + "\n\n".join(b.text for b in group)
        anchor = group[0].anchor if len(groups) > 1 else None
        out.append({
            "id": f"{version}:{rec['page']}#{idx}",
            "version": version,
            "doc_type": rec["doc_type"],
            "section_title": title,
            "page": rec["page"],
            "url": rec["url"] + (f"#{anchor}" if anchor else ""),
            "heading_path": list(path),
            "chunk_index": idx,
            "content": content,
            "token_count": count(content) + count.overhead,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/parsed/docs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/chunks/chunks.jsonl"))
    ap.add_argument("--target", type=int, default=450, help="Soft chunk size in tokens")
    ap.add_argument("--max-tokens", type=int, default=512, help="Hard cap incl. prefix + special tokens")
    args = ap.parse_args()

    count = TokenCounter()
    records = [json.loads(line) for line in args.inp.open(encoding="utf-8")]

    chunks: list[dict] = []
    for rec in records:
        chunks.extend(chunk_record(rec, count, args.target, args.max_tokens))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Stats
    by_type: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        by_type[c["doc_type"]].append(c["token_count"])
    per_page = Counter(c["page"] + c["version"] for c in chunks)
    print(f"{len(records)} pages -> {len(chunks)} chunks -> {args.out}")
    print(f"single-chunk pages: {sum(1 for v in per_page.values() if v == 1)}/{len(per_page)}")
    for dt, toks in sorted(by_type.items()):
        toks.sort()
        print(f"  {dt:13} n={len(toks):5}  min={toks[0]:4}  median={toks[len(toks)//2]:4}  "
              f"p90={toks[int(len(toks)*.9)]:4}  max={toks[-1]:4}")
    over = [c["id"] for c in chunks if c["token_count"] > args.max_tokens]
    if over:
        print(f"WARNING: {len(over)} chunks exceed --max-tokens: {over[:5]}")


if __name__ == "__main__":
    main()
