#!/usr/bin/env python3
"""Fetch selected PostgreSQL docs sections and parse them into JSONL.

Usage:
    pip install requests beautifulsoup4 markdownify
    python fetch_parse_docs.py --contact you@example.com

Raw HTML is cached in data/raw/<version>/<page>.html (never re-fetched).
Parsed records go to data/parsed/docs.jsonl, one record per page.
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urldefrag

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

BASE = "https://www.postgresql.org/docs/{version}/"

# doc_type -> (root pages, optional filename prefix filter for followed links)
SECTIONS = {
    "sql_command": (["sql-commands.html"], "sql-"),
    "functions": (["functions.html"], None),
    "indexes_perf": (["indexes.html", "performance-tips.html"], None),
}


def fetch(url: str, path: Path, session: requests.Session, delay: float) -> str:
    """Return HTML for url, using the on-disk cache when present."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    time.sleep(delay)
    return resp.text


def find_links(html: str, prefix: str | None) -> list[str]:
    """Collect same-directory .html links from a root page's TOC."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#docContent") or soup
    scope = content.select("div.toc") or [content]
    names: list[str] = []
    for block in scope:
        for a in block.find_all("a", href=True):
            href, _ = urldefrag(a["href"])
            if not href.endswith(".html") or "/" in href or href.startswith("http"):
                continue
            if prefix and not href.startswith(prefix):
                continue
            if href not in names:
                names.append(href)
    return names


def parse_page(html: str) -> tuple[str, str]:
    """Return (title, markdown text) for one docs page."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#docContent")
    if content is None:
        return "", ""
    for sel in ("div.navheader", "div.navfooter", "div.toc", "script", "style"):
        for el in content.select(sel):
            el.decompose()
    heading = content.find(["h1", "h2"])
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    text = markdownify(str(content), heading_style="ATX")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True, help="email for the User-Agent")
    ap.add_argument("--versions", nargs="+", default=["16", "17", "18"])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/parsed/docs.jsonl")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = f"rag-portfolio-crawler ({args.contact})"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    chars: Counter = Counter()
    short: list[str] = []

    with out_path.open("w", encoding="utf-8") as out:
        for version in args.versions:
            base = BASE.format(version=version)
            raw_dir = Path(args.raw_dir) / version
            for doc_type, (roots, prefix) in SECTIONS.items():
                pages: list[str] = []
                for root in roots:
                    root_html = fetch(base + root, raw_dir / root, session, args.delay)
                    pages.append(root)
                    pages += [p for p in find_links(root_html, prefix) if p not in pages]
                for page in pages:
                    html = fetch(base + page, raw_dir / page, session, args.delay)
                    title, text = parse_page(html)
                    if len(text) < 200:
                        short.append(f"{version}/{page}")
                    record = {
                        "version": version,
                        "doc_type": doc_type,
                        "section_title": title,
                        "page": page,
                        "url": base + page,
                        "text": text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[(version, doc_type)] += 1
                    chars[(version, doc_type)] += len(text)

    print(f"{'version':<8}{'doc_type':<15}{'pages':>7}{'chars':>12}")
    for key in sorted(counts):
        print(f"{key[0]:<8}{key[1]:<15}{counts[key]:>7}{chars[key]:>12}")
    if short:
        print(f"\nWarning: {len(short)} pages under 200 chars, check these:")
        for name in short[:20]:
            print("  ", name)


if __name__ == "__main__":
    main()
