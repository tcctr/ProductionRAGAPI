"""Answer generation: a cited answer from the retrieved chunks, written by an LLM.

Talks to any OpenAI-compatible chat server, e.g. llama.cpp:
    llama-server -m <model>.gguf --port 8082 -c 16384 --reasoning off

Settings (environment variables):
    LLM_URL      default http://localhost:8082/v1/chat/completions
    LLM_MODEL    model name sent with each request (llama-server serves one model and ignores it)
    LLM_TIMEOUT  seconds to wait for an answer, default 120
"""
import os

import requests

from app.observability import note
from app.versions import VERSIONS

LLM_URL = os.getenv("LLM_URL", "http://localhost:8082/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "local")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

# Low temperature: answers should stick to the sources, not be creative.
TEMPERATURE = 0.2
MAX_ANSWER_TOKENS = 1024

SYSTEM_PROMPT = """\
You answer questions about PostgreSQL using only the numbered documentation excerpts in the user's message.

- Use only facts stated in the excerpts. Do not add facts from memory.
- Cite the excerpts that support each statement with their numbers, like [1] or [2][3].
- Each excerpt starts with the PostgreSQL versions it applies to. If the question names a version, or \
the excerpts differ between versions, say which version each statement applies to.
- If the excerpts do not answer the question, say that the provided documentation does not cover it. \
Do not guess.
- Be concise. Put SQL in markdown code blocks."""

# Added to SYSTEM_PROMPT when the question compares versions and a term search came with the excerpts.
VERSION_RULES = """
- Before the excerpts, a term search lists which versions' documentation contains key terms from the \
question. It searched all of the documentation provided to you, not just the excerpts, so you may state \
its results as facts; cite them as (term search).
- The documentation covers PostgreSQL 16, 17 and 18 only. If something is documented in a version but \
not in the earlier ones covered, say it was added in that version. If it is already in 16, say it is \
available in 16, 17 and 18 and may be older.
- A term being present does not mean the feature asked about is: MERGE and RETURNING both exist in 16, \
but MERGE ... RETURNING may not. Check the excerpts for the feature itself."""


def format_source(n: int, chunk: dict) -> str:
    """One numbered excerpt. A chunk merged across versions gets all of them in its breadcrumb:
    its text says "PostgreSQL 18 > ..." but it applies to every version in chunk["versions"]."""
    content = chunk["content"]
    versions = ", ".join(str(v) for v in chunk["versions"])
    own = f"PostgreSQL {chunk['version']}"
    if content.startswith(own):
        content = f"PostgreSQL {versions}" + content[len(own):]
    return f"[{n}] {content}"


def format_presence(presence: dict[str, list[int]]) -> str:
    """The term search as text: "- `casefold`: in 18; not in 16, 17"."""
    lines = []
    for term, found in presence.items():
        missing = [v for v in VERSIONS if v not in found]
        parts = []
        if found:
            parts.append("in " + ", ".join(map(str, found)))
        if missing:
            parts.append("not in " + ", ".join(map(str, missing)))
        lines.append(f"- `{term}`: {'; '.join(parts)}")
    return "\n".join(lines)


def format_sources(chunks: list[dict], presence: dict[str, list[int]] | None = None) -> str:
    """The numbered excerpts, preceded by the term search when there is one."""
    sources = "\n\n---\n\n".join(format_source(n, c) for n, c in enumerate(chunks, start=1))
    if presence:
        sources = (f"Term search (exact text, all PostgreSQL 16, 17 and 18 documentation provided):\n"
                   f"{format_presence(presence)}\n\n---\n\n{sources}")
    return sources


def build_messages(question: str, chunks: list[dict],
                   presence: dict[str, list[int]] | None = None) -> list[dict]:
    """Chat messages for the LLM. Excerpts are numbered 1..k in the order of chunks, so a
    citation [n] in the answer refers to chunks[n - 1]."""
    system = SYSTEM_PROMPT + (VERSION_RULES if presence else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Documentation excerpts:\n\n{format_sources(chunks, presence)}"
                                    f"\n\n---\n\nQuestion: {question}"},
    ]


def generate(question: str, chunks: list[dict], presence: dict[str, list[int]] | None = None) -> str:
    """Answer question from chunks (and the term search when comparing versions).
    Raises requests.RequestException or ValueError on failure."""
    resp = requests.post(LLM_URL, timeout=LLM_TIMEOUT, json={
        "model": LLM_MODEL,
        "messages": build_messages(question, chunks, presence),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_ANSWER_TOKENS,
        # Qwen-style models think before answering unless told not to; with the answer already
        # in the excerpts that only adds latency. Servers that don't know this field ignore it.
        "chat_template_kwargs": {"enable_thinking": False},
    })
    resp.raise_for_status()
    try:
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"unexpected LLM response: {resp.text[:200]}") from e
    usage = data.get("usage") or {}
    note(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
    if not answer or not answer.strip():
        raise ValueError("LLM returned an empty answer")
    return answer.strip()
