# the golden set: questions written from known chunks, so the answer key comes free.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.http import models

from config import CORPORA, IndexConfig, require_env
from index import Encoders, get_client, with_qdrant_retry
from retrieval import CONTENT_KEY, KnowledgeBase, chunk_key

GOLDENS_PATH = Path(__file__).resolve().parents[1] / "evals" / "goldens.jsonl"
GENERATOR_MODEL = "gpt-4o"

# the constraint is gpt-4o's tokens-per-minute ceiling, not concurrency: an unpaced batch
# spends a minute's budget in seconds and the 429 loses every question generated so far.
# pacing is derived from the ceiling rather than hardcoded, so a tier change is one edit.
GENERATOR_TPM = 450_000            # this account's gpt-4o limit
TOKENS_PER_PROMPT = 900            # a ~700-token chunk plus instructions, rounded up
# half the ceiling. at 80 prompts this is not the binding constraint — latency and
# max_concurrency are — and it only starts to bite if PER_CORPUS or CORPORA grows.
GENERATOR_RPS = GENERATOR_TPM / TOKENS_PER_PROMPT / 60 / 2

# below this a chunk is usually a heading, a nav stub or a license header — nothing a
# real question could be about, and a question written from one measures nothing.
MIN_CHARS = 400
PER_CORPUS = 20

# a golden set generated from chunks is biased toward lexical overlap: the model writes
# the question while looking at the passage, so it reuses the passage's own words and
# BM25 wins by construction. the fix is not to suppress the bias but to make it a
# variable — two styles per chunk, every metric reported per style. "hybrid helps" is
# then a claim about a kind of query rather than an average over an accident.
STYLES: dict[str, str] = {
    "literal": (
        "Write the question as a developer who already knows the API would type it: "
        "use the exact identifiers, error strings, flags or class names the chunk uses."
    ),
    "conceptual": (
        "Write the question as someone who does NOT know the API names would ask it: "
        "describe the goal or the symptom in plain words. Do not reuse the chunk's "
        "distinctive identifiers, class names or error strings verbatim."
    )
}

PROMPT = """You are writing one evaluation question for a documentation search system.

Below is a single chunk of documentation. Write one question that this chunk answers.

{style}

Rules:
- Answerable from this chunk alone.
- Self-contained: no "this document", "the passage", "it", or any reference to the
  chunk itself. It has to make sense typed into a search box by someone who has never
  seen this text.
- One sentence. No preamble, no quotes, no numbering. Return only the question.

CHUNK:
{chunk}"""

@dataclass(frozen=True) # fields may not be assigned to after instance creation
class Golden:
    """one question and the chunk it was written from."""
    question: str
    style: str
    chunk: str              # chunk_key of the gold passage — the answer key
    source: str             # cfg.base
    chunk_fingerprint: str  # the chunking this key is valid under
    head: str               # first line of the chunk, so the file is readable

def sample_chunks(
    client, alias: str, n: int, oversample: int = 4
) -> list[str]:
    """random points from a live collection, filtered down to ones worth asking about.

    Qdrant samples server-side, so this does not scroll the whole collection. the
    sample is not reproducible from a seed — the saved file is the artifact, not the
    procedure that produced it.
    """
    points = with_qdrant_retry(lambda: client.query_points(
        alias,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        limit=n * oversample,
        with_payload=True
    )).points

    seen: set[str] = set()
    texts: list[str] = []
    for point in points:
        text = (point.payload or {}).get(CONTENT_KEY, "")
        if len(text) < MIN_CHARS or chunk_key(text) in seen:
            continue
        seen.add(chunk_key(text))
        texts.append(text)
    return texts[:n]

def generate(texts: list[str], cfg: IndexConfig) -> list[Golden]:
    """one call per (chunk, style), batched and paced. gpt-4o at temperature 0."""
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI

    # max_bucket_size=1 so an idle gap cannot accrue credit and spend it as a burst,
    # which is the shape of the failure the limiter exists to prevent.
    llm = ChatOpenAI(
        model=GENERATOR_MODEL,
        temperature=0,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=GENERATOR_RPS, max_bucket_size=1
        )
    )
    jobs = [(text, style) for text in texts for style in STYLES]
    prompts = [
        PROMPT.format(style=STYLES[style], chunk=text) for text, style in jobs
    ]
    replies = llm.batch(prompts, config={"max_concurrency": 8})

    return [
        Golden(
            question=reply.content.strip().strip('"'),
            style=style,
            chunk=chunk_key(text),
            source=cfg.base,
            chunk_fingerprint=cfg.chunk_fingerprint,
            head=" ".join(text.split())[:120]
        )
        for (text, style), reply in zip(jobs, replies)
    ]

def write_goldens(goldens: list[Golden], path: Path = GOLDENS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for golden in goldens:
            handle.write(json.dumps(asdict(golden)) + "\n")
    return path

def load_goldens(path: Path = GOLDENS_PATH) -> list[Golden]:
    if not path.exists():
        raise FileNotFoundError(
            f"no golden set at {path}. build one with `python scripts/goldens.py`."
        )
    with path.open() as handle:
        return [Golden(**json.loads(line)) for line in handle if line.strip()]

def stale_sources(goldens: list[Golden], configs) -> dict[str, str]:
    """corpora whose chunking has moved since their questions were written.

    an n_docs bump or a model swap leaves the answer keys valid; a resplit does not,
    because the gold chunk's text — and so its key — no longer exists in the index.
    that is the whole reason the check is against chunk_fingerprint and not the alias.
    """
    current = {cfg.base: cfg.chunk_fingerprint for cfg in configs}
    return {
        golden.source: golden.chunk_fingerprint
        for golden in goldens
        if current.get(golden.source) != golden.chunk_fingerprint
    }

def main() -> None:
    load_dotenv()
    require_env("QDRANT_URL", "QDRANT_KEY", "OPENAI_API_KEY")

    encoders = Encoders.for_config(CORPORA[0])
    kb = KnowledgeBase(get_client(), CORPORA, encoders).ensure_indexes()

    goldens: list[Golden] = []
    for cfg in CORPORA:
        texts = sample_chunks(kb.client, cfg.alias, PER_CORPUS)
        print(f"-> {cfg.base}: sampled {len(texts)} chunks from '{cfg.alias}'")
        goldens += generate(texts, cfg)
        print(f"-> {cfg.base}: wrote {len(texts) * len(STYLES)} questions")

    path = write_goldens(goldens)
    print(f"\n-> {len(goldens)} questions -> {path}")
    print("   read them before you trust them: a question you would never type is a "
          "question the metric should not count.")

if __name__ == "__main__":
    main()
