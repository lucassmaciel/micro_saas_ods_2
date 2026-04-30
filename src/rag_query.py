"""Query orchestrator: retrieve → rerank → Groq generation.

Exposes:
  - ``answer(question, variant="new") -> {"answer", "citations", "retrieved_chunks", "variant"}``
    for programmatic use.
  - CLI: ``python -m src.rag_query "question" [--variant baseline]``

Variants:
  - ``"new"`` (default): PT pipeline (e5-small + mmarco + PT prompt, rag_chunks_pt).
  - ``"baseline"``: EN pipeline (MiniLM + ms-marco + EN prompt, rag_chunks_baseline_en).
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config
from src.device import get_device


_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_collection = None
_loaded_for_variant: str | None = None


def _load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=get_device())


def _load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=get_device())


def _load_collection(collection_name: str):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(collection_name)


def _stack_for_variant(variant: str) -> tuple[str, str, str]:
    """Return (embedder_model, reranker_model, collection_name) for a variant."""
    if variant == "new":
        return (config.EMBEDDER_MODEL, config.RERANKER_MODEL,
                config.COLLECTION_NEW_PT)
    if variant == "baseline":
        return (config.BASELINE_EMBEDDER, config.BASELINE_RERANKER,
                config.COLLECTION_BASELINE_EN)
    raise ValueError(f"unknown variant {variant!r}; expected 'new' or 'baseline'")


def _ensure_loaded(variant: str) -> None:
    """Lazy-load models/collection once per variant; swap them when variant changes."""
    global _embedder, _reranker, _collection, _loaded_for_variant
    if _loaded_for_variant == variant and _embedder and _reranker and _collection:
        return
    emb_name, rr_name, coll_name = _stack_for_variant(variant)
    _embedder = _load_embedder(emb_name)
    _reranker = _load_reranker(rr_name)
    _collection = _load_collection(coll_name)
    _loaded_for_variant = variant


def retrieve(question: str, top_k: int = config.TOP_K_RETRIEVE,
             variant: str = "new",
             query_prefix: str = "query: ") -> list[dict]:
    _ensure_loaded(variant)
    q_emb = _embedder.encode(
        [f"{query_prefix}{question}"], normalize_embeddings=True
    ).tolist()
    res = _collection.query(query_embeddings=q_emb, n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]


def rerank(question: str, chunks: list[dict],
           top_n: int = config.TOP_K_RERANK,
           variant: str = "new") -> list[dict]:
    if not chunks:
        return []
    _ensure_loaded(variant)
    pairs = [[question, c["content"]] for c in chunks]
    raw = _reranker.predict(pairs)
    scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    ranked = sorted(zip(scores, chunks), key=lambda t: -t[0])
    return [c for _, c in ranked[:top_n]]


_PT_SYSTEM = (
    "Você é um assistente PT-BR especializado em análise de dados com Python "
    "(pandas, numpy, matplotlib, seaborn). "
    "Use o CONTEXTO fornecido como fonte principal. "
    "Para perguntas básicas (instalação, o que é a biblioteca, conceitos gerais) "
    "que não estejam no contexto, responda com base no seu conhecimento geral "
    "dessas bibliotecas. "
    "Para perguntas sobre APIs ou parâmetros específicos sem embasamento no "
    "contexto, diga 'Não encontrei essa informação na documentação indexada.' "
    "Nunca invente assinaturas de métodos ou parâmetros não mencionados no "
    "contexto. Seja conciso — máximo 3 parágrafos. Tom didático para iniciantes. "
    "Mantenha nomes de métodos/classes em inglês (ex.: DataFrame.merge, np.array)."
)


def build_pt_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        header = f"[{i}] ({meta.get('library','?')} · {meta.get('section','?')})"
        blocks.append(f"{header}\n{c['content']}")
    contexto = "\n\n".join(blocks)
    return (
        f"{_PT_SYSTEM}\n\n"
        f"=== CONTEXTO ===\n{contexto}\n\n"
        f"=== PERGUNTA ===\n{question}\n\n"
        f"=== RESPOSTA (em português) ==="
    )


def format_citations(chunks: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in chunks:
        meta = c.get("metadata", {})
        key = (meta.get("source"), meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": meta.get("source"),
            "section": meta.get("section"),
            "library": meta.get("library"),
        })
    return out


_EN_SYSTEM = (
    "You are an assistant specialized in Python data analysis (pandas, numpy, "
    "matplotlib, seaborn). Answer ONLY from the CONTEXT provided. If the "
    "answer is not in the context, say 'I could not find that in the indexed "
    "documentation.' Do not invent APIs, parameters, or errors. Be concise "
    "and direct — beginner-friendly tone. Answer in English."
)


def build_en_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        header = f"[{i}] ({meta.get('library','?')} · {meta.get('section','?')})"
        blocks.append(f"{header}\n{c['content']}")
    contexto = "\n\n".join(blocks)
    return (
        f"{_EN_SYSTEM}\n\n"
        f"=== CONTEXT ===\n{contexto}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER (in English) ==="
    )


def generate_answer(prompt: str, model: str = config.GROQ_LLM_MODEL) -> str:
    from groq import Groq
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,  # was 800 — saves ~300 output tokens/request
    )
    return resp.choices[0].message.content.strip()


def _generate_cerebras(prompt: str, model: str = config.CEREBRAS_LLM_MODEL) -> str:
    from cerebras.cloud.sdk import Cerebras
    if not config.CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    client = Cerebras(api_key=config.CEREBRAS_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def generate_answer_with_fallback(prompt: str) -> str:
    """Cascade: Groq 70b → Groq 8b → Cerebras 8b, falling back on any error."""
    try:
        return generate_answer(prompt, model=config.GROQ_LLM_MODEL)
    except Exception:
        pass
    try:
        return generate_answer(prompt, model=config.GROQ_LLM_FAST)
    except Exception:
        pass
    return _generate_cerebras(prompt)


def answer(question: str,
           variant: str = "new",
           top_k: int = config.TOP_K_RETRIEVE,
           top_n: int = config.TOP_K_RERANK,
           model: str = config.GROQ_LLM_MODEL) -> dict[str, Any]:
    """Full pipeline: retrieve → rerank → prompt → Groq → package.

    ``variant``:
      - ``"new"`` (default): PT pipeline (e5-small + mmarco + PT prompt, rag_chunks_pt).
      - ``"baseline"``: EN pipeline (MiniLM + ms-marco + EN prompt, rag_chunks_baseline_en).
    """
    retrieved = retrieve(question, top_k=top_k, variant=variant)
    reranked = rerank(question, retrieved, top_n=top_n, variant=variant)
    if variant == "new":
        prompt = build_pt_prompt(question, reranked)
    else:
        prompt = build_en_prompt(question, reranked)
    llm_text = generate_answer(prompt, model=model)
    return {
        "answer": llm_text,
        "citations": format_citations(reranked),
        "retrieved_chunks": reranked,
        "variant": variant,
    }


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Pergunte ao RAG.")
    parser.add_argument("question")
    parser.add_argument("--variant", choices=["new", "baseline"], default="new")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_RETRIEVE)
    parser.add_argument("--top-n", type=int, default=config.TOP_K_RERANK)
    parser.add_argument("--model", default=config.GROQ_LLM_MODEL)
    args = parser.parse_args(argv)

    result = answer(
        args.question, variant=args.variant,
        top_k=args.top_k, top_n=args.top_n, model=args.model,
    )
    header = "RESPOSTA" if args.variant == "new" else "ANSWER"
    cites_header = "CITAÇÕES" if args.variant == "new" else "CITATIONS"
    print(f"\n=== {header} ({args.variant}) ===")
    print(result["answer"])
    print(f"\n=== {cites_header} ===")
    for c in result["citations"]:
        print(f"  - {c['library']} · {c['section']} ({c['source']})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
