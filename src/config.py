"""Centralized configuration: model names, paths, hyperparameters.

Loaded once at import time. Values can be overridden by environment variables
(prefix GROQ_ or RAG_) for experimentation without code changes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
MEDIUM_DIR = DATA_DIR / "medium"
EVAL_DIR = DATA_DIR / "eval"
TRANSLATIONS_CACHE = DATA_DIR / "translations_cache.jsonl"

# --- Chroma collections ---
COLLECTION_BASELINE_EN = "rag_chunks_baseline_en"
COLLECTION_NEW_PT = "rag_chunks_pt"

# --- Models ---
EMBEDDER_MODEL = os.environ.get("RAG_EMBEDDER", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.environ.get("RAG_RERANKER", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
BASELINE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Groq (RAG answer generation + optional translation fallback) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# llama-3.1-8b-instant: ~6× higher free-tier token quota than 70b + significantly
# faster inference. Quality is adequate for context-constrained documentation RAG.
# Switch back to "llama-3.3-70b-versatile" via RAG_LLM_MODEL env var if needed.
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_LLM_FAST = os.environ.get("GROQ_LLM_MODEL_FAST", "llama-3.1-8b-instant")

# --- Cerebras (fallback LLM when Groq rate-limits) ---
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
CEREBRAS_LLM_MODEL = os.environ.get("CEREBRAS_LLM_MODEL", "llama-3.3-70b")
# Translation uses the fast/cheap model — Groq free tier has ~5x more TPD on
# 8b-instant vs 70b, and translation on a glossary-constrained task is already
# very high with 8b. The 70b stays as the default for RAG answer generation.
GROQ_TRANSLATION_MODEL = os.environ.get("GROQ_TRANSLATION_MODEL", GROQ_LLM_FAST)

# --- Gemini (Google AI Studio, free tier) — default translation provider ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# gemini-2.0-flash: 1M TPD / 1.5k RPD / 15 RPM free, excellent PT-BR quality.
GEMINI_TRANSLATION_MODEL = os.environ.get("GEMINI_TRANSLATION_MODEL", "gemini-2.0-flash")

# Translation provider: "gemini" or "groq". Defaults to gemini when GOOGLE_API_KEY
# is available (better quality + higher free quota), falls back to groq otherwise.
TRANSLATION_PROVIDER = os.environ.get(
    "TRANSLATION_PROVIDER",
    "gemini" if GOOGLE_API_KEY else "groq",
).lower()

# --- Ingestion: libraries that skip the EN→PT path ---
# Every lib now has rich PT Google-translated coverage (pandas 47, numpy 31,
# matplotlib 38, seaborn 14), and the glossary_repair module fixes API-term
# mistranslations post-extraction. Translating the /en subset would only add
# duplicated content and burn LLM tokens. Skip them all by default.
SKIP_EN_FOR_LIBRARIES = tuple(
    x.strip()
    for x in os.environ.get(
        "RAG_SKIP_EN_LIBS", "pandas,numpy,matplotlib,seaborn"
    ).split(",")
    if x.strip()
)

# --- Retrieval ---
# TOP_K_RETRIEVE: candidates fetched from ChromaDB before reranking.
# Reduced 15→10 to halve cross-encoder CPU time (main latency bottleneck).
TOP_K_RETRIEVE = 10
# TOP_K_RERANK: chunks sent to the LLM as context.
# Reduced 5→3 saves ~380 input tokens per request (~31% total token reduction).
TOP_K_RERANK = 3

# --- Chunking ---
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP_WORDS = 80
NOISE_THRESHOLD = 0.5

# --- Libraries tracked ---
LIBRARIES = ("pandas", "numpy", "matplotlib", "seaborn")
