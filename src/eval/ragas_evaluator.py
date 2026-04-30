from __future__ import annotations

from datetime import datetime
import json
import argparse
import re
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

# Importar do projeto
try:
    from .. import config
    from ..rag_query import answer
except ImportError:
    # Fallback para execução direta do script (sem python -m)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src import config
    from src.rag_query import answer

# =========================
# MODO OTIMIZADO / BARATO
# =========================
#
# Alterações:
# - REMOVE métricas RAGAS que usam LLM:
#     ❌ faithfulness
#     ❌ answer_relevancy
#
# - Mantém apenas métricas locais:
#     ✅ semantic similarity
#     ✅ context recall
#     ✅ context precision
#
# - Reduz tamanho dos contextos
# - Reutiliza modelo de embeddings
# - Não chama API extra do RAGAS
#
# Economia:
# ~80-95% menos tokens
# =========================


@dataclass
class EvalSample:
    """Estrutura de uma amostra de avaliação."""

    question: str
    ground_truth: str
    answer: str | None = None
    contexts: list[str] | None = None

    # métricas locais/baratas
    answer_semantic_similarity: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================
# CARREGAR MODELO UMA ÚNICA VEZ
# =========================================================

_EMBED_MODEL = None


def get_embedding_model():
    global _EMBED_MODEL

    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(config.EMBEDDER_MODEL)

    return _EMBED_MODEL


# =========================================================
# DATASET
# =========================================================

def load_eval_dataset(path: Path = None) -> list[dict]:
    """Carrega dataset de avaliação em formato JSONL."""

    if path is None:
        candidates = [
            config.EVAL_DIR / "eval_dataset.jsonl",
            config.EVAL_DIR / "ragas_dataset_template.jsonl",
        ]

        path = None

        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                f"Dataset nao encontrado. Esperado um de: {candidates}"
            )

    if not path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {path}")

    samples = []

    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[!] Erro JSON na linha {line_num}: {e}")

            break

        except (UnicodeDecodeError, UnicodeError):
            continue

    if not samples:
        raise ValueError(f"Nenhuma amostra valida foi carregada de {path}")

    return samples


# =========================================================
# UTIL
# =========================================================

def compress_text(text: str) -> str:
    """Normaliza espaços. Truncamento removido para preservar contexto nas métricas."""
    if not text:
        return ""
    return " ".join(text.split())


# =========================================================
# PIPELINE RAG
# =========================================================

def run_rag_pipeline(
    question: str,
    variant: str = "new",
) -> tuple[str, list[str]]:
    """
    Executa o pipeline RAG e retorna resposta + contextos.
    """

    try:
        result = answer(question, variant=variant)

        answer_text = result.get("answer", "")

        # Limitar quantidade e tamanho dos contextos
        retrieved_chunks = result.get("retrieved_chunks", [])[:3]

        contexts = [
            compress_text(chunk["content"])
            for chunk in retrieved_chunks
        ]

        return answer_text, contexts

    except Exception as e:
        print(f"[!] Erro ao processar pergunta: {e}")
        return "", []


# =========================================================
# MÉTRICAS LOCAIS
# =========================================================

def calculate_semantic_similarity(
    answer_text: str,
    ground_truth: str,
) -> float:
    """
    Similaridade semântica local (sem API).
    """

    try:
        from sentence_transformers import util

        model = get_embedding_model()

        emb1 = model.encode(answer_text, convert_to_tensor=True)
        emb2 = model.encode(ground_truth, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(emb1, emb2).item()

        return float(similarity)

    except Exception as e:
        print(f"[!] Erro ao calcular similaridade: {e}")
        return 0.0


def calculate_context_recall(
    ground_truth: str,
    contexts: list[str],
) -> float:
    """
    Recall baseado em similaridade semântica (embeddings).
    """
    if not contexts:
        return 0.0

    try:
        from sentence_transformers import util
        import numpy as np

        model = get_embedding_model()

        # Codifica o ground truth e os contextos
        gt_emb = model.encode(ground_truth, convert_to_tensor=True)
        ctx_embs = model.encode(contexts, convert_to_tensor=True)

        # Calcula similaridade de cosseno entre o ground truth e todos os contextos
        similarities = util.pytorch_cos_sim(gt_emb, ctx_embs)[0].cpu().numpy()

        # O context recall será a similaridade do contexto mais relevante retornado
        return float(np.max(similarities))

    except Exception as e:
        print(f"[!] Erro ao calcular context recall: {e}")
        return 0.0

def calculate_context_precision(
    question: str,
    contexts: list[str],
) -> float:
    """
    Precision semântica local.
    """

    if not contexts:
        return 0.0

    try:
        from sentence_transformers import util

        model = get_embedding_model()

        q_emb = model.encode(question, convert_to_tensor=True)

        ctx_embs = model.encode(
            contexts,
            convert_to_tensor=True,
        )

        similarities = util.pytorch_cos_sim(
            q_emb,
            ctx_embs
        )[0].cpu().numpy()

        return float(np.mean(similarities))

    except Exception as e:
        print(f"[!] Erro ao calcular context precision: {e}")
        return 0.0


# =========================================================
# EVALUATE SAMPLE
# =========================================================

def evaluate_sample(
    sample: dict,
    variant: str = "new",
) -> EvalSample:
    """
    Avaliação BARATA.
    NÃO usa métricas RAGAS com LLM.
    """

    question = sample["question"]
    ground_truth = sample["ground_truth"]

    answer_text, contexts = run_rag_pipeline(
        question,
        variant=variant,
    )

    eval_sample = EvalSample(
        question=question,
        ground_truth=ground_truth,
        answer=answer_text,
        contexts=contexts,
    )

    # ====================================
    # MÉTRICAS LOCAIS (SEM TOKENS)
    # ====================================

    eval_sample.answer_semantic_similarity = (
        calculate_semantic_similarity(
            answer_text,
            ground_truth,
        )
    )

    eval_sample.context_recall = (
        calculate_context_recall(
            ground_truth,
            contexts,
        )
    )

    eval_sample.context_precision = (
        calculate_context_precision(
            question,
            contexts,
        )
    )

    return eval_sample


# =========================================================
# PRINT SAMPLE
# =========================================================

def print_sample_results(
    sample: EvalSample,
    idx: int,
) -> None:

    print(f"\n{'=' * 80}")
    print(f"AMOSTRA {idx}")
    print(f"{'=' * 80}")

    print(f"[?] Pergunta:")
    print(sample.question)

    print(f"\n[Reference] Ground Truth:")
    print(sample.ground_truth[:300])

    print(f"\n[Bot] Resposta:")
    print((sample.answer or "")[:300])

    print(f"\n[Docs] Contextos:")
    for i, ctx in enumerate(sample.contexts or [], 1):
        print(f"  [{i}] {ctx[:150]}...")

    print(f"\n[Metrics]")
    print(
        f"  - Similaridade Semantica: "
        f"{sample.answer_semantic_similarity:.3f}"
    )

    print(
        f"  - Context Recall: "
        f"{sample.context_recall:.3f}"
    )

    print(
        f"  - Context Precision: "
        f"{sample.context_precision:.3f}"
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(samples: list[EvalSample]) -> None:

    print(f"\n\n{'=' * 80}")
    print("RESUMO AGREGADO")
    print(f"{'=' * 80}")

    metrics_names = [
        "answer_semantic_similarity",
        "context_recall",
        "context_precision",
    ]

    for metric in metrics_names:

        values = [
            getattr(s, metric)
            for s in samples
            if getattr(s, metric) is not None
        ]

        if not values:
            print(f"\n{metric}: N/A")
            continue

        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"\n  {metric}:")
        print(f"    - Media: {mean:.3f}")
        print(f"    - Desvio Padrao: {std:.3f}")
        print(f"    - Minimo: {min_val:.3f}")
        print(f"    - Maximo: {max_val:.3f}")

    print(f"\n  Total de amostras avaliadas: {len(samples)}")


# =========================================================
# SAVE
# =========================================================

def save_results(
    samples: list[EvalSample],
    output_path: Path,
) -> None:

    results = {
    "metadata": {
        "total_samples": len(samples),
        "variant": "new",
        "mode": "optimized_local_metrics",
        "timestamp": datetime.now().strftime("%d/%m/%Y às %H:%M"),
    },
        "samples": [
            s.to_dict()
            for s in samples
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[OK] Resultados salvos em: {output_path}")


# =========================================================
# MAIN
# =========================================================

def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Avaliação RAG barata"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Numero maximo de amostras",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ragas_results_optimized.json",
        help="Arquivo de saída",
    )

    parser.add_argument(
        "--variant",
        choices=["new", "baseline"],
        default="new",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = parser.parse_args(argv)

    # ====================================
    # DATASET
    # ====================================

    print("[*] Carregando dataset...")

    dataset = load_eval_dataset()

    if args.sample:
        dataset = dataset[:args.sample]

    print(f"[OK] {len(dataset)} amostras carregadas")

    # ====================================
    # AVALIAÇÃO
    # ====================================

    print(f"\n[*] Iniciando avaliacao...")

    samples = []

    for i, sample_data in enumerate(
        tqdm(dataset, desc="Avaliando"),
        1,
    ):

        sample = evaluate_sample(
            sample_data,
            variant=args.variant,
        )

        samples.append(sample)

        if not args.quiet:
            print_sample_results(sample, i)

    # ====================================
    # RESUMO
    # ====================================

    print_summary(samples)

    # ====================================
    # SAVE
    # ====================================

    output_path = Path(args.output)

    save_results(samples, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())