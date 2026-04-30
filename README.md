# Data Analyst Assistant

Assistente RAG em PT-BR para análise de dados com Python, focado em `pandas`, `numpy`, `matplotlib` e `seaborn`. O sistema combina um backend FastAPI com retrieval sobre documentação indexada e um frontend Next.js.

## Stack

| Camada | Tecnologia |
|---|---|
| Frontend | Next.js 13 + React 18 + Tailwind CSS |
| Backend | FastAPI + ChromaDB |
| Embedder | `intfloat/multilingual-e5-small` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | Groq (70b → 8b) → Cerebras (fallback) |
| Ingestão | trafilatura + chunker customizado |

## Estrutura

```
micro_saas_ods_2/
├── app/                    # Frontend Next.js
│   ├── app/                # App Router (layout, page, globals.css)
│   ├── components/ui/      # Button, Badge, Card, Collapsible, Input
│   ├── hooks/              # useChatHistory
│   ├── lib/                # Utilitários
│   └── next.config.js      # Rewrite /backend/ask → backend:7860/ask
│
├── backend/
│   └── app.py              # FastAPI: GET /health, POST /ask
│
├── src/
│   ├── config.py           # Modelos, paths, hiperparâmetros
│   ├── rag_query.py        # Pipeline: retrieve → rerank → prompt → LLM
│   ├── device.py           # CPU/GPU detection
│   ├── html_extractor.py   # Extração de HTML oficial (trafilatura)
│   ├── medium_extractor.py # Extração de artigos Medium
│   ├── translator.py       # Tradução EN→PT (Gemini/Groq)
│   ├── glossary_repair.py  # Correção de termos Python mal traduzidos
│   ├── indexer.py          # Indexação no ChromaDB
│   └── eval/
│       ├── ragas_evaluator.py  # Avaliação RAG (similaridade + recall + precision)
│       └── ragas_dashboard.py  # Dashboard de resultados
│
├── data/
│   ├── main.py             # Pipeline de ingestão
│   ├── {pandas,numpy,matplotlib,seaborn}/pt/  # HTMLs traduzidos
│   ├── chroma_db/          # Índice vetorial persistente
│   └── eval/               # Dataset e resultados de avaliação
│
├── scripts/
│   ├── smoke_test.py       # Teste rápido do backend
│   └── deploy_space.py     # Deploy no Hugging Face Space
│
├── tests/                  # Testes pytest
├── Dockerfile              # Deploy HF Space
└── pyproject.toml          # Dependências Python (uv)
```

## Variáveis de Ambiente

Crie um `.env` na raiz:

```env
# Obrigatório
GROQ_API_KEY=...

# Opcional
CEREBRAS_API_KEY=...
HF_SPACE_URL=https://seu-space.hf.space
```

## Instalação

```bash
# Dependências Python
uv sync

# Dependências do frontend
cd app && npm install && cd ..
```

## Preparar o índice vetorial

Se `data/chroma_db/` não existir:

```bash
uv run python data/main.py
```

## Rodar localmente

**Terminal 1 — backend:**

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 7860 --reload
```

**Terminal 2 — frontend:**

```bash
cd app && npm run dev
```

Acesse: [http://localhost:3000](http://localhost:3000)

O frontend redireciona `/backend/ask` para `http://127.0.0.1:7860/ask` automaticamente.

## Testes

```bash
# Python
uv run pytest

# Frontend (type-check + build)
cd app && npm run build
```

## Avaliação RAG

```bash
uv run python -m src.eval.ragas_evaluator --sample 50
```

Métricas calculadas localmente (sem chamar LLM externo):
- `answer_semantic_similarity` — similaridade coseno entre resposta e ground truth
- `context_recall` — cobertura de tokens do ground truth nos contextos recuperados
- `context_precision` — similaridade semântica média entre contextos e pergunta

## Deploy

| Destino | Arquivo |
|---|---|
| Backend (HF Space) | `Dockerfile` + `backend/requirements.txt` |
| Frontend (Vercel) | `app/vercel.json` — defina `HF_SPACE_URL` nas env vars da Vercel |
