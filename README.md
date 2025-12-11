# Galactic Menu RAG

Sistema RAG (Retrieval-Augmented Generation) per la ricerca di piatti in menu galattici, basato sul framework [datapizza-ai](https://docs.datapizza.ai/).

## Struttura del Progetto

```
galactic-menu-rag/
├── src/
│   ├── main.py                 # CLI entry point
│   ├── ingestion/              # Pipeline di ingestion
│   │   ├── pipeline.py        # Orchestrazione ingestion
│   │   └── parsers.py          # Parser custom per menu PDF (LlamaParse + LLM)
│   ├── retrieval/              # Pipeline di retrieval
│   │   ├── pipeline.py         # Orchestrazione retrieval
│   │   ├── engines.py          # RetrievalPipeline con DagPipeline
│   │   ├── query_filter.py     # Estrazione filtri metadati (Qdrant)
│   │   └── filter.py           # Filtro LLM finale
│   ├── models/
│   │   └── dish.py             # Modello dati Dish
│   └── utils/
│       └── config.py            # Configurazioni globali
├── Dataset/
│   ├── knowledge_base/         # Base di conoscenza
│   │   ├── menu/               # Menu PDF dei ristoranti
│   │   ├── blogpost/           # Blog HTML
│   │   ├── codice_galattico/   # Regolamenti PDF
│   │   └── misc/               # Manuale cucina, distanze CSV
│   ├── domande.csv             # Domande di test
│   └── ground_truth/           # Ground truth e mapping
└── qdrant_storage/             # Storage locale Qdrant
```

## Componenti Principali

### Ingestion Pipeline
- **Parser Custom** (`GalacticMenuParser`): Estrae dati strutturati da PDF menu usando LlamaParse e LLM
- **NodeSplitter**: Divide i nodi in chunk di max 1000 caratteri
- **OpenAI Embedder**: Genera embeddings (text-embedding-3-small)
- **Qdrant Vectorstore**: Memorizza i chunk con metadati (piatto, ristorante, pianeta, chef, ingredienti, tecniche)

### Retrieval Pipeline
- **QueryFilterExtractor**: Estrae filtri espliciti (pianeta, ristorante, chef, ingredienti, tecniche) e ottimizza la query per ricerca semantica
- **Hybrid Search**: 
  1. Ricerca con filtri metadati Qdrant (precisa)
  2. Fallback a ricerca solo semantica se nessun risultato (robusta a typo)
- **DishFilter**: Filtro LLM finale per verifica rigorosa dei candidati

## Utilizzo

### Setup Qdrant
```bash
docker pull qdrant/qdrant
```
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

### Ingestion
```bash
uv run -m src.main ingest --data_dir Dataset/knowledge_base/menu
```

### Retrieval
```bash
# Tutte le domande
uv run -m src.main retrieve --questions_file Dataset/domande.csv

# Solo domande Easy
uv run -m src.main retrieve --questions_file Dataset/domande.csv --difficulty Easy

# Con output personalizzato
uv run -m src.main retrieve --questions_file Dataset/domande.csv --output_file risultati.csv --difficulty Medium
```

## Tecnologie

- **Framework**: datapizza-ai
- **Vector Store**: Qdrant (localhost:6333)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI gpt-4o-mini
- **PDF Parser**: LlamaParse (cloud API)

## Output

- **Risultati**: CSV con `row_id` e `result` (comma-separated dish IDs)
- **Debug**: File JSON intermedi salvati in `.debug/`
