"""Engine di ricerca ibrida per il sistema RAG."""

import logging
import os
import traceback
from typing import List
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from src.retrieval.filter import DishFilter
from src.retrieval.query_filter import QueryFilterExtractor
from src.utils.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_DEFAULT_TOP_K
)

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Pipeline di retrieval usando DagPipeline di Datapizza AI.
    
    Implementa una ricerca ibrida con fallback:
    1. Ricerca con filtri metadati Qdrant (precisa ma sensibile a typo)
    2. Fallback a ricerca solo semantica se nessun risultato (robusta a typo)
    """
    
    def __init__(self):
        """
        Inizializza la pipeline di retrieval.
        
        Configura i moduli e costruisce il DAG:
        - QueryFilterExtractor: estrae filtri e ottimizza query
        - Embedder: genera embeddings della query
        - Retriever: ricerca in Qdrant
        - DishFilter: filtro LLM finale
        
        Raises:
            ValueError: Se OPENAI_API_KEY non è configurata
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")
        
        self.embedder = OpenAIEmbedder(
            model_name=EMBEDDING_MODEL,
            api_key=api_key
        )
        
        self.retriever = QdrantVectorstore(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION_NAME
        
        self.query_filter_extractor = QueryFilterExtractor()
        self.filter = DishFilter()
        
        self.dag = DagPipeline()
        self.dag.add_module("query_filter_extractor", self.query_filter_extractor)
        self.dag.add_module("embedder", self.embedder)
        self.dag.add_module("retriever", self.retriever)
        self.dag.add_module("filter", self.filter)
        
        self.dag.connect("query_filter_extractor", "embedder", target_key="text", source_key="search_query")
        self.dag.connect("query_filter_extractor", "retriever", target_key="query_filter", source_key="query_filter")
        self.dag.connect("embedder", "retriever", target_key="query_vector")
        self.dag.connect("retriever", "filter", target_key="chunks")
        self.dag.connect("query_filter_extractor", "filter", target_key="query", source_key="query")
    
    def search(self, query: str, top_k: int = RETRIEVAL_DEFAULT_TOP_K) -> List[str]:
        """
        Esegue la ricerca usando DagPipeline con fallback.
        
        Strategia:
        1. Prova prima con filtri metadati Qdrant (precisa ma sensibile a typo)
        2. Se nessun risultato, fallback a ricerca solo semantica (robusta a typo)
        
        Args:
            query: La domanda dell'utente
            top_k: Numero massimo di risultati da recuperare
        
        Returns:
            Lista di nomi di piatti che soddisfano la query
        """
        try:
            logger.info(f"[RetrievalPipeline] Inizio ricerca per query: '{query}'")
            logger.info(f"[RetrievalPipeline] Parametri: top_k={top_k}, collection={self.collection_name}")
            
            logger.info("[RetrievalPipeline] STEP 1: Esecuzione DAG con filtri metadati...")
            result_with_filter = self.dag.run({
                "query_filter_extractor": {"query": query},
                "retriever": {
                    "collection_name": self.collection_name,
                    "k": top_k
                }
            })
            
            chunks_with_filter = result_with_filter.get("retriever", [])
            final_result_with_filter = result_with_filter.get("filter", [])
            filter_info = result_with_filter.get("query_filter_extractor", {})
            
            chunks_count = len(chunks_with_filter) if isinstance(chunks_with_filter, list) else 0
            logger.info(f"[RetrievalPipeline] STEP 1 completato: {chunks_count} chunks recuperati da Qdrant")
            logger.info(f"[RetrievalPipeline] STEP 1: {len(final_result_with_filter)} piatti dopo filtro LLM")
            
            if final_result_with_filter and len(final_result_with_filter) > 0:
                logger.info(f"[RetrievalPipeline] ✓ Ricerca con filtri metadati riuscita: {len(final_result_with_filter)} piatti trovati")
                logger.debug(f"[RetrievalPipeline] Piatti trovati: {final_result_with_filter}")
                return final_result_with_filter
            
            logger.warning("[RetrievalPipeline] ✗ Nessun risultato con filtri metadati, attivazione fallback...")
            logger.info("[RetrievalPipeline] STEP 2: Ricerca solo semantica (senza filtri metadati)...")
            
            search_query = filter_info.get("search_query", query)
            logger.info(f"[RetrievalPipeline] Query ottimizzata per ricerca semantica: '{search_query}'")
            
            logger.debug("[RetrievalPipeline] Generazione embedding della query...")
            query_vector = self.embedder.embed(search_query)
            logger.debug(f"[RetrievalPipeline] Embedding generato: dimensione {len(query_vector)}")
            
            logger.info(f"[RetrievalPipeline] Ricerca in Qdrant (collection={self.collection_name}, k={top_k}, senza filtri)...")
            chunks_without_filter = self.retriever.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=None,
                k=top_k
            )
            
            chunks_count = len(chunks_without_filter) if isinstance(chunks_without_filter, list) else 0
            logger.info(f"[RetrievalPipeline] STEP 2 completato: {chunks_count} chunks recuperati da ricerca semantica")
            
            original_query = filter_info.get("query", query)
            logger.info(f"[RetrievalPipeline] Applicazione filtro LLM finale su {chunks_count} chunks...")
            final_result_without_filter = self.filter._run(
                query=original_query,
                chunks=chunks_without_filter
            )
            
            logger.info(f"[RetrievalPipeline] STEP 2: {len(final_result_without_filter)} piatti dopo filtro LLM")
            logger.info(f"[RetrievalPipeline] ✓ Ricerca semantica completata: {len(final_result_without_filter)} piatti trovati")
            logger.debug(f"[RetrievalPipeline] Piatti trovati: {final_result_without_filter}")
            return final_result_without_filter
            
        except Exception as e:
            logger.error(f"[RetrievalPipeline] ✗ Errore durante la ricerca: {e}")
            logger.debug(traceback.format_exc())
            return []


class HybridSearchEngine:
    """
    Wrapper per mantenere compatibilità con il codice esistente.
    
    Internamente usa RetrievalPipeline basato su DagPipeline per implementare
    la ricerca ibrida con fallback.
    """
    
    def __init__(self):
        """Inizializza il search engine con una nuova RetrievalPipeline."""
        self.pipeline = RetrievalPipeline()

    def search(self, query: str, top_k: int = RETRIEVAL_DEFAULT_TOP_K) -> List[str]:
        """
        Esegue ricerca ibrida e restituisce i nomi dei piatti.
        
        Args:
            query: Query dell'utente
            top_k: Numero massimo di risultati
            
        Returns:
            Lista di nomi di piatti che soddisfano la query
        """
        return self.pipeline.search(query, top_k)
