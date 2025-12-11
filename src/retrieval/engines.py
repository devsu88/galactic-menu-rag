import os
from typing import List
from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from src.retrieval.filter import DishFilter
from src.retrieval.query_filter import QueryFilterExtractor

class RetrievalPipeline:
    """
    Pipeline di retrieval usando DagPipeline di Datapizza AI.
    """
    def __init__(self):
        # Setup componenti
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")
        
        self.embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key=api_key
        )
        
        self.retriever = QdrantVectorstore(host="localhost", port=6333)
        self.collection_name = "galactic_menu"
        
        self.query_filter_extractor = QueryFilterExtractor()
        self.filter = DishFilter()
        
        # Costruisci DagPipeline
        self.dag = DagPipeline()
        self.dag.add_module("query_filter_extractor", self.query_filter_extractor)
        self.dag.add_module("embedder", self.embedder)
        self.dag.add_module("retriever", self.retriever)
        self.dag.add_module("filter", self.filter)
        
        # Connetti i moduli
        # query_filter_extractor -> embedder (search_query ottimizzata per ricerca semantica)
        self.dag.connect("query_filter_extractor", "embedder", target_key="text", source_key="search_query")
        # query_filter_extractor -> retriever (query_filter) - passa il filtro Qdrant
        self.dag.connect("query_filter_extractor", "retriever", target_key="query_filter", source_key="query_filter")
        # embedder -> retriever (query_vector)
        self.dag.connect("embedder", "retriever", target_key="query_vector")
        # retriever -> filter (chunks)
        self.dag.connect("retriever", "filter", target_key="chunks")
        # query_filter_extractor -> filter (query)
        self.dag.connect("query_filter_extractor", "filter", target_key="query", source_key="query")
    
    def search(self, query: str, top_k: int = 50) -> List[str]:
        """
        Esegue la ricerca usando DagPipeline con fallback:
        1. Prima prova con filtri metadati (precisi ma possono fallire con typo)
        2. Se non trova risultati, prova solo ricerca semantica senza filtri
        
        Args:
            query: La domanda dell'utente
            top_k: Numero massimo di risultati da recuperare
        
        Returns:
            Lista di nomi di piatti che soddisfano la query
        """
        try:
            print(f"[DEBUG] Query: {query}")
            
            # STEP 1: Prova prima con filtri metadati
            result_with_filter = self.dag.run({
                "query_filter_extractor": {"query": query},
                "retriever": {
                    "collection_name": self.collection_name,
                    "k": top_k
                }
            })
            
            # Controlla se ci sono risultati con filtro
            chunks_with_filter = result_with_filter.get("retriever", [])
            final_result_with_filter = result_with_filter.get("filter", [])
            
            print(f"[DEBUG] Ricerca con filtri metadati:")
            print(f"  - Chunks trovati: {len(chunks_with_filter) if isinstance(chunks_with_filter, list) else 'N/A'}")
            print(f"  - Piatti finali: {len(final_result_with_filter)}")
            
            # Risultati con filtri metadati
            if final_result_with_filter and len(final_result_with_filter) > 0:
                print(f"[DEBUG] Trovati {len(final_result_with_filter)} piatti con filtri metadati")
                return final_result_with_filter
            
            # STEP 2: Fallback - ricerca solo semantica senza filtri
            print(f"[DEBUG] Nessun risultato con filtri metadati, provo ricerca solo semantica...")
            
            # Estrai solo la query ottimizzata per la ricerca semantica
            filter_info = result_with_filter.get("query_filter_extractor", {})
            search_query = filter_info.get("search_query", query)
            
            # Crea un nuovo DAG run senza filtri (passando None come query_filter)
            # Per fare questo, dobbiamo eseguire manualmente i passaggi senza filtri
            query_vector = self.embedder.embed(search_query)
            
            # Ricerca senza filtri
            chunks_without_filter = self.retriever.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=None,  # Nessun filtro
                k=top_k
            )
            
            print(f"[DEBUG] Ricerca solo semantica:")
            print(f"  - Chunks trovati: {len(chunks_without_filter) if isinstance(chunks_without_filter, list) else 'N/A'}")
            
            if chunks_without_filter and len(chunks_without_filter) > 0:
                print(f"[DEBUG] Trovati {len(chunks_without_filter)} piatti con ricerca semantica")
            
            # Applica il filtro LLM finale sui risultati senza filtri metadati
            original_query = filter_info.get("query", query)
            final_result_without_filter = self.filter._run(
                query=original_query,
                chunks=chunks_without_filter
            )
            
            print(f"[DEBUG] Piatti finali dopo filtro LLM: {len(final_result_without_filter)}")
            return final_result_without_filter
            
            # total_result = final_result_with_filter + final_result_without_filter
            # print(f"[DEBUG] Totale risultati: {len(total_result)}")
            # return total_result
            
        except Exception as e:
            print(f"Error in retrieval pipeline: {e}")
            import traceback
            traceback.print_exc()
            return []


class HybridSearchEngine:
    """
    Wrapper per mantenere compatibilità con il codice esistente.
    Internamente usa RetrievalPipeline basato su DagPipeline.
    """
    def __init__(self):
        self.pipeline = RetrievalPipeline()

    def search(self, query: str, top_k: int = 50) -> List[str]:
        """
        Esegue ricerca semantica e poi filtering puntuale via LLM.
        Restituisce i nomi dei piatti.
        """
        return self.pipeline.search(query, top_k)
