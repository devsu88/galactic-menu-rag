"""Pipeline di ingestion per processare menu PDF e salvarli in Qdrant."""

import logging
import os
import glob
from typing import List
from datapizza.pipeline.pipeline import IngestionPipeline
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.embedders import ChunkEmbedder
from datapizza.modules.splitters import NodeSplitter
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.core.vectorstore import VectorConfig

from src.ingestion.parsers import GalacticMenuParser
from src.utils.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_NAME,
    NODE_SPLITTER_MAX_CHAR
)

logger = logging.getLogger(__name__)


def list_pdf_files(directory: str) -> List[str]:
    """
    Lista tutti i file PDF in una directory.
    
    Args:
        directory: Path della directory da esplorare
        
    Returns:
        Lista di path assoluti dei file PDF trovati
    """
    return glob.glob(os.path.join(directory, "*.pdf"))


def run_ingestion(data_dir: str):
    """
    Esegue la pipeline di ingestion per processare menu PDF.
    
    La pipeline:
    1. Inizializza Qdrant vector store (localhost:6333)
    2. Crea la collection "galactic_menu" se non esiste
    3. Configura i moduli: parser, splitter, embedder
    4. Processa tutti i PDF nella directory specificata
    5. Salva i chunk con embeddings in Qdrant
    
    Args:
        data_dir: Path della directory contenente i file PDF da processare
        
    Raises:
        ValueError: Se OPENAI_API_KEY non è configurata
    """
    logger.info("Inizializzazione Ingestion Pipeline...")
    
    vector_store = QdrantVectorstore(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )
    
    vector_store.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vector_config=[VectorConfig(name=EMBEDDING_NAME, dimensions=EMBEDDING_DIMENSIONS)]
    )
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")
        
    embedder_client = OpenAIEmbedder(model_name=EMBEDDING_MODEL, api_key=api_key)
    
    pipeline = IngestionPipeline(
        modules=[
            GalacticMenuParser(),
            NodeSplitter(max_char=NODE_SPLITTER_MAX_CHAR),
            ChunkEmbedder(client=embedder_client, embedding_name=EMBEDDING_NAME)
        ],
        vector_store=vector_store,
        collection_name=QDRANT_COLLECTION_NAME
    )
    
    files = list_pdf_files(data_dir)
    
    if not files:
        logger.warning(f"Nessun file PDF trovato in {data_dir}")
        return

    logger.info(f"Trovati {len(files)} files da processare.")
    
    for file in files:
        try:
            pipeline.run(file_path=file)
            logger.info(f"Ingested: {os.path.basename(file)}")
        except Exception as e:
            logger.error(f"Failed to ingest {file}: {e}")
            
    logger.info("Ingestion completata con successo.")

