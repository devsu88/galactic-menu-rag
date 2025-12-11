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

def list_pdf_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, "*.pdf"))

def run_ingestion(data_dir: str):
    print("Inizializzazione Ingestion Pipeline...")
    
    # 2. Setup Vector Store
    # Qdrant locale su localhost (default port 6333)
    vector_store = QdrantVectorstore(
        host="localhost",
        port=6333
    )
    
    # Create collection if not exists (dimensione embedding ada-002/3-small è 1536)
    vector_store.create_collection(
        collection_name="galactic_menu",
        vector_config=[VectorConfig(name="default", dimensions=1536)]
    )
    
    # 3. Setup Pipeline
    # Usiamo NodeSplitter per gestire descrizioni lunghe e ChunkEmbedder per l'efficienza
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")
        
    embedder_client = OpenAIEmbedder(model_name="text-embedding-3-small", api_key=api_key)
    
    pipeline = IngestionPipeline(
        modules=[
            GalacticMenuParser(),
            NodeSplitter(max_char=1000), # Divide nodi troppo grandi
            ChunkEmbedder(client=embedder_client, embedding_name="default") # Wrapper per batching
        ],
        vector_store=vector_store,
        collection_name="galactic_menu"
    )
    
    # 4. Get Files
    files = list_pdf_files(data_dir)
    
    if not files:
        print(f"Nessun file PDF trovato in {data_dir}")
        return

    print(f"Trovati {len(files)} files da processare.")
    
    # 5. Run Pipeline    
    for file in files:
        try:
            pipeline.run(file_path=file)
            print(f"Ingested: {os.path.basename(file)}")
        except Exception as e:
            print(f"Failed to ingest {file}: {e}")
            
    print("Ingestion completata con successo.")

