"""Configurazioni globali per il progetto."""

import logging
import sys
from typing import List

# Directory
DEBUG_DIR = ".debug"
DISH_MAPPING_PATH = "Dataset/ground_truth/dish_mapping.json"
OUTPUT_DIR = ".output"

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "galactic_menu"

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_NAME = "default"

# LLM
LLM_MODEL = "gpt-4o-mini"

# LlamaParse
LLAMAPARSE_LANGUAGE = "it"

# NodeSplitter
NODE_SPLITTER_MAX_CHAR = 1000

# Retrieval
RETRIEVAL_DEFAULT_TOP_K = 50

# Pianeti disponibili
PLANETS: List[str] = [
    "Pandora",
    "Ego",
    "Cybertron",
    "Montressor",
    "Krypton",
    "Namecc",
    "Klyntar",
    "Asgard",
    "Tatooine",
    "Arrakis"
]


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configura il logger principale del progetto.
    
    Args:
        level: Livello di logging (default: INFO)
        
    Returns:
        Logger configurato
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)