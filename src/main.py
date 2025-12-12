"""Entry point principale per il sistema RAG Galactic Menu."""

import argparse
import logging
import os
from dotenv import load_dotenv

from src.utils.config import OUTPUT_DIR, setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def main():
    """
    Entry point principale del sistema RAG.
    
    Gestisce i comandi CLI per ingestion e retrieval:
    - ingest: Esegue la pipeline di ingestion dei menu PDF
    - retrieve: Esegue la pipeline di retrieval per rispondere alle domande
    """
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Galactic Menu RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Comandi disponibili")

    ingest_parser = subparsers.add_parser("ingest", help="Esegue la pipeline di ingestion")
    ingest_parser.add_argument("--data_dir", type=str, required=True, help="Path della cartella Knowledge Base")

    retrieve_parser = subparsers.add_parser("retrieve", help="Esegue la pipeline di retrieval")
    retrieve_parser.add_argument("--questions_file", type=str, required=True, help="Path del file domande.csv")
    retrieve_parser.add_argument("--output_file", type=str, help="Path del file di output risultati.csv", default=os.path.join(OUTPUT_DIR, "risultati.csv"))
    retrieve_parser.add_argument("--difficulty", type=str, choices=["Easy", "Medium", "Hard", "Impossible"], 
                                help="Filtra le domande per difficoltà (Easy, Medium, Hard, Impossible). Se non specificato, processa tutte le domande.")

    args = parser.parse_args()

    if args.command == "ingest":
        from src.ingestion.pipeline import run_ingestion
        logger.info(f"Avvio Ingestion da: {args.data_dir}")
        run_ingestion(args.data_dir)
        
    elif args.command == "retrieve":
        from src.retrieval.pipeline import run_retrieval
        difficulty = args.difficulty if hasattr(args, 'difficulty') else None
        if difficulty:
            logger.info(f"Avvio Retrieval per: {args.questions_file} (difficoltà: {difficulty})")
        else:
            logger.info(f"Avvio Retrieval per: {args.questions_file} (tutte le difficoltà)")
        run_retrieval(args.questions_file, args.output_file, difficulty=difficulty)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

