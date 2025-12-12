"""Pipeline di retrieval per processare domande e generare risultati."""

import logging
import pandas as pd
import json
import os
import csv
from src.retrieval.engines import HybridSearchEngine
from src.utils.config import DISH_MAPPING_PATH, DEBUG_DIR

logger = logging.getLogger(__name__)


def run_retrieval(questions_file: str, output_file: str, difficulty: str = None):
    """
    Esegue la pipeline di retrieval per processare domande e generare risultati.
    
    Il processo:
    1. Carica il mapping nomi piatti -> IDs
    2. Inizializza il search engine
    3. Legge le domande dal CSV
    4. Filtra per difficoltà se specificata
    5. Per ogni domanda: ricerca piatti, mappa a IDs, salva risultati
    6. Salva risultati finali in CSV e debug in JSON
    
    Args:
        questions_file: Path del file CSV con le domande
        output_file: Path del file CSV di output con i risultati
        difficulty: Opzionale, filtra domande per difficoltà (Easy, Medium, Hard, Impossible)
    """
    logger.info("Caricamento mapping piatti...")
    
    if not os.path.exists(DISH_MAPPING_PATH):
        logger.error(f"File mapping {DISH_MAPPING_PATH} non trovato.")
        return

    with open(DISH_MAPPING_PATH, "r") as f:
        dish_mapping = json.load(f)
    
    logger.info("Inizializzazione Search Engine...")
    engine = HybridSearchEngine()
    
    logger.info(f"Leggendo domande da {questions_file}...")
    df = pd.read_csv(questions_file)
    
    df['original_index'] = df.index
    
    if difficulty:
        original_count = len(df)
        df = df[df['difficoltà'] == difficulty].copy()
        filtered_count = len(df)
        logger.info(f"Filtrate {original_count} domande per difficoltà '{difficulty}': {filtered_count} domande selezionate")
        if filtered_count == 0:
            logger.error(f"Nessuna domanda trovata con difficoltà '{difficulty}'")
            return
    
    results = []
    debug_results = []
    
    total_questions = len(df)
    logger.info(f"Inizio elaborazione {total_questions} domande...")

    for index, row in df.iterrows():
        question = row['domanda']
        row_id = int(row['original_index']) + 1
        
        logger.info("=" * 80)
        logger.info(f"[{row_id}/{total_questions}] Elaborazione domanda {row_id}")
        logger.info(f"Domanda: '{question}'")
        logger.info("=" * 80)
        
        found_names = engine.search(question)
        logger.info(f"[Pipeline] Ricerca completata: {len(found_names)} piatti trovati")
        logger.debug(f"[Pipeline] Nomi piatti trovati: {found_names}")
        
        debug_results.append({
            "row_id": row_id,
            "question": question,
            "found_dishes": found_names
        })
        
        logger.info(f"[Pipeline] Mapping nomi piatti -> IDs...")
        found_ids = []
        unmapped_names = []
        for name in found_names:
            dish_id = dish_mapping.get(name)
            
            if dish_id is not None:
                found_ids.append(dish_id)
                logger.debug(f"[Pipeline] ✓ '{name}' -> ID {dish_id}")
            else:
                unmapped_names.append(name)
                logger.warning(f"[Pipeline] ✗ Piatto '{name}' trovato dall'LLM ma non nel mapping.")

        if unmapped_names:
            logger.warning(f"[Pipeline] {len(unmapped_names)} piatti non mappati: {unmapped_names}")

        found_ids = sorted(list(set(found_ids)))
        result_str = ",".join(map(str, found_ids))
        
        logger.info(f"[Pipeline] Risultato finale: {len(found_ids)} IDs unici -> '{result_str}'")
        
        if not result_str:
            logger.warning(f"[Pipeline] ⚠ Nessun risultato per domanda {row_id}")

        results.append({
            "row_id": row_id,
            "result": result_str
        })
        
        logger.info(f"[Pipeline] Domanda {row_id} completata\n")

    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_filename = os.path.join(DEBUG_DIR, "retrieval_results.json")
    with open(debug_filename, 'w', encoding='utf-8') as f:
        json.dump(debug_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Risultati di debug salvati in {debug_filename}")
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Elaborazione completata. Risultati salvati in {output_file}")

