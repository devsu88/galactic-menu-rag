import pandas as pd
import json
import os
import csv
from src.retrieval.engines import HybridSearchEngine

from src.utils.config import DISH_MAPPING_PATH, DEBUG_DIR

def run_retrieval(questions_file: str, output_file: str, difficulty: str = None):
    print(f"Caricamento mapping piatti...")
    
    if not os.path.exists(DISH_MAPPING_PATH):
        print(f"Errore: File mapping {DISH_MAPPING_PATH} non trovato.")
        return

    with open(DISH_MAPPING_PATH, "r") as f:
        dish_mapping = json.load(f)
    
    print("Inizializzazione Search Engine...")
    engine = HybridSearchEngine()
    
    print(f"Leggendo domande da {questions_file}...")
    df = pd.read_csv(questions_file)
    
    # Salva l'indice originale (corrisponde alla riga nel CSV, 0-based)
    df['original_index'] = df.index
    
    # Filtra per difficoltà se specificata
    if difficulty:
        original_count = len(df)
        df = df[df['difficoltà'] == difficulty].copy()
        filtered_count = len(df)
        print(f"Filtrate {original_count} domande per difficoltà '{difficulty}': {filtered_count} domande selezionate")
        if filtered_count == 0:
            print(f"Errore: Nessuna domanda trovata con difficoltà '{difficulty}'")
            return
    
    results = []
    debug_results = []  # Per salvare i risultati prima del mapping
    
    total_questions = len(df)
    print(f"Inizio elaborazione {total_questions} domande...")

    for index, row in df.iterrows():
        question = row['domanda']
        # row_id deve corrispondere alla riga originale nel CSV (1-based)
        row_id = int(row['original_index']) + 1
        
        print(f"[{row_id}/{total_questions}] Processing: {question}...")
        
        # Search
        found_names = engine.search(question)
        print(f"\n\n\n[DEBUG] Found names: {found_names}\n\n\n")
        
        # Salva i risultati prima del mapping per debug
        debug_results.append({
            "row_id": row_id,
            "question": question,
            "found_dishes": found_names
        })
        
        # Map to IDs
        found_ids = []
        for name in found_names:
            # Try exact match
            dish_id = dish_mapping.get(name)
            
            # Fallback: Try case insensitive or partial match if strict fails?
            # Per l'MVP manteniamo strict match su quanto restituito dall'LLM
            # L'LLM ha visto i nomi candidati che vengono dal DB, quindi dovrebbe restituire quei nomi esatti.
            
            if dish_id is not None:
                found_ids.append(dish_id)
            else:
                print(f"  Warning: Piatto '{name}' trovato dall'LLM ma non nel mapping.")

        # Format Result
        # Sort numeric IDs
        found_ids = sorted(list(set(found_ids))) # Deduplicate and sort
        result_str = ",".join(map(str, found_ids))
        
        # Fallback se vuoto (il requisito dice "non può essere vuoto")
        if not result_str:
            print(f"  Warning: Nessun risultato per domanda {row_id}. Metto default placeholder (gestire meglio in prod).")
            # Per ora lasciamo vuoto o mettiamo un placeholder?
            # Il prompt dice: "Si assume che esista sempre almeno un piatto". 
            # Se il nostro sistema non lo trova, è un errore di retrieval.
            pass 

        results.append({
            "row_id": row_id,
            "result": result_str
        })

    # Salva i risultati di debug (prima del mapping)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_filename = os.path.join(DEBUG_DIR, "retrieval_results.json")
    with open(debug_filename, 'w', encoding='utf-8') as f:
        json.dump(debug_results, f, indent=2, ensure_ascii=False)
    print(f"Risultati di debug salvati in {debug_filename}")
    
    # Save Output
    output_df = pd.DataFrame(results)
    # Usa QUOTE_NONNUMERIC per quotare sempre le stringhe (result) ma non i numeri (row_id)
    output_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Elaborazione completata. Risultati salvati in {output_file}")

