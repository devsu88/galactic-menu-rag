"""Prompt templates centralizzati per il sistema RAG."""

from typing import List


def EXTRACT_STRUCTURED_DATA_PROMPT(text: str) -> str:
    """
    Genera il prompt per l'estrazione di dati strutturati da menu PDF.
    
    Args:
        text: Testo estratto dal menu PDF
        
    Returns:
        Prompt formattato per l'LLM
    """
    return f"""
        Sei un assistente che estrae dati strutturati da menu di ristoranti galattici.
        
        Analizza attentamente il seguente testo e individua per ciascun piatto tutte le informazioni che lo riguardano e riportale ESATTAMENTE come sono scritti:
        - Nome del piatto
        - Descrizione
        - Pianeta
        - Chef
        - Licenze dello chef
        - Ingredienti
        - Tecniche utilizzate nella preparazione del piatto

        Restituisci un JSON valido con questa struttura ESATTA:
        {{
            "restaurant": {{
                "name": "Nome Ristorante",
                "planet": "Nome Pianeta",
                "chef": {{ "name": "Nome Chef", "licenses": ["LTK III", "P V"] }}
            }},
            "dishes": [
                {{
                    "name": "Nome Piatto",
                    "description": "Descrizione del piatto",
                    "ingredients": ["Ingrediente 1", "Ingrediente 2"],
                    "techniques": ["Tecnica 1", "Tecnica 2"]
                }}
            ]
        }}
        
        TESTO MENU:
        {text}
        """


def EXTRACT_FILTERS_PROMPT(query: str, planets_list: str) -> str:
    """
    Genera il prompt per l'estrazione di filtri espliciti dalla query.
    
    Supporta sia filtri positivi (IN) che negativi (OUT) per ingredienti e tecniche.
    
    Args:
        query: Query dell'utente
        planets_list: Lista dei pianeti disponibili (formattata come stringa)
        
    Returns:
        Prompt formattato per l'LLM
    """
    return f"""
        Analizza la seguente domanda e identifica se contiene filtri espliciti e precisi.
        Restituisci un JSON con questa struttura:
        {{
            "planet": "Nome del pianeta, se menzionato esplicitamente e presente nella lista dei pianeti ({planets_list}), altrimenti null",
            "restaurant_name": "Nome del ristorante, se menzionato esplicitamente, altrimenti null",
            "chef_name": "Nome dello chef se menzionato esplicitamente, altrimenti null",
            "ingredients_in": "Lista degli ingredienti che DEVONO essere presenti (es. ['Ingrediente 1', 'Ingrediente 2']), altrimenti null",
            "ingredients_out": "Lista degli ingredienti che NON devono essere presenti (es. ['Ingrediente 1']), altrimenti null. Cerca frasi come 'senza', 'non contiene', 'non impiegare', 'non usare', 'escludere'",
            "techniques_in": "Lista delle tecniche che DEVONO essere presenti (es. ['Tecnica 1', 'Tecnica 2']), altrimenti null",
            "techniques_out": "Lista delle tecniche che NON devono essere presenti (es. ['Tecnica 1']), altrimenti null. Cerca frasi come 'senza', 'non usa', 'non impiegare', 'non utilizza', 'escludere'"
        }}
        
        IMPORTANTE: 
        - Estrai ingredienti e tecniche SOLO se sono menzionati esplicitamente e precisamente nella domanda.
        - Per ingredienti/tecniche IN: cerca frasi come "con", "usando", "che contiene", "che impiega", "che utilizza"
        - Per ingredienti/tecniche OUT: cerca frasi come "senza", "non contiene", "non impiegare", "non usare", "non utilizza", "escludere"
        - Per ingredienti: estrai il nome esatto (compreso di caratteri speciali)
        - Per tecniche: estrai il nome esatto (compreso di caratteri speciali)
        - Se un ingrediente/tecnica è menzionato senza indicazione esplicita di inclusione/esclusione, mettilo in IN
        
        Esempi:
        - "piatti con X" -> X in ingredients_in
        - "piatti senza X" -> X in ingredients_out
        - "piatti usando tecnica Y" -> Y in techniques_in
        - "piatti senza tecnica Y" -> Y in techniques_out
        
        Domanda: "{query}"
        
        Restituisci SOLO il JSON, senza altro testo.
        """


def EXTRACT_SEARCH_QUERY_PROMPT(query: str) -> str:
    """
    Genera il prompt per l'ottimizzazione della query per ricerca semantica.
    
    Args:
        query: Query originale dell'utente
        
    Returns:
        Prompt formattato per l'LLM
    """
    return f"""
        Analizza la seguente domanda e crea una query ottimizzata per la ricerca semantica di piatti.
        
        La query deve contenere:
        - Nomi di ingredienti menzionati
        - Nomi di tecniche menzionate
        - Informazioni essenziali (pianeta, ristorante, chef se menzionati)
        
        IMPORTANTE: Se la domanda chiede ingredienti e/o tecniche specifici, crea una query che includa:
        - Il nome esatto dell'ingrediente e/o tecnica (compreso anche di caratteri speciali)
        - La parola "piatto", "ingrediente" e/o "tecnica" per contesto
        - Esempi: "Piatto con ingrediente X", "Piatto con tecnica Y"
        
        Domanda originale: "{query}"
        
        Restituisci SOLO la query ottimizzata, senza altro testo o spiegazioni.
        """


def VERIFY_DISHES_PROMPT(query: str, candidates: List[dict]) -> str:
    """
    Genera il prompt per la verifica rigorosa dei piatti candidati.
    
    Args:
        query: Query originale dell'utente
        candidates: Lista di dizionari contenenti informazioni sui piatti candidati
        
    Returns:
        Prompt formattato per l'LLM
    """
    import json
    candidates_json = json.dumps(candidates, indent=2, ensure_ascii=False)
    
    return f"""
        Sei un giudice culinario rigoroso. Il tuo scopo è quello di selezionare i piatti che soddisfano ESATTAMENTE la richiesta dell'utente.
        Analizza la domanda per individuare le informazioni rilevanti:
        - Pianeta
        - Ristorante
        - Chef
        - Ingredienti
        - Tecniche
        
        Domanda Utente: "{query}"
        
        IMPORTANTE: 
        - Se la domanda menziona un pianeta specifico, cerca tutti i piatti che hanno quel pianeta (planet);
        - Se la domanda menziona un ristorante specifico, cerca tutti i piatti che hanno quel ristorante (restaurant_name);
        - Se la domanda menziona un chef specifico, cerca tutti i piatti che hanno quel chef (chef_name);
        - Se la domanda menziona un ingrediente che DEVE essere presente (es. "con X", "che contiene X"), cerca quell'ingrediente ESATTO nella lista degli ingredienti del piatto (ingredients); ignora descrizioni aggiuntive, concentrati solo sul nome dell'ingrediente.
        - Se la domanda menziona un ingrediente che NON deve essere presente (es. "senza X", "non contiene X"), verifica che quell'ingrediente NON sia nella lista degli ingredienti del piatto.
        - Se la domanda menziona una tecnica che DEVE essere presente (es. "usando Y", "con tecnica Y"), cerca quella tecnica ESATTA nella lista delle tecniche del piatto (techniques); ignora descrizioni aggiuntive, concentrati solo sul nome della tecnica.
        - Se la domanda menziona una tecnica che NON deve essere presente (es. "senza tecnica Y", "non usa Y"), verifica che quella tecnica NON sia nella lista delle tecniche del piatto.
        
        Ecco una lista di piatti candidati (con pianeta, ristorante, chef, ingredienti e tecniche):
        {candidates_json}
        
        Compito: Restituisci una lista JSON di stringhe contenente SOLO i nomi dei piatti che soddisfano ESATTAMENTE la richiesta dell'utente.
        Se nessun piatto soddisfa la richiesta, restituisci una lista vuota [].
        
        Output format: ["Nome Piatto A", "Nome Piatto B"]
        """

