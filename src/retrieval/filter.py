import os
import json
from typing import List, Any
from datapizza.core.models import PipelineComponent
from datapizza.clients.openai import OpenAIClient

class DishFilter(PipelineComponent):
    """
    Modulo custom per filtrare i risultati della ricerca usando LLM.
    Verifica che i piatti candidati soddisfino esattamente i vincoli della query.
    """
    def __init__(self):
        super().__init__()
        self.llm_client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )
    
    def _run(self, query: str, chunks: List[Any], **kwargs) -> List[str]:
        """
        Input:
            - query: La domanda dell'utente
            - chunks: Lista di risultati dalla ricerca vettoriale (Node objects con metadata)
        Output:
            - Lista di nomi di piatti che soddisfano la query
        """
        print(f"[DEBUG DishFilter] Ricevuti {len(chunks) if chunks else 0} chunks")
        
        if not chunks:
            print("[DEBUG DishFilter] Nessun chunk ricevuto, ritorno lista vuota")
            return []
        
        # Estrai informazioni dai chunks (Node objects)
        candidates = []
        for i, chunk in enumerate(chunks):
            # I chunks sono Node objects con metadata
            metadata = getattr(chunk, 'metadata', {}) or {}
            dish_name = metadata.get('dish_name')
            
            # print(f"[DEBUG DishFilter] Chunk {i}: dish_name={dish_name}, metadata_keys={list(metadata.keys()) if metadata else []}")
            
            if dish_name:
                # Estrai ingredienti e tecniche dai metadati
                ingredients = metadata.get('raw_ingredients', [])
                techniques = metadata.get('raw_techniques', [])
                
                # Se raw_ingredients non è una lista, prova a parsarlo
                if not isinstance(ingredients, list):
                    if isinstance(ingredients, str):
                        ingredients = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
                    else:
                        ingredients = []
                
                if not isinstance(techniques, list):
                    if isinstance(techniques, str):
                        techniques = [t.strip() for t in techniques.split(',') if t.strip()]
                    else:
                        techniques = []
                
                candidates.append({
                    "name": dish_name,
                    "planet": metadata.get('planet'),
                    "restaurant_name": metadata.get('restaurant_name'),
                    "chef_name": metadata.get('chef_name'),
                    "ingredients": ingredients,
                    "techniques": techniques
                })
                # print(f"[DEBUG DishFilter] Aggiunto candidato: {dish_name} con {len(ingredients)} ingredienti")
        
        if not candidates:
            print("[DEBUG DishFilter] Nessun candidato estratto dai chunks")
            return []
        
        print(f"[DEBUG DishFilter] Totale candidati: {len(candidates)}")
        # Usa LLM per verificare quali candidati soddisfano la query
        verified_names = self._verify_with_llm(query, candidates)
        print(f"[DEBUG DishFilter] Piatti verificati: {verified_names}")
        return verified_names
    
    def _verify_with_llm(self, query: str, candidates: List[dict]) -> List[str]:
        prompt = f"""
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
        - Se la domanda menziona un ingrediente specifico, cerca quell'ingrediente ESATTO nella lista degli ingredienti del piatto (ingredients); ignora descrizioni aggiuntive, concentrati solo sul nome dell'ingrediente.
        - Se la domanda menziona una tecnica specifica, cerca quella tecnica ESATTA nella lista delle tecniche del piatto (techniques); ignora descrizioni aggiuntive, concentrati solo sul nome della tecnica.
        
        Ecco una lista di piatti candidati (con pianeta, ristorante, chef, ingredienti e tecniche):
        {json.dumps(candidates, indent=2, ensure_ascii=False)}
        
        Compito: Restituisci una lista JSON di stringhe contenente SOLO i nomi dei piatti che soddisfano ESATTAMENTE la richiesta dell'utente.
        Se nessun piatto soddisfa la richiesta, restituisci una lista vuota [].
        
        Output format: ["Nome Piatto A", "Nome Piatto B"]
        """
        
        try:
            # print(f"[DEBUG DishFilter] Prompt: {prompt}")
            response = self.llm_client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception:
            return []

