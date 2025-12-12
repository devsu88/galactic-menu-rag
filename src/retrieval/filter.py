"""Filtro LLM finale per verificare rigorosamente i piatti candidati."""

import logging
import os
import json
from typing import List, Any
from datapizza.core.models import PipelineComponent
from datapizza.clients.openai import OpenAIClient
from src.utils.prompts import VERIFY_DISHES_PROMPT
from src.utils.config import LLM_MODEL

logger = logging.getLogger(__name__)


class DishFilter(PipelineComponent):
    """
    Filtro LLM finale per verificare che i piatti candidati soddisfino esattamente i vincoli della query.
    
    Riceve i chunk dalla ricerca vettoriale e usa un LLM per verificare rigorosamente
    quali piatti soddisfano tutti i requisiti della query (pianeta, ristorante, chef, ingredienti, tecniche).
    """
    
    def __init__(self):
        """Inizializza il componente con il client LLM."""
        super().__init__()
        self.llm_client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=LLM_MODEL
        )
    
    def _run(self, query: str, chunks: List[Any], **kwargs) -> List[str]:
        """
        Filtra i chunk candidati usando un LLM per verificare che soddisfino la query.
        
        Args:
            query: La domanda dell'utente
            chunks: Lista di Node objects dalla ricerca vettoriale con metadata
            **kwargs: Argomenti aggiuntivi per compatibilità
            
        Returns:
            Lista di nomi di piatti che soddisfano esattamente la query
        """
        chunks_count = len(chunks) if chunks else 0
        logger.info(f"[DishFilter] Ricevuti {chunks_count} chunks da filtrare")
        
        if not chunks:
            logger.warning("[DishFilter] Nessun chunk ricevuto, ritorno lista vuota")
            return []
        
        candidates = []
        for i, chunk in enumerate(chunks):
            metadata = getattr(chunk, 'metadata', {}) or {}
            dish_name = metadata.get('dish_name')
            
            if dish_name:
                ingredients = metadata.get('raw_ingredients', [])
                techniques = metadata.get('raw_techniques', [])
                
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
                
                candidate_info = {
                    "name": dish_name,
                    "planet": metadata.get('planet'),
                    "restaurant_name": metadata.get('restaurant_name'),
                    "chef_name": metadata.get('chef_name'),
                    "ingredients": ingredients,
                    "techniques": techniques
                }
                candidates.append(candidate_info)
                logger.debug(f"[DishFilter] Candidato {i+1}: {dish_name} (ingredienti: {len(ingredients)}, tecniche: {len(techniques)})")
        
        if not candidates:
            logger.warning("[DishFilter] Nessun candidato estratto dai chunks")
            return []
        
        logger.info(f"[DishFilter] Estratti {len(candidates)} candidati da {chunks_count} chunks")
        logger.info(f"[DishFilter] Applicazione filtro LLM su {len(candidates)} candidati...")
        verified_names = self._verify_with_llm(query, candidates)
        logger.info(f"[DishFilter] ✓ Filtro LLM completato: {len(verified_names)} piatti verificati su {len(candidates)} candidati")
        logger.debug(f"[DishFilter] Piatti verificati: {verified_names}")
        return verified_names
    
    def _verify_with_llm(self, query: str, candidates: List[dict]) -> List[str]:
        """
        Verifica con LLM quali candidati soddisfano esattamente la query.
        
        Args:
            query: Query originale dell'utente
            candidates: Lista di dizionari con informazioni sui piatti candidati
            
        Returns:
            Lista di nomi di piatti che soddisfano la query
        """
        logger.debug(f"[DishFilter] Invio richiesta LLM per verificare {len(candidates)} candidati")
        logger.debug(f"[DishFilter] Query: '{query}'")
        
        prompt = VERIFY_DISHES_PROMPT(query, candidates)
        
        try:
            logger.debug("[DishFilter] Invocazione LLM...")
            response = self.llm_client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```json", "").replace("```", "").strip()
            verified = json.loads(content)
            logger.debug(f"[DishFilter] Risposta LLM ricevuta: {len(verified)} piatti verificati")
            return verified
        except json.JSONDecodeError as e:
            logger.error(f"[DishFilter] ✗ Errore nel parsing JSON dalla risposta LLM: {e}")
            logger.debug(f"[DishFilter] Contenuto risposta: {content[:200] if 'content' in locals() else 'N/A'}")
            return []
        except Exception as e:
            logger.error(f"[DishFilter] ✗ Errore durante la verifica LLM: {e}")
            return []

