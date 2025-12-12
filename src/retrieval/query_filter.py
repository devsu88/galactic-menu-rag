"""Modulo per estrarre filtri espliciti dalle query e costruire filtri Qdrant."""

import logging
import os
import json
from typing import Dict, Optional, List, Any
from datapizza.core.models import PipelineComponent
from datapizza.clients.openai import OpenAIClient
from qdrant_client import models
from src.utils.prompts import EXTRACT_FILTERS_PROMPT, EXTRACT_SEARCH_QUERY_PROMPT
from src.utils.config import LLM_MODEL, PLANETS

logger = logging.getLogger(__name__)


class QueryFilterExtractor(PipelineComponent):
    """
    Estrae filtri espliciti dalle query e costruisce filtri Qdrant.
    
    Analizza la query per identificare:
    - Pianeta
    - Ristorante
    - Chef
    - Ingredienti IN (devono essere presenti)
    - Ingredienti OUT (non devono essere presenti)
    - Tecniche IN (devono essere presenti)
    - Tecniche OUT (non devono essere presenti)
    
    Supporta filtri negativi per gestire query come:
    - "piatti senza ingrediente X"
    - "piatti che non usano tecnica Y"
    
    Genera anche una query ottimizzata per la ricerca semantica.
    """
    
    def __init__(self):
        """Inizializza il componente con il client LLM."""
        super().__init__()
        self.llm_client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=LLM_MODEL
        )

    def _get_planet_name(self) -> str:
        """
        Restituisce la lista dei pianeti disponibili come stringa formattata.
        
        Returns:
            Stringa con i pianeti separati da virgola
        """
        return ", ".join(PLANETS)
    
    def _run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Analizza la query e estrae filtri espliciti.
        
        Args:
            query: Query dell'utente
            **kwargs: Argomenti aggiuntivi per compatibilità
            
        Returns:
            Dict con:
            - query_filter: Filtro Qdrant (se presente, altrimenti None)
            - query: Query originale
            - search_query: Query ottimizzata per ricerca semantica
        """
        logger.info(f"[QueryFilterExtractor] Analisi query: '{query}'")
        
        filters = self._extract_filters(query)
        logger.info(f"[QueryFilterExtractor] Filtri estratti:")
        logger.info(f"  - Planet: {filters.get('planet')}")
        logger.info(f"  - Restaurant: {filters.get('restaurant_name')}")
        logger.info(f"  - Chef: {filters.get('chef_name')}")
        logger.info(f"  - Ingredients IN: {filters.get('ingredients_in')}")
        logger.info(f"  - Ingredients OUT: {filters.get('ingredients_out')}")
        logger.info(f"  - Techniques IN: {filters.get('techniques_in')}")
        logger.info(f"  - Techniques OUT: {filters.get('techniques_out')}")
        
        search_query = self._extract_search_query(query)
        logger.info(f"[QueryFilterExtractor] Query ottimizzata per ricerca semantica: '{search_query}'")
        
        qdrant_filter = self._build_qdrant_filter(filters)
        if qdrant_filter:
            logger.info(f"[QueryFilterExtractor] ✓ Filtro Qdrant costruito con successo")
            logger.debug(f"[QueryFilterExtractor] Dettagli filtro: must={len(qdrant_filter.must) if hasattr(qdrant_filter, 'must') and qdrant_filter.must else 0}, must_not={len(qdrant_filter.must_not) if hasattr(qdrant_filter, 'must_not') and qdrant_filter.must_not else 0}")
        else:
            logger.info(f"[QueryFilterExtractor] Nessun filtro Qdrant costruito (nessun filtro esplicito trovato)")
        
        return {
            "query": query,
            "search_query": search_query,
            "query_filter": qdrant_filter if qdrant_filter else None
        }
    
    def _extract_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Estrae filtri espliciti dalla query usando LLM.
        
        Estrae filtri positivi (IN) e negativi (OUT) per:
        - Pianeta (se presente nella lista dei pianeti)
        - Ristorante
        - Chef
        - Ingredienti IN/OUT (nomi esatti)
        - Tecniche IN/OUT (nomi esatti)
        
        Args:
            query: Query dell'utente
            
        Returns:
            Dict con i filtri estratti (None per quelli non trovati)
        """
        planets_list = self._get_planet_name()
        prompt = EXTRACT_FILTERS_PROMPT(query, planets_list)
        
        try:
            response = self.llm_client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```json", "").replace("```", "").strip()
            filters = json.loads(content)
            
            # Normalizza ingredients_in
            for key in ["ingredients_in", "ingredients_out"]:
                if key in filters:
                    if filters[key] is None or filters[key] == "null":
                        filters[key] = None
                    elif isinstance(filters[key], str):
                        try:
                            filters[key] = json.loads(filters[key])
                        except:
                            filters[key] = [filters[key]]
                    elif not isinstance(filters[key], list):
                        filters[key] = None
            
            # Normalizza techniques_in
            for key in ["techniques_in", "techniques_out"]:
                if key in filters:
                    if filters[key] is None or filters[key] == "null":
                        filters[key] = None
                    elif isinstance(filters[key], str):
                        try:
                            filters[key] = json.loads(filters[key])
                        except:
                            filters[key] = [filters[key]]
                    elif not isinstance(filters[key], list):
                        filters[key] = None
            
            # Manteniamo compatibilità con vecchio formato per retrocompatibilità
            # Se ci sono solo ingredients/techniques (vecchio formato), li mettiamo in _in
            if "ingredients" in filters and "ingredients_in" not in filters:
                filters["ingredients_in"] = filters.pop("ingredients")
                filters["ingredients_out"] = None
            if "techniques" in filters and "techniques_in" not in filters:
                filters["techniques_in"] = filters.pop("techniques")
                filters["techniques_out"] = None
            
            return filters
        except Exception as e:
            logger.error(f"Errore nell'estrazione filtri: {e}")
            return {
                "planet": None, 
                "restaurant_name": None, 
                "chef_name": None, 
                "ingredients_in": None,
                "ingredients_out": None,
                "techniques_in": None,
                "techniques_out": None
            }
    
    def _extract_search_query(self, query: str) -> str:
        """
        Estrae una query ottimizzata per la ricerca semantica.
        
        Rimuove testo descrittivo e mantiene solo:
        - Nomi di ingredienti menzionati
        - Nomi di tecniche menzionate
        - Informazioni essenziali (pianeta, ristorante, chef)
        
        Args:
            query: Query originale dell'utente
            
        Returns:
            Query ottimizzata per ricerca semantica
        """
        prompt = EXTRACT_SEARCH_QUERY_PROMPT(query)
        
        try:
            response = self.llm_client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```", "").strip()
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith(('query', 'query ottimizzata', 'risposta')):
                    return line
            return next((line.strip() for line in lines if line.strip()), query)
        except Exception as e:
            logger.warning(f"Errore nell'estrazione query ottimizzata: {e}, uso query originale")
            return query
    
    def _build_qdrant_filter(self, filters: Dict[str, Optional[str]]):
        """
        Costruisce un filtro Qdrant usando gli oggetti models di Qdrant.
        
        Crea condizioni positive (must) e negative (must_not) per ogni filtro presente.
        Riferimento: https://qdrant.tech/documentation/concepts/filtering/
        
        Args:
            filters: Dict con i filtri estratti dalla query
            
        Returns:
            Filtro Qdrant se ci sono condizioni, altrimenti None
        """
        must_conditions = []
        must_not_conditions = []

        logger.debug(f"[QueryFilterExtractor] Costruzione filtro Qdrant da: {filters}")
        
        # Filtri positivi (must)
        if filters.get("planet"):
            logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST: planet={filters['planet']}")
            must_conditions.append(
                models.FieldCondition(
                    key="planet",
                    match=models.MatchValue(value=filters["planet"])
                )
            )
        
        if filters.get("restaurant_name"):
            logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST: restaurant_name={filters['restaurant_name']}")
            must_conditions.append(
                models.FieldCondition(
                    key="restaurant_name",
                    match=models.MatchValue(value=filters["restaurant_name"])
                )
            )

        if filters.get("chef_name"):
            logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST: chef_name={filters['chef_name']}")
            must_conditions.append(
                models.FieldCondition(
                    key="chef_name",
                    match=models.MatchValue(value=filters["chef_name"])
                )
            )

        # Ingredienti IN (devono essere presenti)
        if filters.get("ingredients_in"):
            for ingredient in filters["ingredients_in"]:
                logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST: ingredient IN={ingredient}")
                must_conditions.append(
                    models.FieldCondition(
                        key="raw_ingredients",
                        match=models.MatchAny(any=[ingredient])
                    )
                )

        # Tecniche IN (devono essere presenti)
        if filters.get("techniques_in"):
            for technique in filters["techniques_in"]:
                logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST: technique IN={technique}")
                must_conditions.append(
                    models.FieldCondition(
                        key="raw_techniques",
                        match=models.MatchAny(any=[technique])
                    )
                )

        # Ingredienti OUT (non devono essere presenti)
        if filters.get("ingredients_out"):
            for ingredient in filters["ingredients_out"]:
                logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST_NOT: ingredient OUT={ingredient}")
                must_not_conditions.append(
                    models.FieldCondition(
                        key="raw_ingredients",
                        match=models.MatchAny(any=[ingredient])
                    )
                )

        # Tecniche OUT (non devono essere presenti)
        if filters.get("techniques_out"):
            for technique in filters["techniques_out"]:
                logger.debug(f"[QueryFilterExtractor] Aggiunta condizione MUST_NOT: technique OUT={technique}")
                must_not_conditions.append(
                    models.FieldCondition(
                        key="raw_techniques",
                        match=models.MatchAny(any=[technique])
                    )
                )
        
        # Costruisci il filtro solo se ci sono condizioni
        if not must_conditions and not must_not_conditions:
            logger.debug("[QueryFilterExtractor] Nessuna condizione trovata, ritorno None")
            return None
        
        filter_dict = {}
        if must_conditions:
            filter_dict["must"] = must_conditions
            logger.info(f"[QueryFilterExtractor] Filtro MUST: {len(must_conditions)} condizioni")
        if must_not_conditions:
            filter_dict["must_not"] = must_not_conditions
            logger.info(f"[QueryFilterExtractor] Filtro MUST_NOT: {len(must_not_conditions)} condizioni")
        
        qdrant_filter = models.Filter(**filter_dict)
        
        # Log dettagliato del filtro completo
        logger.info("[QueryFilterExtractor] " + "=" * 70)
        logger.info("[QueryFilterExtractor] FILTRO QDRANT COMPLETO:")
        logger.info("[QueryFilterExtractor] " + "-" * 70)
        
        if must_conditions:
            logger.info(f"[QueryFilterExtractor] MUST ({len(must_conditions)} condizioni):")
            for i, condition in enumerate(must_conditions, 1):
                key = condition.key
                if hasattr(condition.match, 'value'):
                    value = condition.match.value
                    logger.info(f"[QueryFilterExtractor]   {i}. {key} = '{value}'")
                elif hasattr(condition.match, 'any'):
                    values = condition.match.any
                    logger.info(f"[QueryFilterExtractor]   {i}. {key} IN {values}")
                else:
                    logger.info(f"[QueryFilterExtractor]   {i}. {key} = {condition.match}")
        
        if must_not_conditions:
            logger.info(f"[QueryFilterExtractor] MUST_NOT ({len(must_not_conditions)} condizioni):")
            for i, condition in enumerate(must_not_conditions, 1):
                key = condition.key
                if hasattr(condition.match, 'value'):
                    value = condition.match.value
                    logger.info(f"[QueryFilterExtractor]   {i}. {key} != '{value}'")
                elif hasattr(condition.match, 'any'):
                    values = condition.match.any
                    logger.info(f"[QueryFilterExtractor]   {i}. {key} NOT IN {values}")
                else:
                    logger.info(f"[QueryFilterExtractor]   {i}. {key} != {condition.match}")
        
        logger.info("[QueryFilterExtractor] " + "=" * 70)
        
        return qdrant_filter

