import os
import json
from typing import Dict, Optional, List, Any
from datapizza.core.models import PipelineComponent
from datapizza.clients.openai import OpenAIClient
from qdrant_client import models


class QueryFilterExtractor(PipelineComponent):
    """
    Modulo che analizza la query per estrarre filtri espliciti (pianeta, ristorante, ingrediente, tecnica)
    e costruisce un filtro Qdrant per ottimizzare la ricerca vettoriale.
    """
    def __init__(self):
        super().__init__()
        self.llm_client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

    def _get_planet_name(self) -> str:
        """
        Ritorna la lista dei pianeti.
        """
        planets = [
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
        return ", ".join(planets)
    
    def _run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Analizza la query e estrae filtri espliciti.
        
        Returns:
            Dict con:
            - query_filter: Filtro Qdrant (se presente)
            - query: Query originale (per passarla al filtro LLM successivo)
            - search_query: Query ottimizzata per la ricerca semantica (senza testo descrittivo)
        """
        # Usa LLM per estrarre filtri espliciti dalla query
        filters = self._extract_filters(query)
        print(f"[DEBUG QueryFilterExtractor] Filtri estratti: {filters}")
        
        # Estrai query ottimizzata per ricerca semantica (solo ingredienti/tecniche chiave)
        search_query = self._extract_search_query(query)
        print(f"[DEBUG QueryFilterExtractor] Query ricerca semantica: {search_query}")
        
        # Costruisci filtro Qdrant se ci sono filtri espliciti
        qdrant_filter = self._build_qdrant_filter(filters)
        print(f"[DEBUG QueryFilterExtractor] Filtro Qdrant costruito: {qdrant_filter}")
        
        return {
            "query": query,
            "search_query": search_query,  # Query ottimizzata per ricerca semantica
            "query_filter": qdrant_filter if qdrant_filter else None
        }
    
    def _extract_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Estrae filtri espliciti dalla query usando LLM.
        NOTA: Estraiamo solo filtri espliciti e precisi (pianeta, ristorante, chef, ingredienti, tecniche).
        """
        prompt = f"""
        Analizza la seguente domanda e identifica se contiene filtri espliciti e precisi.
        Restituisci un JSON con questa struttura:
        {{
            "planet": "Nome del pianeta, se menzionato esplicitamente e presente nella lista dei pianeti ({self._get_planet_name()}), altrimenti null",
            "restaurant_name": "Nome del ristorante, se menzionato esplicitamente, altrimenti null",
            "chef_name": "Nome dello chef se menzionato esplicitamente, altrimenti null",
            "ingredients": "Lista degli ingredienti se menzionati esplicitamente (es. ['Ingrediente 1', 'Ingrediente 2']), altrimenti null",
            "techniques": "Lista delle tecniche se menzionate esplicitamente (es. ['Tecnica 1', 'Tecnica 2']), altrimenti null"
        }}
        
        IMPORTANTE: 
        - Estrai ingredienti e tecniche SOLO se sono menzionati esplicitamente e precisamente nella domanda.
        - Per ingredienti: estrai il nome esatto (compreso di caratteri speciali)
        - Per tecniche: estrai il nome esatto (compreso di caratteri speciali)
        
        Domanda: "{query}"
        
        Restituisci SOLO il JSON, senza altro testo.
        """
        
        try:
            response = self.llm_client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```json", "").replace("```", "").strip()
            filters = json.loads(content)
            
            # Normalizza ingredienti e tecniche: assicurati che siano liste o None
            if "ingredients" in filters:
                if filters["ingredients"] is None or filters["ingredients"] == "null":
                    filters["ingredients"] = None
                elif isinstance(filters["ingredients"], str):
                    # Se è una stringa, prova a parsarla come JSON
                    try:
                        filters["ingredients"] = json.loads(filters["ingredients"])
                    except:
                        filters["ingredients"] = [filters["ingredients"]]
                elif not isinstance(filters["ingredients"], list):
                    filters["ingredients"] = None
            
            if "techniques" in filters:
                if filters["techniques"] is None or filters["techniques"] == "null":
                    filters["techniques"] = None
                elif isinstance(filters["techniques"], str):
                    try:
                        filters["techniques"] = json.loads(filters["techniques"])
                    except:
                        filters["techniques"] = [filters["techniques"]]
                elif not isinstance(filters["techniques"], list):
                    filters["techniques"] = None
            
            return filters
        except Exception as e:
            print(f"[DEBUG QueryFilterExtractor] Errore nell'estrazione filtri: {e}")
            return {"planet": None, "restaurant_name": None, "chef_name": None, "ingredients": None, "techniques": None}
    
    def _extract_search_query(self, query: str) -> str:
        """
        Estrae una query ottimizzata per la ricerca semantica, rimuovendo testo descrittivo
        e mantenendo solo ingredienti, tecniche e informazioni chiave.
        """
        prompt = f"""
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
        
        try:
            response = self.llm_client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```", "").strip()
            # Rimuovi eventuali prefissi come "Query:" o "Query ottimizzata:"
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith(('query', 'query ottimizzata', 'risposta')):
                    return line
            # Fallback: usa la prima riga non vuota
            return next((line.strip() for line in lines if line.strip()), query)
        except Exception:
            # Fallback: usa la query originale
            return query
    
    def _build_qdrant_filter(self, filters: Dict[str, Optional[str]]):
        """
        Costruisce un filtro Qdrant usando gli oggetti models di Qdrant.
        Qdrant filter syntax: https://qdrant.tech/documentation/concepts/filtering/
        """
        conditions = []

        print(f"[DEBUG QueryFilterExtractor] Filtri: {filters}")
        
        # Filtro per pianeta
        if filters.get("planet"):
            conditions.append(
                models.FieldCondition(
                    key="planet",
                    match=models.MatchValue(value=filters["planet"])
                )
            )
        
        # Filtro per ristorante
        if filters.get("restaurant_name"):
            conditions.append(
                models.FieldCondition(
                    key="restaurant_name",
                    match=models.MatchValue(value=filters["restaurant_name"])
                )
            )

        # Filtro per chef
        if filters.get("chef_name"):
            conditions.append(
                models.FieldCondition(
                    key="chef_name",
                    match=models.MatchValue(value=filters["chef_name"])
                )
            )

        # Filtro per ingredienti
        if filters.get("ingredients"):
            for _filter in filters["ingredients"]:
                conditions.append(
                    models.FieldCondition(
                        key="raw_ingredients",
                        match=models.MatchAny(any=[_filter])
                    )
                )

        # Filtro per tecniche
        if filters.get("techniques"):
            for _filter in filters["techniques"]:
                conditions.append(
                    models.FieldCondition(
                        key="raw_techniques",
                        match=models.MatchAny(any=[_filter])
                    )
                )
        
        if not conditions:
            return None
        
        # Costruisci filtro Qdrant (Must = AND logic)
        return models.Filter(must=conditions)

