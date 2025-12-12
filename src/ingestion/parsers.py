"""Parser custom per estrarre dati strutturati da menu PDF galattici."""

import logging
import os
import json
from typing import List, Any
from datapizza.core.models import PipelineComponent
from datapizza.type.type import Node
from datapizza.clients.openai import OpenAIClient
from llama_cloud_services import LlamaParse
from src.models.dish import Dish
from src.utils.config import DEBUG_DIR, LLM_MODEL, LLAMAPARSE_LANGUAGE
from src.utils.prompts import EXTRACT_STRUCTURED_DATA_PROMPT

logger = logging.getLogger(__name__)

class GalacticMenuParser(PipelineComponent):
    """
    Parser custom per estrarre dati strutturati da menu PDF.
    
    Utilizza LlamaParse per l'estrazione del testo e un LLM per convertire
    il testo grezzo in dati strutturati (ristorante, piatti, ingredienti, tecniche).
    """
    
    def __init__(self):
        super().__init__()
        
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            verbose=True,
            language=LLAMAPARSE_LANGUAGE
        )
        
        self.client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=LLM_MODEL
        ) 

    def _run(self, file_path: str, **kwargs) -> Node:
        """
        Processa un file PDF di menu e restituisce un Node root con i piatti come children.
        
        Args:
            file_path: Path del file PDF da processare
            **kwargs: Argomenti aggiuntivi per compatibilità con PipelineComponent
            
        Returns:
            Node root contenente tutti i piatti estratti come children.
            Se l'estrazione fallisce, restituisce un nodo vuoto con metadata di errore.
        """
        text_content = self._extract_text_with_llama(file_path)

        logger.debug(f"Text content estratto: {len(text_content)} caratteri")
        
        if not text_content:
            logger.warning(f"Nessun testo estratto da {file_path}")
            return Node(content="Menu vuoto", metadata={"file_path": file_path, "error": "no_text_extracted"})
        
        structured_data = self._extract_structured_data(text_content)

        logger.debug(f"Dati strutturati estratti: {len(structured_data.get('dishes', []))} piatti")
        
        os.makedirs(DEBUG_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        debug_file = os.path.join(DEBUG_DIR, f"{base_name}.json")
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        nodes = []
        
        restaurant_info = structured_data.get("restaurant", {})
        restaurant_name = restaurant_info.get("name", "Unknown")
        planet = restaurant_info.get("planet", "Unknown")
        chef_info = restaurant_info.get("chef", {})
        chef_name = chef_info.get("name", "Unknown")
        
        dishes_data = structured_data.get("dishes", [])
        
        for dish_data in dishes_data:
            dish = Dish(
                name=dish_data.get("name"),
                restaurant_name=restaurant_name,
                planet=planet,
                chef_name=chef_name,
                ingredients=dish_data.get("ingredients", []),
                techniques=dish_data.get("techniques", []),
                description=dish_data.get("description", "")
            )
            
            node_text = f"Piatto: {dish.name}\nRistorante: {dish.restaurant_name}\nDescrizione: {dish.description}\nIngredienti: {', '.join(dish.ingredients)}\nTecniche: {', '.join(dish.techniques)}"
            
            node = Node(
                content=node_text,
                metadata=dish.to_metadata()
            )
            nodes.append(node)
        
        root_node = Node(
            content=f"Menu del ristorante {restaurant_name}",
            metadata={
                "restaurant_name": restaurant_name,
                "planet": planet,
                "chef_name": chef_name,
                "file_path": file_path
            },
            children=nodes
        )
        
        return root_node

    def _extract_text_with_llama(self, path: str) -> str:
        """
        Estrae il testo grezzo da un PDF usando LlamaParse.
        
        Args:
            path: Path del file PDF
            
        Returns:
            Testo estratto dal PDF, stringa vuota in caso di errore
        """
        try:
            result = self.parser.parse(path)
            pages_with_text = [page for page in result.pages if page.text]
            full_text = "\n".join([page.text for page in pages_with_text])
            return full_text
        except Exception as e:
            logger.error(f"Errore durante l'estrazione testo con LlamaParse: {e}")
            return ""

    def _extract_structured_data(self, text: str) -> dict:
        """
        Estrae dati strutturati dal testo del menu usando un LLM.
        
        Args:
            text: Testo grezzo estratto dal PDF
            
        Returns:
            Dizionario con struttura {"restaurant": {...}, "dishes": [...]}
            Restituisce {"dishes": []} in caso di errore
        """
        prompt = EXTRACT_STRUCTURED_DATA_PROMPT(text)
        
        try:
            response = self.client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            content = content.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(content)
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Errore nel parsing JSON dalla risposta LLM: {e}")
            return {"dishes": []}
        except Exception as e:
            logger.error(f"Errore durante l'estrazione dati strutturati: {e}")
            return {"dishes": []}

