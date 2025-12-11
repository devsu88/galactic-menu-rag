import os
import json
from typing import List, Any
from datapizza.core.models import PipelineComponent
from datapizza.type.type import Node
from datapizza.clients.openai import OpenAIClient
# from datapizza.modules.parsers.docling import DoclingParser # Removed
from llama_cloud_services import LlamaParse
from src.models.dish import Dish
from src.utils.config import DEBUG_DIR

class GalacticMenuParser(PipelineComponent):
    def __init__(self):
        super().__init__()
        
        # Inizializza LlamaParse
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            verbose=True,
            language="it" # I menu sono in italiano
        )
        
        # Inizializza client OpenAI
        self.client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        ) 

    def _run(self, file_path: str, **kwargs) -> Node:
        """
        Input: path del file PDF
        Output: Singolo Node root che contiene i piatti come children
        """
        # 1. Estrai testo grezzo usando LlamaParse
        text_content = self._extract_text_with_llama(file_path)

        print(f"\n\n\n[DEBUG] Text content: {text_content}\n\n\n")
        
        if not text_content:
            # Restituiamo un nodo vuoto per non bloccare la pipeline
            return Node(content="Menu vuoto", metadata={"file_path": file_path, "error": "no_text_extracted"})
        
        # 2. Usa LLM per estrarre JSON strutturato
        structured_data = self._extract_structured_data(text_content)

        print(f"\n\n\n[DEBUG] Structured data: {structured_data}\n\n\n")
        
        # Salva il JSON estratto nella cartella .debug per debug
        os.makedirs(DEBUG_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        debug_file = os.path.join(DEBUG_DIR, f"{base_name}.json")
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        # 3. Converti in Nodi Datapizza
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
            
            # Creiamo un Nodo per ogni piatto
            # Il testo del nodo è una combinazione descrittiva per la ricerca semantica
            node_text = f"Piatto: {dish.name}\nRistorante: {dish.restaurant_name}\nDescrizione: {dish.description}\nIngredienti: {', '.join(dish.ingredients)}\nTecniche: {', '.join(dish.techniques)}"
            
            node = Node(
                content=node_text,
                metadata=dish.to_metadata()
            )
            nodes.append(node)
        
        # 4. Creiamo un Node root che contiene tutti i piatti come children
        # Questo è compatibile con NodeSplitter che si aspetta un singolo Node
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
        try:
            # LlamaParse.parse() restituisce un JobResult object
            result = self.parser.parse(path)
            
            # Accedere direttamente alle pagine
            pages_with_text = [page for page in result.pages if page.text]
            full_text = "\n".join([page.text for page in pages_with_text])
            
            return full_text
            
        except Exception as e:
            return ""

    def _extract_structured_data(self, text: str) -> dict:
        prompt = f"""
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
        
        try:
            response = self.client.invoke(prompt)
            content = response.text if hasattr(response, 'text') else str(response)
            
            # Pulizia markdown json se presente
            content = content.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(content)
            
            return parsed_json
        except json.JSONDecodeError:
            return {"dishes": []}
        except Exception:
            return {"dishes": []}

