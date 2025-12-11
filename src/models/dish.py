from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Dish:
    name: str
    restaurant_name: str
    planet: Optional[str] = None
    chef_name: Optional[str] = None
    ingredients: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    def to_metadata(self) -> Dict:
        """Converte l'oggetto in un dizionario piatto per i metadati del Nodo"""
        return {
            "dish_name": self.name,
            "restaurant_name": self.restaurant_name,
            "planet": self.planet,
            "chef_name": self.chef_name,
            "ingredients": ", ".join(self.ingredients), # Utile per ricerca full-text
            "techniques": ", ".join(self.techniques),
            "raw_ingredients": self.ingredients, # Utile per filtering esatto
            "raw_techniques": self.techniques
        }

