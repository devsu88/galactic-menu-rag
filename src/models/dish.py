"""Modello dati per rappresentare un piatto del menu galattico."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Dish:
    """
    Modello dati per rappresentare un piatto del menu.
    
    Attributes:
        name: Nome del piatto
        restaurant_name: Nome del ristorante
        planet: Nome del pianeta (opzionale)
        chef_name: Nome dello chef (opzionale)
        ingredients: Lista degli ingredienti
        techniques: Lista delle tecniche di preparazione
        description: Descrizione del piatto (opzionale)
    """
    name: str
    restaurant_name: str
    planet: Optional[str] = None
    chef_name: Optional[str] = None
    ingredients: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    def to_metadata(self) -> Dict:
        """
        Converte l'oggetto in un dizionario per i metadati del Node Datapizza.
        
        Include sia versioni formattate (stringhe) che raw (liste) per:
        - Ricerca full-text (stringhe separate da virgola)
        - Filtri esatti Qdrant (liste)
        
        Returns:
            Dict con i metadati del piatto formattati per Qdrant
        """
        return {
            "dish_name": self.name,
            "restaurant_name": self.restaurant_name,
            "planet": self.planet,
            "chef_name": self.chef_name,
            "ingredients": ", ".join(self.ingredients),
            "techniques": ", ".join(self.techniques),
            "raw_ingredients": self.ingredients,
            "raw_techniques": self.techniques
        }

