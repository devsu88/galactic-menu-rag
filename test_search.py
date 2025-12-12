from qdrant_client import models
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.embedders.openai import OpenAIEmbedder
import os
from dotenv import load_dotenv
load_dotenv()

embedder = OpenAIEmbedder(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
vector_store = QdrantVectorstore(host="localhost", port=6333)

collection_name = "galactic_menu"

# Search for similar chunks
query = "Quali piatti sono preparati usando la tecnica Surgelamento Antimaterico a Risonanza Inversa senza impiegare Foglie di Mandragora?"
query_vector = embedder.embed(query)
results = vector_store.search(
    collection_name=collection_name,
    query_vector=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="raw_techniques",
                match=models.MatchAny(
                    any=['Cottura Olografica Quantum Fluttuante'],
                ),
            ),
            models.FieldCondition(
                key="raw_techniques",
                match=models.MatchAny(
                    any=['Decostruzione Interdimensionale Lovecraftiana'],
                ),
            )
        ],
    ),
    k=5
)

for chunk in results:
    print(f"Chunk\n{chunk.text}\n\n")
    # print(f"Metadata: {chunk.metadata}")