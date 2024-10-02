import os
import uuid
import datetime
import lancedb
from dotenv import load_dotenv
from typing import List, Dict, Optional, Literal, Type
from lancedb.embeddings import EmbeddingFunctionRegistry, get_registry
from lancedb.pydantic import Vector, LanceModel
from lancedb.table import Table
from lancedb.rerankers import LinearCombinationReranker, ColbertReranker, CohereReranker
from openai import OpenAI

client = OpenAI()
load_dotenv()


# Manage API keys centrally
class APIKeyManager:
    def __init__(self):
        self.cohere_key = os.getenv("COHERE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = self.openai_key


api_keys = APIKeyManager()


# Manage embedding functions
class EmbeddingManager:
    def __init__(self):
        self.registry = EmbeddingFunctionRegistry().get_instance()
        self.csv_registry = get_registry()
        self.openai = self.setup_openai_embedding()
        self.func = self.setup_sentence_transformers_embedding()

    def setup_openai_embedding(self):
        return self.registry.get("openai").create(name="text-embedding-3-small")

    def setup_sentence_transformers_embedding(self):
        return self.csv_registry.get("sentence-transformers").create(device="cuda")


embedding_manager = EmbeddingManager()


class DocumentSchema(LanceModel):
    class Config:
        arbitrary_types_allowed = True
    session_id: str
    thread_id: str
    vector: Vector(embedding_manager.openai.ndims()) = embedding_manager.openai.VectorField()
    text: str = embedding_manager.openai.SourceField()
    entities: str
    bit_map_object: Optional[str]
    image_tags: str


class TranscriptSchema(LanceModel):
    class Config:
        arbitrary_types_allowed = True
    session_id: str
    thread_id: str
    vector: Vector(embedding_manager.openai.ndims()) = embedding_manager.openai.VectorField()
    text: str = embedding_manager.openai.SourceField()
    entities: str


class RerankerFactory:
    rerankers = {
        "linear": lambda weight: LinearCombinationReranker(weight=weight),
        "cohere": lambda api_key: CohereReranker(api_key=api_key),
        "colbert": ColbertReranker
    }

    @staticmethod
    def get_reranker(name: str, **kwargs):
        return RerankerFactory.rerankers[name](**kwargs)


# VectorStoreManager for managing vector store operations
class VectorStoreManager:
    def __init__(self, store_name="captain_logs"):
        _EMBEDDINGS_DIMENSIONS = 512
        self.instance_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(uuid.uuid4())
        self.store_name = store_name
        self.db = self.connect_to_db()

    def connect_to_db(self) -> lancedb.DBConnection:
        db_path = self.store_name
        return lancedb.connect(str(db_path))


def table_search(query: str,
                 _table: Table,
                 prefilter: Optional[str] = None,
                 text_field: Optional[str | List[str]] = None,
                 search_type: Optional[Literal['fts', 'hybrid', None]] = None,
                 limit: int = 10,
                 threshold: Optional[float | None] = 0.9,
                 ) -> list:

    if search_type == 'fts' and text_field is None:
        raise "Missing text field to perform Full-Text-Search"

    reranker = RerankerFactory.get_reranker("cohere", api_key=api_keys.cohere_key)
    if search_type == 'hybrid':
        results = _table.search(query,
                               query_type="hybrid").limit(limit).rerank(reranker=reranker,
                                                                        column=text_field).to_list()
        results = [result for result in results if result['_relevance_score'] > threshold]

    elif search_type == 'fts':
        _table.create_fts_index(text_field, replace=True)
        results = _table.search(query).limit(limit)

    else:
        results = _table.search(query).limit(limit)

    if prefilter:
        results = results.where(prefilter, prefilter=True)

    return results.to_list() if results else []


def notes_to_table(document: str,
                    session_name: str,
                    thread_id: str | None,
                    entities: str,
                    image_tags: str,
                    bit_map_object: str):

    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    session_id = f'{session_name}_{time_stamp}'
    vectorstore = VectorStoreManager('captain_logs')

    response = client.embeddings.create(
        input=document,
        model="text-embedding-3-small"
    )
    embedded = response.data[0].embedding
    payload = {"session_id": session_id,
               "thread_id": thread_id,
               "text": document,
               "vector": embedded,
               "entities": entities,
               "image_tags": image_tags,
               "bit_map_object": bit_map_object
               }
    return vectorstore.db.create_table("work_notes", exist_ok=True, data=[payload], schema=DocumentSchema)


def transcript_to_table(transcript: str,
                        session_name: str,
                        thread_id: str | None,
                        entities: str):
    ### TBD THIS WILL REQUIRE MULTIPLE LAYERS OF CHUNKING BY SPEAKER AND THE LENGTH OF THE SPEECH ###
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    session_id = f'{session_name}_{time_stamp}'
    vectorstore = VectorStoreManager('captain_logs')

    response = client.embeddings.create(
        input=transcript,
        model="text-embedding-3-small"
    )
    embedded = response.data[0].embedding
    payload = {"session_id": session_id,
               "thread_id": thread_id,
               "text": transcript,
               "vector": embedded,
               "entities": entities
               }
    return vectorstore.db.create_table("transcripts", exist_ok=True, data=[payload], schema=TranscriptSchema)