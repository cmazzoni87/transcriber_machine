import os
import uuid
import datetime
import lancedb
import re
from typing import List, Dict, Optional, Literal, Type, Any
from lancedb.embeddings import EmbeddingFunctionRegistry, get_registry
from lancedb.pydantic import Vector, LanceModel
from lancedb.table import Table
from lancedb.rerankers import LinearCombinationReranker, ColbertReranker, CohereReranker
from openai import OpenAI
from tools.txt_preprocessor import split_transcript, extract_speakers
import streamlit as st
from pathlib import Path
import time

# get path to project root
storage_root = Path(__file__).parent


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
os.environ["COHERE_KEY"] = st.secrets["COHERE_KEY"]
os.environ["CO_API_KEY"] = st.secrets["COHERE_KEY"]


client = OpenAI(api_key=st.secrets["OPENAI_KEY"])


# Manage API keys centrally
class APIKeyManager:
    def __init__(self):
        self.cohere_key = st.secrets["COHERE_KEY"]
        self.openai_key = st.secrets["OPENAI_KEY"]
        os.environ["OPENAI_KEY"] = self.openai_key
        os.environ["COHERE_KEY"] = self.cohere_key


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
        self.store_name = storage_root / store_name
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
                 threshold: Optional[float | None] = 0.6,
                 ) -> list:

    if search_type == 'fts' and text_field is None:
        raise "Missing text field to perform Full-Text-Search"

    reranker = RerankerFactory.get_reranker("cohere", api_key=api_keys.cohere_key)

    if search_type == 'hybrid':
        _table.create_fts_index(text_field, replace=True)
        results = _table.search(query, query_type="hybrid").rerank(reranker=reranker).limit(limit)

    elif search_type == 'fts':
        _table.create_fts_index(text_field, replace=True)
        results = _table.search(query).select([text_field]).limit(limit)
    else:
        results = _table.search(query).limit(limit)

    if prefilter:
        try:
            results = results.where(prefilter, prefilter=True).to_list()
        except Exception as e:
            try:
                results = results.to_list()
            except Exception as e:
                results = []

    try:
        listed_results = results.to_list()
    except Exception as e:
        listed_results = []
    if search_type == 'hybrid':
        final_results = [result for result in listed_results if result['_relevance_score'] > threshold]
    else:
        final_results = results
    return final_results


def notes_to_table(document: str,
                   session_id: str,
                   thread_id: str | None,
                   entities: str,
                   image_tags: str | None,
                   bit_map_object: str | None,
                   vectorstore: Type[VectorStoreManager()]):

    response = client.embeddings.create(
        input=document,
        model="text-embedding-3-small"
    )
    if image_tags == "" or image_tags is None:
        image_tags = "No Image"
    if bit_map_object == "" or bit_map_object is None:
        bit_map_object = "No Image"

    embedded = response.data[0].embedding
    payload = {"session_id": session_id,
               "thread_id": thread_id,
               "text": document,
               "vector": embedded,
               "entities": entities,
               "image_tags": image_tags,
               "bit_map_object": bit_map_object
               }
    try:
        vectorstore.db.create_table("work_notes",
                                exist_ok=True,
                                data=[payload],
                                schema=DocumentSchema)
    except:
        vectorstore.db.create_table("work_notes",
                                exist_ok=True,
                                data=[payload]
                                )



def transcript_to_table(transcript: str,
                        session_id: str,
                        thread_id: str | None,
                        vectorstore: Type[VectorStoreManager()]):

    transcript_chunks = split_transcript(transcript)
    payload = []
    for chunk in transcript_chunks:
        chunk_str = "\n".join(chunk)
        entities = ", ".join(extract_speakers(chunk_str))
        response = client.embeddings.create(
            input=chunk_str,
            model="text-embedding-3-small"
        )
        embedded = response.data[0].embedding
        payload.append({"session_id": session_id,
                        "thread_id": thread_id,
                        "text": chunk_str,
                        "vector": embedded,
                        "entities": entities
                        })
    try:
        vectorstore.db.create_table("transcripts",
                                exist_ok=True,
                                data=payload,
                                schema=TranscriptSchema)
    except:
        time.sleep(1)
        vectorstore.db.create_table("transcripts",
                                exist_ok=True,
                                data=payload
                                )


def get_transcripts(query: str,
                    thread_id: Optional[str],
                    prefilter: Optional[str],
                    limit: int,
                    vectorstore: VectorStoreManager,
                    search_type: str = 'hybrid',
                    threshold: Optional[float] = 0.6,
                    table_path_str: str = 'transcripts') -> List[Dict[str, Any]]:
    """
    Retrieves transcripts from the vectorstore using advanced search.

    Args:
        thread_id (str or None): The thread ID to filter by.
        prefilter (str or None): SQL expression for prefiltering (e.g., date ranges).
        limit (int): Maximum number of results to return.
        vectorstore (VectorStoreManager): The vectorstore manager instance.
        search_type (str): Type of search to perform ('hybrid' or 'fts').
        threshold (float, optional): Relevance score threshold for filtering results.
        table_path_str (str): The path to the table in the vectorstore.

    Returns:
        List[Dict[str, Any]]: A list of transcript records matching the search criteria.
    """

    # Open the transcripts table
    # table_path_str = "transcripts"
    transcript_table = vectorstore.db.open_table(table_path_str.lower())

    # Build prefilter conditions
    prefilter_conditions = []
    if thread_id:
        prefilter_conditions.append(f'thread_id = "{thread_id}"')
    if prefilter:
        prefilter_conditions.append(prefilter)
    # Combine prefilter conditions
    prefilter_str = ' AND '.join(prefilter_conditions) if prefilter_conditions else None

    # Perform the search using table_search function
    payload = table_search(
        query=query,
        _table=transcript_table,
        prefilter=prefilter_str,
        text_field='text',
        search_type=search_type,
        limit=limit,
        threshold=threshold,
    )
    if payload:
        _results = [{'text': record['text'], 'speakers': record['entities']} for record in payload]
    else:
        _results = []
    return _results


if __name__ == "__main__":

    vector = VectorStoreManager(store_name=r"C:\Users\cmazz\PycharmProjects\transcriber_machine\storage\captain_logs")
    aa = vector.db.open_table("transcripts")
    query = "what is causing the air issues in NYC?"
    thread_id = 'Host_Guest_20241011130826'
    aa.search(query).to_list()
    results = get_transcripts(query=query,
                              thread_id=thread_id,
                              prefilter=None,
                              limit=10,
                              search_type="vector",
                              vectorstore=vector)
    print(results)


