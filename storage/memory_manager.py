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

os.environ["OPENAI_KEY"] = st.secrets["OPENAI_KEY"]
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
    _table.create_fts_index(text_field, replace=True)
    if search_type == 'hybrid':
        results = _table.search(query, query_type="hybrid").rerank(reranker=reranker).limit(limit)

    elif search_type == 'fts':
        results = _table.search(query).limit(limit)

    else:
        results = _table.search(query).limit(limit)

    if prefilter:
        results = results.where(prefilter, prefilter=True)
    try:
        listed_results = results.to_list()
    except Exception as e:
        listed_results = []
    final_results = [result for result in listed_results if result['_relevance_score'] > threshold]
    return final_results


def notes_to_table(document: str,
                   session_id: str,
                   thread_id: str | None,
                   entities: str,
                   image_tags: str | None,
                   bit_map_object: str | None,
                   vectorstore: Type[VectorStoreManager()]):

    # time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # session_id = f'{session_id}_{time_stamp}'
    # vectorstore = VectorStoreManager('captain_logs')

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

    vectorstore.db.create_table("work_notes",
                                exist_ok=True,
                                data=[payload],
                                schema=DocumentSchema)


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
    vectorstore.db.create_table("transcripts",
                                exist_ok=True,
                                data=payload,
                                schema=TranscriptSchema)


def get_transcripts(query: str,
                    thread_id: Optional[str],
                    prefilter: Optional[str],
                    limit: int,
                    vectorstore: VectorStoreManager,
                    search_type: str = 'hybrid',
                    threshold: Optional[float] = 0.9) -> List[Dict[str, Any]]:
    """
    Retrieves transcripts from the vectorstore using advanced search.

    Args:
        thread_id (str or None): The thread ID to filter by.
        prefilter (str or None): SQL expression for prefiltering (e.g., date ranges).
        limit (int): Maximum number of results to return.
        vectorstore (VectorStoreManager): The vectorstore manager instance.
        search_type (str): Type of search to perform ('hybrid' or 'fts').
        threshold (float, optional): Relevance score threshold for filtering results.

    Returns:
        List[Dict[str, Any]]: A list of transcript records matching the search criteria.
    """
    # Open the transcripts table
    transcript_table = vectorstore.db.open_table("transcripts")

    # Build prefilter conditions
    prefilter_conditions = []
    if thread_id:
        prefilter_conditions.append(f"thread_id = '{thread_id}'")
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
    results = get_transcripts(query="what are the 3 layers of AI", thread_id="Jane_Joe_20241004004215", prefilter=None, limit=10, vectorstore=VectorStoreManager())
    print(results)


# # Example transcript
# transcript = """
# Claudio: Is that I. Let me just share my screen. It's not finished, but I think that maybe we can flush it out today ,...... how do I want to set up. Need to take that?
# Rupinder: No, that you will have to on a fire tread.
# Claudio: Okay.
# Rupinder: And I'll use this fire tread to tell him.
# Claudio: So I think the agenda. Right, it's, it starts off with.
# Rupinder: But let's go. Yeah, you have the very first line with the generative AI stack.
# Claudio: Right, right. And then. Right. So technically the agenda and these kind of go hand in hand where I kind of go into a high level into how we define generative AI and then zoom in into bedrock and talk about how bedrock can be used for the application development.
# Rupinder: But pitch it.
# Claudio: Okay. So actually I have a couple ideas. So, you know, generative AI is such a broad concept right now and it's so, and it's so quickly evolving. Here at AWS, we've decided that we want to break this down into three layers. One is the most like most core infrastructure layer, which is where GPU's and the resources come in. And it's very much catered for data scientists and domain .... Like my, I call it the elevator pitch. The idea is to press a button and by the time I get to.
# Rupinder: The top, I think, you know what we should do is if you want to talk about this, right, so you say that before we dive into bedrock, just want to level set the view and the vision we have in, in Amazon about generative AI stack.
# Claudio: Okay.
# Rupinder: And I mean, maybe you can make notes or you can kind of record kind of sometimes ramble on. Right. So, okay. Also, and we cater for the full stack of people who are building generative reals. So at the bottom layer is where we provide the infrastructure for fine tuning a model for running, for people who want to have more control over how they want the inferencing to happen.
# Claudio: So would you say, would you describe the Amazon Q layer being used by power users. I've heard that before and I don't know if that is the correct term to use in this presentation. And I say that because he used the term advanced practitioners. So I just want to know if that is part of the language that we need to incorporate in this presentation.
# Rupinder: So advanced practitioners is the bottom layer, right?
# Claudio: Any infrastructure?
# """
#
# txt = split_transcript(transcript, max_tokens=100, percent_threshold=0.6)
# print(txt)
#
# transcript_to_table(transcript, "session_1", "thread_1", VectorStoreManager())

#
# lancedb_ = VectorStoreManager()
# notes = lancedb_.db.open_table("work_notes")
# transcripts = lancedb_.db.open_table("transcripts")
#
# print(table_search("Claudio", notes)[0]['text'])
# print(table_search("Smokey the bear", transcripts)[0]['text'])
