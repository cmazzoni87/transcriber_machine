from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from typing import TypedDict
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from storage.memory_manager import get_transcripts, VectorStoreManager, storage_root
import os
import streamlit as st

store_name = storage_root / "captain_logs"
vectorstore = VectorStoreManager(store_name=store_name)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]


llm_groq = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.35,
    max_tokens=8000,
    timeout=None,
    max_retries=4,
)
llm_groq_tiny = ChatOpenAI(temperature=0.1,
                           model="gpt-4o-mini",
                           max_tokens=250,
                           api_key=st.secrets["OPENAI_KEY"])

# llm_groq_tiny = ChatGroq(
#     model="llama3-8b-8192",
#     temperature=0.35,
#     max_tokens=2000,
#     timeout=None,
#     max_retries=4,
# )

# Define the decomposition patterns

decomposition = """
Decompose the parent question on the components needed to answer the question (Chains). Depending on the complexity of the question select one of the following decomposition chains.
**Simple 2-hop Chain:** 
- This reasoning is straightforward, requiring two pieces of information that build on each other.

**3-hop Chain:** 
- This reasoning requires more steps, where each answer leads to the next piece of information.

**2-hop Diverging Chain:** 
- The question requires reasoning in two different directions: first identifying one aspect, then querying another related aspect.

**4-hop Intersecting Reasoning:** 
- Two separate pieces of information must be connected to answer the final question.

**Two Chain with Multiple Paths:**
- This involves a chain of reasoning that takes multiple paths: understanding different relationships before arriving at the final answer.

**4-hop Intersecting Reasoning with Branches:** 
- This type of reasoning requires piecing together different parts of information (e.g., history, geography) to answer the final question.

Parent Question:
{question}
"""

class QuestionDecomposition(BaseModel):
    sub_questions: List[str] = Field(
        description="The sub-questions which answer content can be used to answer the parent question")


class IdentifyData(BaseModel):
    relevant_data: str = Field(description="The relevant needed to answer the question")


def invoke_formatter(_message: dict, prompt: str, style: BaseModel, model, input_vars=['{question}']) -> dict:
    """
    Analyze the sentiment of a document given a context.

    :param _message: The _message to analyze
    :param prompt: The prompt to analyze
    :param style: The style of the prompt
    :param model: The model to analyze
    """
    parser = JsonOutputParser(pydantic_object=style)
    prompt_template = PromptTemplate(
        template=f"{prompt}" + "\n{format_instructions}\n",
        input_variables=input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt_template | model | parser

    return chain.invoke(_message)


# Define the State class for managing the input, sub-questions, and results
class State(TypedDict):
    input_question: str
    sub_questions: List[str]
    results: List[str]


def ai_librarian(question: str, thread_id: str, data_type_selection: str, filters: dict) -> list:
    docs_state = []
    # Define a simple question decomposition function using Bedrock
    search_source = data_type_selection
    if search_source == "Report":
        search_source = "work_notes"

    def decompose_question(state: State) -> State:
        if search_source != "work_notes":
            # Decompose the question using Bedrock's LLM
            payload = invoke_formatter({'question': state['input_question']}, decomposition, QuestionDecomposition,
                                       llm_groq)
        else:
            payload = {'sub_questions': [state['input_question']]}

        state['sub_questions'] = [q.strip() for q in payload['sub_questions'] if q.strip()]
        return state

    def retriever(sub_question: str) -> list:
        docs = []
        payload = get_transcripts(query=sub_question,
                                  thread_id=thread_id,
                                  prefilter=None,
                                  limit=5,
                                  search_type="vector",
                                  vectorstore=vectorstore,
                                  table_path_str=search_source)
        raw_docs = [pl['text'] for pl in payload]
        # append raw docs to docs_state if raw_docs not in docs_state
        for doc in raw_docs:
            if doc not in docs_state:
                docs_state.append(doc)
                docs.append(doc)
        # get unique docs
        docs = list(set(docs))
        return docs

    def run_retriever(state: State) -> State:
        SUMMARY_PROMPT = """
        Given a question and a body of text, identify if the information needed to answer the question is present and summarize the text 
        Do not make up answers, if no relevant text is found then return NONE.
        \n\nQuestion:{question}
        \nData Available:
        \n{data}
        """
        results = []
        for sub_question in state['sub_questions']:
            result = retriever(sub_question)
            if not result:
                continue
            identified_result = invoke_formatter({"question": sub_question, "data": result},
                                                 SUMMARY_PROMPT,
                                                 IdentifyData,
                                                 llm_groq_tiny)
            if identified_result['relevant_data'] == "NONE":
                continue
            try:
                results.append({"relevant": identified_result['relevant_data'], "context": result})
            except KeyError:
                print(f"Error parsing relevant data {result}\n\n for question {sub_question}")

        state['results'] = results
        return state

    # Initialize the LangGraph state graph with the defined state
    graph = StateGraph(State)

    # Add nodes to the graph: decompose_question and run_retriever
    graph.add_node("decompose_question", decompose_question)
    graph.add_node("run_retriever", run_retriever)

    # Set up the flow: Start with decomposition, then run retriever, and end
    graph.set_entry_point("decompose_question")
    graph.add_edge("decompose_question", "run_retriever")
    graph.add_edge("run_retriever", END)

    # Compile the graph into a runnable application
    app = graph.compile()

    # Example invocation with an input question
    return app.invoke({"input_question": question})


if __name__ == "__main__":
    question = "What is the hazard or wild fires?"
    thread_id = 'Jane_Joe_20241004010814'
    filters = {}
    ai_librarian(question, thread_id, filters)