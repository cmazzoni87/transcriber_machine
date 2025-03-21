from pydantic import BaseModel
from agents import ASSISTANT, ACTION_ITEMS, SENTIMENT_ANALYSIS, CONVERSATION_SUMMARY, KEY_DECISIONS, QA_ANSWER
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.structure import MeetingAnalysis, ConversationSummary, SentimentAnalysis, ActionItems, KeyDecisions, AnswerWithSources
from tools.txt_preprocessor import json_to_markdown
from agents.question_decomposition import ai_librarian
import asyncio
import datetime
import streamlit as st
import os

KEY_AI = st.secrets["OPENAI_KEY"]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
os.environ["COHERE_KEY"] = st.secrets["COHERE_KEY"]
os.environ["CO_API_KEY"] = st.secrets["COHERE_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# llm_groq = ChatGroq(
#     model="llama-3.1-70b-versatile",
#     temperature=0.35,
#     max_tokens=8000,
#     timeout=None,
#     max_retries=4,
# )


def task_breakdown_fail_over_response(_message: dict,
                   input_vars=None,
                   prompt: str = ASSISTANT,
                   pydantic_style: BaseModel = MeetingAnalysis,
                   temp=0.35) -> dict:
    """
    Analyze the sentiment of a document given a context.
    :param _message: The _message to analyze
    :param input_vars: The input variables to use
    :param pydantic_style: The pydantic style to use
    :param prompt: The prompt to use
    """
    if input_vars is None:
        input_vars = ["transcript"]
    input_str = ''
    for i in input_vars:
        input_str += f'{{{i}}}'
    model = ChatOpenAI(temperature=temp, model="gpt-4o-mini", max_tokens=8000, api_key=KEY_AI)
    # model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.35, max_tokens=8000, timeout=None,  max_retries=4,)
    parser = JsonOutputParser(pydantic_object=pydantic_style)
    prompt_template = PromptTemplate(
        template=f"{prompt}" + "\n{format_instructions}\n" + input_str,
        input_variables=input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt_template | model | parser

    return chain.invoke(_message)


async def task_breakdown(
    _message: dict,
    input_vars=None,
    prompt: str = ASSISTANT,
    pydantic_style: BaseModel = MeetingAnalysis,
    temp=0.35,
) -> dict:
    """
    Analyze the sentiment of a document given a context.
    """
    if input_vars is None:
        input_vars = ["transcript"]
    input_str = "".join(f"{{{i}}}" for i in input_vars)
    # model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.35, max_tokens=2000, timeout=None,  max_retries=4,)
    model = ChatOpenAI(
        temperature=temp, model="gpt-4o-mini", max_tokens=8000, api_key=KEY_AI
    )
    parser = JsonOutputParser(pydantic_object=pydantic_style)
    prompt_template = PromptTemplate(
        template=f"{prompt}\n{{format_instructions}}\n{input_str}",
        input_variables=input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt_template | model | parser

    return await chain.ainvoke(_message)


async def retry_task_breakdown(
    _message: dict,
    input_vars=None,
    prompt: str = ASSISTANT,
    pydantic_style: BaseModel = MeetingAnalysis,
    temp=0.35,
    retries=3,
    delay=1,
) -> dict:
    """
    Retry the task_breakdown function if it fails, with a delay between retries.
    """
    for attempt in range(retries):
        try:
            return await task_breakdown(
                _message=_message,
                input_vars=input_vars,
                prompt=prompt,
                pydantic_style=pydantic_style,
                temp=temp,
            )
        except Exception as e:
            if attempt < retries - 1:
                print(
                    f"Task failed, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})"
                )
                await asyncio.sleep(delay)
            else:
                print(f"Task failed after {retries} retries: {e}")
                return None


async def async_notes_agent(transcript_, background=""):
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        transcript_ = f"Today's Date: {today}\n" + transcript_
        task_args = {
            "transcript": f"{transcript_}",
            "background": f"{background}",
        }
        input_vars = ["transcript", "background"]

        tasks = [
            retry_task_breakdown(
                task_args,
                input_vars,
                pydantic_style=ConversationSummary,
                prompt=CONVERSATION_SUMMARY,
            ),
            retry_task_breakdown(
                task_args,
                input_vars,
                pydantic_style=ActionItems,
                prompt=ACTION_ITEMS,
            ),
            retry_task_breakdown(
                task_args,
                input_vars,
                pydantic_style=SentimentAnalysis,
                prompt=SENTIMENT_ANALYSIS,
            ),
            retry_task_breakdown(
                task_args,
                input_vars,
                pydantic_style=KeyDecisions,
                prompt=KEY_DECISIONS,
            ),
        ]

        results_list = await asyncio.gather(*tasks)

        (
            summary,
            action_items,
            sentiment_analysis,
            potential_priorities,
        ) = results_list

        # check if any of the results are None
        counter = 0
        for result in results_list:
            if result is None:
                if counter == 0:
                    summary = task_breakdown({"transcript": f"{transcript_}",
                                          "background": f"{background}"},
                                         ["transcript", "background"],
                                         pydantic_style=ConversationSummary,
                                         prompt=CONVERSATION_SUMMARY)
                elif counter == 1:
                    action_items = task_breakdown({"transcript": f"{transcript_}",
                                       "background": f"{background}"},
                                      ["transcript", "background"],
                                      pydantic_style=ActionItems,
                                      prompt=ACTION_ITEMS)
                elif counter == 2:
                    sentiment_analysis = task_breakdown({"transcript": f"{transcript_}",
                                             "background": f"{background}"},
                                            ["transcript", "background"],
                                            pydantic_style=SentimentAnalysis,
                                            prompt=SENTIMENT_ANALYSIS)
                elif counter == 3:
                    potential_priorities = task_breakdown({"transcript": f"{transcript_}",
                                    "background": f"{background}"},
                                    ["transcript", "background"],
                                    pydantic_style=KeyDecisions,
                                    prompt=KEY_DECISIONS)
            counter += 1

        results = {
            "action_items": action_items,
            "sentiment_analysis": sentiment_analysis,
            "conversation_summary": summary,
            "key_decisions": potential_priorities,
        }
        try:
            markdown_result = json_to_markdown(results)
        except Exception as e:
            print(f"Error converting to markdown: {e}")
            model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.05, max_tokens=5000, timeout=None,
                             max_retries=4, )
            prompt = f"""
            Generate a report using the dictionary below, the keys are the titles and the values are the content.
            Content should be formatted in markdown. Do not include any information that is not present in the dictionary.
            Content:
            {str(results)}
            """
            markdown_result = model.invoke(prompt).content
        return markdown_result

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def notes_agent(transcript_, background=""):
    vals = asyncio.run(async_notes_agent(transcript_, background))
    return vals


def check_sentence_is_similar(sentence_generated, sentence_original):
    """
    Check if the generated sentence is similar to the original sentence
    """
    sentence_generated = sentence_generated.lower()
    sentence_original = sentence_original.lower()
    if sentence_generated in sentence_original or sentence_original in sentence_generated:
        return True
    return False


def chat_agent(user_input, thread_id, data_type_selection, filter_params=None, username=None):

    try:
        context = ai_librarian(user_input, thread_id, data_type_selection, filter_params, username=username)
        # if context is empty return None
        if not context:
            return None
        answer = task_breakdown_fail_over_response({"question": f"{user_input}",
                                 "context": f"{str(context)}"},
                                ["question", "context"],
                                pydantic_style=AnswerWithSources,
                                prompt=QA_ANSWER,
                                temp=0.05)

        return answer

    except Exception as e:
        print(f"Error processing: {e}")
        return None


if __name__ == '__main__':
    path = r'C:\Users\cmazz\PycharmProjects\transcriber_machine\documents\upinder_transcript.txt'
    with open(path, 'r') as file:
        transcript = file.read()
    time_start = datetime.datetime.now()
    print(notes_agent(transcript))
    time_finish = datetime.datetime.now()
    time_diff = time_finish - time_start
    print("time to run notes_agent {}".format(time_diff.total_seconds()))

    # snippets = str([{
    #      'text': "Claudio: So would you say, would you describe the Amazon Q layer being used by power users. I've heard that before and I don't know if that is the correct term to use in this presentation. And I say that because he used the term advanced practitioners. So I just want to know if that is part of the language that we need to incorporate in this presentation.\nRupinder: So advanced practitioners is the bottom layer, right?\nClaudio: Any infrastructure?",
    #      'speakers': 'Claudio, Rupinder'}, {
    #      'text': "Claudio: Okay. So actually I have a couple ideas. So, you know, generative AI is such a broad concept right now and it's so, and it's so quickly evolving. Here at AWS, we've decided that we want to break this down into three layers. One is the most like most core infrastructure layer, which is where GPU's and the resources come in. And it's very much catered for data scientists and domain .... Like my, I call it the elevator pitch. The idea is to press a button and by the time I get to.",
    #      'speakers': 'Claudio'}, {
    #      'text': 'Rupinder: The top, I think, you know what we should do is if you want to talk about this, right, so you say that before we dive into bedrock, just want to level set the view and the vision we have in, in Amazon about generative AI stack.\nClaudio: Okay.',
    #      'speakers': 'Claudio, Rupinder'}])
    # question = "What are the three layers of the generative AI stack at AWS?"
    # query = "what is causing the air issues in NYC?"
    # thread_id = 'host_guest_20241011124110'
    # print(chat_agent(query, thread_id))


