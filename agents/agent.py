from pydantic import BaseModel
from agents import ASSISTANT, REVIEWER, ACTION_ITEMS, SENTIMENT_ANALYSIS, CONVERSATION_SUMMARY, PRIORITIZE
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.structure import MeetingAnalysis, ConversationSummary, SentimentAnalysis, PotentialPriority, ActionItems, Priorities
from tools.txt_preprocessor import json_to_markdown
import datetime
import streamlit as st

KEY_AI = st.secrets["OPENAI_KEY"]


def task_breakdown(_message: dict,
                   input_vars=None,
                   prompt: str = ASSISTANT,
                   pydantic_style: BaseModel = MeetingAnalysis) -> dict:
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
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=7000, api_key=KEY_AI)
    parser = JsonOutputParser(pydantic_object=pydantic_style)
    prompt_template = PromptTemplate(
        template=f"{prompt}" + "\n{format_instructions}\n" + input_str,
        input_variables=input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt_template | model | parser

    return chain.invoke(_message)


def notes_agent(transcript_, background=''):
    try:
        # Sample transcript from text file
        today = datetime.date.today().strftime("%Y-%m-%d")
        transcript_ = f"Today's Date: {today}\n" + transcript_
        summary = task_breakdown({"transcript": f"{transcript_}",
                                  "background": f"{background}"},
                                 ["transcript", "background"],
                                 pydantic_style=ConversationSummary,
                                 prompt=CONVERSATION_SUMMARY)

        action_items = task_breakdown({"transcript": f"{transcript_}",
                                       "background": f"{background}"},
                                      ["transcript", "background"],
                                      pydantic_style=ActionItems,
                                      prompt=ACTION_ITEMS)

        sentiment_analysis = task_breakdown({"transcript": f"{transcript_}",
                                             "background": f"{background}"},
                                            ["transcript", "background"],
                                            pydantic_style=SentimentAnalysis,
                                            prompt=SENTIMENT_ANALYSIS)

        potential_priorities = task_breakdown({"transcript": f"{transcript_}",
                                    "background": f"{background}"},
                                    ["transcript", "background"],
                                    pydantic_style=Priorities,
                                    prompt=PRIORITIZE)

        results = {"action_items": action_items,
                   "sentiment_analysis": sentiment_analysis,
                   "conversation_summary": summary,
                   "potential_priorities": potential_priorities}

        markdown_result = json_to_markdown(results)
        return markdown_result

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


if __name__ == '__main__':
    path = r'C:\Users\cmazz\PycharmProjects\transcriber_machine\documents\upinder_transcript.txt'
    with open(path, 'r') as file:
        transcript = file.read()
    print(notes_agent(transcript))
