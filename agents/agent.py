from pydantic import BaseModel
from agents import ASSISTANT, ACTION_ITEMS, SENTIMENT_ANALYSIS, CONVERSATION_SUMMARY, KEY_DECISIONS, QA_ANSWER
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.structure import MeetingAnalysis, ConversationSummary, SentimentAnalysis, ActionItems, KeyDecisions, AnswerWithSources
from tools.txt_preprocessor import json_to_markdown
import datetime
import streamlit as st

KEY_AI = st.secrets["OPENAI_KEY"]


def task_breakdown(_message: dict,
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
                                    pydantic_style=KeyDecisions,
                                    prompt=KEY_DECISIONS)

        results = {
            "action_items": action_items,
            "sentiment_analysis": sentiment_analysis,
            "conversation_summary": summary,
            "key_decisions": potential_priorities
                   }

        markdown_result = json_to_markdown(results)
        return markdown_result

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def chat_agent(user_input, context):
    try:
        answer = task_breakdown({"question": f"{user_input}",
                                 "context": f"{context}"},
                                ["question", "context"],
                                pydantic_style=AnswerWithSources,
                                prompt=QA_ANSWER,
                                temp=0.05)

        return answer

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


# if __name__ == '__main__':
    # path = r'C:\Users\cmazz\PycharmProjects\transcriber_machine\documents\upinder_transcript.txt'
    # with open(path, 'r') as file:
    #     transcript = file.read()
    # print(notes_agent(transcript))
    # snippets = str([{
    #      'text': "Claudio: So would you say, would you describe the Amazon Q layer being used by power users. I've heard that before and I don't know if that is the correct term to use in this presentation. And I say that because he used the term advanced practitioners. So I just want to know if that is part of the language that we need to incorporate in this presentation.\nRupinder: So advanced practitioners is the bottom layer, right?\nClaudio: Any infrastructure?",
    #      'speakers': 'Claudio, Rupinder'}, {
    #      'text': "Claudio: Okay. So actually I have a couple ideas. So, you know, generative AI is such a broad concept right now and it's so, and it's so quickly evolving. Here at AWS, we've decided that we want to break this down into three layers. One is the most like most core infrastructure layer, which is where GPU's and the resources come in. And it's very much catered for data scientists and domain .... Like my, I call it the elevator pitch. The idea is to press a button and by the time I get to.",
    #      'speakers': 'Claudio'}, {
    #      'text': 'Rupinder: The top, I think, you know what we should do is if you want to talk about this, right, so you say that before we dive into bedrock, just want to level set the view and the vision we have in, in Amazon about generative AI stack.\nClaudio: Okay.',
    #      'speakers': 'Claudio, Rupinder'}])
    # question = "What are the three layers of the generative AI stack at AWS?"
    # print(chat_agent(question, snippets))


# {'answer': 'The three layers of the generative AI stack at AWS include: 1) the core infrastructure layer, which is focused on GPUs and resources for data scientists; 2) the advanced practitioners layer; and 3) the top layer, which is not explicitly detailed in the provided context.', 'references': [{'source': "Claudio: Okay. So actually I have a couple ideas. So, you know, generative AI is such a broad concept right now and it's so, and it's so quickly evolving. Here at AWS, we've decided that we want to break this down into three layers. One is the most like most core infrastructure layer, which is where GPU's and the resources come in. And it's very much catered for data scientists and domain .... Like my, I call it the elevator pitch. The idea is to press a button and by the time I get to.", 'speaker': 'Claudio'}, {'source': 'Rupinder: So advanced practitioners is the bottom layer, right?', 'speaker': 'Rupinder'}]}
