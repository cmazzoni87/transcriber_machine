import subprocess
import os
import re
from typing import List, Tuple, Union
from collections import defaultdict
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OpenAIEmbeddings
import streamlit as st



os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
os.environ["COHERE_KEY"] = st.secrets["COHERE_KEY"]
os.environ["CO_API_KEY"] = st.secrets["COHERE_KEY"]

def convert_markdown_to_pdf(mark_downfile_name: str) -> str:
    """
    Converts a Markdown file to a PDF document using the mdpdf command line application.

    Args:
        mark_downfile_name (str): Path to the input Markdown file.

    Returns:
        str: Path to the generated PDF file.
    """
    output_file = os.path.splitext(mark_downfile_name)[0] + '.pdf'

    # Command to convert markdown to PDF using mdpdf
    cmd = ['mdpdf', '--output', output_file, mark_downfile_name]

    # Execute the command
    subprocess.run(cmd, check=True)
    return output_file


def markdown_to_pdf(markdown_text: str):
    """
    Converts a markdown text to a PDF file, including images.

    :param markdown_text: The input markdown text as a string.
    """
    # Convert markdown to HTML
    # html_content = markdown.markdown(markdown_text)
    # # Inject CSS to resize images
    # css = """
    # <style>
    #     img {
    #         max-width: 100%;
    #         height: auto;
    #         max-height: 500px;
    #     }
    # </style>
    # """
    # # Add CSS to the HTML content
    # html_content = css + html_content
    # # save the html content to a file
    # with open(os.path.join(DATA_DIR, 'output.html'), 'w') as file:
    #     file.write(html_content)
    #
    # # Define the output path
    # output_path = os.path.join(DATA_DIR, 'output.pdf')
    #
    # # Convert HTML to PDF using WeasyPrint
    # HTML(string=html_content, base_url=DATA_DIR).write_pdf(output_path)
    # print(f"PDF generated and saved to {output_path}")
    output_path = "WIP"
    return output_path


def json_to_markdown(data: dict) -> str:
    markdown = []

    # Conversation Summary
    if "conversation_summary" in data:
        markdown.append("## Conversation Summary\n")
        summary = data["conversation_summary"]
        markdown.append(f"- **{summary['topic']}:** {summary['summary']}")
        markdown.append("")

    # Action Items
    if "action_items" in data:
        markdown.append("## Action Items\n")
        for item in data["action_items"]["action_items"]:
            markdown.append(f"- **Description:** {item['description']}")
            markdown.append(f"  - **Responsible Party:** {item['responsible_party']}")
            if item['deadline']:
                markdown.append(f"  - **Deadline:** {item['deadline']}")
            if item['additional_notes']:
                markdown.append(f"  - **Additional Notes:** {item['additional_notes']}")
            markdown.append("")

    # Sentiment Analysis
    if "sentiment_analysis" in data:
        markdown.append("## Sentiment Analysis\n")
        markdown.append(f"- **Overall Sentiment:** {data['sentiment_analysis']['overall_sentiment']}\n")
        for sentiment in data['sentiment_analysis']['detailed_sentiment']:
            markdown.append(f"- **{sentiment['speaker']}** ({sentiment['sentiment']}): {sentiment['remarks']}")
        markdown.append("")

    # Potential Priorities
    if "key_decisions" in data["key_decisions"]:
        markdown.append("## Potential Priorities\n")
        for priority in data["key_decisions"]["key_decisions"]:
            markdown.append(f"- **Decision:** {priority['decision']}")
            markdown.append(f"  - **Description:** {priority['description']}")
            markdown.append(f"  - **Reasoning:** {priority['reasoning']}")
            markdown.append("")

    return "\n".join(markdown)


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into words based on whitespace.
    """
    return text.strip().split()


def parse_transcript(transcript: str) -> List[str]:
    """
    Parses the transcript into a list of strings formatted as "Speaker: Text".
    """
    lines = transcript.strip().split('\n')
    dialogue = []
    speaker = None
    text_accumulator = []

    for line in lines:
        # Match lines that start with a speaker label
        match = re.match(r'^(\w+):\s*(.*)', line)
        if match:
            # Save the previous dialogue if any
            if speaker and text_accumulator:
                combined_text = f"{speaker}: {' '.join(text_accumulator).strip()}"
                dialogue.append(combined_text)
                text_accumulator = []
            speaker = match.group(1)
            text = match.group(2)
            text_accumulator.append(text)
        else:
            # Continuation of the previous speaker's dialogue
            text_accumulator.append(line.strip())

    # Add the last accumulated dialogue
    if speaker and text_accumulator:
        combined_text = f"{speaker}: {' '.join(text_accumulator).strip()}"
        dialogue.append(combined_text)

    return dialogue


def chunk_transcript(dialogue: List[str], max_tokens: int = 100) -> List[List[str]]:
    """
    Chunks the dialogue into pieces based on the maximum token limit.
    Each chunk is a list of strings formatted as "Speaker: Text".
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    for entry in dialogue:
        tokens = tokenize(entry)
        num_tokens = len(tokens)

        # Start a new chunk if the token limit is exceeded
        if current_token_count + num_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0

        current_chunk.append(entry)
        current_token_count += num_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_chunk_size(chunk: List[str]) -> int:
    """
    Returns the total number of tokens in the chunk.
    """
    total_tokens = sum(len(tokenize(entry)) for entry in chunk)
    return total_tokens


def get_total_transcript_size(chunks: List[List[str]]) -> int:
    """
    Returns the total number of tokens in the transcript.
    """
    total_tokens = sum(get_chunk_size(chunk) for chunk in chunks)
    return total_tokens


def determine_primary_speaker(chunk: List[str]) -> str:
    """
    Determines the primary speaker in a chunk based on the number of tokens spoken.
    """
    speaker_token_counts = defaultdict(int)

    for entry in chunk:
        # Extract speaker and text
        match = re.match(r'^(\w+):\s*(.*)', entry)
        if match:
            speaker = match.group(1)
            text = match.group(2)
            tokens = tokenize(text)
            num_tokens = len(tokens)
            speaker_token_counts[speaker] += num_tokens

    # Find the speaker with the maximum token count
    primary_speaker = max(speaker_token_counts, key=speaker_token_counts.get)
    return primary_speaker


def further_split_chunk(chunk: List[str], primary_speaker: str) -> List[str]:
    """
    Splits a large chunk into smaller, semantically coherent chunks.
    Adds the primary speaker's name at the beginning of each chunk.
    """
    # Combine the text from the chunk
    combined_text = '\n'.join(chunk)

    # Initialize the SemanticChunker
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )

    # Split the text
    docs = text_splitter.create_documents([combined_text])

    # Each doc represents a semantic chunk
    new_chunks = []
    for doc in docs:
        chunk_text = doc.page_content.strip()
        # Add the primary speaker's name at the beginning
        chunk_with_speaker = f"{primary_speaker} (Continued):\n{chunk_text}"
        new_chunks.append(chunk_with_speaker)

    return new_chunks


def process_chunks(chunks: List[List[str]], percent_threshold: float = 0.2) -> List[Union[List[str], str]]:
    """
    Processes the initial chunks to further split large chunks, retaining the primary speaker.
    """
    # Compute the total transcript size
    total_transcript_size = get_total_transcript_size(chunks)

    processed_chunks = []
    for chunk in chunks:
        chunk_size = get_chunk_size(chunk)
        if chunk_size > percent_threshold * total_transcript_size:
            # Determine the primary speaker in the chunk
            primary_speaker = determine_primary_speaker(chunk)
            # Split the chunk further
            new_chunks = further_split_chunk(chunk, primary_speaker)
            processed_chunks.extend(new_chunks)
        else:
            processed_chunks.append(chunk)
    return processed_chunks


def display_processed_chunks(chunks: List[Union[List[str], str]]):
    """
    Displays the processed chunks in a readable format.
    """
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} ---\n")
        if isinstance(chunk, list):
            # Original chunk format (list of strings)
            for entry in chunk:
                print(entry)
        else:
            # Chunk from SemanticChunker (string)
            print(chunk)


def split_transcript(transcript: str,
                     max_tokens: int = 100,
                     percent_threshold: float = 0.6) -> List[Union[List[str], str]]:
    """
    Splits a transcript into smaller, semantically coherent chunks based on token limits and a percentage threshold.
    """
    # Parse the transcript
    dialogue = parse_transcript(transcript)
    # Chunk the transcript
    chunks = chunk_transcript(dialogue, max_tokens=max_tokens)
    # Process the chunks with a percent threshold (e.g., 20% of total transcript size)
    return process_chunks(chunks, percent_threshold=percent_threshold)


def extract_speakers(transcript: str):
    """
    Extracts all unique speakers from the transcript using regex.
    Speakers are defined as words followed by a colon.
    """
    # Regular expression to match a speaker name followed by a colon
    pattern = r'^(\w+):'

    # Use findall to get all matches of speaker names
    speakers = re.findall(pattern, transcript, re.MULTILINE)

    # Return unique speakers
    return list(set(speakers))

#
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
# #
# # Parse the transcript
# dialogue = parse_transcript(transcript)
#
# # Chunk the transcript
# chunks = chunk_transcript(dialogue, max_tokens=100)
#
# # Process the chunks with a percent threshold (e.g., 20% of total transcript size)
# processed_chunks = process_chunks(chunks, percent_threshold=0.6)
# #
# # # Display the processed chunks
# display_processed_chunks(processed_chunks)
