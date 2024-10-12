from pydub import AudioSegment
from tools import DATA_DIR
import assemblyai as aai
import os
import streamlit as st
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from openai import OpenAI
# import soundfile as sf
# import sounddevice as sd
# import io
import re


KEY_AI = st.secrets["OPENAI_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
# Set your AssemblyAI API key from an environment variable
aai.settings.api_key = st.secrets["ASSEMBLYAI_KEY"]


def process_audio_old(file_path):
    # Transcription configuration with speaker labeling enabled
    config = aai.TranscriptionConfig(speaker_labels=True)

    # Create a transcriber instance
    transcriber = aai.Transcriber()

    # Perform transcription and speaker diarization
    transcript = transcriber.transcribe(
        file_path,
        config=config
    )
    file_name = os.path.basename(file_path).split(".")[0]
    # Format the output text with speaker labels
    diarized_text = []
    for utterance in transcript.utterances:
        diarized_text.append(f"Speaker {utterance.speaker}: {utterance.text}")

    transcript = "\n".join(diarized_text)
    # save the transcript to a file
    with open(os.path.join(DATA_DIR, f"{file_name}_transcript.txt"), "w") as f:
        f.write(transcript)

    return transcript


def cleanup(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass


def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")


def ensure_speaker_labels(transcript: str) -> str:
    """
    Ensures that every line in the transcript starts with 'Speaker (number):'.

    Args:
        transcript (str): Original transcript.

    Returns:
        str: Transcript with properly formatted speaker labels.
    """
    # Regex to match lines that start with "Speaker (number):"
    speaker_pattern = re.compile(r'^(Speaker \d+):')

    lines = transcript.strip().split('\n')
    formatted_lines = []
    current_speaker = None

    for line in lines:
        line = line.strip()
        if speaker_pattern.match(line):
            # If line starts with 'Speaker (number):', update current_speaker
            current_speaker = speaker_pattern.match(line).group(1)
            # replace the Speaker number with a alphanumeric character A, B, C, D, ...
            if current_speaker.split(" ")[1].isnumeric():
                int_2_txt = chr(65 + int(current_speaker.split(" ")[1]))
                current_speaker_str = current_speaker.replace(str(current_speaker.split(" ")[1]), int_2_txt)
                line = line.replace(current_speaker, current_speaker_str)
            if line != "":
                formatted_lines.append(line)
        else:
            # If line doesn't start with 'Speaker (number):', assume the current speaker continues
            if current_speaker:
                # replace the Speaker number with a alphanumeric character A, B, C, D, ...
                if current_speaker.split(" ")[1].isnumeric():
                    int_2_txt = chr(65 + int(current_speaker.split(" ")[1]))
                    current_speaker = current_speaker.replace(str(current_speaker.split(" ")[1]), int_2_txt)
                if line != "":
                    formatted_lines.append(f"{current_speaker}: {line}")

    return '\n'.join(formatted_lines)


def process_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Deepgram's API with speaker diarization.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Formatted transcript with speaker labels.

    Raises:
        Exception: If transcription fails or API returns an error.
    """
    # Retrieve Deepgram API Key from Streamlit secrets
    deepgram_api_key = st.secrets["DEEPGRAM_KEY"]

    if not deepgram_api_key:
        raise ValueError("Deepgram API Key not found in Streamlit secrets.")

    # Initialize Deepgram client
    deepgram = DeepgramClient(deepgram_api_key)

    # Read the audio file
    try:
        with open(file_path, "rb") as file:
            buffer_data = file.read()
    except Exception as e:
        raise Exception(f"Failed to read audio file: {e}")

    # Configure Deepgram options for audio analysis
    options = PrerecordedOptions(
        model="nova-2",  # Choose the appropriate model
        smart_format=True,  # Enables smart formatting (e.g., dates, times)
        diarize=True,  # Enables speaker diarization
        punctuate=True,  # Adds punctuation to the transcript
        language="en-US"  # Set the language as per your audio
    )

    # Prepare the file source
    payload: FileSource = {
        "buffer": buffer_data,
    }

    # Call the transcribe_file method with the payload and options
    try:
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    except Exception as e:
        raise Exception(f"Deepgram API request failed: {e}")
    response = response.to_dict()
    # Check if the transcription was successful
    if not response.get('results') or not response['results'].get('channels'):
        raise Exception("No transcription results found.")

    # Process the transcription results
    try:
        transcript = response['results']['channels'][0]['alternatives'][0]['paragraphs']['transcript']
        # remove new line if the new line does not start with the word Speaker
        return ensure_speaker_labels(transcript)

    except Exception as e:
        raise Exception(f"Failed to process transcription results: {e}")

#
# def text_to_speech(response: str):
#     client = OpenAI()
#     spoken_response = client.audio.speech.create(
#         model="tts-1",
#         voice="echo",
#         response_format="opus",
#         input=response
#     )
#
#     buffer = io.BytesIO()
#     for chunk in spoken_response.iter_bytes(chunk_size=4096):
#         buffer.write(chunk)
#     buffer.seek(0)
#
#     with sf.SoundFile(buffer, 'r') as sound_file:
#         data = sound_file.read(dtype='int16')
#         sd.play(data, sound_file.samplerate)
#         sd.wait()


def text_to_speech(response: str):
    client = OpenAI()
    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="echo",
            input=response,
    ) as response:
        response.stream_to_file("speech.mp3")



if __name__ == '__main__':
    import time
    start = time.time()
    # audio_path = r"C:\Users\cmazz\Downloads\AB1_Discussion.m4a"
    audio_path = r"C:\Users\cmazz\Downloads\20230607_me_canadian_wildfires.mp3"
    result = process_audio(audio_path)
    print(result)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    # start = time.time()
    # # audio_path = r"C:\Users\cmazz\Downloads\AB1_Discussion.m4a"
    # audio_path = r"C:\Users\cmazz\Downloads\20230607_me_canadian_wildfires.mp3"
    # result = process_audio_old(audio_path)
    # print(result)
    # end = time.time()
    # print(f"Time taken: {end - start} seconds")

