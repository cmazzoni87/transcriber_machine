from pydub import AudioSegment
from tools import DATA_DIR
import assemblyai as aai
import os
import streamlit as st


# Set your AssemblyAI API key from an environment variable
aai.settings.api_key = st.secrets["ASSEMBLYAI_KEY"]


def process_audio(file_path):
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


# if __name__ == '__main__':
#     import time
#     # import Path
#     from pathlib import Path
#     start = time.time()
#     audio_path = r"C:\Users\cmazz\Downloads\AB1_Discussion.m4a"
#     # input_path = audio_path
#     # output_path = audio_path
#     # convert_to_wav(input_path, output_path)
#     result = process_audio(audio_path)
#     print(result)
#     end = time.time()
#     print(f"Time taken: {end - start} seconds")
    # import torch
    #check if cuda is available
    # print(torch.cuda.is_available())

    # Start by making sure the `assemblyai` package is installed.
    # If not, you can install it by running the following command:
    # pip install -U assemblyai
    #
    # Note: Some macOS users may need to use `pip3` instead of `pip`.


