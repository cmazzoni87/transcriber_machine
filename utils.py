import streamlit as st
import base64
import re


def parse_transcript(transcript):
    """Parses the transcript and returns a dictionary of speakers and their lines."""
    speaker_pattern = r'^(Speaker [A-Z][a-zA-Z0-9]*):\s*(.*)'
    lines = transcript.strip().split('\n')
    speakers = {}
    for line in lines:
        match = re.match(speaker_pattern, line)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()
            if speaker not in speakers:
                speakers[speaker] = {'lines': [], 'word_count': 0}
            speakers[speaker]['lines'].append(text)
            speakers[speaker]['word_count'] += len(text.split())
    return speakers


def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    allowed_extensions = {'mp3', 'wav', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def replace_speaker_names(transcript, name_mapping):
    """
    Replaces speaker labels in the transcript with provided names.
    """
    for original_name, new_name in name_mapping.items():
        if new_name.strip():
            # Use word boundaries to ensure exact matches
            transcript = re.sub(rf'\b{re.escape(original_name)}:', f'{new_name}:', transcript)
    return transcript


def is_alphanumeric(s):
    """
    Check if the string is alphanumeric.
    """
    s = s.replace(' ', '_')
    return s.isalnum()


def autoplay_audio(file_path: str):
    """
    Embeds an autoplaying audio player in the Streamlit app.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(
                md,
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.error("Audio file not found.")
