# audio_processing.py

import streamlit as st
import os
import re
import datetime
import base64
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from utils import allowed_file, parse_transcript, replace_speaker_names, is_alphanumeric
from storage.memory_manager import VectorStoreManager, notes_to_table, transcript_to_table, storage_root
from storage.qcs_database_manager import download_databases, upload_databases, get_gcs_client
from models import Thread, Transcript, Report, Speakers
from database import get_all_thread_ids, store_thread_id, upload_user_data_directory_to_cloud
from globals import GCS_BUCKET_NAME
from tools.diarization import process_audio, text_to_speech
from agents.agent import notes_agent
from st_copy_to_clipboard import st_copy_to_clipboard as st_copy_button
from mutagen import File as MutagenFile

def upload_page():
    st.header('Upload and Process Audio')
    if st.button('Logout', key='logout_button'):
        st.session_state.logged_in = False
        st.session_state.pop('db_session', None)
        st.session_state.pop('lancedb', None)
        st.rerun()

    db_session = st.session_state.get('db_session', None)
    lancedb = st.session_state.get('lancedb', None)

    if db_session is None or lancedb is None:
        st.error("Database session is not initialized. Please log in again.")
        return

    context = st.text_area('Context (optional)', '', key='context_input')
    uploaded_images = st.file_uploader(
        'Upload images (optional)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key='image_uploader'
    )

    # Initialize session state variables if they don't exist
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'speakers' not in st.session_state:
        st.session_state.speakers = None
    if 'name_mapping' not in st.session_state:
        st.session_state.name_mapping = {}
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'names_confirmed' not in st.session_state:
        st.session_state.names_confirmed = False
    if 'thread_id_to_use' not in st.session_state:
        st.session_state.thread_id_to_use = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'thread_selected' not in st.session_state:
        st.session_state.thread_selected = False
    if 'file_datetime' not in st.session_state:
        st.session_state.file_datetime = None
    if 'markdown_result' not in st.session_state:
        st.session_state.markdown_result = None

    allowed_extensions = {'mp3', 'wav', 'flac', 'm4a'}
    uploaded_file = st.file_uploader('Choose an audio file', type=list(allowed_extensions), key='audio_uploader')

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            if st.button('Process Audio', key='process_audio_button'):
                filename = secure_filename(uploaded_file.name)

                uploads_dir = Path('uploads')
                uploads_dir.mkdir(parents=True, exist_ok=True)
                file_path = uploads_dir / filename

                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success('File uploaded successfully')

                # Try to get the creation time from the file's metadata
                try:
                    audio = MutagenFile(file_path)
                    file_datetime = None
                    if audio is not None and audio.tags is not None:
                        date_tags = ['date', 'creation_date', 'year', 'recording_date', 'TDRC', 'TDEN']
                        for tag in date_tags:
                            if tag in audio.tags:
                                date_str = str(audio.tags[tag][0])
                                try:
                                    file_datetime = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                                    break
                                except ValueError:
                                    try:
                                        file_datetime = datetime.datetime.strptime(date_str, '%Y')
                                        break
                                    except ValueError:
                                        pass
                    if file_datetime is None:
                        file_datetime = datetime.datetime.now()
                except Exception:
                    file_datetime = datetime.datetime.now()

                st.session_state.file_datetime = file_datetime

                # Process the audio file and generate the transcript
                with st.spinner('Transcribing audio...'):
                    try:
                        transcript = process_audio(file_path)
                        st.session_state.transcript = transcript
                        st.success('Audio processed successfully')
                    except Exception as e:
                        st.error(f'Error processing audio: {e}')
                        return

                # Parse the transcript to get speakers and their lines
                st.session_state.speakers = parse_transcript(st.session_state.transcript)
                st.session_state.names_confirmed = False  # Reset confirmation
                st.rerun()
        else:
            st.error('File type not allowed')

    # After transcript is available and names are not yet confirmed
    if st.session_state.transcript and not st.session_state.names_confirmed:
        # Proceed to speaker identification without displaying the transcript
        min_word_count = 20
        speakers_to_rename = {
            speaker: data for speaker, data in st.session_state.speakers.items()
            if data['word_count'] >= min_word_count
        }

        st.header('Identify Speakers')
        st.write('Please provide names for the following speakers based on the sample lines.')

        with st.form('speaker_naming_form'):
            for speaker, data in speakers_to_rename.items():
                st.subheader(f'{speaker}')
                sample_lines = []
                for line in data['lines'][:2]:
                    if len(line.split()) > 35:
                        words = ' '.join(line.split()[:35]) + " ..."
                    else:
                        words = line
                    sample_lines.append(words)
                for line in sample_lines:
                    st.write(f'- {line}')
                name = st.text_input(f'Name for {speaker}', key=f'name_input_{speaker}')
                st.session_state.name_mapping[speaker] = name
            if st.form_submit_button('Confirm Speaker Names'):
                # Replace speaker labels in the transcript
                st.session_state.transcript = replace_speaker_names(
                    st.session_state.transcript,
                    st.session_state.name_mapping
                )
                st.session_state.names_confirmed = True
                st.rerun()

    # After names are confirmed
    if st.session_state.names_confirmed:
        # Display the updated transcript with names
        st.subheader('Updated Transcript with Names')
        st.text_area('Transcript', st.session_state.transcript, height=300, key='updated_transcript_display')
        st_copy_button(
            st.session_state.transcript,
            'Copy Transcript 📋',
            after_copy_label='Copied! ✅',
            key='copy_updated_transcript'
        )

    # Proceed with processing if not done yet
    if st.session_state.names_confirmed and not st.session_state.processing_done:
        # Proceed with processing if thread is selected
        if not st.session_state.thread_selected:
            # Get available thread_ids
            available_thread_ids = get_all_thread_ids(db_session)

            # Add a selection for thread ID generation method
            st.subheader('Select Thread ID Generation Method')

            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.2em;">Thread ID</span>
                <span style="font-size: 1em; margin-left: 5px;" title="A thread ID is a unique identifier that connects related conversations to one another">ℹ️</span>
            </div>
            """, unsafe_allow_html=True)

            thread_id_method = st.radio(
                'Choose how to set the Thread ID:',
                ('Automatic Generation', 'Custom Input'),
                key='thread_id_method_radio'
            )

            if thread_id_method == 'Automatic Generation':
                available_thread_ids_extended = ['Generate New'] + available_thread_ids
                selected_thread_id = st.selectbox(
                    'Select an existing thread ID or generate a new one:',
                    available_thread_ids_extended,
                    key='thread_id_selectbox'
                )
                custom_thread_id = None  # Not used in this method
            else:
                selected_thread_id = None  # Not used in this method
                custom_thread_id = st.text_input('Enter a custom Thread ID (alphanumeric only):', key='custom_thread_id_input')

            if st.button('Select Thread ID', key='select_thread_id_button'):
                if thread_id_method == 'Automatic Generation':
                    if selected_thread_id == 'Generate New':
                        # Automatic thread_id generation logic
                        file_datetime = st.session_state.file_datetime
                        timestamp = file_datetime.strftime("%Y%m%d%H%M%S")
                        participants = '_'.join([
                            name.strip().replace(' ', '') for name in st.session_state.name_mapping.values() if name.strip()
                        ])
                        thread_id_to_use = f"{participants}_{timestamp}"
                        # Ensure uniqueness
                        if db_session.query(Thread).filter_by(thread_id=thread_id_to_use).first():
                            st.error('Automatically generated Thread ID already exists. Please try again.')
                            return
                        # Store the new thread_id in the database
                        store_thread_id(thread_id_to_use, db_session)
                        st.success(f'Automatically generated Thread ID "{thread_id_to_use}" has been set successfully!')
                    else:
                        # Use the selected existing thread_id
                        thread_id_to_use = selected_thread_id
                else:
                    # Custom Thread ID path
                    if not custom_thread_id:
                        st.error('Please enter a Thread ID.')
                        return
                    if not is_alphanumeric(custom_thread_id):
                        st.error('Thread ID must be alphanumeric without spaces or special characters.')
                        return
                    # Check uniqueness
                    if db_session.query(Thread).filter_by(thread_id=custom_thread_id).first():
                        st.error('Thread ID already exists. Please choose a different one.')
                        return
                    thread_id_to_use = custom_thread_id
                    # Store the new custom thread_id in the database
                    store_thread_id(thread_id_to_use, db_session)
                    st.success(f'Custom Thread ID "{thread_id_to_use}" has been set successfully!')

                # Generate session_id using file creation time
                file_datetime = st.session_state.file_datetime
                session_id = f"session_id_{file_datetime.strftime('%Y%m%d%H%M%S')}"

                # Store the selected thread_id and session_id in session state
                st.session_state.thread_id_to_use = thread_id_to_use
                st.session_state.session_id = session_id
                st.session_state.thread_selected = True
                st.rerun()
        else:
            # Proceed with processing
            # Store the transcript in lancedb
            transcript_to_table(
                transcript=st.session_state.transcript,
                thread_id=st.session_state.thread_id_to_use,
                session_id=st.session_state.session_id,
                vectorstore=lancedb
            )

            # Store the transcript in the database
            new_transcript = Transcript(
                thread_id=st.session_state.thread_id_to_use,
                content=st.session_state.transcript
            )
            db_session.add(new_transcript)
            db_session.commit()

            # Store the speakers in the database
            speakers_list = list(set(st.session_state.name_mapping.values()))
            new_speakers = Speakers(
                thread_id=st.session_state.thread_id_to_use,
                speakers_list=json.dumps(speakers_list)
            )
            db_session.add(new_speakers)
            db_session.commit()

            # Generate notes using the updated transcript
            with st.spinner('Generating notes...'):
                try:
                    # Pass the context to notes_agent
                    markdown_result = notes_agent(st.session_state.transcript, context)
                    if markdown_result is None:
                        raise ValueError("notes_agent returned None")
                    st.success('Notes generated successfully')

                    # Process images if any
                    if uploaded_images:
                        image_descriptions = []
                        for image_file in uploaded_images:
                            # Placeholder for image description logic
                            description = f"Image Description: Placeholder description."
                            image_descriptions.append((image_file, description))

                        # Append images and descriptions to the report
                        markdown_result += '\n\n### Related Images\n'
                        for idx, (image_file, description) in enumerate(image_descriptions):
                            # Read image data
                            image_bytes = image_file.read()
                            # Reset file pointer to beginning
                            image_file.seek(0)
                            # Encode image in base64
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            image_extension = image_file.type.split('/')[-1]
                            image_data_url = f"data:image/{image_extension};base64,{image_base64}"
                            # Append image and description to markdown
                            markdown_result += f"#### Image {idx+1}\n"
                            markdown_result += f"![]({image_data_url})\n\n"
                            markdown_result += f"{description}\n\n"

                    # Store the notes in lancedb
                    document = markdown_result.split('\n\n### Related Images\n')[0]
                    entities = ', '.join([
                        name.strip() for name in st.session_state.name_mapping.values() if name.strip()
                    ])
                    notes_to_table(
                        document=document,
                        session_id=st.session_state.session_id,
                        thread_id=st.session_state.thread_id_to_use,
                        entities=entities,
                        image_tags=None,
                        bit_map_object=None,
                        vectorstore=lancedb
                    )

                    # Store the notes in the database
                    new_report = Report(
                        thread_id=st.session_state.thread_id_to_use,
                        content=markdown_result
                    )
                    db_session.add(new_report)
                    db_session.commit()

                    # Store the markdown result in session state
                    st.session_state.markdown_result = markdown_result

                except Exception as e:
                    st.error(f'Error generating notes: {e}')
                    return

                # Set processing done flag
                st.session_state.processing_done = True
                st.rerun()

    # Display the generated notes if processing is done
    if st.session_state.processing_done:
        st.subheader('Generated Notes')
        st.markdown(st.session_state.markdown_result, unsafe_allow_html=True)
        if "\n\n### Related Images\n" in st.session_state.markdown_result:
            markdown_result_copy = st.session_state.markdown_result.split("\n\n### Related Images\n")[0]
        else:
            markdown_result_copy = st.session_state.markdown_result
        st_copy_button(
            markdown_result_copy,
            'Copy Report 📋',
            after_copy_label='Copied! ✅',
            key='copy_report'
        )
        st.write('')
        # Upload the user data directory to the cloud
        upload_user_data_directory_to_cloud(storage_root / st.session_state.username, st.session_state.username)

        if st.button('Reset for Next Upload', key='reset_button'):
            # Reset session state variables
            st.session_state.transcript = None
            st.session_state.speakers = None
            st.session_state.name_mapping = {}
            st.session_state.processing_done = False
            st.session_state.names_confirmed = False
            st.session_state.thread_id_to_use = None
            st.session_state.session_id = None
            st.session_state.thread_selected = False
            st.session_state.file_datetime = None
            st.session_state.markdown_result = None
            st.rerun()
