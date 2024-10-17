# chatbot.py

import streamlit as st
from agents.agent import chat_agent
from tools.diarization import process_audio, text_to_speech
from utils import autoplay_audio
from database import get_speakers_for_thread, get_all_thread_ids
from st_copy_to_clipboard import st_copy_to_clipboard as st_copy_button


def chatbot_page():
    st.header('Chatbot Interface')
    if st.button('Logout', key='chatbot_logout_button'):
        st.session_state.logged_in = False
        st.rerun()

    db_session = st.session_state.get('db_session', None)

    if db_session is None:
        st.error("Database session is not initialized. Please log in again.")
        return

    # Get all available thread_ids
    available_thread_ids = get_all_thread_ids(db_session)

    if not available_thread_ids:
        st.warning('No threads available. Please upload and process audio files first.')
        return

    # Select Thread ID from the sidebar
    selected_thread_id = st.sidebar.selectbox(
        'Select Thread ID (Conversation Identifier ID)',
        available_thread_ids,
        key='thread_id_selectbox_chatbot'
    )

    # Fetch the list of speakers for the selected thread_id from the database
    speakers = get_speakers_for_thread(selected_thread_id, db_session)
    speaker_options = ['All Speakers'] + speakers

    # Add the Data Type Dropdown to the sidebar
    data_type_selection = st.sidebar.selectbox(
        'Select Data Type',
        ['Transcripts', 'Report'],
        key='data_type_selectbox_chatbot'
    )

    # Add speaker selection to the sidebar
    selected_speaker = st.sidebar.selectbox(
        'Select Speaker',
        speaker_options,
        key='speaker_selectbox'
    )

    # Initialize session state for chatbot messages
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize session state for sound path
    if 'sound_path' not in st.session_state:
        st.session_state.sound_path = None

    # User input
    user_input = st.text_input('You:', key='user_input')

    if st.button('Send', key='send_button'):
        if user_input.strip() == '':
            st.error('Please enter a question.')
        else:
            # Prepare filter parameters
            filter_params = {
                'thread_id': selected_thread_id,
                'speaker': None if selected_speaker == 'All Speakers' else selected_speaker,
                'limit': 10,
                'search_type': 'semantic',
            }
            # Generate response using chat_agent function
            with st.spinner('Generating response...'):
                try:
                    # Pass the data_type_selection and filter_params to chat_agent
                    chat_response = chat_agent(
                        user_input,
                        selected_thread_id,
                        data_type_selection,
                        filter_params,
                        st.session_state.username
                    )
                    if not chat_response:
                        st.error('No relevant information found for the selected thread.')
                        return
                    response = chat_response['answer']
                    references = chat_response.get('references', [])
                    # Prepend the conversation to the chat history
                    st.session_state.chat_history.insert(0, {'speaker': 'Assistant', 'message': response, 'references': references})
                    st.session_state.chat_history.insert(0, {'speaker': 'You', 'message': user_input})
                    sound_path = text_to_speech(response)
                    st.session_state.sound_path = sound_path  # Store in session_state
                except Exception as e:
                    st.error(f'Error generating response: {e}')
                    return

    # Define CSS styles for the chat bubbles
    st.markdown(
        """
        <style>
        /* Chat messages container */
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        /* User message bubble */
        .user-message {
            align-self: flex-start;
            background-color: #2E2E2E; /* Dark gray */
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            word-wrap: break-word;
        }
        /* Assistant message bubble */
        .assistant-message {
            align-self: flex-end;
            background-color: #404040; /* Slightly lighter dark gray */
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            word-wrap: break-word;
        }
        /* References styling */
        .references {
            align-self: flex-end;
            color: #BFBFBF; /* Light gray */
            font-size: 0.9em;
            margin: 5px 5px 15px 5px;
            max-width: 70%;
            word-wrap: break-word;
            text-align: right;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Display chat history with newest messages at the top
    chat_container = st.container()
    with chat_container:
        for chat_entry in st.session_state.chat_history:
            speaker = chat_entry['speaker']
            message = chat_entry['message']
            references = chat_entry.get('references', [])
            if speaker == 'You':
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="user-message">
                            {message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="assistant-message">
                            {message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # If there are references, display them
                if references:
                    refs_html = '<div class="references"><em>Sources:</em><br/>'
                    for ref in references:
                        source_text = ref.get('source', '')
                        source_speaker = ref.get('speaker', '')
                        refs_html += f"<em>- {source_speaker}: {source_text}</em><br/>"
                    refs_html += '</div>'
                    st.markdown(refs_html, unsafe_allow_html=True)

    # Redisplay the audio player if sound_path exists
    if 'sound_path' in st.session_state and st.session_state.sound_path:
        st.write("# Auto Response")
        autoplay_audio(st.session_state.sound_path)
