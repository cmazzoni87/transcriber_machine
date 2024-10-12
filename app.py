import os
import re
import datetime
import base64
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet
from mutagen import File as MutagenFile
from tools.diarization import process_audio, text_to_speech
from agents.agent import notes_agent, chat_agent
from storage.memory_manager import (
    VectorStoreManager,
    notes_to_table,
    transcript_to_table,
    storage_root,
)

# Import the copy to clipboard component
from st_copy_to_clipboard import st_copy_to_clipboard as st_copy_button


# Initialize lancedb
lancedb = VectorStoreManager(storage_root / "captain_logs")


# Encryption key retrieval
def get_encryption_key():
    """Retrieve the encryption key from Streamlit secrets."""
    key = st.secrets["ENCRYPTION_KEY"]
    if not key:
        raise ValueError("ENCRYPTION_KEY not set in Streamlit secrets.")
    return key


def encrypt_data(data: str) -> str:
    """Encrypts the data using the encryption key."""
    key = get_encryption_key()
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data.decode()


def decrypt_data(encrypted_data: str) -> str:
    """Decrypts the data using the encryption key."""
    key = get_encryption_key()
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data.encode())
    return decrypted_data.decode()


# Database setup
engine = create_engine(f'sqlite:///{str(storage_root)}/creds.db')
Base = declarative_base()


# Define the User model
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(150), nullable=False)
    encrypted_info = Column(String(500), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def store_encrypted_info(self, info):
        self.encrypted_info = encrypt_data(info)

    def retrieve_encrypted_info(self):
        if self.encrypted_info:
            return decrypt_data(self.encrypted_info)
        return None


# Define the Thread model
class Thread(Base):
    __tablename__ = 'thread'
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(150), unique=True, nullable=False)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()


# Allowed file extensions
allowed_extensions = {'mp3', 'wav', 'flac', 'm4a'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def is_valid_email(email):
    """Simple regex check for email format."""
    regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    return re.match(regex, email)


def parse_transcript(transcript):
    """Parses the transcript and returns a dictionary of speakers and their lines."""
    speaker_pattern = r'^(Speaker [A-Z][a-zA-Z0-9]*):\s*(.*)'
    # speaker_pattern = r'^(Speaker [0-9]*):\s*(.*)'
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


def replace_speaker_names(transcript, name_mapping):
    """Replaces speaker labels in the transcript with provided names."""
    for original_name, new_name in name_mapping.items():
        if new_name.strip():
            # Use word boundaries to ensure exact matches
            transcript = re.sub(rf'\b{re.escape(original_name)}:', f'{new_name}:', transcript)
    return transcript


# Function to store a new thread_id
def store_thread_id(thread_id):
    """Stores a new thread_id in the database."""
    existing_thread = db_session.query(Thread).filter_by(thread_id=thread_id).first()
    if not existing_thread:
        new_thread = Thread(thread_id=thread_id)
        db_session.add(new_thread)
        db_session.commit()


# Function to get all available thread_ids
def get_all_thread_ids():
    """Retrieves all thread_ids from the database."""
    threads = db_session.query(Thread).all()
    return [thread.thread_id for thread in threads]


# Helper function to check if a string is alphanumeric
def is_alphanumeric(s):
    """Check if the string is alphanumeric."""
    s = s.replace(' ', '_')
    return s.isalnum()


def main():
    st.title('Audio Processing App')

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        page = st.sidebar.selectbox('Navigation', ['Login', 'Create Account'])
        if page == 'Login':
            login_page()
        elif page == 'Create Account':
            create_account_page()
    else:
        page = st.sidebar.selectbox('Navigation', ['Upload & Process Audio', 'Chatbot'])
        if page == 'Upload & Process Audio':
            upload_page()
        elif page == 'Chatbot':
            chatbot_page()

def login_page():
    st.header('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        user = db_session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success('Logged in successfully!')
            st.rerun()
        else:
            st.error('Invalid username or password')

def create_account_page():
    st.header('Create Account')
    secret_key = st.text_input('Secret Key')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Create Account'):
        if secret_key != os.getenv("ACCOUNT_CREATION_KEY"):
            st.error('Invalid secret key')
        elif db_session.query(User).filter_by(username=username).first():
            st.error('Username already exists')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db_session.add(new_user)
            db_session.commit()
            st.success('Account created successfully!')
            st.rerun()

def upload_page():
    st.header('Upload and Process Audio')
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.rerun()

    recipient_email = st.text_input('Recipient Email')
    context = st.text_area('Context (optional)', '')
    uploaded_images = st.file_uploader('Upload images (optional)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    # Initialize session state variables
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

    uploaded_file = st.file_uploader('Choose an audio file', type=list(allowed_extensions))

    if uploaded_file is not None and recipient_email:
        if not is_valid_email(recipient_email):
            st.error('Please enter a valid email address.')
            return

        if allowed_file(uploaded_file.name):
            if st.button('Process Audio'):
                filename = secure_filename(uploaded_file.name)
                file_path = os.path.join('uploads', filename)
                os.makedirs('uploads', exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success('File uploaded successfully')

                # Try to get the creation time from the file's metadata
                try:
                    audio = MutagenFile(file_path)
                    file_datetime = None
                    if audio is not None and audio.tags is not None:
                        # Try common tags for date
                        date_tags = ['date', 'creation_date', 'year', 'recording_date', 'TDRC', 'TDEN']
                        for tag in date_tags:
                            if tag in audio.tags:
                                date_str = str(audio.tags[tag][0])
                                try:
                                    # Try parsing the date string
                                    file_datetime = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                                    break
                                except ValueError:
                                    try:
                                        file_datetime = datetime.datetime.strptime(date_str, '%Y')
                                        break
                                    except ValueError:
                                        pass
                    if file_datetime is None:
                        # If date not found in tags, use current time
                        file_datetime = datetime.datetime.now()
                except Exception:
                    # If any error occurs, use current time
                    file_datetime = datetime.datetime.now()

                # Store the file_datetime in session state
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
    elif uploaded_file is not None and not recipient_email:
        st.error('Please enter the recipient email before uploading the file.')

    # After transcript is available
    if st.session_state.transcript and not st.session_state.names_confirmed:
        # Filter out speakers who spoke very little (e.g., less than 20 words)
        min_word_count = 20
        speakers_to_rename = {
            speaker: data for speaker, data in st.session_state.speakers.items()
            if data['word_count'] >= min_word_count
        }

        # Get names for speakers using a form
        st.header('Identify Speakers')
        st.write('Please provide names for the following speakers based on the sample lines.')

        with st.form('speaker_naming_form'):
            for speaker, data in speakers_to_rename.items():
                st.subheader(f'{speaker}')
                sample_lines = []
                for line in data['lines'][:2]:
                    if len(line.split()) > 35:
                        words = ' '.join(line.split()[:35]) + " ..."  # Display first 35 words
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
    if st.session_state.names_confirmed and not st.session_state.processing_done:
        # Display the updated transcript
        st.subheader('Updated Transcript')
        st.text_area('Transcript', st.session_state.transcript, height=300)
        # Add "Copy Transcript" button using the component
        st_copy_button(st.session_state.transcript, 'Copy Transcript 📋', after_copy_label='Copied! ✅')

        if not st.session_state.thread_selected:
            # Get available thread_ids
            available_thread_ids = get_all_thread_ids()

            # Add a selection for thread ID generation method
            st.subheader('Select Thread ID Generation Method')

            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.2em;">Thread ID</span>
                <span style="font-size: 1em; margin-left: 5px;" title="A thread Id is a unique identifier that connects related conversations to one another">ℹ️</span>
            </div>
            """, unsafe_allow_html=True)

            thread_id_method = st.radio(
                'Choose how to set the Thread ID:',
                ('Automatic Generation', 'Custom Input')
            )

            if thread_id_method == 'Automatic Generation':
                available_thread_ids_extended = ['Generate New'] + available_thread_ids
                selected_thread_id = st.selectbox(
                    'Select an existing thread ID or generate a new one:',
                    available_thread_ids_extended
                )
                custom_thread_id = None  # Not used in this method
            else:
                selected_thread_id = None  # Not used in this method
                custom_thread_id = st.text_input('Enter a custom Thread ID (alphanumeric only):')

            if st.button('Select Thread ID'):
                if thread_id_method == 'Automatic Generation':
                    if selected_thread_id == 'Generate New':
                        # Existing automatic thread_id generation logic
                        file_datetime = st.session_state.file_datetime
                        timestamp = file_datetime.strftime("%Y%m%d%H%M%S")
                        participants = '_'.join([
                            name.strip().replace(' ', '') for name in st.session_state.name_mapping.values() if name.strip()
                        ])
                        thread_id_to_use = f"{participants}_{timestamp}"
                        # Ensure uniqueness (optional, as timestamp is likely unique)
                        if db_session.query(Thread).filter_by(thread_id=thread_id_to_use).first():
                            st.error('Automatically generated Thread ID already exists. Please try again.')
                            return
                        # Store the new thread_id in the database
                        store_thread_id(thread_id_to_use)
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
                    store_thread_id(thread_id_to_use)
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
                            # description = describe_image(image_file)
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

                except Exception as e:
                    st.error(f'Error generating notes: {e}')
                    return

            # Display the markdown result
            st.subheader('Generated Notes')
            st.markdown(markdown_result, unsafe_allow_html=True)
            if "\n\n### Related Images\n" in markdown_result:
                markdown_result_copy = markdown_result.split("\n\n### Related Images\n")[0]
            else:
                markdown_result_copy = markdown_result
            # Add "Copy Report" button using the component
            st_copy_button(markdown_result_copy, 'Copy Report 📋', after_copy_label='Copied! ✅')
            st.write('')
            # Reset processing flags for next upload
            st.session_state.processing_done = True

    if st.session_state.processing_done:
        if st.button('Reset for Next Upload'):
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
            st.rerun()


def autoplay_audio(file_path: str):
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


autoplay_audio("local_audio.mp3")

def chatbot_page():
    st.header('Chatbot Interface')
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.rerun()

    # Get all available thread_ids
    available_thread_ids = get_all_thread_ids()

    if not available_thread_ids:
        st.warning('No threads available. Please upload and process audio files first.')
        return

    selected_thread_id = st.sidebar.selectbox('Select Thread ID (Conversations identifier ID)', available_thread_ids)

    # Add the Data Type Dropdown
    data_type_selection = st.selectbox('Select Data Type', ['Transcripts', 'Report'])

    # Add widgets for search type, prefilter, keyword search, limit
    search_type = 'semantic'
    if st.sidebar.checkbox('Use Full-Text Search (Useful when searching for specific keywords)'):
        search_type = 'fts'

    prefilter = st.sidebar.text_input('Prefilter (SQL expression)')
    if not prefilter.strip():
        prefilter = None

    key_word_search = st.sidebar.text_input('Keyword Search')
    if not key_word_search.strip():
        key_word_search = None

    limit = st.sidebar.number_input('Number of Results', min_value=1, max_value=100, value=10)

    # Initialize session state for chatbot messages
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input('You:', key='user_input')

    if st.button('Send'):
        if user_input.strip() == '':
            st.error('Please enter a question.')
        else:
            # Print parameters in the terminal
            # print(f"Selected Thread ID: {selected_thread_id}")
            # print(f"Data Type: {data_type_selection}")
            # print(f"Search Type: {search_type}")
            # print(f"Filter results by: {prefilter}")
            # print(f"Keyword Search: {key_word_search}")
            # print(f"Top number of references: {limit}")
            # print(f"User Input: {user_input}")
            filer_params = {
                'keyword': key_word_search,
                'thread_id': selected_thread_id,
                'prefilter': prefilter,
                'limit': limit,
                'search_type': search_type,
            }
            # Generate response using chat_agent function
            with st.spinner('Generating response...'):
                try:
                    # Pass the data_type_selection to chat_agent
                    chat_response = chat_agent(
                        user_input,
                        selected_thread_id,
                        data_type_selection,
                        filer_params
                    )
                    if not chat_response:
                        st.error('No relevant information found for the selected thread.')
                        return
                    response = chat_response['answer']
                    references = chat_response.get('references', [])
                    # Append the conversation to the chat history
                    st.session_state.chat_history.append({'speaker': 'You', 'message': user_input})
                    st.session_state.chat_history.append({'speaker': 'Assistant', 'message': response, 'references': references})
                    sound_path = text_to_speech(chat_response['answer'])
                    st.write("# Auto Response")
                    autoplay_audio(sound_path)
                except Exception as e:
                    st.error(f'Error generating response: {e}')
                    return

    # Display chat history
    for chat_entry in st.session_state.chat_history:
        speaker = chat_entry['speaker']
        message = chat_entry['message']
        if speaker == 'You':
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:** {message}", unsafe_allow_html=True)
            # If there are references, display them
            references = chat_entry.get('references', [])
            if references:
                st.markdown('---')
                st.markdown('**Sources:**')
                for ref in references:
                    source_text = ref.get('source', '')
                    source_speaker = ref.get('speaker', '')
                    # Display the source text and speaker
                    st.markdown(f"- **{source_speaker}**: {source_text}")


if __name__ == '__main__':
    main()
