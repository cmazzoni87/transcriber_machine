import os
import re
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet
# from dotenv import load_dotenv
from tools.diarization import process_audio
from agents.agent import notes_agent
# from tools.emailer import send_email_with_mailgun
# from tools.stats import analyze_transcript  # Assuming you have this module

# Load environment variables
# load_dotenv()


# Encryption key retrieval
# def get_encryption_key():
#     """Retrieve the encryption key from environment variables."""
#     key = os.getenv("ENCRYPTION_KEY")
#     if not key:
#         raise ValueError("ENCRYPTION_KEY not set in environment variables.")
#     return key

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
engine = create_engine('sqlite:///creds.db')
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
        upload_page()


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

                # Process the audio file and generate the transcript
                with st.spinner('Processing audio...'):
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
                sample_lines = data['lines'][:2]  # Get first two lines
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

        # Generate notes using the updated transcript
        with st.spinner('Generating notes...'):
            try:
                # Pass the context to notes_agent
                pdf_path, markdown_result = notes_agent(st.session_state.transcript, context)
                st.success('Notes generated successfully')
            except Exception as e:
                st.error(f'Error generating notes: {e}')
                return

        # Display the markdown result
        st.subheader('Generated Notes')
        st.markdown(markdown_result)

        # # Perform additional analyses and display images
        # st.subheader('Transcript Analysis')
        # analysis_images = analyze_transcript(st.session_state.transcript)
        #
        # for title, image_path in analysis_images.items():
        #     st.subheader(title)
        #     st.image(image_path, use_column_width=True)

        # Send the PDF via email automatically
        with st.spinner('Sending email...'):
            try:
                # send_email_with_mailgun(
                #     subject="Meeting Notes",
                #     body="Please find the attached meeting notes.",
                #     recipient_email=recipient_email,
                #     file_path=pdf_path
                # )
                st.success('Email sent successfully!')
            except Exception as e:
                st.error(f'Error sending email: {e}')

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
            st.rerun()


if __name__ == '__main__':
    main()
