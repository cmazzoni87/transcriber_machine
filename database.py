from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import streamlit as st
from models import User, Thread, Transcript, Report, Speakers, Base
import json
from sqlalchemy.ext.declarative import declarative_base
from storage.qcs_database_manager import download_databases, upload_databases, get_gcs_client
from storage.memory_manager import VectorStoreManager, storage_root


# Base = declarative_base()
GCS_BUCKET_NAME = 'meeting-notes-storage-1122987'  # Replace with your bucket name


def get_speakers_for_thread(thread_id, db_session):
    """Retrieves speakers for a given thread_id from the database."""
    speakers_records = db_session.query(Speakers).filter_by(thread_id=thread_id).all()
    speakers_set = set()
    for record in speakers_records:
        speakers_list = json.loads(record.speakers_list)
        speakers_set.update(speakers_list)
    return sorted(list(speakers_set))

@st.cache_resource
def get_global_db_session():
    """
    Creates and caches the global database session.
    Ensures the tables are created if they do not exist.
    """
    creds_db_path = storage_root / "creds.db"
    engine = create_engine(f'sqlite:///{creds_db_path}', connect_args={"check_same_thread": False})
    # Ensure tables are created
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

global_db_session = get_global_db_session()


def initialize_local_databases_for_user(username):
    local_user_data_directory = storage_root / username
    local_user_data_directory.mkdir(parents=True, exist_ok=True)

    # For SQL database
    user_db_path = local_user_data_directory / 'user_db.db'
    engine = create_engine(f'sqlite:///{user_db_path}')
    # Ensure tables are created using the same Base
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    st.session_state['db_session'] = db_session

    # For vector store
    lancedb_path = local_user_data_directory / 'captain_logs'
    lancedb = VectorStoreManager(str(lancedb_path))
    st.session_state['lancedb'] = lancedb


def create_user_data_directory_in_cloud(user_data_directory):
    """Creates a user data directory in the cloud."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    # Create an empty blob to represent the directory
    blob = bucket.blob(user_data_directory)  #+ '/')
    blob.upload_from_string('')
    print(f'Created user data directory {user_data_directory} in cloud.')


def download_user_data_directory_from_cloud(user_data_directory, username):
    """
    Downloads the user's data directory from the cloud to a local directory.
    """
    local_user_data_directory = storage_root / username
    local_user_data_directory.mkdir(parents=True, exist_ok=True)
    download_databases(
        bucket_name=GCS_BUCKET_NAME,
        destination_folder=local_user_data_directory,
        prefix=user_data_directory
    )
    return local_user_data_directory

def upload_user_data_directory_to_cloud(user_data_directory, username):
    """
    Uploads the user's data directory to the cloud.
    """
    local_user_data_directory = storage_root / username
    upload_databases(
        bucket_name=GCS_BUCKET_NAME,
        source_folder=local_user_data_directory,
        prefix='',
        overwrite=True
    )

def store_thread_id(thread_id, db_session):
    """
    Stores a new thread_id in the database.
    """
    existing_thread = db_session.query(Thread).filter_by(thread_id=thread_id).first()
    if not existing_thread:
        new_thread = Thread(thread_id=thread_id)
        db_session.add(new_thread)
        db_session.commit()

def get_all_thread_ids(db_session):
    """
    Retrieves all thread_ids from the database.
    """
    if db_session is None:
        st.error("Database session is not initialized.")
        return []
    threads = db_session.query(Thread).all()
    return [thread.thread_id for thread in threads]
