
import json
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Thread,  Speakers, Base
from storage.qcs_database_manager import (download_creds_db_from_cloud,
                                          upload_creds_db_to_cloud,
                                          download_databases,
                                          upload_databases, get_gcs_client)
from storage.memory_manager import VectorStoreManager, storage_root

GCS_BUCKET_NAME = 'meeting-notes-storage-1122987'

# Ensure storage_root is resolved and the directory exists
storage_root = storage_root.resolve()
storage_root.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_global_db_session():
    """
    Creates and caches the global database session.
    Ensures the creds.db is synchronized with GCS.
    """
    creds_db_path = storage_root / "creds.db"

    # Check if creds.db exists locally
    if not creds_db_path.exists():
        # Attempt to download creds.db from GCS
        try:
            download_creds_db_from_cloud(creds_db_path)
            # st.write("Downloaded creds.db from cloud storage.")
        except Exception as e:
            # st.write("creds.db not found in cloud storage. Creating a new one.")
            # Create the SQLite database
            engine = create_engine(f'sqlite:///{creds_db_path}', connect_args={"check_same_thread": False})
            Base.metadata.create_all(engine)
            # Upload the new creds.db to GCS
            upload_creds_db_to_cloud(creds_db_path)
    # else:
    #     st.write("Using existing creds.db.")

    # Now create the engine and session
    engine = create_engine(f'sqlite:///{creds_db_path}', connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


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


def create_user_data_directory_in_cloud(username):
    """Creates a user data directory in the cloud."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    # Create an empty blob to represent the directory
    user_data_directory = f'user_data/{username}/'
    blob = bucket.blob(user_data_directory)
    blob.upload_from_string('')
    print(f'Created user data directory {user_data_directory} in cloud.')


def download_user_data_directory_from_cloud(username):
    """
    Downloads the user's data directory from the cloud to a local directory.
    """
    local_user_data_directory = storage_root / username
    local_user_data_directory.mkdir(parents=True, exist_ok=True)
    download_databases(
        bucket_name=GCS_BUCKET_NAME,
        destination_folder=local_user_data_directory,
        prefix=f'user_data/{username}/'
    )
    return local_user_data_directory


def upload_user_data_directory_to_cloud(username):
    """
    Uploads the user's data directory to the cloud.
    """
    local_user_data_directory = storage_root / username
    upload_databases(
        bucket_name=GCS_BUCKET_NAME,
        source_folder=local_user_data_directory,
        prefix=f'user_data/{username}/',
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


def get_speakers_for_thread(thread_id, db_session):
    """Retrieves speakers for a given thread_id from the database."""
    speakers_records = db_session.query(Speakers).filter_by(thread_id=thread_id).all()
    speakers_set = set()
    for record in speakers_records:
        speakers_list = json.loads(record.speakers_list)
        speakers_set.update(speakers_list)
    return sorted(list(speakers_set))


global_db_session = get_global_db_session()