# storage/qcs_database_manager.py

from google.cloud import storage
from pathlib import Path
import streamlit as st
from google.oauth2 import service_account

GCS_BUCKET_NAME = 'meeting-notes-storage-1122987'


def get_gcs_client():
    """
    Initializes and returns a Google Cloud Storage client using Streamlit secrets.
    """
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)
    return client


def download_creds_db_from_cloud(creds_db_path):
    """
    Downloads creds.db from GCS to the specified local path.
    """
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob('creds.db')
    if blob.exists():
        blob.download_to_filename(str(creds_db_path))
    else:
        raise FileNotFoundError("creds.db not found in GCS.")


def upload_creds_db_to_cloud(creds_db_path):
    """
    Uploads creds.db from the specified local path to GCS.
    """
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob('creds.db')
    blob.upload_from_filename(str(creds_db_path))
    st.write("Uploaded creds.db to cloud storage.")


def download_databases(bucket_name, destination_folder, prefix=''):
    """
    Downloads all files from GCS bucket with the given prefix to the local destination folder.
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # Compute relative path
        relative_path = Path(blob.name).relative_to(prefix)
        destination_path = Path(destination_folder) / relative_path
        if blob.name.endswith('/'):
            continue  # Skip directories
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination_path))


def upload_databases(bucket_name, source_folder, prefix='', overwrite=True):
    """
    Uploads all files from the local source folder to GCS bucket with the given prefix.
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    source_folder = Path(source_folder)
    for file_path in source_folder.rglob('*'):
        if file_path.is_file():
            blob_name = f"{prefix}{file_path.relative_to(source_folder).as_posix()}"
            blob = bucket.blob(blob_name)
            if not overwrite and blob.exists():
                continue
            blob.upload_from_filename(str(file_path))



