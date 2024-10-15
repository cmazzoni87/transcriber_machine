import datetime
import os
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account


def get_gcs_client():
    # Load the service account info from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)
    return client


def download_databases(bucket_name, destination_folder, prefix='databases/'):
    """Downloads all files from a specified GCS bucket and prefix to the local destination folder."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for blob in blobs:
        # Remove the prefix from the blob name to create a relative path
        relative_path = os.path.relpath(blob.name, prefix)
        # Skip the blob if it's the directory itself or if the blob represents a directory (ends with '/')
        if blob.name == prefix or relative_path == '.' or blob.name.endswith('/'):
            continue
        destination_path = os.path.join(destination_folder, relative_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)
        print(f'Downloaded {blob.name} to {destination_path}')


def upload_databases(bucket_name, source_folder, prefix='databases/', overwrite=False):
    """Uploads all files from the local source folder to a specified GCS bucket and prefix."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Create the relative path
            relative_path = os.path.relpath(local_path, source_folder)
            blob_name = os.path.join(prefix, relative_path).replace('\\', '/')
            blob = bucket.blob(blob_name)

            if not overwrite and blob.exists():
                # Append timestamp to the blob name for versioning
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                name, ext = os.path.splitext(blob_name)
                versioned_blob_name = f"{name}_{timestamp}{ext}"
                blob = bucket.blob(versioned_blob_name)
                print(f'Uploading versioned file {local_path} to {versioned_blob_name}')
            else:
                print(f'Uploading {local_path} to {blob_name}')

            blob.upload_from_filename(local_path)

# def main():
#     GCS_BUCKET_NAME = 'meeting-notes-storage-1122987'
#     from pathlib import Path
#     # get the directory where this file is
#     LOCAL_DB_FOLDER = str(Path(__file__).parent)
#     # Download the databases before starting the app
#     download_databases(bucket_name=GCS_BUCKET_NAME, destination_folder=LOCAL_DB_FOLDER)
#     # upload_databases(bucket_name=GCS_BUCKET_NAME, source_folder=r'C:\Users\cmazz\PycharmProjects\transcriber_machine\documents')

def get_gcs_client():
    """Creates a Google Cloud Storage client using credentials from Streamlit secrets."""
    # Load credentials from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)
    return client

# get_gcs_client()