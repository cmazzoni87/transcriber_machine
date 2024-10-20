# globals.py

import streamlit as st
from cryptography.fernet import Fernet

# Define your GCS bucket name here
GCS_BUCKET_NAME = 'meeting-notes-storage-1122987'


def initialize_globals():
    """
    Initialize global session state variables.
    """
    if 'db_session' not in st.session_state:
        st.session_state.db_session = None
    if 'lancedb' not in st.session_state:
        st.session_state.lancedb = None


def get_encryption_key():
    """
    Retrieve the encryption key from Streamlit secrets.
    """
    key = st.secrets.get("ENCRYPTION_KEY")
    if not key:
        raise ValueError("ENCRYPTION_KEY not set in Streamlit secrets.")
    return key


def encrypt_data(data: str) -> str:
    """
    Encrypts the data using the encryption key.
    """
    key = get_encryption_key()
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data.decode()


def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypts the data using the encryption key.
    """
    key = get_encryption_key()
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data.encode())
    return decrypted_data.decode()