# login.py

import streamlit as st
import os
from database import global_db_session, create_user_data_directory_in_cloud, initialize_local_databases_for_user, \
    download_user_data_directory_from_cloud, upload_creds_db_to_cloud
from models import User
from globals import encrypt_data, decrypt_data
from storage.memory_manager import storage_root


def login_page():
    st.header('Login')
    username = st.text_input('Username', key='login_username_input')
    password = st.text_input('Password', type='password', key='login_password_input')
    if st.button('Login', key='login_button'):
        user = global_db_session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            # Download the user's data directory from the cloud
            download_user_data_directory_from_cloud(username)
            # Initialize local databases with the user's data
            initialize_local_databases_for_user(username)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success('Logged in successfully!')
            st.rerun()
        else:
            st.error('Invalid username or password')


def create_account_page():
    st.header('Create Account')
    secret_key = st.text_input('Secret Key', key='secret_key_input', type='password')
    username = st.text_input('Username', key='create_username_input')
    password = st.text_input('Password', type='password', key='create_password_input')

    if st.button('Create Account', key='create_account_button'):
        account_creation_key = os.getenv("ACCOUNT_CREATION_KEY")
        if not account_creation_key:
            st.error("ACCOUNT_CREATION_KEY is not set in environment variables.")
            return

        if secret_key != account_creation_key:
            st.error('Invalid secret key')
        elif global_db_session.query(User).filter_by(username=username).first():
            st.error('Username already exists')
        else:
            # Create the user data directory in the cloud
            create_user_data_directory_in_cloud(username)
            # Initialize local databases for the user
            initialize_local_databases_for_user(username)
            # Create the user entry
            new_user = User(username=username, data_directory=username)
            new_user.set_password(password)
            global_db_session.add(new_user)
            global_db_session.commit()
            # Upload creds.db to cloud to ensure it's updated
            creds_db_path = storage_root / 'creds.db'
            upload_creds_db_to_cloud(creds_db_path)
            st.success('Account created successfully!')
            st.rerun()
