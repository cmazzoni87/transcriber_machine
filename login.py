import streamlit as st
import os
from database import global_db_session, create_user_data_directory_in_cloud, initialize_local_databases_for_user, download_user_data_directory_from_cloud
from models import User
# from globals import encrypt_data, decrypt_data


def login_page():
    st.header('Login')
    username = st.text_input('Username', key='login_username_input')
    password = st.text_input('Password', type='password', key='login_password_input')
    if st.button('Login', key='login_button'):
        user = global_db_session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            # Download the user's data directory from the cloud
            local_user_data_directory = download_user_data_directory_from_cloud(user.data_directory, username)
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
            # Generate a unique data directory for the user
            user_data_directory = f"{username}/"  # Adjust path as needed
            # Create the directory on the cloud
            create_user_data_directory_in_cloud(user_data_directory)
            # Initialize empty databases in the user's directory
            initialize_local_databases_for_user(username)
            # Create the user entry
            new_user = User(username=username, data_directory=user_data_directory)
            new_user.set_password(password)
            global_db_session.add(new_user)
            global_db_session.commit()
            st.success('Account created successfully!')
            st.rerun()
