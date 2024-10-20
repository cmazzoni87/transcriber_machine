import streamlit as st
from login import login_page, create_account_page
from audio_processing import upload_page
from chatbot import chatbot_page
from globals import initialize_globals


def main():
    # st.set_page_config(page_title='Audio Processing App', layout='wide')
    st.title('Audio Processing App')
    initialize_globals()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        page = st.sidebar.selectbox('Navigation', ['Login', 'Create Account'], key='navigation_selectbox')
        if page == 'Login':
            login_page()
        elif page == 'Create Account':
            create_account_page()
    else:
        page = st.sidebar.selectbox('Navigation', ['Upload & Process Audio', 'Chatbot'], key='navigation_selectbox_logged_in')
        if page == 'Upload & Process Audio':
            upload_page()
        elif page == 'Chatbot':
            chatbot_page()


if __name__ == "__main__":
    main()
