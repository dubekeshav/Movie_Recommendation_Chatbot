import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import shelve
from dotenv import load_dotenv
from backend.ModelInitialization import compile_pipeline, init_prompt, init_env_vars, load_documents, split_docs, indexing_documents

st.set_page_config(page_title="Movie Recommendation App", layout="wide")

# st.markdown("""
#     <style>
#         .reportview-container {
#             margin-top: -2em;
#         }
#         #MainMenu {visibility: hidden;}
#         .stDeployButton {display:none;}
#         footer {visibility: hidden;}
#         #stDecoration {display:none;}
#     </style>
# """, unsafe_allow_html=True)

load_dotenv()
init_env_vars()
db = load_documents()
indexing_documents(split_docs(db))

# st.title("Movie Recommendation App")

USER_AVATAR = '🥷'
BOT_AVATAR = '🐼'

llm = compile_pipeline()
prompt = init_prompt()

def load_chat_sessions():
    with shelve.open('./data/movie_chat_sessions.db') as db:
        return db.get('chat_sessions', [])

def save_chat_sessions(chat_sessions):
    with shelve.open('./data/movie_chat_sessions.db') as db:
        db['chat_sessions'] = chat_sessions

def load_chat_history(chat_name):
    with shelve.open('./data/movie_chat_history.db') as db:
        return db.get(chat_name, [])

def save_chat_history(chat_name, messages):
    with shelve.open('./data/movie_chat_history.db') as db:
        db[chat_name] = messages

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = ["New Chat"]
if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = st.session_state.chat_sessions[0]
if 'new_chat_name' not in st.session_state:
    st.session_state.new_chat_name = ""

with st.sidebar:
    st.sidebar.subheader("Chat Sessions")
    for chat_name in st.session_state.chat_sessions:
        cols = st.sidebar.columns([0.8, 0.2])
        with cols[0]:
            if cols[0].button(chat_name, key=f"select_{chat_name}"):
                st.session_state.selected_chat = chat_name
        with cols[1]:
            if cols[1].button("🗑️", key=f"delete_{chat_name}"):
                st.session_state.chat_sessions.remove(chat_name)
                if chat_name in st.session_state.chat_histories: #Check for key existance
                    del st.session_state.chat_histories[chat_name]
                save_chat_sessions(st.session_state.chat_sessions)
                if os.path.exists(f'./data/movie_chat_history.db'):
                    os.remove(f'./data/movie_chat_history.db')
                if st.session_state.chat_sessions:
                    st.session_state.selected_chat = st.session_state.chat_sessions[0]
                else:
                    st.session_state.chat_sessions = ["New Chat"]
                    st.session_state.selected_chat = "New Chat"
                st.rerun()

    st.session_state.new_chat_name = st.text_input("New Chat Name", value=st.session_state.new_chat_name)
    if st.button("Add Chat"):
        if st.session_state.new_chat_name and st.session_state.new_chat_name not in st.session_state.chat_sessions:
            st.session_state.chat_sessions.append(st.session_state.new_chat_name)
            save_chat_sessions(st.session_state.chat_sessions)
            st.session_state.selected_chat = st.session_state.new_chat_name
            st.session_state.new_chat_name = "" #clear the input
            if st.session_state.new_chat_name in st.session_state.chat_histories:
                pass
            else:
                st.session_state.chat_histories[st.session_state.new_chat_name] = []
            st.rerun()

if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

if st.session_state.selected_chat not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.selected_chat] = load_chat_history(st.session_state.selected_chat)

messages = st.session_state.chat_histories[st.session_state.selected_chat]

for message in messages:
    col1, col2 = st.columns([0.25, 0.75])
    avatar = USER_AVATAR if message['role'] == 'user' else BOT_AVATAR

    if message['role'] == 'user':
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 5px; margin-left: auto; max-width: 100%;'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 5px; max-width: 100%;'>{message['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask a Question about Movies"):
    messages.append({'role': 'user', 'content': prompt})

    col1, col2 = st.columns([0.25, 0.75])

    with st.chat_message('user', avatar=USER_AVATAR):
        st.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 5px; margin-left: auto; max-width: 100%;'>{prompt}</div>", unsafe_allow_html=True)

    with st.chat_message('bot', avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ''
        try:
            query_response = llm.invoke({'question': prompt, 'context': []})
            full_response = query_response['answer']
        except Exception as e:
            print(str(e))
            full_response = f'An Error occurred: {e}'
        message_placeholder.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 5px; max-width: 100%;'>{full_response}</div>", unsafe_allow_html=True)
    messages.append({'role': 'bot', 'content': full_response})
    st.session_state.chat_histories[st.session_state.selected_chat] = messages
    save_chat_history(st.session_state.selected_chat, messages)