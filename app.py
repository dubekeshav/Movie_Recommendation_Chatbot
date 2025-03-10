import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import shelve
import pandas as pd
import csv
from dotenv import load_dotenv
from backend.ModelInitialization import compile_pipeline, init_prompt, init_env_vars, init_llm, prepare_documents_for_splitting, split_document_into_chunks, indexing_documents

st.set_page_config(page_title="Movie Recommendation App", layout="wide")

load_dotenv()
init_env_vars()
llm = init_llm()

file_path = "/Users/mohitbhoir/Git/Movie_Recommendation_Chatbot/constant/output_movies_copy.txt"
df = pd.read_csv(file_path, sep="^", quoting=csv.QUOTE_ALL)
docs = prepare_documents_for_splitting(df)
all_splits = split_document_into_chunks(docs)
indexing_documents(all_splits)

prompt = init_prompt()
app = compile_pipeline(llm)

USER_AVATAR = '🥷'
BOT_AVATAR = '🐼'

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

    # Use the compiled pipeline to get the answer with structured context
    inputs = {"question": prompt, "llm": llm}
    response = app.invoke(inputs)
    full_response = response.get("answer", "No answer returned.")

    with st.chat_message('bot', avatar=BOT_AVATAR):
        st.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 5px; max-width: 100%;'>{full_response}</div>", unsafe_allow_html=True)
    messages.append({'role': 'bot', 'content': full_response})
    st.session_state.chat_histories[st.session_state.selected_chat] = messages
    save_chat_history(st.session_state.selected_chat, messages)