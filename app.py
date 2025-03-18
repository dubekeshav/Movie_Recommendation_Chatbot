import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import shelve
from dotenv import load_dotenv
from backend.ModelInitialization import compile_pipeline, init_prompt, init_env_vars, init_llm

st.set_page_config(page_title="Movie Recommendation App", layout="wide")

# Move the chatbot title to the top-left corner
st.title("CineMate 🎬")

load_dotenv()
init_env_vars()
llm = init_llm()

prompt = init_prompt()
app = compile_pipeline(llm)

USER_AVATAR = '🥷'
BOT_AVATAR = '🐼'

# Set background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("bg.png");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
            if cols[0].button(chat_name, key=f"select_{chat_name}", help="Select this chat session"):
                st.session_state.selected_chat = chat_name
        with cols[1]:
            if cols[1].button("🗑️", key=f"delete_{chat_name}"):
                st.session_state.chat_sessions.remove(chat_name)
                if chat_name in st.session_state.chat_histories: 
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
    with st.form(key='new_chat_form'):
        if st.form_submit_button("Add Chat"):
            new_chat_name = st.session_state.new_chat_name or "New Chat"
            if new_chat_name not in st.session_state.chat_sessions:
                st.session_state.chat_sessions.append(new_chat_name)
            else:
                base_name = new_chat_name
                counter = 1
                while new_chat_name in st.session_state.chat_sessions:
                    new_chat_name = f"{base_name}_{counter}"
                    counter += 1
                st.session_state.chat_sessions.append(new_chat_name)
            save_chat_sessions(st.session_state.chat_sessions)
            st.session_state.selected_chat = new_chat_name
            st.session_state.new_chat_name = ""  # clear the input immediately
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

    # Ensure structured formatting in responses
    formatted_response = ""

    if isinstance(full_response, dict):  # Check if full_response is a dictionary
        formatted_response = f"""
        - **Title:** {full_response.get('title', 'Unknown Title')}
        - **Year:** {full_response.get('year', 'Unknown')}
        - **IMDb Rating:** {full_response.get('rating', 'Not Rated')}
        - **Duration:** {full_response.get('duration', 'Unknown')} min
        - **Genres:** {', '.join(full_response.get('genres', ['Unknown']))}
        - **Synopsis:** {full_response.get('synopsis', 'No description available.')}
        - **Director:** {', '.join(full_response.get('director', ['Unknown']))}
        - **Writer:** {', '.join(full_response.get('writer', ['Unknown']))}
        - **Stars:** {', '.join(full_response.get('stars', ['Unknown']))}
        - **Streaming Availability:** {full_response.get('streaming', 'Not Available')}
        """
    elif isinstance(full_response, list):  # Handle multiple recommendations
        for movie in full_response:
            if isinstance(movie, dict):  # Ensure each movie is a dictionary
                formatted_response += f"""
                - **Title:** {movie.get('title', 'Unknown Title')}
                - **Year:** {movie.get('year', 'Unknown')}
                - **IMDb Rating:** {movie.get('rating', 'Not Rated')}
                - **Duration:** {movie.get('duration', 'Unknown')} min
                - **Genres:** {', '.join(movie.get('genres', ['Unknown']))}
                - **Synopsis:** {movie.get('synopsis', 'No description available.')}
                - **Director:** {', '.join(movie.get('director', ['Unknown']))}
                - **Writer:** {', '.join(movie.get('writer', ['Unknown']))}
                - **Stars:** {', '.join(movie.get('stars', ['Unknown']))}
                - **Streaming Availability:** {movie.get('streaming', 'Not Available')}
                \n---
                """
    else:  # If full_response is a string, return it as-is
        formatted_response = full_response

    with st.chat_message('bot', avatar=BOT_AVATAR):
        st.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 5px; max-width: 100%;'>{formatted_response}</div>", unsafe_allow_html=True)

    messages.append({'role': 'bot', 'content': formatted_response})
    st.session_state.chat_histories[st.session_state.selected_chat] = messages
    save_chat_history(st.session_state.selected_chat, messages)