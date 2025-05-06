# Disable Streamlit file watcher before any other imports
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Import necessary libraries
import streamlit as st
import shelve
import dbm
import glob
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.models.ModelInitialization import compile_pipeline, init_prompt, init_env_vars, init_llm

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Set up page configuration
st.set_page_config(page_title="Movie Recommendation App", layout="wide")

# Simple styling for the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #2b313e;
    }
    .bot-message {
        background-color: #1e2533;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("CineMate 🎬 - Movie Recommendation")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Initialize environment and models
load_dotenv()
init_env_vars()
llm = init_llm()

# Set up prompt and pipeline
prompt = init_prompt()
app = compile_pipeline(llm)

# Set avatars
USER_AVATAR = '🥷'
BOT_AVATAR = '🐼'

# Database functions
def clear_db_files(db_name):
    pattern = f'./data/{db_name}.db*'
    files = glob.glob(pattern)
    for file in files:
        try:
            os.remove(file)
            print(f"Removed old database file: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")

def load_chat_sessions():
    try:
        with shelve.open('./data/movie_chat_sessions', flag='c', protocol=4) as db:
            return db.get('chat_sessions', [])
    except dbm.error:
        clear_db_files('movie_chat_sessions')
        with shelve.open('./data/movie_chat_sessions', flag='c', protocol=4) as db:
            return []

def save_chat_sessions(chat_sessions):
    try:
        with shelve.open('./data/movie_chat_sessions', flag='c', protocol=4) as db:
            db['chat_sessions'] = chat_sessions
    except dbm.error:
        clear_db_files('movie_chat_sessions')
        with shelve.open('./data/movie_chat_sessions', flag='c', protocol=4) as db:
            db['chat_sessions'] = chat_sessions

def load_chat_history(chat_name):
    try:
        with shelve.open('./data/movie_chat_history', flag='c', protocol=4) as db:
            return db.get(chat_name, [])
    except dbm.error:
        clear_db_files('movie_chat_history')
        with shelve.open('./data/movie_chat_history', flag='c', protocol=4) as db:
            return []

def save_chat_history(chat_name, messages):
    try:
        with shelve.open('./data/movie_chat_history', flag='c', protocol=4) as db:
            db[chat_name] = messages
    except dbm.error:
        clear_db_files('movie_chat_history')
        with shelve.open('./data/movie_chat_history', flag='c', protocol=4) as db:
            db[chat_name] = messages

def delete_chat_history(chat_name):
    try:
        with shelve.open('./data/movie_chat_history', flag='c', protocol=4) as db:
            if chat_name in db:
                del db[chat_name]
                print(f"Deleted chat history for: {chat_name}")
                return True
    except dbm.error as e:
        print(f"Error deleting chat history: {e}")
    return False

# Initialize session state
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = ["New Chat"]
if 'selected_chat' not in st.session_state:
    st.session_state.selected_chat = st.session_state.chat_sessions[0]
if 'new_chat_name' not in st.session_state:
    st.session_state.new_chat_name = ""

# Sidebar for chat sessions
with st.sidebar:
    st.sidebar.subheader("Chat Sessions")
    
    for chat_name in st.session_state.chat_sessions:
        cols = st.sidebar.columns([0.8, 0.2])
        with cols[0]:
            if cols[0].button(chat_name, key=f"select_{chat_name}", use_container_width=True):
                st.session_state.selected_chat = chat_name
        with cols[1]:
            if cols[1].button("🗑️", key=f"delete_{chat_name}"):
                st.session_state.chat_sessions.remove(chat_name)
                if chat_name in st.session_state.chat_histories: 
                    del st.session_state.chat_histories[chat_name]
                
                delete_chat_history(chat_name)
                save_chat_sessions(st.session_state.chat_sessions)
                
                if st.session_state.chat_sessions:
                    st.session_state.selected_chat = st.session_state.chat_sessions[0]
                else:
                    st.session_state.chat_sessions = ["New Chat"]
                    st.session_state.selected_chat = "New Chat"
                    
                st.rerun()

    # New chat controls
    st.session_state.new_chat_name = st.text_input("New Chat Name", value=st.session_state.new_chat_name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add Chat", use_container_width=True):
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
            st.session_state.new_chat_name = ""
            st.rerun()
            
    with col2:
        if st.button("Rename", use_container_width=True):
            old_name = st.session_state.selected_chat
            new_name = st.session_state.new_chat_name
            
            if new_name and new_name != old_name:
                index = st.session_state.chat_sessions.index(old_name)
                st.session_state.chat_sessions[index] = new_name
                
                if old_name in st.session_state.chat_histories:
                    st.session_state.chat_histories[new_name] = st.session_state.chat_histories[old_name]
                    del st.session_state.chat_histories[old_name]
                
                st.session_state.selected_chat = new_name
                
                save_chat_sessions(st.session_state.chat_sessions)
                save_chat_history(new_name, st.session_state.chat_histories[new_name])
                
                st.session_state.new_chat_name = ""
                
                st.rerun()

# Initialize chat histories
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

if st.session_state.selected_chat not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.selected_chat] = load_chat_history(st.session_state.selected_chat)

# Display chat messages
messages = st.session_state.chat_histories[st.session_state.selected_chat]

for message in messages:
    avatar = USER_AVATAR if message['role'] == 'user' else BOT_AVATAR
    message_class = "user-message" if message['role'] == 'user' else "bot-message"
    
    with st.chat_message(message['role'], avatar=avatar):
        st.markdown(f"<div class='chat-message {message_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about movies or get recommendations..."):
    messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user', avatar=USER_AVATAR):
        st.markdown(f"<div class='chat-message user-message'>{prompt}</div>", unsafe_allow_html=True)

    with st.chat_message('bot', avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            # Use the compiled pipeline to get the answer
            inputs = {"question": prompt, "llm": llm}
            response = app.invoke(inputs)
            
            # Get the simplified text response
            text_response = response.get("answer", "No answer returned.")
            
            st.markdown(f"<div class='chat-message bot-message'>{text_response}</div>", unsafe_allow_html=True)
            
            # Save the response
            messages.append({'role': 'bot', 'content': text_response})
            
            # Save updated chat history
            st.session_state.chat_histories[st.session_state.selected_chat] = messages
            save_chat_history(st.session_state.selected_chat, messages)