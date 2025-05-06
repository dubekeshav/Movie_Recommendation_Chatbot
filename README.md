# Movie Recommendation Chatbot (CineMate 🎬)

A conversational AI assistant that provides movie recommendations and information using natural language processing. This chatbot leverages a language model to understand user queries and provides personalized movie recommendations based on a vector database of movie information.

## Overview

CineMate is a Streamlit-based chatbot application designed to help users discover movies and get information about them. The application uses a combination of vector search and language models to provide contextually relevant responses about movies.

## Features

- **Movie Recommendations**: Get personalized movie recommendations based on your preferences
- **Movie Information**: View detailed information about movies including cast, crew, ratings, and plot
- **Conversation Memory**: The chatbot remembers previous interactions for follow-up questions
- **Chat History**: Save and manage multiple chat sessions

## Project Structure

```
Movie_Recommendation_Chatbot/
├── app.py                      # Main streamlit application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── constant/                   # Constants and data processing scripts
│   ├── load_movies_data.py
│   └── process_movies_data.py
├── data/                       # Data storage
│   ├── movie_chat_history.db   # Chat history database
│   ├── movie_chat_sessions.db  # Chat sessions database
│   ├── cache/                  # Cache data
│   ├── chat/                   # Additional chat history files
│   ├── movies/                 # Movie datasets directory
│   ├── movies_data/            # Raw movie data directory
│   ├── processed/              # Processed data files
│   └── raw/                    # Raw data files
├── research/                   # Research notebooks
└── src/                        # Source code
    ├── data_processing/        # Data processing scripts
    │   ├── DataProcessing.py
    │   ├── fetch_update_omdb.py
    │   ├── load_movies_data.py
    │   ├── PineconeDataUpload.py
    │   ├── process_movies_data.py
    │   └── tmdb_fetch_data.py
    ├── models/                 # Model definition and initialization
    │   ├── ModelInitialization.py
    │   └── State.py
    └── utils/                  # Utility functions
```

## Setup Instructions

### 1. Create a virtual environment:
```bash
python -m venv .venv
```

### 2. Activate the virtual environment:

**For macOS/Linux:**
```bash
source .venv/bin/activate
```

**For Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install the dependencies:
Run the following code in the root directory of the project:
```bash
pip install -r requirements.txt
```

### 4. Install watchdog for better file watching:
```bash
pip install watchdog
```

### 5. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 6. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter your movie-related questions or requests in the chat input
2. The chatbot will respond with relevant movie information and recommendations
3. You can ask for:
   - Movie recommendations based on genres (e.g., "Recommend some good sci-fi movies")
   - Information about specific movies (e.g., "Tell me about Inception")
   - Similar movies to ones you like (e.g., "What movies are similar to The Matrix?")
4. Create new chat sessions using the "Add Chat" button
5. Rename chats by entering a new name and clicking "Rename"
6. Delete chats using the trash icon

## Technical Architecture

The application is built with the following components:

- **Streamlit**: Powers the web interface with reactive components
- **LangChain**: Orchestrates the language model and vector store
- **LangGraph**: Creates a directed graph for conversation flow
- **Groq**: Provides the LLM for natural language understanding
- **Pinecone**: Vector database for storing and retrieving movie information
- **HuggingFace Embeddings**: Converts text to vector embeddings

## Data Flow

1. User inputs a question about movies
2. The question is processed and converted to embeddings
3. Relevant movie data is retrieved from Pinecone vector store
4. The LLM generates a contextual response using the retrieved data
5. The response is presented to the user in the chat interface

## Troubleshooting

If you encounter file watcher errors:
- The app is configured to use watchdog by default
- If issues persist, run: `streamlit run --server.fileWatcherType none app.py`

## License

This project is open source and available under the MIT License.