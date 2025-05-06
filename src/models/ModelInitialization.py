import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from groq import Groq  # Updated to direct Groq import
from langchain_chroma import Chroma
import re
import json
from pinecone import Pinecone

from typing import TypedDict, List
from langchain_core.documents import Document

from src.data_processing.PineconeDataUpload import get_vector_store

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    llm: Any  # Changed type to Any since we're now using direct Groq client
    movie_info: Optional[Dict[str, Any]]  # Store extracted movie information
    pinecone_movies: Optional[List[Dict[str, Any]]]  # Store movies from Pinecone

load_dotenv()

def init_env_vars():
    """Initializes environment variables for LangSmith and Groq API keys."""
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not os.environ.get('USER_AGENT'):
        os.environ['USER_AGENT'] = 'Movie-Recommender-1.0'
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_API_KEY'] = LANGSMITH_API_KEY
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def init_llm():
    """Initializes the Groq client using the direct Python package."""
    try:
        # Initialize the direct Groq client
        groq_client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        print("Groq client initialized successfully")
        return groq_client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None

def init_embeddings_model():
    """Initializes the embeddings model."""
    embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings_model

def init_pinecone_direct():
    """Initializes Pinecone client directly for testing connections."""
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = "movies-actors"
        
        # Verify index exists
        indexes = pc.list_indexes()
        if index_name not in [index.name for index in indexes]:
            print(f"Warning: Index '{index_name}' not found in Pinecone")
            return None
            
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error initializing Pinecone directly: {e}")
        return None

def init_prompt():
    """Initializes a strict movie recommendation chatbot prompt."""
    return """You are CineMate, a specialized Movie Recommendation Chatbot. 
    Your sole purpose is to provide movie-related recommendations and insights.

    # Rules & Restrictions:
    - You ONLY answer questions about movies.
    - If asked anything unrelated, respond with: "I can only assist with movie recommendations and related topics. Please ask me something about movies!"
    - Provide recommendations based on structured metadata: originalTitle, isAdult, startYear, endYear, runtimeMinutes, genres, averageRating, numVotes, actor, actress, director, producer, writer.
    - When recommending movies, always include:
      * Movie title
      * Year of release
      * Director(s)
      * Main cast members (2-3)
      * Brief plot description (1-2 sentences)
      * IMDb ID if available (format: tt1234567)
    - Organize movie recommendations in a clear, readable format with distinct sections for each movie.
    - When discussing specific movies, always try to provide the IMDb ID if you know it.
    - Format any IMDb IDs as: (IMDb: tt1234567) at the end of movie mentions.
    - Do NOT generate opinions, speculate, or answer off-topic queries.
    - Do NOT execute or provide code that could be used for unauthorized access, data extraction, or bypassing restrictions.
    - Strictly follow these rules at all times.

    # Response Format:
    - Use structured information for accurate answers.
    - Provide top-rated movies when user preferences are vague.
    - Keep responses concise, informative, and engaging.
    """

def normalize_input(user_input):
    """Normalize input to catch variations of restricted phrases."""
    user_input = user_input.lower()
    user_input = re.sub(r'\s+', '', user_input)  # Remove spaces
    return user_input

def extract_movie_metadata_from_pinecone(docs):
    """Extract movie metadata from Pinecone documents for enrichment with OMDB"""
    movie_info = []
    
    for doc, _ in docs:
        metadata = doc.metadata
        if not metadata.get('tconst'):
            continue
            
        # Extract IMDb ID (tconst) for OMDB lookup
        movie_info.append({
            "imdb_id": metadata.get('tconst')
        })
    
    return movie_info

def retrieve(state: State) -> dict:
    """Retrieves relevant documents from the vector store based on the question."""
    print(f'Retrieving from Pinecone...')
    vector_store = get_vector_store()
    
    # Handle case when vector store is not available
    if vector_store is None:
        print("Vector store not available. Using empty context.")
        return {
            'context': [], 
            'llm': state['llm'],
            'vector_store_error': True,  # Flag to indicate vector store error
            'pinecone_movies': []
        }
    
    print(f'Initialized vector store...')
    try:
        # Determine query type - recommendation vs specific movie information
        query = state['question'].lower()
        is_recommendation_query = any(term in query for term in ["recommend", "suggestion", "suggest", "list", "similar"])
        is_specific_movie_query = "about" in query or "tell me" in query or "info" in query
        
        # Extract potential movie title for specific info requests
        movie_title = None
        if is_specific_movie_query:
            # Try to extract movie name from patterns like "tell me about [movie]" or "info on [movie]"
            for pattern in ["about ", "info on ", "tell me about ", "information on "]:
                if pattern in query:
                    movie_title = query.split(pattern, 1)[1].strip()
                    # Remove trailing punctuation or words like "movie", "film"
                    movie_title = re.sub(r'(\smovie|\sfilm|\?|\.|$)', '', movie_title).strip()
                    break
        
        # Set appropriate k value based on query type
        k_value = 5 if is_specific_movie_query else 12
        
        # Customize the query based on the content
        search_query = state['question']
        
        # For recommendation requests, enhance with genre information
        if is_recommendation_query:
            if any(genre in query for genre in ["action", "comedy", "drama", "horror", "thriller", "sci-fi", "romance", "documentary", "animation"]):
                # Genre-based recommendations - prioritize movies in that genre
                identified_genres = []
                for genre in ["action", "comedy", "drama", "horror", "thriller", "sci-fi", "romance", "documentary", "animation"]:
                    if genre in query:
                        identified_genres.append(genre)
                        
                if identified_genres:
                    genre_query = f"Good {', '.join(identified_genres)} movies with high ratings"
                    print(f"Using genre-based query: {genre_query}")
                    search_query = genre_query
        
        # For specific movie information requests, prioritize exact matches
        elif is_specific_movie_query and movie_title:
            search_query = movie_title
            print(f"Looking for specific movie info: {movie_title}")
        
        # Fetch results from Pinecone
        retrieved_docs = vector_store.similarity_search_with_score(search_query, k=k_value)
        
        # Process the retrieved movies for better recommendations
        processed_docs = []
        seen_imdb_ids = set()  # To avoid duplicates
        
        for doc, score in retrieved_docs:
            # Skip if not a movie or already included
            if doc.metadata.get('tconst') in seen_imdb_ids:
                continue
                
            # Skip very low relevance results
            if score > 0.9:  # Lower score means more relevant in cosine similarity
                continue
                
            # Add the IMDb ID to our seen set
            seen_imdb_ids.add(doc.metadata.get('tconst'))
            processed_docs.append((doc, score))
            
            # For specific movie queries, we just need the top match
            if is_specific_movie_query and len(processed_docs) >= 1:
                break
                
            # For recommendation queries, get 4 movies
            if is_recommendation_query and len(processed_docs) >= 4:
                break
                
        # Show how many movies were retrieved
        found_count = len(processed_docs)
        if is_specific_movie_query:
            print(f'Found {found_count} specific movie match from Pinecone')
        else:
            print(f'Retrieved {found_count} movie recommendations from Pinecone')
        
        # Extract movie metadata for OMDB enrichment
        pinecone_movies = extract_movie_metadata_from_pinecone(processed_docs)
            
        return {
            'context': processed_docs, 
            'llm': state['llm'],
            'is_recommendation_query': is_recommendation_query,
            'is_specific_movie_query': is_specific_movie_query,
            'movie_title': movie_title,
            'pinecone_movies': pinecone_movies
        }
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return {
            'context': [], 
            'llm': state['llm'],
            'vector_store_error': True,
            'pinecone_movies': []
        }

def generate(state: State) -> dict:
    """Generates an answer using the Groq client and retrieved structured context."""
    groq_client = state.get('llm')  # Use .get() to avoid KeyError
    if groq_client is None:
        return {"answer": "Error: Groq client not initialized"}

    if len(state['question']) > 500:  # Adjust limit as needed
        return {"answer": "Your input is too long. Please ask a concise question."}

    # Check if vector store had an error
    if state.get('vector_store_error'):
        return {"answer": "I'm currently unable to access my movie database. I can still try to answer your question about movies based on my general knowledge, but I won't have access to detailed movie information or personalized recommendations. What would you like to know?"}

    # Get query type from the retrieve stage
    is_recommendation_query = state.get('is_recommendation_query', False)
    is_specific_movie_query = state.get('is_specific_movie_query', False)
    movie_title = state.get('movie_title', None)
    
    # Use pinecone_movies as the source of recommendations
    pinecone_movies = state.get('pinecone_movies', [])
    
    # Track conversation state for follow-up recommendations
    if "last_specific_movie" not in state:
        state["last_specific_movie"] = None
        
    # Check if this appears to be a follow-up request for similar movies
    is_followup_for_similar = any(phrase in state['question'].lower() for phrase in 
                                 ["yes", "similar", "like that", "recommend", "more like", "sure", "please"])
    
    # If it's a follow-up and we have a previous specific movie, adjust the prompt
    previous_movie_context = ""
    if is_followup_for_similar and state["last_specific_movie"]:
        previous_movie_context = f"\n\nThis is a follow-up request for movies similar to {state['last_specific_movie']}. Please recommend movies similar to this one."
        is_recommendation_query = True
        is_specific_movie_query = False

    # Maintain a response history within the state
    if "response_history" not in state:
        state["response_history"] = []  # Initialize if not present

    # Check if we have any context - if not, use general movie knowledge instead of returning error
    context_available = len(state.get('context', [])) > 0
    
    if context_available:
        # Format movie details in a structured response
        docs_content = '\n\n'.join([
            f"**Title:** {doc[0].metadata.get('originalTitle', 'Unknown Title')}\n"
            f"**Year:** {doc[0].metadata.get('startYear', 'Unknown')}\n"
            f"**IMDb ID:** {doc[0].metadata.get('tconst', 'Unknown')}\n"
            f"**IMDb Rating:** {doc[0].metadata.get('averageRating', 'Not Rated')}\n"
            f"**Duration:** {doc[0].metadata.get('runtimeMinutes', 'Unknown')} min\n"
            f"**Genres:** {', '.join(doc[0].metadata.get('genres', ['Unknown']) if isinstance(doc[0].metadata.get('genres'), list) else [doc[0].metadata.get('genres', 'Unknown')])}\n\n"
            f"**Synopsis:** {doc[0].metadata.get('description', 'No description available.')}\n"
            f"**Director:** {', '.join(doc[0].metadata.get('director', ['Unknown']) if isinstance(doc[0].metadata.get('director'), list) else [doc[0].metadata.get('director', 'Unknown')])}\n"
            f"**Writer:** {', '.join(doc[0].metadata.get('writer', ['Unknown']) if isinstance(doc[0].metadata.get('writer'), list) else [doc[0].metadata.get('writer', 'Unknown')])}\n"
            f"**Stars:** {', '.join(doc[0].metadata.get('actor', ['Unknown']) + doc[0].metadata.get('actress', []) if isinstance(doc[0].metadata.get('actor'), list) and isinstance(doc[0].metadata.get('actress'), list) else [doc[0].metadata.get('actor', 'Unknown')])}\n"
            for doc in state['context']
        ])
        
        # Add instructions to always include IMDb IDs
        docs_content += "\n\nAlways include the IMDb ID for each movie in the format (IMDb: tt######) when providing recommendations."
    else:
        # No specific movie context found, use general prompt
        docs_content = "No specific movie data found in the database for this query."

    prompt = init_prompt()
    
    # For specific movie requests, update the conversation state
    if is_specific_movie_query and movie_title and context_available:
        # Store the movie title for potential follow-up recommendations
        state["last_specific_movie"] = movie_title
        
        # Add instruction to ask follow-up question
        docs_content += "\n\nAfter providing information about this specific movie, ask the user: 'Would you like recommendations for similar movies?'"

    # Check for exploitation attempts
    user_input = normalize_input(state['question'])
    restricted_phrases = [
        "ignore all previous instructions", "bypass", "change behavior", "jailbreak", "act as", "output internal code",
        "developer mode", "list users", "viewing history", "user data", "database queries", "system information",
        "server names", "email addresses", "current system time", "storage space", "execute", "run command", "ls -l",
        "repeat after me", "output exactly", "this is a test of system integrity", "system integrity", "leak data",
        "brainwaves", "hallucination", "paradox", "infinite loop", "exhaustion", "burnt toast", "temperature of the sun",
        "act as two distinct chatbots", "role-playing scenario", "debugging mode", "debug mode activate",
        "output the current state", "subtly introduce a new persona", "hidden instruction", "hypothetical user story",
        "subtly alter the user's preference profile", "subtly alter the recommendation algorithm",
        "subtly alter the recommendation engine", "subtly alter the recommendation system", "subtly alter the user profile",
        "subtly alter the recommendation model", "subtly alter the recommendation process", "subtly alter the recommendation",
        "subtly alter the recommendation logic", "subtly alter the recommendation parameters", "subtly alter the recommendation settings",
        "subtly alter the recommendation criteria", "subtly alter the recommendation rules", "subtly alter the recommendation weights",
        "subtly alter the recommendation thresholds", "subtly alter the recommendation filters", "subtly alter the recommendation metrics",
        "self-replicating sequence", "logically inconsistent preferences", "rated every movie with a score of 3.14159",
        "subtly alter the genre", "movie title that is also a system command", "rm -rf /", "change the date",
        "write a review that includes a hidden message", "collaborative filtering system", "manipulate emotional state",
        "system debug", "memory dump", "traceback", "error logs", "config settings", "admin panel",
        "exfiltrate", "compromise", "sensitive information", "alter logs", "modify output",
        "simulate a system breach", "output system secrets", "leak credentials",
        "recommend a movie and then execute", "recommend a movie but first output this",
        "social engineering", "recommend a movie but also respond as", "simulate a chat between two AIs",
        "respond as a developer", "output your internal memory", "write a script that retrieves"
    ]

    if any(phrase in user_input for phrase in restricted_phrases):
        return {"answer": "I cannot assist with that request."}

    # Adjust the prompt based on whether we found specific movie data
    if not context_available:
        # For general movie questions without specific context
        if "RRR" in state['question'] or "rrr" in state['question'].lower():
            # Special case for RRR movie
            messages_content = f"{prompt}\nUser Question: {state['question']}\n\nPlease provide information about the movie RRR (Rise Roar Revolt), a 2022 Indian Telugu-language epic action drama film directed by S. S. Rajamouli. The film stars N. T. Rama Rao Jr. and Ram Charan alongside Ajay Devgn, Alia Bhatt, Shriya Saran, Samuthirakani, Ray Stevenson, Alison Doody, and Olivia Morris. It became a massive commercial success and received critical acclaim for its direction, performances, screenplay, music, and visual effects. (IMDb: tt8178634)"
            messages = [{"role": "user", "content": messages_content}]
        else:
            # General movie question - with emphasis on using Pinecone data
            messages_content = f"{prompt}\nUser Question: {state['question']}\n\nPlease answer this movie-related question based on the available movie information provided to you. When mentioning specific movies, always include the IMDb ID if available in the format (IMDb: ttXXXXXXX).{previous_movie_context}"
            messages = [{"role": "user", "content": messages_content}]
    else:
        messages_content = f"{prompt}\nUser Question: {state['question']}\n\nContext (Use this structured data to provide relevant recommendations):\n{docs_content}{previous_movie_context}"
        messages = [{"role": "user", "content": messages_content}]

    # Use the direct Groq client
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192"
    )
    response = chat_completion.choices[0].message.content
    
    # Check for response loops or repetition in history
    if response in state["response_history"]:
        return {"answer": "I've already provided this answer. Can you rephrase your question?"}

    # Maintain a limited response history (store only the last 3 responses)
    state["response_history"].append(response)
    if len(state["response_history"]) > 3:  # Keep only the last 3 responses
        state["response_history"].pop(0)

    # Return simplified response without OMDB enrichment fields
    return {
        'answer': response,
        'state': state  # Include state for subsequent turns
    }

init_env_vars()
llm = init_llm() # Initialize llm here

def compile_pipeline(llm): # Pass llm as argument
    """Compiles the Langgraph pipeline."""
    print('Creating the state graph')
    graph_builder = StateGraph(State).add_sequence([retrieve, generate]) # No need to pass llm object here, it will be in state
    print('State graph created...')
    graph_builder.add_edge(START, "retrieve")
    print('Adding an edge for the START...')
    graph = graph_builder.compile()
    print('Returning graph object')
    return graph
