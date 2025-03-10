import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing import TypedDict, List
from langchain_core.documents import Document
from dotenv import load_dotenv
# import fetch_update_omdb
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from typing import TypedDict, List
from langchain_core.documents import Document

from backend.PineconeDataUpload import get_vector_store

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    llm: ChatGroq  # Add LLM to the state

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
    """Initializes the language model (LLM)."""
    try:
        llm = ChatGroq(model_name='llama3-8b-8192', api_key=os.getenv('GROQ_API_KEY'))
        print("LLM Initialized with Groq")
        return llm
    except Exception as e:
        print(f"Error initializing LLM with Groq: {e}")
        return None

def init_embeddings_model():
    """Initializes the embeddings model."""
    embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings_model

def init_prompt():
    """Initializes the prompt from Langchain Hub."""
    prompt = hub.pull('rlm/rag-prompt')
    return prompt

def retrieve(state: State) -> dict:
    """Retrieves relevant documents from the vector store based on the question."""
    print(f'Retrieving...........')
    vector_store = get_vector_store()
    print(f'Initialized vector store...')
    retrieved_docs = vector_store.similarity_search(state['question'], k=5)
    print(f'Retrieved documents: {retrieved_docs}')
    return {'context': retrieved_docs, 'llm': state['llm']}  # Ensure LLM is passed

def generate(state: State) -> dict:
    """Generates an answer using the LLM and retrieved structured context."""
    llm = state.get('llm')  # Use .get() to avoid KeyError
    if llm is None:
        return {"answer": "Error: LLM not initialized"}
    print(f'{llm} is fine..........')
    # Combine retrieved documents; each document is a structured comma-separated string.
    docs_content = '\n\n'.join([doc.page_content for doc in state['context']])
    # Update the prompt input to include instructions for structured context
    prompt = init_prompt()
    prompt_input = {
        'question': state['question'],
        'context': docs_content,
        'instruction': (
            "The context provided is a structured, comma-separated string containing movie metadata "
            "in the following order: originalTitle, isAdult, "
            "startYear, endYear, runtimeMinutes, genres, averageRating, numVotes, actor, actress, "
            "director, producer, writer. Use this structured context to generate an accurate answer."
        )
    }
    messages = prompt.invoke(prompt_input)
    response = llm.invoke(messages)
    print(f'Returning generated response ...')
    return {'answer': response.content}

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

# if __name__ == "__main__":
#     docs = prepare_documents_for_splitting(df)
#     all_splits = split_document_into_chunks(docs)
#     indexing_documents(all_splits) # Index documents only once

#     app = compile_pipeline(llm) # Pass llm to compile_pipeline

#     if llm is None:
#         print("Error: LLM not initialized. Exiting.")
#         exit(1)
        
#     inputs = {"question": "Recommend me a brad pitt movie", 'llm': llm} # Pass llm in the input state
#     response = app.invoke(inputs)
#     print(f"Final Response: {response}")



# def generate_and_store_embeddings():
#     """Generates embeddings using BAAI/bge-large-en and stores them in ChromaDB."""
#     import pandas as pd
#     from langchain_huggingface import HuggingFaceEmbeddings
#     from langchain_chroma import Chroma

#     # Load and clean movie data
#     file_path = "/mnt/data/output_movies_copy.txt"
#     df = pd.read_csv(file_path, sep="^", quoting=3)
#     df.columns = [col.replace('"', '') for col in df.columns]
#     df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
#     df.replace(r'\\N', None, inplace=True, regex=True)
#     numeric_cols = ["startYear", "endYear", "runtimeMinutes", "averageRating", "numVotes"]
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Initialize the embeddings model using BAAI/bge-large-en
#     embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

#     # Initialize ChromaDB storage
#     persist_directory = "./chroma_db"
#     vector_store = Chroma(embedding_function=embedding_model, persist_directory=persist_directory)

#     # Prepare documents for embedding
#     def create_movie_documents(df):
#         docs = []
#         for _, row in df.iterrows():
#             metadata = {
#                 "tconst": row["tconst"],
#                 "titleType": row["titleType"],
#                 "startYear": row["startYear"],
#                 "genres": row["genres"],
#                 "actor": row["actor"],
#                 "actress": row["actress"],
#                 "director": row["director"],
#                 "producer": row["producer"],
#                 "writer": row["writer"],
#             }
#             content = f"{row['primaryTitle']} ({row['startYear']}) - {row['genres']}"
#             docs.append((content, metadata))
#         return docs

#     movie_documents = create_movie_documents(df)

#     # Batch insert documents into ChromaDB
#     batch_size = 5000  # Prevent batch overflow errors
#     for i in range(0, len(movie_documents), batch_size):
#         batch = movie_documents[i : i + batch_size]
#         texts, metadatas = zip(*batch)
#         vector_store.add_texts(texts=texts, metadatas=list(metadatas))
#         print(f"Inserted batch {i // batch_size + 1}")

#     print("✅ Embeddings generated and stored in ChromaDB!")

# if __name__ == "__main__":
#     Check for command-line argument "embed" to run embeddings generation
#     if len(sys.argv) > 1 and sys.argv[1] == "embed":
#         generate_and_store_embeddings()
#     else:
#         # Existing main block code
#         docs = prepare_documents_for_splitting(df)
#         all_splits = split_document_into_chunks(docs)
#         indexing_documents(all_splits)  # Index documents only once

#         app = compile_pipeline(llm)  # Pass llm to compile_pipeline

#         if llm is None:
#             print("Error: LLM not initialized. Exiting.")
#             exit(1)
            
#         inputs = {"question": "Recommend me a brad pitt movie", "llm": llm}  # Pass llm in the input state
#         response = app.invoke(inputs)
#         print(f"Final Response: {response}")
        
        

# def init_vector_store():
#     """Initializes the Chroma vector store."""
#     embeddings_model = init_embeddings_model()
#     persist_directory = './chroma_db'
#     if not os.path.exists(persist_directory):
#         os.makedirs(persist_directory)
#     vector_store = Chroma(embedding_function=embeddings_model, persist_directory=persist_directory)
#     return vector_store

# # Pinecone Initialization
# def init_pinecone(api_key: str, environment: str, index_name: str):
#     """Initializes Pinecone."""
#     pinecone.init(api_key=api_key, environment=environment)
#     if index_name not in pinecone.list_indexes():
#         pinecone.create_index(index_name, dimension=1536, metric="cosine") #Adjust dimension to your embedding model dimension.
#     return pinecone.Index(index_name)

# def init_pinecone_vector_store(api_key: str, environment: str, index_name: str):
#     """Initializes the Pinecone vector store."""
#     embeddings_model = init_embeddings_model()
#     pinecone_index = init_pinecone(api_key, environment, index_name)
#     vector_store = Pinecone(pinecone_index, embeddings_model.embed_query, "page_content")
#     return vector_store

# def indexing_documents_pinecone(all_splits: List[Document], api_key: str, environment: str, index_name: str, batch_size=100):
#     """Indexes document splits into the Pinecone vector store in smaller batches."""
#     vector_store = init_pinecone_vector_store(api_key, environment, index_name)

#     # Process documents in chunks
#     for i in range(0, len(all_splits), batch_size):
#         batch = all_splits[i:i + batch_size]
#         try:
#             vector_store.add_documents(documents=batch)
#             print(f"Inserted batch {i // batch_size + 1}/{(len(all_splits) // batch_size) + 1}")
#         except Exception as e:
#             print(f"Error inserting batch {i // batch_size + 1}: {e}")

#     print("Indexing completed successfully.")

# Example usage.
# Replace with your actual Pinecone API key, environment and index name.
# indexing_documents_pinecone(all_splits=your_document_list, api_key="YOUR_API_KEY", environment="YOUR_ENV", index_name="YOUR_INDEX", batch_size=100)

# file_path = "constant\output_movies_copy.txt"
# df = pd.read_csv(file_path, sep="^", quoting=csv.QUOTE_ALL)
# df["primaryTitle"] = df["primaryTitle"].fillna("").astype(str)

# def prepare_documents_for_splitting(df: pd.DataFrame) -> List[Document]:
#     """Prepares a list of Document objects from the DataFrame with structured context."""
#     docs = []
#     for index, row in df.iterrows():
#         structured_str = (
#             f"Movie: {row.get('originalTitle','')}, A-Rated: {row.get('isAdult','')}, Year: {row.get('startYear','')}, "
#             f"Duration: {row.get('runtimeMinutes','')}, Genre: {row.get('genres','')}, "
#             f"Rating: {row.get('averageRating','')}, Number of Votes: {row.get('numVotes','')}, Actor: {row.get('actor','')}, "
#             f"Actress: {row.get('actress','')}, Director: {row.get('director','')}, Producer: {row.get('producer','')}, "
#             f"Writer: {row.get('writer','')}"
#         )
#         doc = Document(page_content=structured_str)
#         docs.append(doc)
#     return docs

# def split_movie_descriptions(docs: List[Document]) -> List[Document]:
#     """Splits movie descriptions into chunks while preserving other metadata."""
#     description_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     all_splits = []
#     for doc in docs:
#         movie_data = doc.page_content.split("Description: ")[0]
#         description = doc.page_content.split("Description: ")[1]
#         description_chunks = description_splitter.split_text(description)
#         for chunk in description_chunks:
#             new_content = f"{movie_data} Description: {chunk}"
#             new_doc = Document(page_content=new_content, metadata=doc.metadata)
#             all_splits.append(new_doc)
#     return all_splits

# def indexing_documents(all_splits: List[Document], batch_size=5000):
#     """Indexes document splits into the vector store in smaller batches."""
#     vector_store = init_vector_store()

#     # Process documents in chunks
#     for i in range(0, len(all_splits), batch_size):
#         batch = all_splits[i:i+batch_size]
#         vector_store.add_documents(documents=batch)
#         print(f"Inserted batch {i//batch_size + 1}/{(len(all_splits)//batch_size) + 1}")

#     print("Indexing completed successfully.")