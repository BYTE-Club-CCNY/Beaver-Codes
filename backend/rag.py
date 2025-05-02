from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
import os
from dotenv import load_dotenv
import openai


# load api key from .env and set up openai
load_dotenv()
api_key = os.getenv("API_KEY")
openai.api_key = api_key

# create new client and collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# embedding function 
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load context data
docs = SimpleDirectoryReader("./data").load_data()

# set up vector store and load data in it
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=embed_model
)

query_engine  = index.as_query_engine()
response = query_engine.query("Hello, what is your name and what is your purpose?")
print(response)
