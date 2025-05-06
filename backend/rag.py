# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from IPython.display import Markdown, display
# import chromadb
# import os
# from dotenv import load_dotenv
# import openai


# # load api key from .env and set up openai
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

# # create new client and collection
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart")

# # embedding function 
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# # load context data
# docs = SimpleDirectoryReader("./data").load_data()

# # set up vector store and load data in it
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     docs, storage_context=storage_context, embed_model=embed_model
# )

# query_engine  = index.as_query_engine()
# response = query_engine.query("Hello, what is your name and what is your purpose?")
# print(response)

print("Running RAG...")

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp import LlamaCPP

# from dotenv import load_dotenv
# import openai
# import os

# load api key from .env and set up openai
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

# print("Checking if API key loaded... \n API KEY:", os.getenv("API_KEY"))


print("Imported packages")

# set up embedding model and load in contextual documents
# embed_model = OpenAIEmbedding()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
docs = SimpleDirectoryReader("./data").load_data()

print(f"Data Read: {docs[0]}")


print("Creating Client and Collection...")
# create new client and collection
db = chromadb.PersistentClient(path="./db")
collection = db.get_or_create_collection("beaverbot")

print("Initializing Vector Store...")
# create vector store
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Building Index...")
# build the index 
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


llm = LlamaCPP(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_new_tokens=512,
    context_window=2048,
    model_kwargs={"n_gpu_layers": 0}, 
    verbose=True,
)

engine = index.as_query_engine(llm=llm)

question = input("Enter a query: ")
# engine = index.as_query_engine()
print("Querying Engine...")
response = engine.query(question)

print(response)




