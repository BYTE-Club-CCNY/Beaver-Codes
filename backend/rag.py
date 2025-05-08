print("Running model...")
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.readers.file import PDFReader
import shutil
from dotenv import load_dotenv
import os
import openai

shutil.rmtree("./db", ignore_errors=True)
print("Imported packages")

# load api key from .env and set up openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

# set up embedding model and load in contextual documents
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
topic = input("select a topic: ")
data_path = f"./data/{topic}"
docs = SimpleDirectoryReader(
    input_dir=data_path,
    file_extractor={
        ".pdf": PDFReader()
        }).load_data()
# print("\n=== LOADED DOCUMENTS ===")
# for i, doc in enumerate(docs):
#     print(f"[{i}] File: {doc.metadata.get('file_name')}")
#     print(doc.text[:200])
#     print("-" * 40)

print("Creating Client and Collection...")
# create new client and collection
db = chromadb.PersistentClient(path="./db")
# reset collection or create new one if one doesnt exist
collection_name = f"beaverbot_{topic}"
try:
    db.delete_collection(collection_name)  # Optional: if you want a clean rebuild
except:
    pass
collection = db.get_or_create_collection(collection_name)
print(f"Loaded {len(docs)} documents.")
print("Initializing Vector Store...")
# create vector store
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("Building Index...")
# build the index 
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
llm = LlamaCPP(
    model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", # NOT TRACKED IN THE GITHUB - MUST BE DOWNLOADED LOCALLY
    # LINK TO EMBEDDING MODEL: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    # PLACE FILE IN backend/models
    temperature=0.7,
    max_new_tokens=256,
    context_window=2048,
    model_kwargs={"n_gpu_layers": 0}, 
    verbose=True,
)
engine = index.as_query_engine(llm=llm)
question = input("Enter a query: ")
# engine = index.as_query_engine()
print("Querying Engine...")
response = engine.query(question)
print("RESPONSE:", response)
print("SOURCE DOCS:\n")
for s in response.source_nodes:
    print("-", s.text[:10])