import streamlit as st
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.readers.file import PDFReader
import shutil
shutil.rmtree("./db", ignore_errors=True)
import time



# backend logic 

def run_rag(topic, question):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    data_path = f"backend/data/{topic.lower().replace(' ', '_')}"
    docs = SimpleDirectoryReader(
        input_dir=data_path,
        file_extractor={
            ".pdf": PDFReader()

            }).load_data()

    print("Creating Client and Collection...")
    # create new client and collection
    db = chromadb.PersistentClient(path="backend/db")

    # reset collection or create new one if one doesnt exist
    collection_name = f"beaverbot_{topic.lower().replace(' ', '_')}"
    try:
        db.delete_collection(collection_name) 
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
        model_path="backend/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", # NOT TRACKED IN THE GITHUB - MUST BE DOWNLOADED LOCALLY
        # LINK TO EMBEDDING MODEL: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
        # PLACE FILE IN backend/models
        temperature=0.7,
        max_new_tokens=256,
        context_window=2048,
        model_kwargs={"n_gpu_layers": 0}, 
        verbose=True,
    )

    engine = index.as_query_engine(llm=llm)
    userQuestion = str(question)
    response = engine.query(userQuestion)

    return str(response)


# frontend

st.set_page_config(page_title = "Beaver Bot", page_icon = "frontend/beaver.png", layout = "centered", initial_sidebar_state = "auto", menu_items = None)
# st.logo("frontend/beaver.png", size="large", link=None, icon_image=None)

st.title("Beaver Codes Chatbot")
st.subheader("For all your CCNY course questions")

with st.sidebar:
    classes = st.selectbox(
        "Select a Course:", ("Data Structures", "Algorithms","Software Engineering", "Theoretical Computer Science", 
        "Modern Distributing Computing",
        "Prog.Language Paradigms", "Computer Organization"), index = None, placeholder = "Choose any CSC Class:",
    )


# This is for when this is the first message
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displays chat history 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Enter your question here: ",
    key= None, max_chars= None, accept_file= "multiple", 
    file_type = ["jpg", "jpeg", "png", "pdf"], #accepts these files types
    disabled=False, on_submit=None, args=None, kwargs=None): 
    # displays user chat in message container
     
    with st.chat_message("user"): 
        st.markdown(query.text) #user's input
        
        # add user's message to chat history
    st.session_state.messages.append({"role": "user","content": query.text})
    
    response = run_rag(classes, query)
    with st.chat_message("assistant"): 
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant","content": response})

    #FIX when have backend to store files and dont need to show image
    if "files" in query and query["files"]: #This displays image, doesn't work on pdf
        st.image(query["files"][0]) 

# Notes:
#THINGS TO CONSIDER: since we using the bucket method with a scroll down menu, how does the user start - disable chat till pick class? - default class?
#
# rn it only has the users input want to have the AIs answer after each submission
# Handling pdfs
# Need to make a page where you select class you want
# Also option for login  
# Maybe a beaver logo on the side to refresh page to pick a diff class
    





























