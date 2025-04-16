import streamlit as st
import time

st.title("Beaver Codes Chatbot")
st.subheader("For all your CCNY course questions")

# This is for when this is the first message
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displays chat history 
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
        
    #FIX when have backend to store files and dont need to show image
    if "files" in query and query["files"]: #This displays image, doesn't work on pdf
        st.image(query["files"][0]) 

# Notes:
# rn it only has the users input want to have the AIs answer after each submission
# Handling pdfs
# Need to make a page where you select class you want
# Also option for login  
# Maybe a beaver logo on the side to refresh page to pick a diff class
    





























