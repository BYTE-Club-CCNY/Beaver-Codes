import streamlit as st

st.title("Beaver Codes Chatbot")
st.subheader("For all your CCNY course questions")
query = st.text_input("Enter your question here: ")
st.write(f"Your question was: {query}")
