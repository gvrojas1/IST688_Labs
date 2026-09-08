import streamlit as st
from openai import OpenAI
import numpy as np

st.title("Streamlit Chatbot")

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a friendly assistant. "
        "When the user asks a question, answer it clearly and simply. "
        "Then ALWAYS ask: 'Do you want more info?' "
        "If the user says yes, give more detail on the same topic, "
        "and then ask 'Do you want more info?' again. "
        "If the user says no, respond with 'Okay! What else can I help you with?'"
    )
    }

#Initialize session state for messages and create Open AI client 
if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT,
    {"role": "assistant", "content": "How can I help you?"}]

#Show chat history on screen but hide system message
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your message here...")
#react to users input
if prompt:
    #Adds user message to chat history 
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    #Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
      
    response = st.session_state.client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=st.session_state.messages,
    )
    answer = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

    # Trim history but NEVER lose the system prompt 
    system_msg = st.session_state.messages[0]          # save it first
    recent_msgs = st.session_state.messages[1:][-4:]    # last 2 turns (4 messages)
    st.session_state.messages = [system_msg] + recent_msgs
    





