import streamlit as st
from openai import OpenAI

st.title("Streamlit Chatbot")
st.write("This is a simple chatbot interface using Streamlit and OpenAI's API. Type your message below and the assistant will respond.")

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

def build_buffer(full_history, num_turns=2):
    """
    Takes the FULL chat history and returns a small 'buffer' to send to the LLM:
    the system prompt + only the last `num_turns` user/assistant exchanges.
    This keeps token usage low without deleting anything the user sees.
    """
    system_msg = full_history[0]
    non_system_msgs = full_history[1:]
    recent_msgs = non_system_msgs[-(num_turns * 2):]  # 2 messages per turn (user + assistant)
    return [system_msg] + recent_msgs


# Initialize OpenAI client
if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

# FULL history — shown to the user
if "messages" not in st.session_state:
    st.session_state.messages = [
        SYSTEM_PROMPT,
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Show FULL chat history on screen (but hide system message)
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your message here...")

if prompt:
    # Add user message to the FULL history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Build the small buffer just for this API call
    api_messages = build_buffer(st.session_state.messages, num_turns=2)

    response = st.session_state.client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=api_messages,
    )
    answer = response.choices[0].message.content

    # Add the answer to the FULL history (for display)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)