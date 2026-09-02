import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("📄 My document question answering")
st.write(
    "Upload a document below and get an instant summary – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=openai_api_key)

# Validate the Key
key_is_valid = False
try:
    client.models.list()
    key_is_valid = True
except Exception as e:
    st.error(f"Invalid API key: {e}")

if key_is_valid:

    # Sidebar options
    st.sidebar.header("Summary Options")

    summary_type = st.sidebar.radio(
        "Choose a summary format:",
        (
            "Summarize in 100 words",
            "Summarize in 2 connecting paragraphs",
            "Summarize in 5 bullet points",
        ),
    )

    use_advanced_model = st.sidebar.checkbox("Use advanced model")

    # Map the checkbox to actual model names
    model = "gpt-4.1" if use_advanced_model else "gpt-4.1-mini"

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    if uploaded_file:

        # Process the uploaded file.
        document = uploaded_file.read().decode()

        # Build the instruction based on the sidebar selection.
        instruction_map = {
            "Summarize in 100 words": "Summarize the document below in exactly 100 words.",
            "Summarize in 2 connecting paragraphs": "Summarize the document below in 2 connecting paragraphs.",
            "Summarize in 5 bullet points": "Summarize the document below in 5 concise bullet points.",
        }
        instruction = instruction_map[summary_type]

        messages = [
            {
                "role": "user",
                "content": f"{instruction}\n\nHere's the document:\n\n{document}",
            }
        ]

        # Generate an answer 
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        st.write_stream(stream)