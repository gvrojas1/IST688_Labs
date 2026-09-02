import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

# Show title and description.
st.title("📄 My document question answering")
st.write(
    "Upload a document below and get an instant summary – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.secrets`.
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

    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
    )

    if uploaded_file:

        # Process the uploaded file based on its type.
        if uploaded_file.type == "application/pdf":
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            document = ""
            for page in pdf_doc:
                document += page.get_text()
            pdf_doc.close()
        else:
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

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)