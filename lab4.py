import streamlit as st
from openai import OpenAI
import sys

# A fix for working with ChromaDB on Streamlit
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

import chromadb
from pathlib import Path
import fitz  # PyMuPDF

# Page setup
st.title("Lab 4: Chatbot using RAG")

#Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = st.session_state.openai_client

# Helper functions
# Extract text from PDF 
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each sylabus to pass to add_to_collection() function to add to ChromaDB collection.
    """
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def add_to_collection(collection, text, file_name):
    #Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    # Get the embedding vector
    embedding_vector = response.data[0].embedding
    #Add embedding and document to the collection (ChromaDB)
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding_vector]
    )

# Populate collection with PDFs
def load_pdfs_into_collection(folder_path, collection):
    """Reads all PDFs in folder_path and adds each one to collection."""
    pdf_files = sorted(Path(folder_path).glob('*.pdf'))
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        add_to_collection(collection, text, pdf_path.name)

def create_lab4_collection():
    """Creates (or opens) the Lab4Collection ChromaDB collection and
    populates it from PDFs only if it's currently empty."""
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
    collection = chroma_client.get_or_create_collection('Lab4Collection')
 
    if collection.count() == 0:
        load_pdfs_into_collection('./Lab-04-Data', collection)
 
    return collection

def get_relevant_context(query, n_results=3):
    """Embeds the query, retrieves the top-n matching documents from the
    collection, and returns a combined context string plus the source filenames."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
 
    docs = results['documents'][0]
    ids = results['ids'][0]
    context_str = "\n\n".join(
        f"--- From {ids[i]} ---\n{docs[i]}" for i in range(len(docs))
    )
    return context_str, ids
 
 
def build_buffer(full_history, num_turns=2):
    """Returns the system prompt + only the last `num_turns` user/assistant
    exchanges, to keep token usage low."""
    system_msg = full_history[0]
    non_system_msgs = full_history[1:]
    recent_msgs = non_system_msgs[-(num_turns * 2):]
    return [system_msg] + recent_msgs

# Build (or reuse) the vector DB 
if 'Lab4_VectorDB' not in st.session_state:
    with st.spinner("Building vector database from course PDFs..."):
        st.session_state.Lab4_VectorDB = create_lab4_collection()

collection = st.session_state.Lab4_VectorDB

    
#### Main App ####

#### Querying a collection, only used for testing

# topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., Gen AI)...')
# if topic:
#     client = st.session_state.openai_client
#     response = client.embeddings.create(
#         input=topic,
#         model='text-embedding-3-small')
#     #Get the embedding
#     query_embedding = response.data[0].embedding
#     #Get the text related to this question (this prompt)
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=3 # The number of closest documents to return
#     )
#     #Display the results
#     st.subheader(f"Results for: {topic}")

#     for i in range(len(results['documents'][0])):
#         doc = results['documents'][0][i]
#         doc_id = results['ids'][0][i]

#         st.write(f'**{i+1}.{doc_id}**')
# else:
#     st.info('Enter a topic in the sidebar to search the collection') 

# 6. CHATBOT (PART B) — RAG-integrated

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful course advisor chatbot for Syracuse iSchool courses. "
        "You will sometimes be given relevant excerpts from course syllabi under "
        "'RELEVANT COURSE INFO'. Use that information to answer the user's question "
        "when it's relevant. If you used it, explicitly say something like "
        "'Based on the course materials I found...' at the start of your answer. "
        "If the retrieved info isn't relevant to the question, answer from general "
        "knowledge and say so instead."
    )
}
 
if "messages" not in st.session_state:
    st.session_state.messages = [
        SYSTEM_PROMPT,
        {"role": "assistant", "content": "Hi! Ask me about any of the courses."}
    ]
 
# Show chat history (skip the system message)
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
prompt = st.chat_input("Ask about a course...")
 
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    # RAG retrieval for this turn 
    context_str, source_ids = get_relevant_context(prompt, n_results=3)
 
    api_messages = build_buffer(st.session_state.messages, num_turns=2)
    augmented_messages = api_messages + [
        {
            "role": "system",
            "content": f"RELEVANT COURSE INFO (sources: {', '.join(source_ids)}):\n{context_str}"
        }
    ]
 
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=augmented_messages,
    )
    answer = response.choices[0].message.content
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
 
