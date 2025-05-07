from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# ✅ Use HuggingFace embeddings (local, free)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# ❌ Disable OpenAI completely
Settings.llm = None

# ✅ Load your documents
documents = SimpleDirectoryReader("data").load_data()

# ✅ Build the index
index = VectorStoreIndex.from_documents(documents)

# ✅ Use a simple retrieval engine (no LLM required)
query_engine = index.as_query_engine()

# ✅ Streamlit interface
st.title("📚 My Custom Chatbot (LLM-Free)")
user_input = st.text_input("Ask me something:")

if user_input:
    response = query_engine.query(user_input)
    st.write("🤖", response.response)
