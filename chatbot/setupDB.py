# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chromadb>=1.5.5",
#     "langchain>=1.2.15",
#     "langchain-community>=0.4.1",
#     "langchain-ollama>=1.0.1",
#     "pypdf>=6.9.2",
#     "streamlit>=1.56.0",
# ]
# ///

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Cargar PDF
loader = PyPDFLoader("docs/curso.pdf")
docs = loader.load()

# 2. Dividir texto
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embeddings (ligero)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Guardar en Chroma
db = Chroma.from_documents(chunks, embeddings, persist_directory="db")

db.persist()

print("Base de datos creada")
