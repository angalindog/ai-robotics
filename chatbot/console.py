# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chromadb>=1.5.5",
#     "langchain-community>=0.4.1",
#     "langchain-ollama>=1.0.1",
# ]
# ///

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# 1. Cargar embeddings (igual que en setup)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Cargar base de datos
db = Chroma(persist_directory="db", embedding_function=embeddings)

# 3. Modelo LLM
llm = OllamaLLM(model="qwen:4b")

print("Chatbot listo (escribe 'salir' para terminar)\n")

while True:
    pregunta = input("Tú: ")

    if pregunta.lower() == "salir":
        break

    # 4. Buscar contexto relevante
    docs = db.similarity_search(pregunta, k=5)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    # 5. Prompt (MUY importante mejorarlo)
    prompt = f"""
Eres un asistente que responde SOLO con base en el contexto proporcionado.

Si la respuesta no está en el contexto, di: "No encontré esa información en los documentos."

Contexto:
{contexto}

Pregunta:
{pregunta}
"""

    # 6. Generar respuesta
    respuesta = llm.invoke(prompt)

    print("\nBot:", respuesta, "\n")
