import os
from dotenv import load_dotenv

# Load environment variables from tools/.env
load_dotenv("tools/.env")

# A more standard browser-like user agent to avoid being blocked
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

from config.models import get_llm, get_embeddings
from rag.ingest import load_documents, split_documents
from rag.retriever import get_or_create_vector_store
from tools.rag_tools import get_retriever_tool
from agents.faq_agent import build_agent

def main():
    embeddings = get_embeddings()
    index_path = "faiss_index"

    if os.path.exists(index_path):
        print("Loading existing vector store...")
        vector_store = get_or_create_vector_store(embeddings)
    else:
        print("No index found. Loading and indexing documents (this will take a few minutes)...")
        print("Loading documents...")
        docs = load_documents()
        print(f"Loaded {len(docs)} documents.")
        
        print("Splitting documents...")
        splits = split_documents(docs)
        print(f"Created {len(splits)} splits.")

        print("Creating vector store...")
        vector_store = get_or_create_vector_store(embeddings, splits)

    print("Setting up tools...")
    tool = get_retriever_tool(vector_store)

    print("Getting LLM...")
    llm = get_llm()
    print("Building agent...")
    agent = build_agent(llm, [tool])

    while True:
        query = input("Ask: ")
        if query == "exit":
            break
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        # Extract and print the last assistant message
        print("\nAssistant: " + response["messages"][-1].content + "\n")

if __name__ == "__main__":
    main()
