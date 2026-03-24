import os
from langchain_community.vectorstores import FAISS

def get_or_create_vector_store(embeddings, docs=None):
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        print("Loading existing vector store from disk...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    if docs is None:
        raise ValueError("Index not found and no documents provided to create it.")

    print("No existing vector store found. Creating new one...")
    # Using the simplified FAISS.from_documents method
    store = FAISS.from_documents(docs, embeddings)
    
    print("Saving vector store to disk...")
    store.save_local(index_path)
    
    return store
