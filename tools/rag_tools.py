import os
from langchain_core.tools import tool
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

def get_retriever_tool(vector_store):
    
    # Configure Reranker if API key is present
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if cohere_api_key and cohere_api_key != "C0diBrne7TeGTGMXlT4Scr1dDfvMSg98tFeWmC8x":
        print("Using Cohere Reranker for improved retrieval accuracy.")
        compressor = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-english-v3.0", top_n=5)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    else:
        print("Using basic similarity search (No reranker configured).")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @tool
    def retrieve_context(query: str):
        """Retrieve relevant ABB product data and catalogue information based on the user's query. 
        Highly effective for finding technical specifications of induction motors and other products."""
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs])

    return retrieve_context
