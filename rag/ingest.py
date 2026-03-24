from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents():
    print("Fetching website...")
    web_loader = WebBaseLoader(["https://new.abb.com/low-voltage/products/building-automation/catalogues-and-brochures"])
    web_docs = web_loader.load()
    print(f"Fetched {len(web_docs)} documents from website.")

    print("Loading PDF (with PyMuPDF)...")
    pdf_loader = PyMuPDFLoader("graph/data/9AKK108468A3912_en_E_B Electrical installation solutions for buildings part B-Technical _EN_ (PDF).pdf")
    pdf_docs = pdf_loader.load()
    print(f"Loaded {len(pdf_docs)} documents from PDF.")

    return web_docs + pdf_docs

def split_documents(docs):
    # Optimizing chunks for technical data retrieval:
    # 500 characters with 10% overlap helps preserve technical specifications (SKUs, Voltage, etc.)
    # within individual chunks while maintaining context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        add_start_index=True
    )
    return splitter.split_documents(docs)
