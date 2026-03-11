from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
from pathlib import Path
from datetime import datetime

try:
    from langchain.embeddings import CacheBackedEmbeddings
except ImportError:
    try:
        from langchain.embeddings.cache import CacheBackedEmbeddings
    except ImportError:
        CacheBackedEmbeddings = None

try:
    from langchain.storage import LocalFileStore
except ImportError:
    LocalFileStore = None


#Extract Data From the PDF File
def load_pdf_file(data):
    # Load all PDF documents from a directory
    return DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader).load()


def _source_date_from_path(source: str) -> Optional[str]:
    # Convert file modified time to YYYY-MM-DD
    try:
        return datetime.fromtimestamp(Path(source).stat().st_mtime).date().isoformat()
    except OSError:
        return None


def add_metadata(docs: List[Document]) -> List[Document]:
    # Attach source, section, and date metadata to documents
    """
    Enrich document metadata with source, section, and date.
    - source: original file path
    - section: page number (1-indexed) when available
    - date: file last-modified date (YYYY-MM-DD) when available
    """
    enriched: List[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        source = metadata.get("source")
        page = metadata.get("page")
        if source:
            metadata["source"] = source
            metadata.setdefault("date", _source_date_from_path(source))
        if page is not None:
            metadata["section"] = f"page {int(page) + 1}"
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))
    return enriched



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    # Keep only minimal metadata fields to reduce payload size
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source', 'section', and 'date' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        metadata = doc.metadata or {}
        minimal = {k: metadata[k] for k in ("source", "section", "date") if metadata.get(k) is not None}
        minimal_docs.append(Document(page_content=doc.page_content, metadata=minimal))
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data, chunk_size: int = 500, chunk_overlap: int = 20):
    # Split documents into overlapping chunks for retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(extracted_data)



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    # Build embeddings with optional local cache
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if CacheBackedEmbeddings is None or LocalFileStore is None:
        return embeddings
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(str(cache_dir))
    return CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace=embeddings.model_name)
