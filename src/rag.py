import os
from pathlib import Path

import certifi
import docx
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# import python_docx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# 1. STORAGE DIRECTORIES & VECTORSTORE INITIALIZATION
# ---------------------------------------------------------------------------

Path("uploads").mkdir(exist_ok=True)
Path("chrom_db").mkdir(exist_ok=True)

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# Initialize local Chroma vector database
# persist_directory ensures chunks and embeddings are saved to disk across app restarts
vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory="chrom_db",
)

# ---------------------------------------------------------------------------
# 2. FILE EXTRACTION UTILITY
# ---------------------------------------------------------------------------


def read_file_text(file_path: str) -> str:
    """
    Extracts raw text content from varied file extensions (.pdf, .docx, .txt, .md, .py, .csv).
    Acts as a lightweight document ingestion router.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    # Case A: PDF files - iterate through all pages and aggregate text
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    # Case B: Microsoft Word (.docx)
    # Iterate through individual paragraphs since Document object has no direct .text attribute
    if suffix == ".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    # Case C: Plain text, markdown, source code, and CSVs
    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unaupposed file type. Upload a PDF, DOCX, TXT, or Markdown file.")


# ---------------------------------------------------------------------------
# 3. DOCUMENT INGESTION & VECTOR INDEXING
# ---------------------------------------------------------------------------


def add_document_to_rag(file_path: str, thread_id: str):
    """
    Processes an uploaded file into chunks, tags them with conversation metadata, embeds them, and persists them into ChromaDB under the given thread_id.
    """
    text = read_file_text(file_path)

    # 1. Extract plain text from the file
    if not text.strip():
        raise ValueError("No text could be extracted from the file.")

    # 2. Split large text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.split_text(text)

    # 3. Convert raw text chunks into LangChain Document objects with metadata
    docs: list[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name,
            },
        )
        for chunk in chunks
    ]

    # 4. Generate embeddings and save to Chroma (synchronous call)
    vectorstore.aadd_documents(docs)

    return {"filename": Path(file_path).name, "chunks": len(docs)}


# ---------------------------------------------------------------------------
# 4. THREAD-ISOLATED SIMILARITY RETRIEVAL
# ---------------------------------------------------------------------------


def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    """
    Performs similarity search in ChromaDB scoped strictly to the current thread_id.
    Formats the top k chunks into an LLM-ready context block with citations.
    """
    docs = vectorstore.similarity_search(query, k=k, filter={"thread_id": thread_id})

    if not docs:
        return "No relevant uploaded document "

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n".join(results)
