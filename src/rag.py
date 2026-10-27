from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitter import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import python_docx

Path("uploads").mkdir(exist_ok=True)
Path("chrom_db").mkdir(exist_ok=True)

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001",
)

vectorstore = Chroma(
    collection_name = "agentic_chatbot_docs",
    embedding_function = embeddings,
    persist_directory = "chrom_db",
)

def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return python_docx.Document(file_path).text

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unaupposed file type. Upload a PDF, DOCX, TXT, or Markdown file.")

def add_document_to_rag(file_path: str, thread_id: str):

    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from the file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap  = 150,
    )

    chunks = splitter.split_text(text)

    docs: list[Document] = [
        Document(
            page_content = chunk,
            metadata = {
                "thread_id": thread_id,
                "source": Path(file_path).name,
            }
        )
        for chunk in chunks
    ]

    vectorstore.aadd_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }

def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(
        query,
        k = k,
        filter = {"thread_id": thread_id}
    )

    if not docs:
        return "No relevant uploaded document "

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)







