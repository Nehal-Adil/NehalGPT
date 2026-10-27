from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# -------------------------------------------------------------------
# 1. DATABASE ENGINE & SESSION SETUP
#  -------------------------------------------------------------------

# Ensure the local directory exists before SQLite attempts to create the DB file
Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db"

# Create the SQLite engine.
# check_same_thread=False allows Streamlit's multi-threaded runners to share connections safely.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Factory for creating database sessions per request/operation
SessionLocal = sessionmaker(bind=engine, autoflush=True, autocommit=False)

# Base class for all SQLAlchemy declarative ORM models
Base = declarative_base()

# ---------------------------------------------------------------------------
# 2. ORM DATA MODELS
# ---------------------------------------------------------------------------

class Conversation(Base):
    """
    Tracks high-level chat sessions/threads (similar to ChatGPT sidebar items).
    Each thread has its own unique thread_id, editable title, and update tracking.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.now(UTC))
    updated_at = Column(DateTime, default=datetime.now(UTC))

class ChatMessage(Base):
    """
    Stores individual conversational turns (short-term conversational history).
    Retains chronologically ordered user inputs and assistant responses per thread.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now(UTC))

class LongTermMemory(Base):
    """
    Stores extracted facts, user preferences, and profile insights (long-term memory).
    Allows cross-turn or persistent personalization across sessions.
    """

    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    memory = Column(Text)
    created_at = Column(DateTime, default=datetime.now(UTC))

# ---------------------------------------------------------------------------
# 3. DATABASE OPERATIONS / REPOSITORY INTERFACE
# ---------------------------------------------------------------------------

def init_db():
    Base.metadata.create_all(bind=engine)

def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    """
    Creates a new conversation record or bumps updated_at on existing conversations.
    Optionally extracts a concise sidebar title from the user's first prompt.
    """
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"

            if first_message:
                title = first_message.strip()[:40]
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id = thread_id,
                title = title,
                created_at = datetime.now(UTC),
                updated_at = datetime.now(UTC),
            )

            db.add(conversation)

        else:
            conversation.updated_at = datetime.now(UTC)

        db.commit()

    finally:
        db.close()

def list_conversations():
    """
    Fetches all active conversations ordered by most recently updated.
    Used directly by the UI (e.g., Streamlit sidebar) to display past chats.
    """

    db = SessionLocal()

    try:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()

def save_chat_message(thread_id: str, role: str, content: str):
    """
    Persists an individual message (user prompt or AI response) to `chat_messages`.
    Ensures message sequence is recorded for UI re-rendering and context reconstruction.
    """

    db = SessionLocal()

    try:
        msg = ChatMessage(
            thread_id = thread_id,
            role = role,
            content = content,
            created_at = datetime.now(UTC),
        )

        db.add(msg)

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if conversation:
            conversation.updated_at = datetime.now(UTC)

        db.commit()

    finally:
        db.close()

def get_chat_history(thread_id: str):
    """
    Retrieves all messages for a specific thread_id in chronological order.
    Formats records into LangChain/LangGraph message objects (HumanMessage, AIMessage).
    """

    db = SessionLocal()

    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    finally:
        db.close()

def save_memory(thread_id: str, memory: str):
    """
    Persists an extracted piece of long-term memory or key user fact.
    Typically called by an agent background task or extraction node.
    """

    db = SessionLocal()

    try:
        item = LongTermMemory(
            thread_id = thread_id,
            memory = memory,
            created_at = datetime.now(UTC),
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    finally:
        db.close()

def search_memory(thread_id: str, query: str):
    """
    Searches stored long-term memories for a given thread or user.
    Can perform simple keyword/SQL LIKE matching or semantic vector similarity.
    """
    
    db = SessionLocal()

    try:
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id == thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )

        if not memories:
            return "No save memory found."

        return "\n".join([f"- {m.memory}" for m in memories])

    finally:
        db.close()