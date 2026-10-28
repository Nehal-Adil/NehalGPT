from dotenv import load_dotenv
import os

load_dotenv()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from agent import get_agent
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
)

from rag import add_document_to_rag
from tools import set_current_thread_id

app = FastAPI()

# BASE_DIR = Path(__file__).resolve().parent
# templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

init_db()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
