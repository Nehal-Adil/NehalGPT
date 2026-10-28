from agent import get_agent
from langchain_core.messages import HumanMessage, SystemMessage
from database import init_db

init_db()

agent = get_agent("gemini-2.5-flash")

config = {"configurable": {"thread_id": "test_thread_id"}}

for message_chunk, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(
                content="My name is Nehal and I'm 27 years old, and I'm a Software Developer, Please remember that."
            )
        ]
    },
    config=config,
    stream_mode="messages",
):
    content = message_chunk.content
    text_to_print = ""

    # Case A: Gemini structured block format
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_to_print += part.get("text", "")
            elif isinstance(part, str):
                text_to_print += part

    # Case B: Standard string format
    elif isinstance(content, str):
        text_to_print = content

    if text_to_print:
        print(text_to_print, end="", flush=True)
