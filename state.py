from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    """State of the application."""
    messages : Annotated[list, add_messages]
    summary : str