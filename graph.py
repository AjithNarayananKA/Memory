# Import Libraries
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama
import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver

from state import State
from chatbot import chatbot
from summary import summary
load_dotenv()

# Initialize StateGraph
graph = StateGraph(State)



# Create Nodes
# 1.chatbot node
# 2.summary node
# 3.conditional edge

def should_continue(state:State)-> str:

    messages = state.get("messages")

    if len(messages) > 6:
        return "SUMMARY"
    
    return END

# Build Graph

 # Add Nodes
graph.add_node("CHATBOT", chatbot)
graph.add_node("SUMMARY", summary)

# Add Edges
graph.add_edge(START, "CHATBOT")
graph.add_conditional_edges("CHATBOT", should_continue)
graph.add_edge("SUMMARY", END)

# Compile Graph with Memory
connection = sqlite3.connect("graph_memory.db", check_same_thread=False)
saver = SqliteSaver(connection)
compiled_graph = graph.compile(checkpointer=saver)

# print graph
# print(compiled_graph.get_graph().draw_mermaid_png)

# Create Chatbot Instance
config = {"configurable":{"thread_id": "1"}}

print("Starting Chatbot...")

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Create initial state
    human_message = HumanMessage(content=user_input)

    # Graph invokation
    response = compiled_graph.invoke({"messages":[human_message]}, config=config)
    print(response)