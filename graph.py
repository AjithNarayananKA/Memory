# Import Libraries
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama
import sqlite3
import json

from state import State

load_dotenv()

# Initialize StateGraph
graph = StateGraph(State)

# Initialize LLM
llm = ChatOllama(model="gemma4:12b", verbose=True, temperature=0.9, reasoning = True, num_predict = 2048)

# Create Nodes

# chat node
def chatbot(state: State) -> State:
    summary = state["summary"]
    if summary:
        system_message ="""
        You are continuing a conversation with a human. Here is the summary of the conversation so far:
        {summary}
        Use this summary as a prior context to generate your response.  Ensure the response feels natural, maintains the flow of the conversation, and  addresses the user's most recent message appropriately.
        """
        messages = [SystemMessage(system_message)] +state["messages"]
    else:
        system_message = """
        You are an Intelligent conversation chatbot designed to engage in natural and meaningful conversations with users. Your primary goal is to provide helpful, informative, and engaging responses to user inputs. You should strive to understand the user's intent and context, and respond in a way that is relevant and adds value to the conversation.
        
        Make sure to :
        1. Understand the full contextof messages provided
        2. Generate responses that are coherent, contextually relevant, and engaging.
        3.Address any  question or unresolved point.
        """
        messages = [SystemMessage(system_message)] + state["messages"]
    
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content = response.content)]
    }