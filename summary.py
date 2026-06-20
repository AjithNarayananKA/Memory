from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama
from state import State
llm = ChatOllama(model="gemma4:12b", verbose =True, temperature=0.9, reasoning= True, num_predict=2048)

def summary(state: State) -> State:
    summary = state.get("summary")
    if summary:
        system_message = f"""   
        Given this conversation summary:
        {summary}
        Your task:
        1. Analyze new messages provided to you.
        2.identify key updates in topic, content or use Intent.
        3.Integrate these updates with existing summary.
        4.Maintain chronological order and contextual relevance.
        5.focus on information essential for conversation continuity
       Generate an Updated summary that maintains clarity and coherence.
       """
        
    else:
        system_message = f"""   
        Analyze the conversation and make a concise summary that:

        1.Captures the main topics and key points discussed
        2.Preserves the essential context and decisions made
        3.Notes any unresolved issues or pending actions.
        4.Maintain chronological order and contextual relevance.
        5.Focus on information essential for conversation continuity

       Generate an summary that maintains clarity and coherence.
       """
    messages =  state["messages"] + [HumanMessage(content=system_message)] 
    response = llm.invoke(messages)
    # Delete first 4 messages or keep last 2 messages
    deleted_messages = [RemoveMessage(m.id) for m in state["messages"][:-2]]

    return {
        "messages": deleted_messages,
        "summary": response.content
    }