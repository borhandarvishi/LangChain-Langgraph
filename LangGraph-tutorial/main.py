from dotenv import load_dotenv
import os

from typing import Annotated, Literal

from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = init_chat_model("gpt-4o-mini", temperature=0)

class State(TypedDict):
    messages: Annotated[list,add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# the way to add a node
graph_builder.add_node("chatbot", chatbot)

# exept the chatbot node, we also have two special nodes: START and END (we added them automatically but the not connected by edges. lets connect them)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# let's run the graph
graph = graph_builder.compile()

user_input = input("User: ")

state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(state["messages"][-1].content)