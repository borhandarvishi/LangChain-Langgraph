

from dotenv import load_dotenv
import os

from typing import Annotated, Literal

from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()
os.environ["OPENAI_API_KEY"] =  os.getenv("OPENAI_API_KEY")


llm = init_chat_model("gpt-4o-mini", temperature=0)



class MessageClassifier(BaseModel):
    message_type: Literal["emotional","logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response." 

# the state is defineing the type of information that we want to have  in our graph.
class State(TypedDict):
    messages: Annotated[list,add_messages]
    message_type: str | None



def classify_message(state: State) -> State:
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "systemt",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])

    return {"message_type": result.message_type}
    
def router(state: State) -> State:
    message_type = state.get("message_type","logical")
    if message_type == "emotional":
        return {"next":"therapist"}
    
    else:
        return {"next":"logical"}   
    

def therapist_agent(state: State) -> State:
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    reply = llm.invoke(messages)
     return {"messages": [{"role": "assistant", "content": reply.content}]}
    

def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}
    


# create a graph using state.
# a node is a function that modify or check the state.
graph_builder = StateGraph(State)

graph_builder.add_node("classifier",classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edge(
    "router",
    lambda state: state.get("next"),
    {
        "therapist": "therapist",
        "logical": "logical"
    }
)                           

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)  

# let's run the graph 
graph = graph_builder.compile()

def run_chatbot(user_message: str):
    state = {"message":[],"message_type": None}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot.")
            break
        state["messages"] = state.get("messages",[]) + [{"role": "user", "content": user_input}]
        
        state = graph.invoke(state)