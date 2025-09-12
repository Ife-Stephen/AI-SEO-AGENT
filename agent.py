import os
import re
from typing import Annotated, Sequence, TypedDict, List, Dict
from dotenv import load_dotenv
import streamlit as st

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from openai import OpenAI

from tools import TOOLS

# Load environment variables locally
load_dotenv()

# Hugging Face settings
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = "deepseek-ai/DeepSeek-R1-0528:novita" 

# First try Streamlit secrets, then environment
API_KEY = st.secrets.get("TOKEN")

if not API_KEY:
    raise RuntimeError("⚠️ Please set TOKEN in Streamlit secrets or .env file")

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# HF OpenAI-compatible client
client = OpenAI(base_url=HF_BASE_URL, api_key=API_KEY)

def call_hf_model(messages: List[Dict[str, str]], model: str = HF_MODEL, max_tokens: int = 512) -> str:
    """Wrapper to call Hugging Face model via OpenAI API style."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    try:
        return response.choices[0].message.content
    except Exception:
        return response.choices[0].message["content"]

# System instructions
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in SEO and content generation for businesses. "
    "You can produce blog outlines, SEO meta descriptions, titles, and captions. "
    "If you need a tool, use:\n"
    "TOOL_CALL: <tool_name> <arg1> <arg2>\n"
    "Example: TOOL_CALL: add 2 3\n"
)

# Main process node
def process_node(state: AgentState) -> AgentState:
    messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in state["messages"]:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        messages_for_model.append({"role": role, "content": m.content})

    ai_text = call_hf_model(messages_for_model)

    # Tool call detection
    match = re.search(r"TOOL_CALL:\s*(\w+)(.*)", ai_text, flags=re.IGNORECASE)
    if match:
        tool_name, args_str = match.group(1).strip(), match.group(2).strip()
        arg_list = []
        for token in args_str.split():
            try:
                arg_list.append(int(token))
            except ValueError:
                try:
                    arg_list.append(float(token))
                except ValueError:
                    arg_list.append(token)

        if tool_name in TOOLS:
            result = TOOLS[tool_name](*arg_list)
            tool_msg = ToolMessage(content=str(result), tool_call_id=tool_name)

            final_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in state["messages"]:
                final_msgs.append({"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content})
            final_msgs.append({"role": "assistant", "content": ai_text})
            final_msgs.append({"role": "tool", "content": str(result)})

            final_ai = call_hf_model(final_msgs)
            return {"messages": list(state["messages"]) + [AIMessage(content=ai_text), tool_msg, AIMessage(content=final_ai)]}

    return {"messages": list(state["messages"]) + [AIMessage(content=ai_text)]}

# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_node("tools", ToolNode(list(TOOLS.values())))
graph.add_edge(START, "process")
graph.add_edge("process", "tools")
graph.add_edge("tools", END)

agent = graph.compile()


