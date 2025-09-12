# tools.py
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

TOOLS = {"add": add}
