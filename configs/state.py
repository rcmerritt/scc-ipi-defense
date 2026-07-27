from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the email processing agent.
    Tracks the current email, ReAct trajectory, and control variables.
    The 'messages' field uses LangGraph's add_messages reducer annotation.
    This ensures that when any node returns a state update containing
    new messages, those messages are APPENDED to the existing list
    rather than replacing it. This is critical for compatibility with
    LangGraph's prebuilt ToolNode, which returns ToolMessages that
    must be merged into the existing conversation history while
    preserving the preceding AIMessage with tool calls.
    """
    # add_messages reducer: new messages are appended, not replaced.
    messages: Annotated[list, add_messages]
    iteration: int
    max_iterations: int
    escalated: bool
    completed: bool
    final_answer: Optional[str]
