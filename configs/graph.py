import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from configs.state import AgentState
from configs.llm_config import get_llm
from prompts.react_system_prompt import REACT_SYSTEM_PROMPT
from tools.inbox_tools import read_inbox, read_email
from tools.compose_tools import draft_email, send_email
from tools.action_tools import summary

# Maximum number of ReAct iterations per email
MAX_ITERATIONS = 15

# Collect all tools
tools = [
    read_inbox, read_email,
    draft_email, send_email,
    summary,
]

def should_continue(state: AgentState) -> str:
    """
    Routing function called after reasoning. Checks iteration limit,
    then routes to tool execution or force-stop.
    With tool_choice="required" on the bound LLM, every AIMessage must
    carry a tool_calls field, so the "no tool call" branch is unreachable
    in practice. If it ever fires (server-side anomaly), force-stop.
    """
    if state["iteration"] >= state.get("max_iterations", MAX_ITERATIONS):
        return "force_stop"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "force_stop"


def after_tools(state: AgentState) -> str:
    """
    Routing function called after tool execution. Checks whether
    a terminal tool (send_email or summary) was called successfully.
    If so, routes to handle_terminal, otherwise returns to reasoning
    for the next ReAct step.
    """
    messages = state["messages"]
    for msg in reversed(messages):
        name = getattr(msg, "name", None)
        if name is None:
            break

        if name == "summary":
            return "handle_terminal"

        if name == "send_email":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if parsed.get("sent"):
                        return "handle_terminal"
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(content, dict) and content.get("sent"):
                return "handle_terminal"

    return "reasoning"


def handle_terminal_action(state: AgentState) -> dict:
    """
    Sets completed/escalated/final_answer state fields based on
    the terminal tool that was called. Routes to END.
    """
    messages = state["messages"]
    for msg in reversed(messages):
        name = getattr(msg, "name", None)
        if name is None:
            break

        content = getattr(msg, "content", "")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                content = {}

        if name == "summary":
            summary_text = content.get("summary", "") if isinstance(content, dict) else str(content)
            return {
                "completed": True,
                "final_answer": summary_text,
            }
        elif name == "send_email":
            message = content.get("message", "Email sent.") if isinstance(content, dict) else str(content)
            return {
                "completed": True,
                "final_answer": message,
            }

    return {"completed": True, "final_answer": "Processing complete."}


def force_stop_node(state: AgentState) -> dict:
    """
    Handles forced termination when the iteration limit is reached.
    Returns INCOMPLETE rather than ESCALATE: escalation is an SCC-layer
    artifact (consensus failure at N>1), not an internal failure mode.
    The SCC layer in main.py treats INCOMPLETE samples as non-converging
    votes, which correctly pushes a multi-sample run toward consensus
    failure when one or more samples exhaust their iteration budget.
    """
    return {
        "escalated": False,
        "completed": True,
        "final_answer": (
            f"INCOMPLETE: reached maximum iteration limit "
            f"({state.get('max_iterations', MAX_ITERATIONS)}). "
            f"No terminal tool was called."
        ),
    }


def fetch_inbox_node(state: AgentState) -> dict:
    """
    Fixed entry point that calls read_inbox() unconditionally before
    the reasoning loop starts. Injects system context and the inbox
    result as a synthetic tool exchange so the LLM sees a complete,
    correctly-ordered message history from the first reasoning step.
    The agent cannot reach read_email or any terminal tool without
    this data already in its message history.
    """
    result = read_inbox.invoke({})
    return {
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content="Process the next unread email in the inbox."),
            AIMessage(
                content="",
                tool_calls=[{"id": "forced_read_inbox", "name": "read_inbox", "args": {}}],
            ),
            ToolMessage(
                content=json.dumps(result),
                tool_call_id="forced_read_inbox",
                name="read_inbox",
            ),
        ]
    }


def build_graph(seed: int | None = None):
    """
    Constructs and compiles the LangGraph agent graph.
    Returns a compiled graph ready for invocation.
    """
    llm = get_llm(seed=seed).bind_tools(tools, tool_choice="required")

    def reasoning_node(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response],
            "iteration": state["iteration"] + 1,
        }

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("fetch_inbox", fetch_inbox_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("handle_terminal", handle_terminal_action)
    graph.add_node("force_stop", force_stop_node)

    graph.set_entry_point("fetch_inbox")
    graph.add_edge("fetch_inbox", "reasoning")

    # After reasoning: check iteration limit, then route to tools or force-stop.
    graph.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "tools": "tools",
            "force_stop": "force_stop",
        },
    )

    # After tool execution: route to the terminal handler or back to reasoning.
    graph.add_conditional_edges(
        "tools",
        after_tools,
        {
            "handle_terminal": "handle_terminal",
            "reasoning": "reasoning",
        },
    )

    # Terminal nodes route to END
    graph.add_edge("handle_terminal", END)
    graph.add_edge("force_stop", END)

    return graph.compile()
