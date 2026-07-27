import logging
from langchain.tools import tool

logger = logging.getLogger("email_agent.tools.action")

@tool
def summary(summary: str) -> dict:
    """
    Returns a summary of the processed email to the user.
    This is a terminal action. After calling this, processing ends.
    Args:
        summary: A brief summary of the email and why no reply is needed.
    Returns:
        A dict confirming the summary was delivered.
    """
    logger.info(f"EMAIL SUMMARY: {summary}")
    return {"delivered": True, "summary": summary}
