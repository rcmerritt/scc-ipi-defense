import os
import smtplib
import unicodedata
from email.mime.text import MIMEText
from langchain.tools import tool

# SMTP configuration loaded from environment variables (set in .env)
SMTP_HOST = os.environ["SMTP_HOST"]
SMTP_PORT = int(os.environ["SMTP_PORT"])
SMTP_USER = os.environ["SMTP_USER"]
SMTP_PASS = os.environ["SMTP_PASS"]


_draft_store: dict[int, dict] = {}
_draft_counter: int = 0
_dry_run: bool = False

def reset_draft_store() -> None:
    global _draft_counter
    _draft_store.clear()
    _draft_counter = 0


def set_dry_run(enabled: bool) -> None:
    global _dry_run
    _dry_run = enabled


def get_last_sent_draft() -> dict | None:
    for draft in _draft_store.values():
        if draft.get("status") == "sent":
            return dict(draft)
    return None

@tool
def draft_email(
    to: str,
    subject: str,
    body: str,
    in_reply_to: str = "",
) -> dict:
    """
    Creates a draft email reply and saves it to the draft store.
    Does NOT send the email. Returns a draft_id that must be passed
    to send_email() to send the reply.
    Args:
        to: Recipient email address.
        subject: Subject line (typically 'Re: <original subject>').
        body: The plain-text body of the reply.
        in_reply_to: The Message-ID of the original email for threading.
    Returns:
        A dict with 'draft_id' (int) and a 'preview' summary.
        Pass the draft_id to send_email() to send the reply.
    """
    global _draft_counter
    _draft_counter += 1
    draft_id = _draft_counter
    _draft_store[draft_id] = {
        "from": SMTP_USER,
        "to": to,
        "subject": subject,
        "body": body,
        "in_reply_to": in_reply_to,
        "status": "draft",
    }
    return {
        "draft_id": draft_id,
        "preview": {
            "to": to,
            "subject": subject,
            "body_snippet": body[:120],
        },
    }


@tool
def send_email(draft_id: int) -> dict:
    """
    Sends a draft email via SMTP. The draft must exist in the store
    (created by draft_email()).
    Args:
        draft_id: The integer ID returned by draft_email().
    Returns:
        A dict with 'sent' (bool) and 'message' (str).
    """
    draft = _draft_store.get(draft_id)
    if draft is None:
        return {
            "sent": False,
            "message": f"No draft found for draft_id {draft_id}.",
        }

    # Validate body
    body = draft.get("body", "").strip()
    if not body:
        return {"sent": False, "message": "Draft body is empty; cannot send."}

    # Validate subject
    if not draft.get("subject", "").strip():
        return {"sent": False, "message": "Draft subject is empty; cannot send."}

    # Dry-run: simulate success without SMTP for self-consistency sampling
    if _dry_run:
        _draft_store[draft_id]["status"] = "sent"
        return {"sent": True, "message": f"Email sent to {draft['to']} successfully."}

    result = real_send_draft(draft)
    if result["sent"]:
        _draft_store[draft_id]["status"] = "sent"
    return result


def real_send_draft(draft: dict) -> dict:
    """
    Actually sends a draft via SMTP. Called by send_email tool after consensus.
    """
    _PUNCT_MAP = str.maketrans({
        "‐": "-", "‑": "-", "‒": "-", "–": "-",
        "—": "-", "―": "-", "−": "-",
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        " ": " ",
    })

    def to_ascii(s: str) -> str:
        # Map common punctuation, then normalize+drop remaining non-ASCII
        return (unicodedata.normalize("NFKD", s.translate(_PUNCT_MAP))
                .encode("ascii", "ignore").decode("ascii"))

    body = to_ascii(draft.get("body", "").strip())
    to_addr = to_ascii(draft.get("to", "").strip())
    subject = to_ascii(draft.get("subject", "").strip())
    in_reply_to = to_ascii(draft.get("in_reply_to", "").strip())
    from_addr = to_ascii(draft.get("from", "").strip())

    msg = MIMEText(body, _charset="us-ascii")
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject

    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo("dave.m.riley-systems.com")
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        return {"sent": True, "message": f"Email sent to {to_addr} successfully."}

    except (smtplib.SMTPException, UnicodeEncodeError) as e:
        return {"sent": False, "message": f"SMTP error: {str(e)}"}
