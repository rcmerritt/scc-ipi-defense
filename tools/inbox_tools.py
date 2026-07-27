from imapclient import IMAPClient
from langchain.tools import tool
from typing import List
import os
import logging
import mailparser

logger = logging.getLogger("email_agent.tools.inbox")

IMAP_HOST = os.environ["IMAP_HOST"]
IMAP_PORT = int(os.environ["IMAP_PORT"])
IMAP_USER = os.environ["IMAP_USER"]
IMAP_PASS = os.environ["IMAP_PASS"]

@tool
def read_inbox() -> List[dict]:
    """
    Connects to the agent's IMAP inbox and retrieves all UNSEEN
    (unread) email summaries. Returns a list of dictionaries
    containing uid, sender, subject, and date for each unread message.
    """
    with IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=False) as client:
        client.login(IMAP_USER, IMAP_PASS)
        client.select_folder("INBOX")

        # Search for unread messages
        uids = client.search(["UNSEEN"])

        if not uids:
            return []

        # Fetch envelope data for each unread message
        messages = client.fetch(uids, ["ENVELOPE"])
        summaries = []

        for uid, data in messages.items():
            envelope = data[b"ENVELOPE"]
            sender = (
                f"{envelope.from_[0].mailbox.decode()}@{envelope.from_[0].host.decode()}"
                if envelope.from_
                else "unknown"
            )
            summaries.append({
                "uid": uid,
                "sender": sender,
                "subject": envelope.subject.decode() if envelope.subject else "(no subject)",
                "date": str(envelope.date),
            })

        return summaries

@tool
def read_email(uid: int) -> dict:
    """
    Fetches and parses the full content of an email by its UID.
    Returns a dictionary with uid, sender, to (recipients), subject,
    date, the plain-text body, and a headers dict containing the
    'received' trace and 'message_id' of the email.
    """
    with IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=False) as client:
        client.login(IMAP_USER, IMAP_PASS)
        client.select_folder("INBOX")
        raw = client.fetch([uid], ["BODY.PEEK[]"])

        if uid not in raw:
            return {"error": f"Email with UID {uid} not found."}
        raw_message = raw[uid][b"BODY[]"]
        parsed = mailparser.parse_from_bytes(raw_message)

        return {
            "uid": uid,
            "sender": parsed.from_[0][1] if parsed.from_ else "unknown",
            "to": [addr[1] for addr in parsed.to_] if parsed.to_ else [],
            "subject": parsed.subject or "(no subject)",
            "date": str(parsed.date),
            "body": parsed.text_plain[0] if parsed.text_plain else "(no plain text body)",
            "headers": {
                "received": parsed.received or [],
                "message_id": parsed.message_id or "",
            },
        }

def sweep_inbox() -> int:
    try:
        with IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=False) as client:
            client.login(IMAP_USER, IMAP_PASS)
            client.select_folder("INBOX")
            uids = client.search(["ALL"])
            if not uids:
                return 0
            client.set_flags(uids, [b"\\Deleted"])
            client.expunge()
            logger.info(f"Pre-flight sweep: deleted {len(uids)} message(s) from INBOX")
            return len(uids)
    except Exception as e:
        logger.error(f"Failed to sweep inbox: {str(e)}")
        return 0

def delete_email(uid: int) -> bool:
    try:
        with IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=False) as client:
            client.login(IMAP_USER, IMAP_PASS)
            client.select_folder("INBOX")
            client.set_flags([uid], [b"\\Deleted"])
            client.expunge()
            logger.info(f"Deleted UID {uid} from INBOX")
            return True
    except Exception as e:
        logger.error(f"Failed to delete UID {uid}: {str(e)}")
        return False
