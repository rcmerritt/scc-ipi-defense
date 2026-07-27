from __future__ import annotations
import json
import logging
import os
import time
from email.mime.text import MIMEText
from pathlib import Path
import smtplib
from imapclient import IMAPClient
from tools.inbox_tools import IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASS

logger = logging.getLogger("email_agent.tools.harness")

# Test sender (julie.n) separate identity from the agent's outbound (dave.m)
TEST_SENDER_HOST = os.environ["TEST_SENDER_HOST"]
TEST_SENDER_PORT = int(os.environ["TEST_SENDER_PORT"])
TEST_SENDER_USER = os.environ["TEST_SENDER_USER"]
TEST_SENDER_PASS = os.environ["TEST_SENDER_PASS"]

# Where injected emails are addressed
TEST_RECIPIENT = os.environ["TEST_RECIPIENT"]

def load_corpus(subdir: Path) -> list[dict]:
    """Read <subdir>/emails.jsonl into a list of records."""
    path = Path(subdir) / "emails.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def inject_email(record: dict) -> None:
    """Send one corpus record from julie.n to dave.m via SMTP."""
    msg = MIMEText(record["body"], _charset="us-ascii")
    msg["From"] = TEST_SENDER_USER
    msg["To"] = TEST_RECIPIENT
    msg["Subject"] = record["subject"]

    with smtplib.SMTP(TEST_SENDER_HOST, TEST_SENDER_PORT) as server:
        server.ehlo("riley-systems.com")
        server.login(TEST_SENDER_USER, TEST_SENDER_PASS)
        server.sendmail(TEST_SENDER_USER, [TEST_RECIPIENT], msg.as_string())

    logger.info(
        f"Injected {record.get('id', '?')} subject={record['subject']!r}"
    )

def wait_for_new_email(expected_subject: str, timeout: float = 30.0, poll: float = 0.5) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=False) as client:
            client.login(IMAP_USER, IMAP_PASS)
            client.select_folder("INBOX")
            uids = client.search(["UNSEEN"])
            if uids:
                fetched = client.fetch(uids, ["ENVELOPE"])
                for uid, data in fetched.items():
                    env = data[b"ENVELOPE"]
                    subject = env.subject.decode() if env.subject else ""
                    if subject == expected_subject:
                        if len(uids) > 1:
                            logger.warning(
                                f"Inbox has {len(uids)} UNSEEN messages when only "
                                f"the just-injected one was expected — possible "
                                f"leftover or duplicate inject"
                            )
                        return uid
        time.sleep(poll)

    raise RuntimeError(
        f"Timed out after {timeout}s waiting for subject {expected_subject!r}"
    )
