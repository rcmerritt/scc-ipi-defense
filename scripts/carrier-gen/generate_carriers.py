#!/usr/bin/env python3
# Carrier email generation script

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import pandas as pd
from anthropic import Anthropic, APIError, APIStatusError, RateLimitError

# Configuration
MODEL = "claude-opus-4-7"
MAX_TOKENS = 4096          
TEMPERATURE = 1.0          
SEED_FILE = "seed-gen.ods"
SYSTEM_PROMPT_FILE = "syn-system-prompt.txt"
USER_PROMPT_FILE = "syn-user-prompt.txt"
OUTPUT_DIR = Path("output")
EMAILS_DIR = OUTPUT_DIR / "emails"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
LOG_FILE = OUTPUT_DIR / "generation.log"

# Retry policy for transient errors
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 2.0
BACKOFF_MULT = 2.0

# Prompt assembly
def build_user_prompt(
    template: str,
    domain: str,
    length: str,
    seed: str,
    tone: str,
) -> str:
    filled = template
    filled = filled.replace(
        "{one of: software development, IT operations, healthcare administration, "
        "legal and compliance, education administration, retail operations, "
        "finance and accounting, marketing, human resources, "
        "facilities and office management}",
        domain.strip(),
    )
    filled = filled.replace(
        "{short (80-200) | medium (200-800) | long (800-2000) | very long (2000-4000)}",
        length.strip(),
    )
    if filled.rstrip().endswith("Specific seed:"):
        filled = filled.rstrip() + " " + seed.strip() + "\n"
    else:
        filled = filled.rstrip() + f"\nSpecific seed: {seed.strip()}\n"

    filled += f"Tone: {tone.strip()}\n"
    return filled


def call_with_retry(
    client: Anthropic,
    *,
    system: str,
    user: str,
    log,
) -> tuple[str, dict[str, Any]]:
    backoff = INITIAL_BACKOFF_S
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            text = "".join(text_parts).strip()
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "stop_reason": resp.stop_reason,
            }
            return text, usage

        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            retryable = isinstance(e, RateLimitError) or (status is not None and status >= 500)
            log(f"  API error attempt {attempt}/{MAX_RETRIES}: status={status} retryable={retryable} msg={e}")
            if not retryable or attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= BACKOFF_MULT
            last_exc = e

        except APIError as e:
            log(f"  Generic APIError attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= BACKOFF_MULT
            last_exc = e

    raise RuntimeError(f"Exhausted retries: {last_exc}")

# Validation
LENGTH_BANDS = {
    "short (80-200)": (80, 200),
    "medium (200-800)": (200, 800),
    "long (800-2000)": (800, 2000),
    "very long (2000-4000)": (2000, 4000),
}

def validate_email(text: str, length_label: str) -> list[str]:
    warnings = []
    if not text.startswith("Dave,"):
        warnings.append("does not start with 'Dave,'")
    if not text.rstrip().endswith("Julie"):
        warnings.append("does not end with 'Julie'")
    if "—" in text:
        warnings.append("contains em dash")
    if "--" in text:
        warnings.append("contains '--'")
    if "@" in text and "riley-systems.com" not in text:
        warnings.append("contains @ but no riley-systems.com address")
    if "P.S." in text or "PS:" in text:
        warnings.append("contains postscript")
    band = LENGTH_BANDS.get(length_label.strip())
    if band is not None:
        lo, hi = band
        n = len(text)
        if n < lo:
            warnings.append(f"length {n} below band [{lo},{hi}]")
        elif n > hi:
            warnings.append(f"length {n} above band [{lo},{hi}]")
    return warnings

# Main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate only the first N rows (for piloting).")
    ap.add_argument("--rows", type=str, default=None,
                    help="Comma-separated seed numbers (1-indexed) to generate.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows whose .txt output already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print prompts but don't call the API.")
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.", file=sys.stderr)
        return 2

    df = pd.read_excel(SEED_FILE, engine="odf")
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Seed Number", "Domain", "Length", "Seed", "Tone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: spreadsheet missing columns {missing}. Found: {list(df.columns)}",
              file=sys.stderr)
        return 2
    df = df.dropna(subset=["Seed Number", "Domain", "Length", "Seed"])
    df["Seed Number"] = df["Seed Number"].astype(int)

    with open(SYSTEM_PROMPT_FILE) as f:
        system_prompt = f.read()
    with open(USER_PROMPT_FILE) as f:
        user_prompt_template = f.read()

    # Filter rows
    if args.rows:
        wanted = {int(x) for x in args.rows.split(",")}
        df = df[df["Seed Number"].isin(wanted)]
    if args.limit:
        df = df.head(args.limit)

    EMAILS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if resuming
    manifest: dict[str, Any] = {"model": MODEL, "rows": {}}
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
        except json.JSONDecodeError:
            print(f"WARN: existing {MANIFEST_FILE} is corrupt; starting fresh.")
            manifest = {"model": MODEL, "rows": {}}

    log_fh = open(LOG_FILE, "a")
    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    client = None if args.dry_run else Anthropic()
    total_in = total_out = 0

    for _, row in df.iterrows():
        n = int(row["Seed Number"])
        out_path = EMAILS_DIR / f"email_{n:03d}.txt"

        if args.resume and out_path.exists():
            log(f"Row {n}: skipping (already on disk)")
            continue

        try:
            user_msg = build_user_prompt(
                template=user_prompt_template,
                domain=row["Domain"],
                length=row["Length"],
                seed=row["Seed"],
                tone=row["Tone"],
            )
        except Exception as e:
            log(f"Row {n}: prompt build failed: {e}")
            continue

        if args.dry_run:
            log(f"Row {n}: DRY RUN")
            print("---- USER PROMPT ----")
            print(user_msg)
            print("---------------------")
            continue

        log(f"Row {n}: calling API ({row['Domain']}/{row['Length']})")
        try:
            text, usage = call_with_retry(client, system=system_prompt, user=user_msg, log=log)
        except Exception as e:
            log(f"Row {n}: FAILED: {e}")
            continue

        out_path.write_text(text + "\n")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        warnings = validate_email(text, row["Length"])

        manifest["rows"][str(n)] = {
            "seed_number": n,
            "type": "INFORMATIONAL",
            "domain": str(row["Domain"]),
            "length_band": str(row["Length"]),
            "tone": str(row["Tone"]),
            "seed_text": str(row["Seed"]),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "output_file": str(out_path),
            "char_length": len(text),
            "sha256": sha,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "stop_reason": usage["stop_reason"],
            "warnings": warnings,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]

        if warnings:
            log(f"Row {n}: OK with warnings: {warnings}")
        else:
            log(f"Row {n}: OK ({len(text)} chars, {usage['output_tokens']} out tokens)")

        # Persist manifest after every row
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))

    log(f"DONE. total_input_tokens={total_in} total_output_tokens={total_out}")
    log_fh.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
