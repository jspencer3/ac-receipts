#!/usr/bin/env python3
"""Extract date, total, sales tax, and vendor from scanned receipts.

Gemini is used only for OCR. All field extraction is done locally in Python.

Usage:
    uv run scan.py [--input-dir inputs] [--output-dir outputs]
                   [--model gemini-3.5-flash-lite] [--sleep 5] [--dry-run]

Outputs:
    outputs/receipts.csv      one row per receipt (multiple per page allowed)
    outputs/ocr/<name>.txt    cached raw OCR text (re-runs skip the API)

Requires GEMINI_API_KEY in the environment (loaded from .env).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

DELIMITER = "=====RECEIPT====="
PROMPT = (
    "This page may contain one or more separate receipts. Transcribe all "
    "text exactly as printed, preserving line breaks. If there is more than "
    f"one receipt, put a line containing only {DELIMITER} between each "
    "receipt. Do not put the delimiter before the first or after the last "
    "receipt. Return only the text."
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".pdf"}
DAILY_REQUEST_LIMIT = 1500  # Gemini free tier RPD ceiling
REQUEST_TIMEOUT_MS = 120_000
QUOTA_NAME = ".quota.json"
CSV_NAME = "receipts.csv"
CSV_FIELDS = ["file", "receipt", "vendor", "date", "total", "sales_tax", "needs_review"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scan")

AMOUNT = r"\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2}|\d*\.\d{2})(?!\d)"
MONTHS = (
    "jan feb mar apr may jun jul aug sep oct nov dec".split()
)
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"), "mdy"),
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})\b"), "mdy2"),
    (
        re.compile(
            r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I
        ),
        "mon",
    ),
    (
        re.compile(
            r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")[a-z]*\.?,?\s+(\d{4})\b", re.I
        ),
        "dmon",
    ),
]


def parse_date(text: str) -> str | None:
    """Return the earliest plausible (not future) date as ISO, or None."""
    found: list[date] = []
    today = date.today()
    for pattern, kind in DATE_PATTERNS:
        for m in pattern.finditer(text):
            a, b, c = m.groups()
            try:
                if kind == "ymd":
                    d = date(int(a), int(b), int(c))
                elif kind == "mdy":
                    d = date(int(c), int(a), int(b))
                elif kind == "mdy2":
                    d = date(2000 + int(c), int(a), int(b))
                elif kind == "mon":
                    d = date(int(c), MONTHS.index(a[:3].lower()) + 1, int(b))
                else:
                    d = date(int(c), MONTHS.index(b[:3].lower()) + 1, int(a))
            except ValueError:
                continue
            if 2000 <= d.year <= today.year and d <= today:
                found.append(d)
    return min(found).isoformat() if found else None


def to_amount(s: str) -> float:
    return float(s.replace(",", ""))


def line_amounts(line: str) -> list[float]:
    return [to_amount(m.group(1)) for m in re.finditer(AMOUNT, line)]


def parse_total(lines: list[str]) -> float | None:
    candidates: list[float] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if "total" not in low or re.search(r"sub\s*-?\s*total|total\s+(?:tax|savings|discount)", low):
            continue
        amounts = line_amounts(line)
        if not amounts and i + 1 < len(lines):
            amounts = line_amounts(lines[i + 1])
        if amounts:
            candidates.append(amounts[-1])
    if candidates:
        return candidates[-1]
    all_amounts = [a for line in lines for a in line_amounts(line)]
    return max(all_amounts) if all_amounts else None


def parse_tax(lines: list[str]) -> float | None:
    taxes: list[float] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if not re.search(r"\b(sales\s+)?tax\b|\bgst\b|\bhst\b|\bvat\b", low):
            continue
        if re.search(r"tax\s*(id|no|number|exempt)|before\s+tax|incl", low):
            continue
        stripped = re.sub(r"\d+(?:\.\d+)?\s*%", "", line)
        amounts = line_amounts(stripped)
        if not amounts and i + 1 < len(lines):
            amounts = line_amounts(lines[i + 1])
        if amounts:
            taxes.append(amounts[-1])
    if not taxes:
        return None
    # Summary line is usually last; "total tax" lines win if present.
    return taxes[-1]


def parse_vendor(lines: list[str]) -> str | None:
    skip = re.compile(
        r"receipt|invoice|welcome|thank|^\W*$|^\d[\d\s\-().]*$|\d{3}[-.\s]\d{4}|www\.|\.com",
        re.I,
    )
    for line in lines[:10]:
        text = line.strip()
        if len(re.findall(r"[A-Za-z]", text)) >= 3 and not skip.search(text):
            return re.sub(r"\s+", " ", text)
    return None


def split_receipts(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(rf"^\s*{re.escape(DELIMITER)}\s*$", text, flags=re.M)]
    return [p for p in parts if p] or [text]


def parse_receipt(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = {
        "vendor": parse_vendor(lines),
        "date": parse_date(text),
        "total": parse_total(lines),
        "sales_tax": parse_tax(lines),
    }
    result["needs_review"] = any(
        result[k] is None for k in ("vendor", "date", "total")
    ) or (
        result["total"] is not None
        and result["sales_tax"] is not None
        and result["sales_tax"] > result["total"]
    )
    return result


def load_quota(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("date") == str(date.today()):
            return data
    return {"date": str(date.today()), "count": 0}


def ocr_with_gemini(client: genai.Client, model: str, path: Path) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    response = client.models.generate_content(
        model=model,
        contents=[PROMPT, types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)],
    )
    return response.text or ""


def ocr_with_retry(
    client: genai.Client, model: str, path: Path, max_retries: int = 5
) -> str:
    delay = 10
    for attempt in range(1, max_retries + 1):
        try:
            return ocr_with_gemini(client, model, path)
        except Exception as exc:
            retryable = any(
                m in str(exc) for m in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "504", "DEADLINE", "timed out", "Timeout")
            )
            if attempt == max_retries or not retryable:
                raise
            log.warning(
                "Rate limited on %s (attempt %d/%d). Backing off %ds.",
                path.name, attempt, max_retries, delay,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="inputs", type=Path)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--sleep", default=5.0, type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        log.error("GEMINI_API_KEY not set. Add it to .env or export it.")
        return 1

    if not args.input_dir.is_dir():
        log.error("Input dir %s does not exist.", args.input_dir)
        return 1
    images = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        log.warning("No images found in %s", args.input_dir)
        return 0

    ocr_dir = args.output_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    quota_path = args.output_dir / QUOTA_NAME
    quota = load_quota(quota_path)

    if args.dry_run:
        for img in images:
            cached = (ocr_dir / f"{img.name}.txt").exists()
            log.info("[dry-run] %s (%s)", img.name, "cached OCR" if cached else "would OCR")
        return 0

    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
    )
    rows: list[dict] = []
    for img in images:
        cache = ocr_dir / f"{img.name}.txt"
        if cache.exists():
            text = cache.read_text()
        else:
            if quota["count"] >= DAILY_REQUEST_LIMIT:
                log.warning("Daily request limit (%d) reached. Re-run later.", DAILY_REQUEST_LIMIT)
                break
            log.info("OCR %s ...", img.name)
            try:
                text = ocr_with_retry(client, args.model, img)
            except Exception as exc:
                log.error("OCR failed for %s: %s", img.name, exc)
                rows.append({"file": img.name, "receipt": 1, "needs_review": True})
                continue
            cache.write_text(text)
            quota["count"] += 1
            quota_path.write_text(json.dumps(quota, indent=2))
            time.sleep(args.sleep)
        for n, part in enumerate(split_receipts(text), start=1):
            rows.append({"file": img.name, "receipt": n, **parse_receipt(part)})

    with (args.output_dir / CSV_NAME).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d row(s) to %s", len(rows), args.output_dir / CSV_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
