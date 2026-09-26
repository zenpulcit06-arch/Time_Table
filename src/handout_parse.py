import pdfplumber
import pathlib
import re
import json
import time
import ollama

# ---- Paths ----
HANDOUTS_DIR = pathlib.Path("../data/handouts")
OUTPUT_DIR = pathlib.Path("../data/output")
FAILED_LOG = pathlib.Path("../data/failed_log.txt")
TOO_LONG_LOG = pathlib.Path("../data/too_long_for_local_llm.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ----
MAX_RETRIES = 3
NUM_CTX = 16384  # was silently defaulting to something much smaller — this was the real bug

# Safety net only now that num_ctx is fixed — your longest handout so far was
# 17,869 chars and succeeded fine. This just catches a genuinely pathological
# outlier (e.g. a broken extraction that duplicates content, or a handout with
# a huge embedded table) rather than doing real work like before.
MAX_HANDOUT_CHARS = 60000

EXTRACTION_PROMPT = """You are extracting structured data from a university course handout. Return ONLY a single valid JSON object — no markdown code fences, no explanation before or after, no trailing commentary.

Use exactly this schema:
{{
  "course_id": string,
  "course_title": string,
  "credits": number or null,
  "instructor_in_charge": string or null,
  "evaluation_components": [
    {{
      "component": string,
      "weightage_percent": number or null,
      "when": string or null,
      "notes": string or null
    }}
  ],
  "attendance_policy": string or null,
  "makeup_policy": string or null,
  "extraction_notes": string or null
}}

Rules:
- Only use information that is literally present in the HANDOUT TEXT below. Never use information from these instructions, this schema, or any example as if it were content from the handout.
- If a value is not clearly and explicitly stated in the handout text, use null. Do NOT infer, guess, or fill in a plausible-sounding value.
- If a field in the handout's own evaluation table contains unresolved placeholder text instead of a real value (meaning the source document itself has not yet been filled in — for instance non-data filler text, an empty dash, or bracketed template text left blank by whoever wrote the handout), set that specific field to null and briefly describe what you saw, in your own words, in "extraction_notes". Do not copy any example wording from these instructions into your output.
- "TBA", "To Be Announced", or similar are legitimate real-world values meaning a date has not been scheduled yet — treat these as an actual string value, not as a placeholder to null out.
- "weightage_percent" must be a number (e.g. 35), not a string, and not include the % sign.
- Do not invent evaluation components that are not listed in the text.
- "extraction_notes" should be null unless there is a genuine, specific issue with the source handout that a human reviewer should check. Do not fill it with a generic or example issue just to have something to say.

HANDOUT TEXT:
\"\"\"
{handout_text}
\"\"\"
"""

REQUIRED_KEYS = {
    "course_id", "course_title", "credits", "instructor_in_charge",
    "evaluation_components", "attendance_policy", "makeup_policy", "extraction_notes"
}


def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    """Stage 1: PDF -> cleaned combined text."""
    texts = []
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            table = page.extract_tables()

            if text and text.strip():
                texts.append(text)

            if table:
                for t in table:
                    cleaned_table = []
                    for row in t:
                        cleaned_row = [
                            re.sub(r'[\uf0b7\uf0a7]+|\s+', ' ', cell).strip()
                            for cell in row
                            if cell is not None and cell != "(%)" and cell != ""
                        ]
                        if cleaned_row:
                            cleaned_table.append(cleaned_row)
                    if cleaned_table:
                        tables.append({"content": cleaned_table, "Page number": page.page_number})

    combined_text = "\n".join(texts)
    for table_dict in tables:
        combined_text += f"\n[Table on Page {table_dict['Page number']}]\n"
        for row in table_dict["content"]:
            combined_text += " | ".join(row) + "\n"

    return combined_text


def call_llm(handout_text: str) -> str:
    """Stage 2: text -> raw LLM response string."""
    prompt = EXTRACTION_PROMPT.format(handout_text=handout_text)
    response = ollama.chat(
        model="qwen2.5:7b",
        format="json",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": NUM_CTX},
    )
    return response["message"]["content"]


def parse_response(raw_response: str) -> dict:
    """Stage 3: raw string -> Python dict. Raises json.JSONDecodeError on invalid JSON."""
    return json.loads(raw_response)


def validate_schema(data: dict) -> list[str]:
    """Stage 4: return a list of problems (empty list = valid)."""
    problems = []
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        problems.append(f"missing keys: {missing}")
    if "evaluation_components" in data and not isinstance(data["evaluation_components"], list):
        problems.append("evaluation_components is not a list")
    return problems


def process_handout(pdf_path: pathlib.Path) -> dict:
    """
    Stages 1-4 with retry and a length-based safety-net skip.
    Returns either a valid extracted dict, or a dict with a "_skip_reason" key
    explaining why it wasn't extracted.
    """
    handout_text = extract_pdf_text(pdf_path)
    text_len = len(handout_text)
    print(f"  handout_text length: {text_len} chars")

    if text_len > MAX_HANDOUT_CHARS:
        print(f"  -> SKIPPED: {text_len} chars exceeds {MAX_HANDOUT_CHARS} safety-net limit")
        return {"_skip_reason": f"too_long ({text_len} chars)"}

    raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_llm(handout_text)
            data = parse_response(raw)
            problems = validate_schema(data)
            if not problems:
                return data

            print(f"  attempt {attempt}: schema problems {problems}")
            debug_path = OUTPUT_DIR / f"{pdf_path.stem}_attempt{attempt}_raw.txt"
            with open(debug_path, "w") as f:
                f.write(raw)

        except json.JSONDecodeError as e:
            print(f"  attempt {attempt}: invalid JSON ({e})")
            debug_path = OUTPUT_DIR / f"{pdf_path.stem}_attempt{attempt}_raw.txt"
            with open(debug_path, "w") as f:
                f.write(raw if raw is not None else "<no response captured>")

    return {"_skip_reason": "failed_after_retries"}


def main():
    pdf_files = sorted(HANDOUTS_DIR.glob("*.pdf"))  # ALL handouts now
    total = len(pdf_files)
    print(f"Found {total} handouts to process.\n")

    too_long_queue = []
    failed_queue = []
    succeeded = 0
    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files, start=1):
        elapsed = time.time() - start_time
        avg_per_file = elapsed / (i - 1) if i > 1 else 0
        eta_remaining = avg_per_file * (total - i + 1)
        print(f"[{i}/{total}] {pdf_path.name}  (elapsed {elapsed/60:.1f}m, ETA {eta_remaining/60:.1f}m)")

        result = process_handout(pdf_path)

        if "_skip_reason" not in result:
            out_path = OUTPUT_DIR / (pdf_path.stem + ".json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  -> saved {out_path.name}")
            succeeded += 1

        elif result["_skip_reason"].startswith("too_long"):
            too_long_queue.append((pdf_path.name, result["_skip_reason"]))
            print(f"  -> queued for bigger LLM: {result['_skip_reason']}")

        else:
            failed_queue.append(pdf_path.name)
            print(f"  -> FAILED after {MAX_RETRIES} attempts")

    total_time = time.time() - start_time
    print(f"\n{succeeded}/{total} succeeded in {total_time/60:.1f} minutes.")

    if too_long_queue:
        with open(TOO_LONG_LOG, "w") as f:
            for name, reason in too_long_queue:
                f.write(f"{name}: {reason}\n")
        print(f"{len(too_long_queue)} routed to too-long queue -> see {TOO_LONG_LOG}")

    if failed_queue:
        with open(FAILED_LOG, "w") as f:
            f.write("\n".join(failed_queue))
        print(f"{len(failed_queue)} genuinely failed -> see {FAILED_LOG}")


if __name__ == "__main__":
    #main()
    text = extract_pdf_text(pathlib.Path("../data/handouts/362_MATH_F214.pdf"))
    print(repr(text[:500]))
    print(len(text))