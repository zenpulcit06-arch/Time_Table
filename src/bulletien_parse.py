import pdfplumber
import pathlib
import re
import json
import ollama

BULLETIN_PDF = pathlib.Path("../data/bulletin.pdf")
OUTPUT_DIR = pathlib.Path("../data/output/bulletin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CTX = 16384
MAX_RETRIES = 3


STANDARD_PROGRAMME_FIRST_PAGE = 200  
STANDARD_PROGRAMME_LAST_PAGE = 300   


DEL_RULE = (
    "Any elective-designated course offered within the student's own department "
    "(matching the department's course-code prefix), not already counted toward "
    "Core requirements. No fixed course list exists in the Bulletin for standard "
    "on-campus programmes — eligible courses must be resolved by department "
    "prefix against the Handout/Timetable datasets."
)
OPEL_RULE = (
    "Any course from a discipline other than the student's own major, subject to "
    "prerequisites and, in some cases, Head of Department approval. No fixed "
    "course list exists in the Bulletin — this is a policy rule, not an "
    "enumerated pool."
)

STRUCTURE_EXTRACTION_PROMPT = """You are extracting a university degree programme's semester-wise course requirements from two paired text blocks taken from adjacent table cells: a TITLES block (course codes and titles, possibly wrapped across multiple lines) and a CREDITS block (one number per course, in the SAME top-to-bottom order as the titles block).

Background context on this university's terminology, so you correctly recognize category names when they appear (this is background knowledge, not content to extract):
- "Humanities Electives" / "HUEL" — elective courses from a shared humanities course pool
- "Discipline Electives" / "DEL" — elective courses specific to the student's own department
- "Open Electives" / "OPEL" — elective courses from any department outside the student's own
- "Core" / "Discipline Core" — compulsory courses specific to the student's programme
- "Foundation" (Science/Mathematics/Engineering Foundation) — compulsory general-education courses taken by most first-year students regardless of discipline

Return ONLY a single valid JSON object — no markdown fences, no commentary.

Schema:
{{
  "entries": [
    {{
      "type": "course" or "category_requirement",
      "course_code": string or null,
      "course_title": string or null,
      "credit_units": number or null,
      "category_label": string or null
    }}
  ]
}}

Rules:
- The Nth course/category you identify in the TITLES block corresponds to the Nth number in the CREDITS block — pair them by this matching order, not by proximity or guesswork.
- "course" entries have a course code (e.g. "CS F211") and a title. "category_requirement" entries are unresolved elective/foundation slots with only a category name (e.g. "Humanities Electives") and no specific course code — use the background terminology above to recognize these correctly, and normalize the category_label to the full name (e.g. "Discipline Electives" not "DEL") even if the source text uses an abbreviation.
- Reassemble titles that wrap across multiple lines into one coherent title, but do NOT merge two genuinely different courses into one, and do NOT invent words not present in the source.
- If you cannot confidently pair a specific credit number to a specific entry, set credit_units to null rather than guessing.
- Only use information literally present in the TITLES and CREDITS blocks below — the background terminology above is for your understanding only, never treat it as content to extract.

TITLES BLOCK:
\"\"\"
{titles_text}
\"\"\"

CREDITS BLOCK:
\"\"\"
{credits_text}
\"\"\"
"""

HUEL_EXTRACTION_PROMPT = """You are extracting a flat list of courses from a university's "Pool of Humanities Electives" listing. The source text was extracted from a two-column PDF table with columns: Course No., Course Title, L, P, U (lecture hours, practical hours, total credit units).

Return ONLY a single valid JSON object — no markdown fences, no commentary.

Schema:
{{
  "courses": [
    {{ "course_code": string, "course_title": string, "credit_units": number or null }}
  ]
}}

Rules:
- Extract every course listed, in any order.
- Course codes look like "HSS F234", "GS F231", etc.
- credit_units is the "U" column value (a single number). If unclear for a given course, use null rather than guessing.
- Only use information literally present in the SOURCE TEXT below.

SOURCE TEXT:
\"\"\"
{pool_text}
\"\"\"
"""

def is_header_or_summary_row(row) -> bool:
    """Skip column-header rows and near-empty summary rows (e.g. just totals like '18  19')."""
    if row[0] and "Semester-wise Pattern" in str(row[0]):
        return True
    if row[1] == "First Semester" or row[3] == "Second Semester":
        return True
    # summary rows: Year column empty and title cells empty/very short
    if not row[0] and (not row[1] or len(str(row[1]).strip()) < 5) and (not row[3] or len(str(row[3]).strip()) < 5):
        return True
    return False


def call_llm(prompt: str) -> str:
    response = ollama.chat(
        model="qwen2.5:7b",
        format="json",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": NUM_CTX},
    )
    return response["message"]["content"]


def call_llm_with_retry(prompt: str, required_key: str):
    """Generic retry wrapper: parse JSON, confirm required_key exists, else retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_llm(prompt)
            data = json.loads(raw)
            if required_key in data:
                return data
            print(f"    attempt {attempt}: missing key '{required_key}'")
        except json.JSONDecodeError as e:
            print(f"    attempt {attempt}: invalid JSON ({e})")
    return None


def build_programme_index() -> dict:
    """
    Scan the Bulletin once for 'Semester-wise Pattern for Students Admitted to
    <Programme> Programme' headers, skipping Table-of-Contents entries (which
    have dotted leaders '……' and don't represent real content pages).
    Returns {programme_name: page_index}.
    """
    index = {}
    pattern = re.compile(
        r"Semester-wise Pattern for Students Admitted to\s+(.+?)\s+Programme", re.IGNORECASE
    )
    with pdfplumber.open(BULLETIN_PDF) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "……" in text or "...................." in text:
                continue  # TOC entry, not real content
            m = pattern.search(text)
            if m:
                name = m.group(1).strip()
                if name not in index:  # keep first occurrence only
                    index[name] = i
    return index


def extract_programme_structure(programme_name: str, page_index: int) -> dict:
    with pdfplumber.open(BULLETIN_PDF) as pdf:
        page = pdf.pages[page_index]
        table = page.extract_table(
            table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
        )

    if not table:
        return {"programme_name": programme_name, "page": page_index, "entries": [], "parse_notes": ["no table extracted"]}

    all_entries = []
    parse_notes = []

    for row in table:
        if is_header_or_summary_row(row):
            continue

        semester_blocks = [(row[1], row[2]), (row[3], row[4])] if len(row) >= 5 else []

        for titles_text, credits_text in semester_blocks:
            if not titles_text or not titles_text.strip():
                continue
            result = call_llm_with_retry(
                STRUCTURE_EXTRACTION_PROMPT.format(
                    titles_text=titles_text, credits_text=credits_text or ""
                ),
                required_key="entries",
            )
            if result:
                all_entries.extend(result["entries"])
            else:
                parse_notes.append(f"failed to parse block: {titles_text[:60]}...")

    return {
        "programme_name": programme_name,
        "page": page_index,
        "entries": all_entries,
        "discipline_elective_rule": DEL_RULE,
        "open_elective_rule": OPEL_RULE,
        "parse_notes": parse_notes,
    }


def extract_huel_pool(first_page: int, last_page: int) -> dict:
    """Extract the shared Humanities elective pool once (pages confirmed at ~333-335)."""
    all_courses = []
    parse_notes = []

    with pdfplumber.open(BULLETIN_PDF) as pdf:
        for page_idx in range(first_page, last_page + 1):
            text = pdf.pages[page_idx].extract_text() or ""
            if not text.strip():
                continue
            result = call_llm_with_retry(
                HUEL_EXTRACTION_PROMPT.format(pool_text=text), required_key="courses"
            )
            if result:
                all_courses.extend(result["courses"])
            else:
                parse_notes.append(f"failed to parse HUEL page {page_idx}")

    seen = {}
    for c in all_courses:
        code = c.get("course_code")
        if code and code not in seen:
            seen[code] = c

    return {"courses": list(seen.values()), "parse_notes": parse_notes}


def save_json(data: dict, filename: str):
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)


def main():
    print("Building programme index...")
    index = build_programme_index()
    print(f"Found {len(index)} programmes:")
    for name, page in index.items():
        print(f"  {name} -> page {page}")
    save_json(index, "_programme_index.json")

    print("\nExtracting shared HUEL pool...")
    huel = extract_huel_pool(first_page=332, last_page=335) 
    print(f"  extracted {len(huel['courses'])} humanities elective courses")
    save_json(huel, "_huel_pool.json")

    print("\nExtracting programme structures...")
    for name, page in index.items():
        print(f"  {name} (page {page})...")
        structure = extract_programme_structure(name, page)
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        save_json(structure, f"{safe_name}.json")
        if structure["parse_notes"]:
            print(f"    -> {len(structure['parse_notes'])} parse issues, see file")

    print("\nDone.")


if __name__ == "__main__":
    main()