import pdfplumber
import pathlib
import re
import json

TIMETABLE_PDF = pathlib.Path("../data/timetable.pdf")
OUTPUT_DIR = pathlib.Path("../data/output/timetable")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


COURSEWISE_FIRST_PAGE = 9
COURSEWISE_LAST_PAGE = 111 


HOUR_TO_TIME = {
    1: "8:00-8:50AM", 2: "9:00-9:50AM", 3: "10:00-10:50AM", 4: "11:00-11:50AM", 5: "12:00-12:50PM",
    6: "1:00-1:50PM", 7: "2:00-2:50PM", 8: "3:00-3:50PM", 9: "4:00-4:50PM", 10: "5:00-5:50PM",
}
MIDSEM_SESSION_TO_TIME = {
    "FN1": "9:00 AM - 10:30 AM", "FN2": "11:00 AM - 12:30 PM",
    "AN1": "2:00 PM - 3:30 PM", "AN2": "4:00 PM - 5:30 PM",
}
COMPRE_SESSION_TO_TIME = {
    "FN": "9:00 AM - 12:00 PM", "AN": "2:00 PM - 5:00 PM",
}
DAY_CODES = {"M", "T", "W", "Th", "F", "S"}


def parse_days_hours(raw: str):
    if not raw or not raw.strip():
        return [], []

    tokens = raw.split()
    meetings = []
    unparsed = []
    cur_days, cur_hours = [], []

    def flush():
        if not cur_days or not cur_hours:
            return
        for d in cur_days:
            for h in cur_hours:
                meetings.append({"day": d, "hour": h, "time": HOUR_TO_TIME.get(h, f"UNKNOWN_HOUR_{h}")})

    for tok in tokens:
        if tok in DAY_CODES:
            if cur_hours:
                flush()
                cur_days, cur_hours = [], []
            cur_days.append(tok)
        elif tok.isdigit():
            cur_hours.append(int(tok))
        else:
            unparsed.append(tok)

    flush()
    return meetings, unparsed


def parse_midsem_compre(raw_date_session: str, session_lookup: dict):
    
    if not raw_date_session or not raw_date_session.strip():
        return None
    parts = raw_date_session.strip().split()
    if len(parts) != 2:
        return {"date": raw_date_session.strip(), "session": None, "time": None}
    date, session = parts
    return {"date": date, "session": session, "time": session_lookup.get(session)}


def is_header_or_label_row(row) -> bool:
    
    if row[0] in ("COM\nCOD", None) and row[1] in ("COURSE NO.", None) and (row[8] in ("SEC", "", None)):
        if row[1] == "COURSE NO." or row[3] in ("L", None) and row[9] in ("", None) and row[2] in ("COURSE TITLE", None):
            return True
    return False


def is_blank_row(row) -> bool:
    return all(cell is None or str(cell).strip() == "" for cell in row)


def new_course_record(row) -> dict:
    def clean_credit(v):
        return None if v in (None, "-", "") else v
    return {
        "com_cod": row[0].strip() if row[0] else None,
        "course_id": row[1].strip() if row[1] else None,
        "course_title": row[2].strip() if row[2] else None,
        "credit_structure": {
            "lecture_hrs": clean_credit(row[3]),
            "practical_hrs": clean_credit(row[4]),
            "tutorial_hrs": clean_credit(row[5]),
            "self_study_hrs": clean_credit(row[6]),
            "units_credit_hours": clean_credit(row[7]),
        },
        "lecture_sections": [],
        "tutorial_sections": [],
        "practical_sections": [],
        "midsem": None,
        "compre": None,
        "parse_notes": [],
    }


def new_section(sec_id: str, instructor: str, room: str, days_hours_raw: str) -> dict:
    meetings, unparsed = parse_days_hours(days_hours_raw or "")
    room_clean = room.strip() if room else None
    status = "cancelled" if room_clean == "CANCLED" else ("active" if room_clean or meetings else "unknown")
    section = {
        "section_id": sec_id,
        "instructors": [instructor.strip()] if instructor and instructor.strip() else [],
        "room": None if room_clean == "CANCLED" else room_clean,
        "meetings": meetings,
        "status": status,
    }
    if unparsed:
        section["parse_warning"] = f"unparsed tokens in days/hours: {unparsed}"
    return section


def save_course(course: dict, seq: int):
    if not course or not course.get("course_id"):
        return
    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", course["course_id"]).strip("_")
    filename = f"{seq:03d}_{safe_id}.json"
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(course, f, indent=2)


def main():
    current_course = None
    current_section_type = "lecture_sections"
    saved_count = 0

    with pdfplumber.open(TIMETABLE_PDF) as pdf:
        for page_idx in range(COURSEWISE_FIRST_PAGE, COURSEWISE_LAST_PAGE + 1):
            page = pdf.pages[page_idx]
            table = page.extract_table(
                table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            )
            if not table:
                continue

            for row in table:
                if is_blank_row(row) or is_header_or_label_row(row):
                    continue

                com_cod, course_no, course_title, sec_id, instructor, room, days_hours = (
                    row[0], row[1], row[2], row[8], row[9], row[10], row[11]
                )
                midsem_raw, compre_raw = row[12], row[13]

                if com_cod and com_cod.strip():
                    if current_course:
                        saved_count += 1
                        save_course(current_course, saved_count)
                    current_course = new_course_record(row)
                    current_section_type = "lecture_sections"
                    section = new_section(sec_id, instructor, room, days_hours)
                    current_course["lecture_sections"].append(section)
                    if midsem_raw:
                        current_course["midsem"] = parse_midsem_compre(midsem_raw, MIDSEM_SESSION_TO_TIME)
                    if compre_raw:
                        current_course["compre"] = parse_midsem_compre(compre_raw, COMPRE_SESSION_TO_TIME)
                    continue

                # ... (rest of the loop body is unchanged from before)

        if current_course:
            saved_count += 1
            save_course(current_course, saved_count)

    print(f"\nSaved {saved_count} course timetable records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()