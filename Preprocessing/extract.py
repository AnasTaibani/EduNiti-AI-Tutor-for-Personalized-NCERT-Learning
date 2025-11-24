import os
import re
import zipfile
from pathlib import Path
import shutil

# ---------------- CONFIG ----------------
SOURCE_DIR = r"C:\Users\Lenovo\Downloads\NCERT_Books_zipfiles"  
OUTPUT_ROOT = r"D:\NCERT_bookss"
# ----------------------------------------

SUBJECT_MAP = {
    # Sanskrit
    "hsk": "Sanskrit",

    # Accountancy
    "eac": "Accountancy",

    # Chemistry
    "ech": "Chemistry",

    # Mathematics
    "emh": "Mathematics",

    # Biology
    "ebo": "Biology",

    # Psychology
    "epy": "Psychology",

    # Geography
    "egy": "Geography",

    # Physics
    "eph": "Physics",

    # Hindi (multiple variants)
    "hat": "Hindi",
    "har": "Hindi",
    "hvt": "Hindi",
    "han": "Hindi",

    # Sociology
    "esy": "Sociology",

    # English (multiple variants)
    "evt": "English",
    "ekl": "English",
    "efl": "English",


    # Political Science
    "eps": "Political_Science",

    # History
    "ehs": "History",

    # Economics
    "eec": "Economics",
    "est": "Economics",

    # Business Studies
    "ebs": "Business_Studies",

    # Urdu (multiple variants)
    "una": "Urdu",
    "udh": "Urdu",
    "uku": "Urdu",
    "uga": "Urdu",

    # Home Science
    "ehe": "Home_Science",

    # Fine Art
    "efa": "Fine_Art",

    # Information Practices
    "eip": "Information_Practices",

    # Computer Science
    "ecs": "Computer_Science",

    # Health & Physical Education
    "ehp": "Health_and_Physical_Education",

    # Biotechnology
    "ebt": "Biotechnology",

    # Sangeet
    "htp": "Sangeet",

    # Knowledge Traditions & Practices of India
    "eks": "Knowledge_Traditions_and_Practices_of_India"
}
# -------------------------------------------------

# Regex: <letter><subjectcode(2-6 letters)><digits(2-4)>
# Example matches: fhks101, FEmh102, x_esc205.pdf
CODE_RE = re.compile(r'([A-Za-z])([A-Za-z]{2,6})(\d{2,4})', flags=re.IGNORECASE)

def find_downloads_path():
    """Auto-detect Downloads (common locations) or fallback to cwd."""
    home = Path.home()
    candidates = [
        home / "Downloads",
        home / "downloads",
        Path.cwd()
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(Path.cwd())

def letter_to_grade(letter: str):
    """Convert letter a..z to grade number: a->1, b->2, ..."""
    letter = letter.lower()
    if 'a' <= letter <= 'z':
        return ord(letter) - ord('a') + 1
    return None

def parse_code_from_string(s: str):
    """
    Parse code in string and map subject code via SUBJECT_MAP.
    Returns (grade_number, subject_name, chapter_number, matched_subject_code) or None.
    """
    m = CODE_RE.search(s)
    if not m:
        return None
    letter, subj_raw, digits = m.group(1), m.group(2).lower(), m.group(3)
    # Try to match the longest subject code substring present in SUBJECT_MAP
    subj_name = None
    subj_code_used = None
    for length in range(len(subj_raw), 1, -1):
        candidate = subj_raw[:length]
        if candidate in SUBJECT_MAP:
            subj_name = SUBJECT_MAP[candidate]
            subj_code_used = candidate
            break
    # fallback direct lookup
    if not subj_name and subj_raw in SUBJECT_MAP:
        subj_name = SUBJECT_MAP[subj_raw]
        subj_code_used = subj_raw

    grade_num = letter_to_grade(letter)
    try:
        dig_val = int(digits)
    except ValueError:
        return None

    # If encoded like 101 -> chapter 1 (common format), else use value if <100
    chap = dig_val % 100 if dig_val >= 100 else dig_val
    if chap == 0:
        chap = dig_val  # fallback

    if grade_num is None or subj_name is None:
        return None

    return (grade_num, subj_name, chap, subj_code_used)

def safe_extract_member(zip_ref: zipfile.ZipFile, member_name: str, dest_path: Path, prefix_name: str = None):
    """
    Safely extract a single member from zip into dest_path.
    Uses member basename and optional prefix to avoid collisions.
    Returns Path of extracted file or None.
    """
    normalized = Path(member_name)
    # skip suspicious entries
    if normalized.is_absolute() or ".." in normalized.parts:
        print(f"⚠️ Skipping suspicious zip entry: {member_name}")
        return None

    member_basename = normalized.name
    if not member_basename:
        # directory entry: create directory structure
        (dest_path / normalized).mkdir(parents=True, exist_ok=True)
        return None

    # prefix to avoid overwrite
    target_name = f"{prefix_name}_{member_basename}" if prefix_name else member_basename
    target = dest_path / target_name
    os.makedirs(dest_path, exist_ok=True)
    try:
        with zip_ref.open(member_name) as source, open(target, "wb") as out_file:
            shutil.copyfileobj(source, out_file)
        return target
    except Exception as e:
        print(f"✖ Failed to extract {member_name}: {e}")
        return None

def process_zip(zip_path: Path, output_root: Path):
    zip_stem = zip_path.stem
    code_in_zipname = parse_code_from_string(zip_stem)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            if code_in_zipname:
                grade_num, subj_name, chap, used_code = code_in_zipname
                grade_label = f"Grade_{grade_num}"
                chapter_label = f"Chapter_{chap}"
                dest_dir = output_root / grade_label / subj_name / chapter_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                print(f"📦 Zip '{zip_path.name}' matched code -> extracting ALL contents into: {dest_dir}")
                for member in members:
                    safe_extract_member(z, member, dest_dir, prefix_name=zip_stem)
                return

            # else inspect members
            any_matched = False
            for member in members:
                base = Path(member).name
                if not base:
                    continue
                parsed = parse_code_from_string(base)
                if parsed:
                    any_matched = True
                    grade_num, subj_name, chap, used_code = parsed
                    grade_label = f"Grade_{grade_num}"
                    chapter_label = f"Chapter_{chap}"
                    dest_dir = output_root / grade_label / subj_name / chapter_label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    safe_extract_member(z, member, dest_dir, prefix_name=zip_stem)
                else:
                    # skip for now; will extract to Unsorted if none matched
                    pass

            if not any_matched:
                unsorted_dir = output_root / "Unsorted" / zip_stem
                unsorted_dir.mkdir(parents=True, exist_ok=True)
                print(f"🔎 No codes found in '{zip_path.name}'. Extracting all to: {unsorted_dir}")
                for member in members:
                    safe_extract_member(z, member, unsorted_dir, prefix_name=zip_stem)
    except zipfile.BadZipFile:
        print(f"✖ Bad zip file: {zip_path.name} (skipping)")
    except Exception as e:
        print(f"✖ Error processing {zip_path.name}: {e}")

def main():
    global SOURCE_DIR
    if not SOURCE_DIR:
        SOURCE_DIR = find_downloads_path()
        print(f"ℹ️ SOURCE_DIR not set. Auto-detected: {SOURCE_DIR}")

    src = Path(SOURCE_DIR)
    if not src.exists():
        print(f"✖ SOURCE_DIR does not exist: {src}")
        return

    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)

    zip_files = sorted([p for p in src.iterdir() if p.suffix.lower() == ".zip"])
    if not zip_files:
        print(f"⚠️ No .zip files found in {src.resolve()}")
        return

    for zf in zip_files:
        print(f"\nProcessing: {zf.name}")
        process_zip(zf, out)

    print("\n✅ Extraction complete. Inspect the 'NCERT_books' directory.")

if __name__ == "__main__":
    main()