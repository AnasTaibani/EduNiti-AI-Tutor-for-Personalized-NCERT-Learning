import os
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Input and output directories
RAW_DIR = Path(r"C:\Users\Lenovo\OneDrive\Desktop\eduniti-majorProject\extractthis")
TEXT_OUT_DIR = Path("NCERT_text")
TEXT_OUT_DIR.mkdir(exist_ok=True, parents=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a single PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text

def read_text_file(path: Path) -> str:
    """Reads a text or CSV file and returns text."""
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            return "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
        else:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"⚠️ Skipped {path}: {e}")
        return ""

def extract_text_from_file(file_path: Path) -> str:
    """Detect file type and extract text."""
    if file_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() in [".txt", ".csv"]:
        return read_text_file(file_path)
    else:
        return ""

def process_all_files(raw_dir: Path, out_dir: Path):
    for grade_dir in tqdm(list(raw_dir.iterdir()), desc="Grades"):
        if not grade_dir.is_dir():
            continue
        for subj_dir in grade_dir.iterdir():
            if not subj_dir.is_dir():
                continue
            for chap_dir in subj_dir.iterdir():
                if not chap_dir.is_dir():
                    continue

                output_folder = out_dir / grade_dir.name / subj_dir.name
                output_folder.mkdir(parents=True, exist_ok=True)
                out_path = output_folder / f"{chap_dir.name}.txt"

                chapter_text = ""
                for file in chap_dir.glob("*"):
                    if file.is_file():
                        extracted = extract_text_from_file(file)
                        chapter_text += extracted + "\n\n"

                if len(chapter_text.strip()) > 50:
                    out_path.write_text(chapter_text.strip(), encoding="utf-8")
                    print(f"✅ Saved {out_path}")
                else:
                    print(f"⚠️ No valid text in {chap_dir}")

if __name__ == "__main__":
    process_all_files(RAW_DIR, TEXT_OUT_DIR)
    print("\n🎯 Text extraction completed. Check NCERT_text folder.")
