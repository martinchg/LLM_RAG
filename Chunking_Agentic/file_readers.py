import fitz  # PyMuPDF
from docx import Document
from pathlib import Path


def read_pdf(path: Path):
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page_number": i, "text": text})
    return pages


def read_docx(path: Path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page_number": 1, "text": text.strip()}]


def read_txt(path: Path):
    return [{
        "page_number": 1,
        "text": path.read_text(encoding="utf-8", errors="ignore").strip()
    }]


def load_text_from_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        print(f"📘 Lecture PDF : {path.name}")
        return read_pdf(path)
    if ext == ".docx":
        print(f"📄 Lecture DOCX : {path.name}")
        return read_docx(path)
    if ext == ".txt":
        print(f"📜 Lecture TXT : {path.name}")
        return read_txt(path)
    raise ValueError(f"Format non supporté : {ext}")
