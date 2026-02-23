import json
import time
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

from Chunking.file_readers import load_text_from_file
from Chunking.process_paragraph import process_paragraph
from Chunking.registry import load_registry, save_registry


SUPPORTED_EXT = [".pdf", ".docx", ".txt"]


def load_json_or_empty(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def main(
    use_multiprocessing=True,
    max_processes=4
):
    root = Path(__file__).parent.parent

    base_docs = root / "base_documents"
    new_docs = root / "nouveaux_documents"
    processed_docs = root / "processed_documents"

    paragraphs_path = root / "paragraphs.json"
    chunks_path = root / "chunks.json"
    registry_path = root / "processed_files.json"

    for d in [base_docs, new_docs, processed_docs]:
        d.mkdir(exist_ok=True)

    processed_registry = load_registry(registry_path)

    new_files = [
        f for f in new_docs.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
        and f.name not in processed_registry
    ]

    if not new_files:
        print("ℹ️ Aucun nouveau document à chunker.")
        return

    print(f"📂 {len(new_files)} nouveau(x) document(s) détecté(s).")

    paragraphs = []
    for file in new_files:
        pages = load_text_from_file(file)
        for page in pages:
            for p in page["text"].split("\n\n"):
                if p.strip():
                    paragraphs.append((p.strip(), file.name, page["page_number"]))

    print(f"📄 {len(paragraphs)} paragraphes à traiter.")

    start = time.time()

    if use_multiprocessing:
        n_proc = min(max_processes, cpu_count())
        with Pool(n_proc) as pool:
            results = pool.map(process_paragraph, paragraphs)
    else:
        results = [process_paragraph(p) for p in paragraphs]

    existing_paragraphs = load_json_or_empty(paragraphs_path)
    existing_chunks = load_json_or_empty(chunks_path)

    new_paragraphs = [r["paragraph"] for r in results]
    new_chunks = [c for r in results for c in r["chunks"]]

    paragraphs_path.write_text(
        json.dumps(existing_paragraphs + new_paragraphs, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    chunks_path.write_text(
        json.dumps(existing_chunks + new_chunks, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    for f in new_files:
        processed_registry.add(f.name)
        shutil.move(f, base_docs / f.name)
        shutil.copy(base_docs / f.name, processed_docs / f.name)

    save_registry(registry_path, processed_registry)

    print(f"✅ Chunking terminé en {time.time() - start:.2f}s")
    print(f"🧩 Nouveaux chunks : {len(new_chunks)}")


if __name__ == "__main__":
    main()
