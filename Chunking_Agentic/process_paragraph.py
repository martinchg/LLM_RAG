from uuid import uuid4
from Chunking_Agentic.agentic_chunker_ollama import AgenticChunker


def process_paragraph(args):
    paragraph_text, file_name, page_number = args
    paragraph_id = str(uuid4())[:8]

    chunker = AgenticChunker(model="mistral")

    chunks = chunker.chunk_paragraph(
        paragraph_text,
        document_name=file_name,
        page_number=page_number,
        parent_id=paragraph_id
    )

    return {
        "paragraph": {
            "paragraph_id": paragraph_id,
            "document_name": file_name,
            "page_number": page_number,
            "text": paragraph_text
        },
        "chunks": chunks
    }
