import re
import logging
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

_GLOBAL_PARENT_COUNTER = 1

# --- CONFIGURATION TOKENIZER ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1" 

try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
except Exception as e:
    logging.warning(f"Warning Tokenizer: {e}")
    TOKENIZER = None

def count_tokens(text: str) -> int:
    if not text: return 0
    if TOKENIZER:
        return len(TOKENIZER.encode(text, add_special_tokens=False))
    else:
        return int(len(text.split()) * 1.3)

def clean_markdown_images(text: str) -> str:
    if not text: return ""
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    return re.sub(r'[ \t]+', ' ', text).strip()

def is_noise(text: str) -> bool:
    if not text: return True
    if len(text.strip()) < 5: return True
    return False

# --- 1. Split par Headers + SAFETY CHECK (Correction ici) ---

def split_markdown_by_headers(markdown_text: str, metadata: Dict, max_parent_tokens: int = 2000) -> List[Dict]:
    """
    Découpe par headers, PUIS vérifie si les blocs ne sont pas trop gros.
    Si un bloc est trop gros (ex: pas de headers), il est redécoupé.
    """
    lines = markdown_text.split('\n')
    initial_parents = [] # Liste temporaire des sections par headers
    current_buffer = []
    current_title = "Introduction" 
    header_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    # --- Phase A : Découpe Sémantique (Headers) ---
    for line in lines:
        match = header_pattern.match(line)
        if match:
            if current_buffer:
                raw_content = "\n".join(current_buffer)
                clean_content = clean_markdown_images(raw_content)
                if not is_noise(clean_content):
                    initial_parents.append({
                        "text": clean_content,
                        "title": current_title,
                        "metadata": metadata
                    })
            current_buffer = [line] 
            current_title = match.group(2).strip().replace('*', '').replace('_', '')
        else:
            current_buffer.append(line)
    
    # Dernier buffer
    if current_buffer:
        raw_content = "\n".join(current_buffer)
        clean_content = clean_markdown_images(raw_content)
        if not is_noise(clean_content):
            initial_parents.append({
                "text": clean_content,
                "title": current_title,
                "metadata": metadata
            })

    # --- Phase B : Safety Splitting (Taille Max) ---
    final_parents = []
    
    # On prépare un splitter pour les cas où le parent est trop gros
    # On met un overlap pour les parents pour ne pas perdre le fil entre deux gros blocs coupés arbitrairement
    safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_parent_tokens,
        chunk_overlap=200,  # Overlap de sécurité pour les parents
        length_function=count_tokens,
        separators=["\n\n", "\n", ".", " ", ""],
        strip_whitespace=True
    )

    for parent in initial_parents:
        text = parent['text']
        token_count = count_tokens(text)

        if token_count <= max_parent_tokens:
            # Cas idéal : le header a suffi à faire une taille correcte
            parent["type"] = "Section"
            final_parents.append(parent)
        else:
            # Cas problématique : Pas de header ou section énorme
            # On découpe ce "gros parent" en plusieurs "sous-parents"
            logging.info(f"Section '{parent['title']}' trop grosse ({token_count} tokens). Découpage forcé.")
            sub_chunks = safety_splitter.split_text(text)
            
            for i, sub_text in enumerate(sub_chunks):
                final_parents.append({
                    "text": sub_text,
                    "type": "Section-Split", # Pour savoir que c'est un découpage mécanique
                    "title": f"{parent['title']} (Part {i+1})", # On garde le titre en ajoutant un suffixe
                    "metadata": parent['metadata']
                })

    return final_parents

# --- 2. Chunking Enfants (Texte et Tableaux) ---

def create_text_child_chunks(text: str, chunk_size_tokens: int, overlap_tokens: int) -> List[str]:
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens, 
        chunk_overlap=overlap_tokens,
        length_function=count_tokens,
        separators=separators,
        keep_separator=True,
        strip_whitespace=True
    )
    return text_splitter.split_text(text)

def chunk_markdown_table(table_text: str, max_tokens: int) -> List[str]:
    # (Votre logique de tableau reste inchangée ici, elle est bonne)
    table_text = re.sub(r'<br\s*/?>', ' ', table_text)
    table_text = re.sub(r'•\s*\n\s*', '• ', table_text)
    table_text = re.sub(r'[ \t]+', ' ', table_text)
    if count_tokens(table_text) <= max_tokens: return [table_text]

    lines = table_text.strip().split('\n')
    separator_idx = -1
    for i, line in enumerate(lines[:10]):
        if re.search(r'\|[\s-]*:?[\s-]*\|', line):
            separator_idx = i; break
    if separator_idx == -1: return create_text_child_chunks(table_text, max_tokens, 50)

    header_block = "\n".join(lines[:separator_idx+1])
    header_tokens = count_tokens(header_block)
    chunks = []; current_chunk_rows = []; current_tokens = header_tokens
    for row in lines[separator_idx+1:]:
        row = row.strip(); 
        if not row: continue
        row_tokens = count_tokens(row)
        if current_tokens + row_tokens > (max_tokens - 10) and current_chunk_rows:
            chunks.append(header_block + "\n" + "\n".join(current_chunk_rows))
            current_chunk_rows = [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_chunk_rows.append(row)
            current_tokens += row_tokens + 1
    if current_chunk_rows: chunks.append(header_block + "\n" + "\n".join(current_chunk_rows))
    return chunks

# --- 3. Main Modifié ---

def create_parent_child_chunks(elements: List[Dict], 
                               parent_size_tokens, 
                               child_size_tokens) -> Tuple[List[Dict], List[Dict]]:
    global _GLOBAL_PARENT_COUNTER
    
    children_export = []
    parents_export = []
    raw_parents_buffer = []
    
    # Étape 1 : Parents (avec limite de taille forcée)
    for el in elements:
        text = el.get('text', '')
        metadata = el.get('metadata', {})
        text = clean_markdown_images(text)
        if is_noise(text): continue
        
        # On passe parent_size_tokens ici
        raw_parents_buffer.extend(split_markdown_by_headers(text, metadata, max_parent_tokens=parent_size_tokens))

    # Étape 2 : Enfants
    base_id = _GLOBAL_PARENT_COUNTER
    
    for p_idx, parent in enumerate(raw_parents_buffer):
        parent_text = parent['text']
        global_parent_id = f"parent_{base_id + p_idx}"
        parent_meta = parent.get('metadata', {})

        parents_export.append({
            "id": global_parent_id,
            "text": parent_text,
            "title": parent.get('title'), # Ajout utile
            "type": parent.get('type', 'Section'),
            "filename": parent_meta.get('filename', 'unknown')
        })

        is_table = parent_text.count('|') > parent_text.count('\n') and "---" in parent_text
        
        # --- NOUVEAUTÉ : On ignore les Tables des Matières ---
        # On s'assure que le titre n'est pas None (sinon ça plante)
        titre = parent.get('title') or ''
        is_toc = "Contents" in titre or "Table of Contents" in parent_text
        
        if is_table and not is_toc:
            child_texts = chunk_markdown_table(parent_text, child_size_tokens)
        else:
            child_texts = create_text_child_chunks(parent_text, child_size_tokens, 50)
        for c_idx, child_text in enumerate(child_texts):
            if is_noise(child_text): continue
            
            filename = parent_meta.get('filename', 'unknown')
            page_info = parent_meta.get('page_label') or parent_meta.get('page_start')

            children_export.append({
                "id": f"{global_parent_id}_child_{c_idx}",
                "parent_id": global_parent_id,
                "text": child_text,
                "page": page_info,
                "filename": filename
            })

    _GLOBAL_PARENT_COUNTER = base_id + len(raw_parents_buffer)
    
    return parents_export, children_export