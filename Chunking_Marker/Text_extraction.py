"""
Text_extraction.py : Version "RAG Colab Speed" (NVIDIA GPU).
- HTML : BeautifulSoup
- DOCX : Unstructured
- EXCEL : Pandas
- PDF : Marker (Batch x4 + CUDA)
- AUDIO : Whisper (FP16 enabled)
"""
import logging
import warnings
import os
import gc
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF

# --- Imports Dynamiques ---
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from unstructured.partition.docx import partition_docx
except ImportError:
    partition_docx = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# --- Device & Optimisation GPU ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Optimisation pour NVIDIA (T4/L4/A100)
    torch.backends.cudnn.benchmark = True 
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 Mode NVIDIA (CUDA) activé sur : {gpu_name}")
else:
    DEVICE = "cpu"
    print("🐢 Mode CPU (Pas de GPU détecté, vérifiez 'Exécution > Modifier le type d'exécution')")

_MARKER_CONVERTER = None
_WHISPER_MODEL = None

def clean_memory():
    """Nettoyage VRAM CUDA"""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# --- 1. HTML (BeautifulSoup) ---
def extract_html(file_path: Path) -> List[Dict[str, Any]]:
    if BeautifulSoup is None: return []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')

            # Sauvetage Maths
            for script in soup.find_all("script", type="math/tex"):
                tex = script.get_text()
                if "mode=display" in str(script):
                    new_tag = soup.new_tag("p")
                    new_tag.string = f" $$ {tex} $$ "
                    script.replace_with(new_tag)
                else:
                    script.replace_with(f" $ {tex} $ ")

            # Sauvetage Images Alt
            for img in soup.find_all("img"):
                alt = img.get("alt", "").strip()
                if alt: img.replace_with(f" [IMAGE: {alt}] ")
                else: img.decompose()

            # Nettoyage
            for garbage in soup(["script", "style", "header", "footer", "nav", "meta", "noscript", "iframe"]):
                garbage.extract()
            
            text = soup.get_text(separator='\n').strip()
            clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(clean_lines)
        
        if clean_text:
            return [{"text": clean_text, "category": "HTML", "metadata": {"filename": file_path.name}}]
    except Exception as e:
        logger.error(f"❌ Erreur HTML : {e}")
    return []

# --- 2. DOCX (Unstructured) ---
def extract_docx(file_path: Path) -> List[Dict[str, Any]]:
    if partition_docx is None: return []
    try:
        elements = partition_docx(filename=str(file_path))
        formatted_text = []
        for el in elements:
            text = str(el).strip()
            if not text: continue
            if el.category == "Title": formatted_text.append(f"\n## {text}")
            elif el.category == "ListItem": formatted_text.append(f"- {text}")
            elif el.category == "Table": formatted_text.append(f"\n{text}\n")
            elif el.category not in ["Header", "Footer"]: formatted_text.append(text)

        final_text = "\n\n".join(formatted_text)
        if final_text:
            return [{"text": final_text, "category": "DOCX", "metadata": {"filename": file_path.name}}]
    except Exception as e:
        logger.error(f"❌ Erreur DOCX : {e}")
    return []

# --- 3. EXCEL (Pandas) ---
def extract_excel(file_path: Path) -> List[Dict[str, Any]]:
    if pd is None: return []
    elements = []
    try:
        excel = pd.ExcelFile(file_path)
        for sheet in excel.sheet_names:
            df = pd.read_excel(excel, sheet_name=sheet)
            df = df.dropna(how='all').dropna(axis=1, how='all')
            if not df.empty:
                text = f"## Feuille: {sheet}\n\n" + df.to_markdown(index=False)
                elements.append({
                    "text": text,
                    "category": "Spreadsheet",
                    "metadata": {"filename": file_path.name, "sheet_name": sheet}
                })
    except Exception: pass
    return elements

# --- 4. PDF (Marker - Optimisé CUDA) ---
def get_marker_converter():
    global _MARKER_CONVERTER
    if _MARKER_CONVERTER is None:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        
        # Configuration Colab / GPU
        # On augmente le batch size car on a plus de VRAM (12-16Go vs 8Go)
        gpu_config = {
            "batch_multiplier": 4,   # X4 Vitesse par rapport à la version Mac
            "ocr_all_pages": False,  # False = plus rapide (utilise texte natif si dispo)
            "disable_image_extraction": True,
            "languages": "fr,en"
        }
        
        # Chargement du modèle sur CUDA explicitement
        model_dict = create_model_dict()
        # Marker gère le device via torch, mais on s'assure que c'est propre
        _MARKER_CONVERTER = PdfConverter(artifact_dict=model_dict, config=gpu_config)
    return _MARKER_CONVERTER

def extract_pdf(file_path: Path) -> List[Dict[str, Any]]:
    clean_memory()
    try:
        try:
            with fitz.open(str(file_path)) as doc: num_pages = len(doc)
        except: num_pages = 1
        
        converter = get_marker_converter()
        # Marker utilise automatiquement CUDA si dispo et torch configuré
        rendered = converter(str(file_path))
        from marker.output import text_from_rendered
        text, _, _ = text_from_rendered(rendered)
        
        if text:
            return [{"text": text, "category": "PDF", "metadata": {"filename": file_path.name, "pages": num_pages}}]
    except Exception as e:
        logger.error(f"❌ Erreur PDF: {e}")
        clean_memory()
    return []

# --- 5. AUDIO (Whisper - Optimisé FP16) ---
def extract_audio(file_path: Path) -> List[Dict[str, Any]]:
    global _WHISPER_MODEL
    clean_memory()
    try:
        if _WHISPER_MODEL is None:
            import whisper
            # On peut utiliser "small" ou "medium" sur Colab car on a de la place
            # Mais "base" reste le plus rapide.
            print("🎙️ Chargement Whisper sur GPU...")
            _WHISPER_MODEL = whisper.load_model("base", device=DEVICE)
        
        # IMPORTANT : fp16=True est beaucoup plus rapide sur NVIDIA
        result = _WHISPER_MODEL.transcribe(str(file_path), fp16=True)
        text = result.get("text", "").strip()
        if text:
            return [{"text": text, "category": "Audio", "metadata": {"filename": file_path.name}}]
    except Exception as e:
        logger.error(f"❌ Erreur Audio: {e}")
    clean_memory()
    return []

# --- ROUTER ---
def extract(file_path: str | Path) -> Tuple[List[Dict], Dict[str, int]]:
    path = Path(file_path)
    if not path.exists(): return [], {}
    
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        return extract_pdf(path), {"PDF": 1}
    elif ext in {".mp3", ".wav", ".m4a"}:
        return extract_audio(path), {"Audio": 1}
    elif ext in {".html", ".htm"}:
        return extract_html(path), {"HTML": 1}
    elif ext == ".docx":
        return extract_docx(path), {"DOCX": 1}
    elif ext in {".xlsx", ".xls"}:
        return extract_excel(path), {"Excel": 1}
    elif ext in {".txt", ".md"}:
        with open(path, 'r', encoding='utf-8') as f: 
            return [{"text": f.read(), "category": "Text", "metadata": {"filename": path.name}}], {"Text": 1}
            
    return [], {}