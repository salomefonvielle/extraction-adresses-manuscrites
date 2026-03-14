# ------------------------------------------------------------
# Imports (dédupliqués & ordonnés)
# ------------------------------------------------------------
import os
import time
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Union, List, Dict

import pandas as pd
import requests
from PIL import Image                     # Pillow – manipulation d'images
from pdf2image import convert_from_bytes  # PDF → images
from PyPDF2 import PdfMerger


# ------------------------------------------------------------
# Configuration du proxy (déjà présent dans votre projet)
# ------------------------------------------------------------
os.environ["no_proxy"] = (
    "100.70.1.199,forge.dgfip.finances.rie.gouv.fr,"
    "pia-exp-back.dev.dgfip,pia-exp-front.dev.dgfip,"
    "10.156.253.10,huggingface.co,10.156.253.13,"
    "10.156.226.144,10.156.226.145"
)

# ------------------------------------------------------------
# Chargement du .env (une seule fois)
# ------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(override=True)

# ------------------------------------------------------------
# Identifiants du serveur VLM
# ------------------------------------------------------------
API_IP = os.getenv("PIA_IP")
API_PORT = os.getenv("PIA_PORT")
API_KEY = os.getenv("PIA_API_KEY")          # clé secrète du VLM
timeout = 90                                 # délai d'attente (secondes)

# ------------------------------------------------------------
# Constantes globales utilisées dans tout le pipeline
# ------------------------------------------------------------
DPI_PDF = 200            # DPI utilisé lors de la conversion PDF → images
MAX_PAGES = 16           # Nombre maximum de pages analysées par document
TESS_CONFIG = "--psm 11"   # même valeur que dans le module Tesseract
TESS_LANG = "fra"          # même langue que dans le module Tesseract


# ----------------------------------------------------------------------
def run_vlm_on_image(
    img: Image.Image,
    doc_idatlas: str,
    page_number: int = 0,
    max_page_number: int = 0,
) -> Dict:
    """
    Envoie une image à un VLM et retourne le JSON décodé **avec le temps
    d’exécution**. Le dictionnaire retourné possède les mêmes clés que le
    résultat de ``ocr_tesseract`` afin d’obtenir une structure identique.
    """
    start = time.time()

    # -------------------------------------------------
    # Prompt (unchanged)
    # -------------------------------------------------
    prompt = (
        "qwenvl markdown\n"
        "Important:\n"
        "1. Preserve the document structure (headers with #, ##, paragraphs).\n"
        "2. If possible, output tables in Markdown format, not LaTeX.\n"
        "3. Do not crop text on the edges.\n"
        "4. Never return blocks of code. Always avoid '```' patterns.\n"
        "5. Never return the texts for logos nor headers nor footers nor page numbers.\n"
    )

    # -------------------------------------------------
    # Encode image – use JPEG (mime‑type image/jpeg)
    # -------------------------------------------------
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # -------------------------------------------------
    # Build payload (unchanged)
    # -------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    url = f"http://{API_IP}:{API_PORT}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": os.getenv("PIA_MODEL", "Qwen3-VL-32B-Instruct-FP8"),
        "messages": messages,
        "temperature": 0,
        # NE PAS forcer json_object : Qwen-VL choisit lui-même la clé JSON
        # ce qui rendait le texte introuvable (clé != "text" / "content")
    }

    # -------------------------------------------------
    # Call VLM – use the global timeout variable
    # -------------------------------------------------
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    # 👇 DEBUG TEMPORAIRE – à supprimer après diagnostic
    raw_response = resp.json()
    raw_text = raw_response["choices"][0]["message"]["content"] or ""

    # Le contenu est maintenant du texte brut (Markdown) directement
    raw_text = resp.json()["choices"][0]["message"]["content"] or ""
    if isinstance(raw_text, (list, tuple)):
        raw_text = " ".join(map(str, raw_text))

    # Compatibilité : on reconstruit un dict minimal comme avant
    resultat_json = {"text": raw_text}
    resultat_json["text"] = raw_text  # garantit un string (déjà le cas)

    # -------------------------------------------------
    # Ajout des métadonnées attendues
    # -------------------------------------------------
    elapsed = time.time() - start
    resultat_json.update(
        {
            "doc_idatlas": doc_idatlas,
            "page_number": page_number,
            "max_page_number": max_page_number,
            "dpi_pdf": DPI_PDF,
            "vlm_time": elapsed,
            "vlm_config": TESS_CONFIG,
            "vlm_lang": TESS_LANG,
        }
    )
    return resultat_json


# ----------------------------------------------------------------------
def process_vlm_csv_and_generate_pdf(
    csv_file_path: Union[str, Path],
    output_pdf_path: Union[str, Path],
    export_csv_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Pipeline VLM qui produit exactement le même schéma de DataFrame que
    ``process_tesseract_csv_and_generate_pdf`` : une ligne **par page** avec les
    colonnes
    ``doc_idatlas, text, page_number, max_page_number, dpi_pdf,
    tess_time, tess_config, tess_lang``.
    """
    # -------------------------------------------------
    # Normalisation des chemins
    # -------------------------------------------------
    csv_path = Path(csv_file_path).expanduser().resolve()
    output_path = Path(output_pdf_path).expanduser().resolve()
    ocr_output_csv = output_path.with_name(output_path.stem + "_ocr.csv")

    # -------------------------------------------------
    # Lecture du CSV manifest
    # -------------------------------------------------
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV introuvable : {csv_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Erreur lecture CSV '{csv_path}' : {exc}") from exc

    required_cols = {"doc_idatlas", "pdf_path"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    # -------------------------------------------------
    # Container for OCR results
    # -------------------------------------------------
    ocr_frames: List[pd.DataFrame] = []

    # -------------------------------------------------
    # ----------- OCR PART – PAGE‑BY‑PAGE -----------------
    # -------------------------------------------------
    for _, row in df.iterrows():
        doc_id = str(row["doc_idatlas"])
        pdf_path = Path(row["pdf_path"]).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        try:
            # Load PDF & convert pages (max MAX_PAGES)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pages = convert_from_bytes(pdf_bytes, dpi=DPI_PDF)[:MAX_PAGES]
            max_page_idx = len(pages) - 1

            # ---- Collect OCR results for all pages of this PDF ----
            for page_idx, img in enumerate(pages):
                rec = run_vlm_on_image(
                    img=img,
                    doc_idatlas=doc_id,
                    page_number=page_idx,
                    max_page_number=max_page_idx,
                )
                cur_doc = [
                    rec["doc_idatlas"],      # doc_idatlas
                    rec["text"],             # texte de la page
                    rec["page_number"],      # page_number
                    rec["max_page_number"],  # max_page_number
                    rec["dpi_pdf"],          # DPI (200)
                    rec["vlm_time"],         # temps d'exécution → tess_time
                    rec["vlm_config"],       # config → tess_config
                    rec["vlm_lang"],         # langue → tess_lang
                ]

                df_ocr = pd.DataFrame(
                    [cur_doc],   # one row → list of rows
                    columns=[
                        "doc_idatlas",
                        "text",
                        "page_number",
                        "max_page_number",
                        "dpi_pdf",
                        "vlm_time",   # keep expected column name
                        "vlm_config",
                        "vlm_lang",
                    ],
                )
                ocr_frames.append(df_ocr)

        except Exception as e:  # pragma: no cover
            # Keep the loop alive – we still want the merged PDF even if OCR fails
            print(f"[WARN] OCR VLM échoué pour {pdf_path} – {e}")

    # -------------------------------------------------
    # ----------- PDF MERGE PART – AFTER OCR -------------
    # -------------------------------------------------
    with PdfMerger() as merger:
        for _, row in df.iterrows():
            pdf_path = Path(row["pdf_path"]).expanduser().resolve()
            merger.append(str(pdf_path))

        # write the combined PDF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fout:
            merger.write(fout)

    # -------------------------------------------------
    # Concaténation & écriture CSV
    # -------------------------------------------------
    if ocr_frames:
        all_ocr_df = pd.concat(ocr_frames, ignore_index=True)
        all_ocr_df.to_csv(ocr_output_csv, index=False)
        print(f"[INFO] CSV OCR combiné écrit ici : {ocr_output_csv}")

        if export_csv_path is not None:
            export_path = Path(export_csv_path).expanduser().resolve()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            all_ocr_df.to_csv(export_path, index=False)
            print(f"[INFO] CSV OCR final également exporté vers : {export_path}")
    else:
        all_ocr_df = pd.DataFrame()
        print("[INFO] Aucun résultat OCR collecté ; aucun CSV généré.")

    return all_ocr_df