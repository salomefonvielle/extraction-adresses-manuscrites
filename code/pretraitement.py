from __future__ import annotations
import pandas as pd, csv
from typing import Optional
import pathlib
from pathlib import Path

PDF_DIREC=pathlib.Path("/home/sfonvielle-stagiai01/projets/new_extraction_adresse/local/datas/inputs/PDF")

#Extraction du chemin du pdf pour une ID_Atlas 

def _compute_pdf_path(doc_id: Optional[str]) -> str:
    """
    Renvoie le chemin complet du PDF correspondant à `doc_id`
    si le fichier <quelque_chose>_<doc_id>.pdf existe dans PDF_DIR,
    sinon une chaîne vide.
    """
    PDF_EXT=".pdf"
    PDF_DIR=PDF_DIREC
    if doc_id is None or pd.isna(doc_id) or str(doc_id).strip() == "":
        return ""

    cleaned_id = str(doc_id).strip().lower()
    # On cherche tout fichier qui se termine par <cleaned_id>.pdf
    candidate = None
    for p in PDF_DIR.rglob(f"*{cleaned_id}{PDF_EXT}"):
        # On prend le premier match trouvé
        candidate = p
        break

    if candidate:
        print(f"[DEBUG] PDF trouvé pour {doc_id} -> {candidate}")
        return str(candidate)
    else:
        print(f"[DEBUG] PDF manquant pour {doc_id}")
        return False 


#Ajout au fichier csv 

def enrich_csv_with_pdf_path(CSV_IN,CSV_OUT) -> None:
    """
    Charge CSV_IN, ajoute la colonne `pdf_path` et l’enregistre
    dans CSV_OUT.
    """
    df = pd.read_csv(CSV_IN, dtype=str)

    if "doc_idatlas" not in df.columns:
        raise KeyError("La colonne 'doc_idatlas' est manquante dans le CSV.")

    # Calculer le chemin de chaque PDF
    df["pdf_path"] = df["doc_idatlas"].apply(_compute_pdf_path)

    # Statistiques de débogage
    non_empty = df["pdf_path"].str.strip() != ""
    print(f"PDF trouvés : {non_empty.sum()} / {len(df)} ({non_empty.mean()*100:.1f} %) [sous forme de chaînes non‑vide]")

    # Écrire le CSV enrichi
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"✅ CSV enrichi écrit dans : {CSV_OUT.resolve()}")


#mise en forme des csv 


def detect_csv_dialect(file_path: Path, sample_size: int = 2048) -> dict:
    """Retourne un dictionnaire d'options compatible avec ``pandas.read_csv``."""
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(sample_size)

    try:
        dialect = csv.Sniffer().sniff(sample)
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:  # fallback très simple
        dialect = csv.get_dialect("excel")
        has_header = True

    return {
        "delimiter": dialect.delimiter,
        "quotechar": dialect.quotechar,
        "quoting": dialect.quoting,
        "escapechar": dialect.escapechar,
        "skipinitialspace": True,
        "engine": "python",          # nécessaire pour les cas non‑standard
        "dtype": str,                # garder tout en texte
        "header": 0 if has_header else None,
    }



def prepare_csv(df_ocr2) : 
    # Nettoyage des noms de colonnes
    # Supprime une colonne “Unnamed: 0” éventuelle (index ajouté par pandas)
    if "Unnamed: 0" in df_ocr2.columns:
        df_ocr2 = df_ocr2.drop(columns=["Unnamed: 0"])
    # Normalise: tout en minuscules, sans espaces
    df_ocr2 = df_ocr2.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    
    # Normalisation du(s) champ(s) de jointure (case‑insensitive)--
    key_col = "doc_idatlas" 
    if key_col not in df_ocr2.columns:
        raise KeyError(f"Colonne {key_col!r} introuvable dans le CSV source")
    df_ocr2[key_col] = df_ocr2[key_col].str.strip().str.lower()   # <-- important

    return df_ocr2
    


