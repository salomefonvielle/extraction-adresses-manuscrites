from __future__ import annotations
import pandas as pd, csv
from typing import Optional
import pathlib
from pathlib import Path


def _compute_pdf_path(doc_id: Optional[str], pdf_dir: Path) -> str:
    """
    Renvoie le chemin complet du PDF correspondant à `doc_id`
    si le fichier <quelque_chose>_<doc_id>.pdf existe dans pdf_dir,
    sinon une chaîne vide.
    """
    PDF_EXT = ".pdf"
    if doc_id is None or pd.isna(doc_id) or str(doc_id).strip() == "":
        return ""

    cleaned_id = str(doc_id).strip().lower()
    for p in pdf_dir.rglob(f"*{cleaned_id}{PDF_EXT}"):
        return str(p)

    return ""


def enrich_csv_with_pdf_path(CSV_IN, CSV_OUT, pdf_dir: Path) -> None:
    """
    Charge CSV_IN, ajoute la colonne `pdf_path` et l'enregistre dans CSV_OUT.
    Le répertoire PDF est passé en paramètre (lu depuis config.yaml).
    """
    df = pd.read_csv(CSV_IN, dtype=str)

    if "doc_idatlas" not in df.columns:
        raise KeyError("La colonne 'doc_idatlas' est manquante dans le CSV.")

    df["pdf_path"] = df["doc_idatlas"].apply(lambda doc_id: _compute_pdf_path(doc_id, pdf_dir))

    non_empty = df["pdf_path"].str.strip() != ""
    print(f"PDF trouvés : {non_empty.sum()} / {len(df)} ({non_empty.mean()*100:.1f} %)")

    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"✅ CSV enrichi écrit dans : {Path(CSV_OUT).resolve()}")


def detect_csv_dialect(file_path: Path, sample_size: int = 2048) -> dict:
    """Retourne un dictionnaire d'options compatible avec ``pandas.read_csv``."""
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(sample_size)

    try:
        dialect = csv.Sniffer().sniff(sample)
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
        has_header = True

    return {
        "delimiter": dialect.delimiter,
        "quotechar": dialect.quotechar,
        "quoting": dialect.quoting,
        "escapechar": dialect.escapechar,
        "skipinitialspace": True,
        "engine": "python",
        "dtype": str,
        "header": 0 if has_header else None,
    }


def prepare_csv(df_ocr2):
    if "Unnamed: 0" in df_ocr2.columns:
        df_ocr2 = df_ocr2.drop(columns=["Unnamed: 0"])
    df_ocr2 = df_ocr2.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))

    key_col = "doc_idatlas"
    if key_col not in df_ocr2.columns:
        raise KeyError(f"Colonne {key_col!r} introuvable dans le CSV source")
    df_ocr2[key_col] = df_ocr2[key_col].str.strip().str.lower()

    return df_ocr2
