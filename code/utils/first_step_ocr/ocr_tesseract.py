# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
# Utility to clean raw OCR output (keeps only alphabetic characters, etc.)
from code.utils.second_step_scores.scores_calcul import clean_text
import os
import pandas as pd
import pytesseract as tess
from pdf2image import convert_from_bytes
import time
import requests
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
from PyPDF2 import PdfMerger
from typing import Union, List

# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------
# DPI to render PDF pages before feeding them to Tesseract
DPI_PDF = 200

# Limit the number of pages processed per PDF for speed/size constraints
MAX_PAGES = 16

# Tesseract page‑segmentation mode (15 = treat the image as a single block of text)
TESS_CONFIG = "--psm 11"

# Language used by Tesseract for OCR recognition
TESS_LANG = "fra"

# ------------------------------------------------------------------
# Environment & HTTP session setup
# ------------------------------------------------------------------
# Restrict OpenMP to a single thread – improves stability on multithreaded machines
os.environ["OMP_THREAD_LIMIT"] = "1"

# Configure a resilient session for HTTP requests (retries on failure)
SESS = requests.Session()
retry = Retry(connect=5, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
SESS.mount('http://', adapter)
SESS.mount('https://', adapter)

# ------------------------------------------------------------------
# Transforms a PDF file into a CSV with one line per page
# including : document id, cleaned OCR text and page information
# ------------------------------------------------------------------
def ocr_tesseract_pdf(chemin: str, idatlas: str) -> List[list]:
    """
    Perform OCR on a PDF file using Tesseract.

    Parameters
    ----------
    chemin : str
        Path to the PDF file to be OCR‑processed.
    idatlas : str
        Document identifier that will be stored in each result row.

    Returns
    -------
    List[list]
        One inner list per processed page containing:
        [doc_id, cleaned_text, page_index, last_page_index,
         DPI, elapsed_time, tess_config, tess_lang]
    """
    # ------------------------------------------------------------------
    # 1️⃣ Load the PDF into a byte buffer and convert to images
    # ------------------------------------------------------------------
    with open(chemin, "rb") as f:
        pdf = f.read()
    pages = convert_from_bytes(pdf, dpi=DPI_PDF)

    # Trim the list of pages if it exceeds MAX_PAGES
    imgs = pages[:MAX_PAGES]
    processed_pages = []

    # ------------------------------------------------------------------
    # 2️⃣ Loop over each page image, run Tesseract and collect results
    # ------------------------------------------------------------------
    for k, img in enumerate(imgs):
        start = time.time()

        # Run Tesseract on the current page image
        raw_text = tess.image_to_string(img, config=TESS_CONFIG, lang=TESS_LANG)

        tess_time = time.time() - start

        # Clean the raw OCR output (lowercase by default)
        cleaned = clean_text(raw_text, upper=False)

        # Build the per‑page record
        cur_doc = [
            idatlas,            # document identifier
            cleaned,            # cleaned OCR text
            k,                  # current page index (0‑based)
            len(imgs) - 1,      # index of the last processed page
            DPI_PDF,            # DPI used when rasterising
            tess_time,          # time spent in Tesseract
            TESS_CONFIG,        # Tesseract configuration string
            TESS_LANG,          # language used
        ]
        processed_pages.append(cur_doc)

    return processed_pages

# ------------------------------------------------------------------
# Process a CSV list of PDFs, capture OCR page by page, merge PDFs
# and output a combined PDF and a CSV with OCR results
# ------------------------------------------------------------------
def ocr_tesseract_csv(
    csv_file_path: Union[str, Path],
    output_pdf_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Read a CSV that lists PDF files, run OCR on each PDF, merge the PDFs
    into a single document and return a DataFrame with all OCR results.
    The function also writes the merged PDF to ``output_pdf_path`` and
    saves the OCR DataFrame to ``<output_pdf_path>_ocr.csv``.

    Parameters
    ----------
    csv_file_path : Union[str, Path]
        Path to a CSV containing at least ``doc_idatlas`` and ``pdf_path`` columns.
    output_pdf_path : Union[str, Path]
        Destination path for the merged PDF file.

    Returns
    -------
    pd.DataFrame
        Concatenated OCR results for every processed document.
    """
    # ------------------------------------------------------------------
    # Normalise input / output paths
    # ------------------------------------------------------------------
    csv_path = Path(csv_file_path).expanduser().resolve()
    output_path = Path(output_pdf_path).expanduser().resolve()
    ocr_output_csv = output_path.with_name(output_path.stem + "_ocr.csv")

    # ------------------------------------------------------------------
    # Load CSV and validate its structure
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {exc}") from exc

    required_cols = {"doc_idatlas", "pdf_path"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Containers for OCR results
    # ------------------------------------------------------------------
    ocr_frames: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Process each row: OCR → PDF merge
    # ------------------------------------------------------------------
    with PdfMerger() as merger:
        for _, row in df.iterrows():
            doc_id = str(row["doc_idatlas"])
            pdf_path = Path(row["pdf_path"]).expanduser().resolve()

            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            # ----- OCR -------------------------------------------------
            try:
                text_ocr = ocr_tesseract_pdf(str(pdf_path), doc_id)
                df_ocr = pd.DataFrame(
                    text_ocr,
                    columns=[
                        "doc_idatlas",
                        "text",
                        "page_number",
                        "max_page_number",
                        "dpi_pdf",
                        "tess_time",
                        "tess_config",
                        "tess_lang",
                    ],
                )
                ocr_frames.append(df_ocr)
            except Exception as e:
                # OCR failures are logged but do not abort the whole pipeline
                print(f"[WARN] OCR failed for {pdf_path} – {e}")

            # ----- PDF merge -------------------------------------------
            merger.append(str(pdf_path))

        # Write the merged PDF to disk
        with output_path.open("wb") as fout:
            merger.write(fout)

    # ------------------------------------------------------------------
    # Combine OCR DataFrames and persist to CSV
    # ------------------------------------------------------------------
    if ocr_frames:
        all_ocr_df = pd.concat(ocr_frames, ignore_index=True)
        all_ocr_df.to_csv(ocr_output_csv, index=False)
        print(f"[INFO] Combined OCR data written to: {ocr_output_csv}")
    else:
        all_ocr_df = pd.DataFrame()
        print("[INFO] No OCR data collected; empty CSV not created.")

    return all_ocr_df