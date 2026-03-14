# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import sys
import pathlib
import logging
import datetime
from pathlib import Path
import yaml
import pandas as pd
import os
import argparse
from contextlib import redirect_stdout, nullcontext

from code.utils.pretraitement import prepare_csv, detect_csv_dialect
from code.utils.first_step_ocr.ocr_vlm import process_vlm_csv_and_generate_pdf
from code.utils.second_step_scores.scores_attribution import compute_best_scores
from code.utils.third_step_extraction.extraction_llm import get_pred


# --------------------------------------------------------------
# Configuration du logger (log dans logs/vlm_llm)
# --------------------------------------------------------------
# 1️⃣ Chemin du répertoire de logs – on le crée toujours
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent          # dossier du script
LOG_DIR   = BASE_DIR / "logs" / "vlm_llm"
LOG_DIR.mkdir(parents=True, exist_ok=True)                 # ← crée tout le chemin

# 2️⃣ Nom du fichier de log
log_file_name = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log"
log_file_path = LOG_DIR / log_file_name

# 3️⃣ Configuration de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),                # console
        logging.FileHandler(log_file_path, encoding="utf-8")  # fichier
    ]
)

log = logging.getLogger(__name__)
# ------------------------------------------------------------------
def run_pipeline_vlm_llm(
    cfg_path: str = str(pathlib.Path(__file__).resolve().parent.parent / "config.yaml"),
    quiet: bool = False,
) -> None:
    """
    Execute the full VLM → LLM extraction pipeline.

    Args:
        cfg_path: Path to the YAML configuration file.
        quiet:   If True, suppress console output (useful for batch runs).
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Resolve paths from configuration
    # ------------------------------------------------------------------
    csv_input = pathlib.Path(cfg["paths"]["input_csv_enrichi"])
    extraction_output = pathlib.Path(cfg["paths"]["extraction_output_vlm_llm"])
    ocr_output = pathlib.Path(cfg["paths"]["ocr_vlm"])

    # ------------------------------------------------------------------
    # Helper to optionally silence stdout
    # ------------------------------------------------------------------
    stdout_ctx = nullcontext() if quiet else redirect_stdout(sys.stdout)

    with stdout_ctx:
        # ------------------------------------------------------------------
        # 1️⃣ OCR avec VLM
        # ------------------------------------------------------------------
        log.info("▶ Ocerisation des documents pdf")
        df_ocr_output = process_vlm_csv_and_generate_pdf(csv_input, ocr_output)

        # ------------------------------------------------------------------
        # 2️⃣ Calcul du meilleur score
        # ------------------------------------------------------------------
        log.info("▶ Calcul du meilleur score pour trouver le paragraphe à analyser")
        csv_opts = detect_csv_dialect(csv_input)
        df_csv_input = pd.read_csv(csv_input, **csv_opts)
        df_csv_input_first = prepare_csv(df_csv_input)
        df_best_scores = compute_best_scores(df_csv_input_first, df_ocr_output)

        # ------------------------------------------------------------------
        # 3️⃣ Prédictions finales LLM
        # ------------------------------------------------------------------
        log.info("▶ Prédictions finales du LLM (extraction d'adresse) …")
        get_pred(df_best_scores, df_csv_input_first, extraction_output)
        log.info(f"▶ Prédictions sauvegardées dans {extraction_output}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: expose a tiny CLI to toggle ``quiet`` mode
    parser = argparse.ArgumentParser(description="Run the VLM → LLM extraction pipeline.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (keep only log file).",
    )
    args = parser.parse_args()
    run_pipeline_vlm_llm(quiet=args.quiet)