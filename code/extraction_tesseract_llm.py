# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import sys
import pathlib
import logging
import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml

from code.utils.pretraitement import prepare_csv, detect_csv_dialect
from code.utils.first_step_ocr.ocr_tesseract import ocr_tesseract_csv
from code.utils.second_step_scores.scores_attribution import compute_best_scores
from code.utils.third_step_extraction.extraction_llm import get_pred

# ------------------------------------------------------------------
# Logger configuration – logs are written in a dedicated folder
# ------------------------------------------------------------------
import pathlib
import sys
import logging
import datetime

# --------------------------------------------------------------
# Configuration du logger (log dans logs/tesseract_llm)
# --------------------------------------------------------------
# 1️⃣ Chemin du répertoire de logs – on le crée toujours
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR   = BASE_DIR / "logs" / "tesseract_llm"
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
def run_pipeline_tesseract_llm(
    cfg_path: str = "/home/sfonvielle-stagiai01/projets/new_extraction_adresse/local/config.yaml",
    quiet: bool = False,
) -> None:
    """
    Execute the full OCR → LLM extraction pipeline.

    Args:
        cfg_path: Path to the YAML configuration file.
        quiet:    If True, suppress console output (only file logs are kept).
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
    extraction_output = pathlib.Path(cfg["paths"]["extraction_output_tess_llm"])
    ocr_output = pathlib.Path(cfg["paths"]["ocr_tess"])

    # ------------------------------------------------------------------
    # Optional silence of stdout (useful for batch runs)
    # ------------------------------------------------------------------
    from contextlib import redirect_stdout, nullcontext

    stdout_ctx = nullcontext() if quiet else redirect_stdout(sys.stdout)

    with stdout_ctx:
        log.info("▶ Démarrage de la pipeline OCR → LLM")

        # ---------------- OCR ----------------
        log.info("▶ Ocerisation des documents pdf")
        df_ocr_output = ocr_tesseract_csv(csv_input, ocr_output)
        log.info(f"✅ OCR terminé – lignes générées : {len(df_ocr_output)}")
        log.debug(f"🧪 Sample OCR rows:\n{df_ocr_output.head().to_string(index=False)}")

        # ---------------- Best‑score ----------------
        log.info("▶ Calcul du meilleur score pour trouver le paragraphe à analyser")
        csv_opts = detect_csv_dialect(csv_input)
        df_csv_input = pd.read_csv(csv_input, **csv_opts)
        log.debug(f"📊 CSV d’entrée – shape : {df_csv_input.shape}")
        df_csv_input_first = prepare_csv(df_csv_input)
        log.debug(f"🔧 CSV pré‑traité – shape : {df_csv_input_first.shape}")

        df_best_scores = compute_best_scores(df_csv_input_first, df_ocr_output)

        # ---------------- LLM extraction ----------------
        log.info("▶ Prédictions finales du LLM (extraction d'adresse) …")
        get_pred(df_best_scores, df_csv_input_first, extraction_output)
        log.info(f"✅ Prédictions sauvegardées dans {extraction_output}")

        # Final sanity‑check
        try:
            df_pred = pd.read_csv(extraction_output)
            log.info(
                f"🔎 Vérification finale – rows : {len(df_pred)} – columns : {list(df_pred.columns)}"
            )
            log.debug(f"🧪 Sample predictions:\n{df_pred.head().to_string(index=False)}")
        except Exception as e:
            log.error(f"❗️ Impossible de charger les prédictions pour vérification : {e}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny CLI to allow turning off console output
    import argparse

    parser = argparse.ArgumentParser(description="Run the Tesseract → LLM extraction pipeline.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (keep only file logs).",
    )
    args = parser.parse_args()
    run_pipeline_tesseract_llm(quiet=args.quiet)