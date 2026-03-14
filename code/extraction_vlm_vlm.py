import sys
import pathlib
import logging
import datetime
from pathlib import Path
import pandas as pd
import time                                 # timing utilities
from code.utils.pretraitement import prepare_csv, detect_csv_dialect
from code.utils.third_step_extraction.extraction_vlm import get_pred_full_vlm
import yaml

# --------------------------------------------------------------
# Configuration du logger (log dans logs/vlm_vlm)
# --------------------------------------------------------------
# 1️⃣ Chemin du répertoire de logs – on le crée toujours
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent          # dossier du script
LOG_DIR   = BASE_DIR / "logs" / "vlm_vlm"
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
def run_pipeline_vlm_vlm(cfg_path: str = "/home/sfonvielle-stagiai01/projets/new_extraction_adresse/local/config.yaml") -> None:
    """
    Exécute le pipeline VLM‑VLM.

    Args:
        cfg_path: Chemin vers le fichier de configuration YAML.
    """
    # ---------- sanity‑check: start ----------
    start_time = time.perf_counter()
    log.info("🚀 VLM‑VLM pipeline started")

    # ------------------------------------------------------------------
    # Chargement de la configuration
    # ------------------------------------------------------------------
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- Input/Output paths ------------------------------------------------
    ocr_input = pathlib.Path(cfg["paths"]["input_csv_enrichi"])
    if not ocr_input.is_file():
        log.error(f"❌ Input CSV not found: {ocr_input}")
        sys.exit(1)

    output_path = pathlib.Path(cfg["paths"]["extraction_output_vlm_vlm"])

    # ---- Step 1 : format CSV ------------------------------------------------
    t1 = time.perf_counter()
    log.info("▶ Mise en forme du CSV")
    csv_opts = detect_csv_dialect(ocr_input)
    df_ocr_input = pd.read_csv(ocr_input, **csv_opts)
    log.debug(f"📊  Rows read from input CSV: {len(df_ocr_input)}")
    df_ocr_input_first = prepare_csv(df_ocr_input)
    log.debug(f"✅  CSV formatted – elapsed {time.perf_counter() - t1:.2f}s")

    # ---- Step 2 : VLM predictions ------------------------------------------
    t2 = time.perf_counter()
    log.info("▶ Prédictions VLM (OCR + extraction d'adresse)")
    df_pred = get_pred_full_vlm(df_ocr_input_first)
    log.debug(f"🗂  Predictions generated: {len(df_pred)} rows – elapsed {time.perf_counter() - t2:.2f}s")

    # ---- Export -------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)   # ensure output folder exists
    df_pred.to_csv(output_path, index=False)
    log.info(f"▶ Prédictions sauvegardées dans {output_path}")

    # ---------- sanity‑check: end ----------
    total_elapsed = time.perf_counter() - start_time
    log.info(f"🏁 VLM‑VLM pipeline finished – total time: {total_elapsed:.2f}s")

# ------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline_vlm_vlm()