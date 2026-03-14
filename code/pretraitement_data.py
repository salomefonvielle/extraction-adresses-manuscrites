# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import pathlib
import sys
import logging
import datetime
import yaml               # new: read the config file
import pandas as pd


# --------------------------------------------------------------
# Configuration du logger (log dans logs/tesseract_llm)
# --------------------------------------------------------------
# 1️⃣ Chemin du répertoire de logs – on le crée toujours
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent          # dossier du script
LOG_DIR   = BASE_DIR / "logs" / "pretraitement"
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
# Fonction principale
# ------------------------------------------------------------------
def run_pretraitement(
    cfg_path: str = "/home/sfonvielle-stagiai01/projets/new_extraction_adresse/local/config.yaml",
    quiet: bool = False,
) -> None:
    """
    Exécute la pipeline d’enrichissement du CSV avec le chemin PDF.

    Args:
        cfg_path: Chemin du fichier de configuration YAML.
        quiet:   Si ``True`` supprime la sortie console (utile en batch).
    """
    # ------------------------------------------------------------------
    # Chargement de la configuration
    # ------------------------------------------------------------------
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # ------------------------------------------------------------------
    # Configuration des chemins – tirés du config.yaml
    # ------------------------------------------------------------------
    CSV_IN  = pathlib.Path(cfg["paths"]["input_csv"])
    CSV_OUT = pathlib.Path(cfg["paths"]["input_csv_enrichi"])
    PDF_DIR = pathlib.Path(cfg["paths"]["pdf_dir"])

    # ------------------------------------------------------------------
    # Vérifications de base
    # ------------------------------------------------------------------
    if not CSV_IN.is_file():
        log.error(f"❌ Le fichier CSV d’entrée n’existe pas : {CSV_IN}")
        sys.exit(1)
    log.info(f"📂 Traitement du fichier : {CSV_IN}")

    if not PDF_DIR.is_dir():
        log.error(f"❌ Le répertoire PDF n’existe pas : {PDF_DIR}")
        sys.exit(1)
    log.info(f"📂 Répertoire PDF trouvé : {PDF_DIR}")

    # ------------------------------------------------------------------
    # 1️⃣ Ajout conditionnel de la colonne « index »
    # ------------------------------------------------------------------
    df = pd.read_csv(
        CSV_IN,
        **cfg.get("pandas", {}).get("read_options", {"dtype": str})
    )
    if "index" not in df.columns:
        df.insert(0, "index", range(1, len(df) + 1))
        log.info("➡️ Ajout d’une colonne « index » 1‑based")
    else:
        if df.columns[0] != "index":
            col = df.pop("index")
            df.insert(0, "index", col)
            log.info("➡️ Column « index » déjà présente, ré‑positionnée en première colonne")

    # ------------------------------------------------------------------
    # 2️⃣ Enrichissement du CSV avec le chemin PDF
    # ------------------------------------------------------------------
    try:
        from code.utils.pretraitement import enrich_csv_with_pdf_path
    except ImportError as e:
        log.error(f"❌ Impossible d’importer la fonction d’enrichissement : {e}")
        sys.exit(1)

    temp_csv = CSV_IN.parent / "tmp_with_index.csv"
    df.to_csv(
        temp_csv,
        **cfg.get("pandas", {}).get("write_options", {"index": False})
    )
    log.debug(f"⏱  CSV temporaire écrit : {temp_csv}")
    log.info("📥 Lancement de l’enrichissement avec les chemins PDFs")
    enrich_csv_with_pdf_path(temp_csv, CSV_OUT)

    # ------------------------------------------------------------------
    # Nettoyage du fichier temporaire
    # ------------------------------------------------------------------
    try:
        temp_csv.unlink()
        log.debug(f"🗑️ Fichier temporaire supprimé : {temp_csv}")
    except Exception:
        log.warning(f"⚠️ Impossible de supprimer le fichier temporaire : {temp_csv}")

    log.info(f"✅ {len(df)} lignes traitées – fichier final écrit dans : {CSV_OUT}")

# ------------------------------------------------------------------
# Entrée du script – petite CLI pour désactiver la console si besoin
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from contextlib import redirect_stdout, nullcontext

    parser = argparse.ArgumentParser(
        description="Pipeline d’enrichissement CSV → PDF"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Supprime la sortie console (conserve seulement le fichier log).",
    )
    args = parser.parse_args()

    # Redirige stdout uniquement quand on ne veut pas de messages console
    stdout_ctx = nullcontext() if args.quiet else redirect_stdout(sys.stdout)
    with stdout_ctx:
        run_pretraitement(quiet=args.quiet)