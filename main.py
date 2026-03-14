import pathlib
import sys
import logging
import yaml

# Fonctions de vos différents pipelines
BASE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
from code.pretraitement_data import run_pretraitement
from code.extraction_vlm_llm import run_pipeline_vlm_llm
from code.extraction_tesseract_llm import run_pipeline_tesseract_llm
from code.extraction_vlm_vlm import run_pipeline_vlm_vlm

# ----------------------------------------------------------------------
# Configuration du logger (utilisé partout dans ce script)
# ----------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs" / "run_all"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"{pathlib.Path(__file__).stem}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),          # console
        logging.FileHandler(log_file, encoding="utf-8")  # fichier
    ],
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Chemin du fichier de configuration (peut être passé en argument si besoin)
# ----------------------------------------------------------------------
DEFAULT_CFG_PATH = str(BASE_DIR / "config.yaml")


def load_cfg(cfg_path: str = DEFAULT_CFG_PATH) -> dict:
    """Lit le fichier YAML et renvoie le dictionnaire de configuration."""
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_run_pretraitement(cfg: dict, cfg_path: str) -> None:
    """
    Lance le pré‑traitement uniquement si le CSV enrichi n'existe pas.

    Le chemin du CSV enrichi est attendu dans ``cfg["paths"]["input_csv_enrichi"]``.
    """
    enriched_csv = pathlib.Path(cfg["paths"]["input_csv_enrichi"])
    if not enriched_csv.is_file():
        log.info("🚀 CSV enrichi introuvable → lancement du pré‑traitement")
        run_pretraitement(cfg_path=cfg_path, quiet=False)
    else:
        log.info(f"✅ CSV enrichi déjà présent : {enriched_csv}")


def run_extraction_pipeline(cfg: dict, cfg_path: str) -> None:
    """
    Sélectionne et exécute la pipeline d'extraction demandée.

    - ``tesseract_llm`` → :func:`run_pipeline_tesseract_llm`
    - ``vlm_llm``       → :func:`run_pipeline_vlm_llm`
    - ``vlm_vlm``       → :func:`run_pipeline_vlm_vlm`
    """
    engine = cfg["pipeline"].get("extraction_engine", "tesseract_llm")
    if engine == "tesseract_llm":
        log.info("▶ Lancement de la pipeline Tesseract → LLM")
        run_pipeline_tesseract_llm(cfg_path=cfg_path)
    elif engine == "vlm_llm":
        log.info("▶ Lancement de la pipeline VLM → LLM")
        run_pipeline_vlm_llm(cfg_path=cfg_path)
    elif engine == "vlm_vlm":
        log.info("▶ Lancement de la pipeline VLM → VLM")
        run_pipeline_vlm_vlm(cfg_path=cfg_path)
    else:
        log.warning(
            f"⚠️ extraction_engine '{engine}' inconnu, utilisation du fallback Tesseract → LLM"
        )
        run_pipeline_tesseract_llm(cfg_path=cfg_path)


def main(cfg_path: str = DEFAULT_CFG_PATH) -> None:
    """Point d'entrée du script."""
    cfg = load_cfg(cfg_path)

    # 1️⃣  Pré‑traitement (si besoin)
    maybe_run_pretraitement(cfg, cfg_path)

    # 2️⃣  Pipeline d'extraction choisie
    run_extraction_pipeline(cfg, cfg_path)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Petite CLI permettant de spécifier un autre fichier de config
    import argparse

    parser = argparse.ArgumentParser(
        description="Orchestre le pré‑traitement + la pipeline d'extraction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CFG_PATH,
        help="Chemin du fichier de configuration YAML.",
    )
    args = parser.parse_args()
    main(cfg_path=args.config)
