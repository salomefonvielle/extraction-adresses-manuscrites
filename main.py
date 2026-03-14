import pathlib
import sys
import logging
import yaml

# Fonctions de vos différents pipelines
BASE_DIR = pathlib.Path(__file__).resolve()
sys.path.append(str(BASE_DIR))                              # ← rendant « BASE_DIR » visible
from code.pretraitement_data import run_pretraitement
from code.extraction_vlm_llm import run_pipeline_vlm_llm
from code.extraction_tesseract_llm import run_pipeline_tesseract_llm
from code.extraction_vlm_vlm import run_pipeline_vlm_vlm
from code.pretraitement_data import run_pretraitement
from code.extraction_vlm_llm import run_pipeline_vlm_llm
from code.extraction_tesseract_llm import run_pipeline_tesseract_llm
from code.extraction_vlm_vlm import run_pipeline_vlm_vlm

# ----------------------------------------------------------------------
# Configuration du logger (utilisé partout dans ce script)
# ----------------------------------------------------------------------
LOG_DIR = pathlib.Path("logs/run_all")
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
DEFAULT_CFG_PATH = (
    "/home/sfonvielle-stagiai01/projets/new_extraction_adresse/local/config.yaml"
)


def load_cfg(cfg_path: str = DEFAULT_CFG_PATH) -> dict:
    """Lit le fichier YAML et renvoie le dictionnaire de configuration."""
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_run_pretraitement(cfg: dict) -> None:
    """
    Lance le pré‑traitement uniquement si le CSV enrichi n’existe pas.

    Le chemin du CSV enrichi est attendu dans ``cfg["paths"]["output_csv"]``.
    """
    enriched_csv = pathlib.Path(cfg["paths"]["input_csv_enrichi"])
    if not enriched_csv.is_file():
        log.info("🚀 CSV enrichi introuvable → lancement du pré‑traitement")
        # La fonction run_pretraitement accepte généralement le chemin de config
        # et éventuellement le mode « quiet ».  On transmet uniquement le cfg_path.
        run_pretraitement(cfg_path=DEFAULT_CFG_PATH, quiet=False)
    else:
        log.info(f"✅ CSV enrichi déjà présent : {enriched_csv}")


def run_extraction_pipeline(cfg: dict) -> None:
    """
    Sélectionne et exécute la pipeline d’extraction demandée.

    - ``tesseract_llm`` → :func:`run_pipeline_tesseract_llm`
    - ``vlm_llm``       → :func:`run_pipeline_vlm_llm`
    - tout autre valeur → fallback sur ``tesseract_llm``
    """
    engine = cfg["pipeline"].get("extraction_engine", "tesseract_llm")
    if engine == "tesseract_llm":
        log.info("▶ Lancement de la pipeline Tesseract → LLM")
        run_pipeline_tesseract_llm(cfg_path=DEFAULT_CFG_PATH)
    elif engine == "vlm_llm":
        log.info("▶ Lancement de la pipeline VLM → LLM")
        run_pipeline_vlm_llm(cfg_path=DEFAULT_CFG_PATH)
    elif engine == "vlm_vlm" : 
        run_pipeline_vlm_vlm(cfg_path=DEFAULT_CFG_PATH)

    else:
        log.warning(
            f"⚠️ extraction_engine '{engine}' inconnu, utilisation du fallback Tesseract → LLM"
        )


def main(cfg_path: str = DEFAULT_CFG_PATH) -> None:
    """Point d’entrée du script."""
    cfg = load_cfg(cfg_path)

    # 1️⃣  Pré‑traitement (si besoin)
    maybe_run_pretraitement(cfg)

    # 2️⃣  Pipeline d’extraction choisie
    run_extraction_pipeline(cfg)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Petite CLI permettant de spécifier un autre fichier de config
    import argparse

    parser = argparse.ArgumentParser(
        description="Orchestre le pré‑traitement + la pipeline d’extraction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CFG_PATH,
        help="Chemin du fichier de configuration YAML.",
    )
    args = parser.parse_args()
    main(cfg_path=args.config)