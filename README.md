# Extraction d'adresses depuis des documents PDF

Pipeline d'extraction automatique d'adresses postales à partir de documents PDF, développé dans le cadre d'un stage à la DGFIP.

## Présentation

Le projet vise à fiabiliser les adresses fiscales en extrayant automatiquement l'adresse de domicile d'un contribuable depuis des documents officiels numérisés (actes notariés, jugements, etc.).

Le pipeline combine OCR et modèles de langage (LLM/VLM) pour identifier et extraire les informations d'adresse de manière robuste, même sur des documents de mauvaise qualité.

## Architecture

Le pipeline se déroule en trois étapes :

```
PDF → [1. OCR] → texte brut → [2. Scoring] → meilleur paragraphe → [3. Extraction LLM] → adresse structurée
```

**Étape 1 — OCR** (`code/utils/first_step_ocr/`)
- `ocr_vlm.py` : transcription via un VLM (Qwen3-VL-32B) appelé par API
- `ocr_tesseract.py` : transcription via Tesseract (fallback local)

**Étape 2 — Scoring** (`code/utils/second_step_scores/`)
- Identifie les paragraphes candidats (mots-clés de civilité)
- Score chaque candidat via fuzzy matching sur le nom, les prénoms, la date de naissance et le code postal

**Étape 3 — Extraction** (`code/utils/third_step_extraction/`)
- `extraction_llm.py` : extraction structurée via un LLM
- `extraction_vlm.py` : extraction directe via un VLM (pipeline vlm_vlm)

## Trois pipelines disponibles

| Pipeline | OCR | Extraction | Usage recommandé |
|---|---|---|---|
| `tesseract_llm` | Tesseract | LLM | Environnement sans GPU |
| `vlm_llm` | VLM | LLM | Meilleure précision OCR |
| `vlm_vlm` | VLM | VLM | Pipeline entièrement VLM |

## Installation

```bash
pip install -r requirements.txt
```

Tesseract doit être installé séparément : [guide d'installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)

## Configuration

1. Copier `.env.example` en `.env` et renseigner les identifiants du serveur VLM :

```bash
cp .env.example .env
```

2. Adapter `config.yaml` avec les chemins vers vos données :

```yaml
paths:
  input_csv: "datas/inputs/fichier_traites.csv"
  pdf_dir: "datas/inputs/PDF"
  input_csv_enrichi: "datas/outputs/fichier_traites_enrichis.csv"
  ...

pipeline:
  extraction_engine: "vlm_llm"  # tesseract_llm | vlm_llm | vlm_vlm
```

## Utilisation

```bash
python main.py
# ou avec un fichier de config personnalisé :
python main.py --config /chemin/vers/config.yaml
```

## Structure du projet

```
.
├── main.py                        # Point d'entrée
├── config.yaml                    # Configuration (chemins, pipeline)
├── requirements.txt
├── .env.example
└── code/
    ├── pretraitement_data.py      # Étape 0 : enrichissement CSV avec chemins PDF
    ├── extraction_vlm_llm.py      # Orchestration pipeline VLM → LLM
    ├── extraction_tesseract_llm.py# Orchestration pipeline Tesseract → LLM
    ├── extraction_vlm_vlm.py      # Orchestration pipeline VLM → VLM
    └── utils/
        ├── pretraitement.py       # Fonctions utilitaires CSV/PDF
        ├── first_step_ocr/        # Modules OCR
        ├── second_step_scores/    # Scoring des paragraphes candidats
        └── third_step_extraction/ # Extraction finale via LLM/VLM
```
