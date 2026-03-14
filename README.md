# Extraction d'adresses depuis des documents PDF

Pipeline d'extraction automatique d'adresses postales à partir de documents officiels numérisés, développé dans le cadre d'un stage en institution.

## Contexte

Une base de données institutionnelle recense des documents officiels (actes notariés, jugements, etc.) dont une partie est scannée, manuscrite ou dactylographiée. L'objectif est d'en extraire automatiquement l'adresse de domicile d'une personne afin de fiabiliser les données de la base.

Le principal défi : les documents **manuscrits ou formulaires scannés** rendent les OCR classiques (Tesseract) inefficaces. Ce projet explore le recours aux **modèles de vision** (VLM) pour surmonter cette limitation.

## Trois pipelines comparées

| Pipeline | OCR | Extraction | Résultat |
|---|---|---|---|
| `tesseract_llm` | Tesseract | LLM | ❌ Échec sur manuscrits |
| `vlm_llm` | VLM (Qwen3-VL) | LLM | ✅ Recommandée |
| `vlm_vlm` | VLM (Qwen3-VL) | VLM | ⚠️ À surveiller (~40s/doc) |

**Conclusion des tests** : Tesseract est exclu pour ce type de documents. La pipeline `vlm_llm` offre la meilleure fiabilité (~2m24s/doc). La pipeline `vlm_vlm` est plus rapide mais introduit des erreurs d'initiative (ex. "avenue" remplacée par "rue").

## Architecture

```
PDF → [1. OCR] → texte brut → [2. Scoring] → meilleur paragraphe → [3. Extraction] → adresse structurée
```

**Étape 1 — OCR** (`code/utils/first_step_ocr/`)
Transcription page par page via VLM ou Tesseract.

**Étape 2 — Scoring** (`code/utils/second_step_scores/`)
Identification du paragraphe contenant l'adresse par fuzzy matching sur le nom, les prénoms, la date de naissance et le code postal.

**Étape 3 — Extraction** (`code/utils/third_step_extraction/`)
Extraction structurée de l'adresse via LLM ou VLM.

## Structure du projet

```
.
├── main.py                          # Point d'entrée
├── config.yaml                      # Chemins et choix de pipeline
├── requirements.txt
├── .env.example                     # Variables d'environnement à renseigner
└── code/
    ├── pretraitement_data.py        # Enrichissement CSV avec chemins PDF
    ├── extraction_vlm_llm.py        # Orchestration pipeline VLM → LLM
    ├── extraction_tesseract_llm.py  # Orchestration pipeline Tesseract → LLM
    ├── extraction_vlm_vlm.py        # Orchestration pipeline VLM → VLM
    └── utils/
        ├── pretraitement.py         # Utilitaires CSV/PDF
        ├── first_step_ocr/          # Modules OCR
        ├── second_step_scores/      # Scoring des paragraphes candidats
        └── third_step_extraction/   # Extraction finale
```

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

2. Adapter `config.yaml` avec les chemins vers vos données et le choix de pipeline :

```yaml
paths:
  input_csv: "datas/inputs/fichier_traites.csv"
  pdf_dir: "datas/inputs/PDF"
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

> **Note** : Ce projet a été développé dans un environnement institutionnel. Les données et les accès au serveur VLM interne ne sont pas fournis.
