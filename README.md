# Extraction d'adresses depuis des documents PDF

Pipeline d'extraction automatique d'adresses postales à partir de documents officiels numérisés, développé dans le cadre d'un stage en institution.

## Contexte

Une base de données institutionnelle recense des documents officiels (actes notariés, jugements, etc.) dont une partie est scannée, manuscrite ou dactylographiée. L'objectif est d'en extraire automatiquement l'adresse de domicile d'une personne afin de fiabiliser une base donnée nationale.

Le principal défi : les documents **manuscrits** rendent les OCR classiques (Tesseract) inefficaces. Ce projet explore le recours aux **modèles de vision** (VLM) pour surmonter cette limitation.

## Trois pipelines comparées

| Pipeline | OCR | Extraction | Durée / doc | Résultat |
|---|---|---|---|---|
| `tesseract_llm` | Tesseract | LLM | — | ❌ Exclu |
| `vlm_llm` | VLM (Qwen3-VL) | LLM | ~2m24s | ✅ Recommandée |
| `vlm_vlm` | VLM (Qwen3-VL) | VLM | ~40s | ⚠️ À surveiller |

### Résultats des tests (2 documents)

Deux types de documents ont été testés : un document **entièrement manuscrit** et un document **mixte (tapuscrit + manuscrit)**.

| | `tesseract_llm` | `vlm_llm` | `vlm_vlm` |
|---|---|---|---|
| **Doc. 1 — Manuscrit** | ❌ Résultat vide (texte non reconnu) | ✅ Adresse correctement extraite | ✅ Adresse correctement extraite |
| **Doc. 2 — Mixte** | ❌ Mauvaise adresse (donateur extrait à la place du donataire) | ⚠️ Coquille sur le nom de l'avenue | ⚠️ Coquille + "avenue" remplacée par "rue" (erreur d'initiative) |

### Conclusion

Tesseract est **exclu** pour ce type de documents manuscrits. La pipeline `vlm_llm` offre la meilleure fiabilité mais implique un coût computationnel et une durée de traitement élevés (~2m24s/doc). La pipeline `vlm_vlm` est plus rapide (~40s/doc) mais introduit des erreurs d'initiative.

Une **industrialisation sur un échantillon plus large** serait nécessaire pour trancher définitivement entre `vlm_llm` et `vlm_vlm`.

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

1. Copier `.env.example` en `.env` et renseigner les identifiants du serveur LLM et VLM :

```bash
cp .env.example .env
```

2. Adapter si besoin les chemins dans `config.yaml` (relatifs à la racine du projet) et choisir la pipeline :

```yaml
pipeline:
  extraction_engine: "vlm_llm"  # tesseract_llm | vlm_llm | vlm_vlm
```

## Utilisation

```bash
python main.py
# ou avec un fichier de config personnalisé :
python main.py --config /chemin/vers/config.yaml
```

> **Note** : Ce projet a été développé dans un environnement institutionnel. Les données et les accès au modèles LLM et VLM internes ne sont pas fournis.

## Perspectives

Les tests ont été réalisés sur 2 documents. Pour valider statistiquement le choix de pipeline, la prochaine étape consiste à **industrialiser l'évaluation à grande échelle** via un script Bash orchestrant le pipeline sur un échantillon représentatif :

- Lancement automatisé en lot (`batch`) sur l'ensemble du corpus
- Collecte des métriques de sortie (adresse extraite, temps de traitement, statut)
- Comparaison systématique des résultats `vlm_llm` vs `vlm_vlm` sur un volume suffisant pour conclure
