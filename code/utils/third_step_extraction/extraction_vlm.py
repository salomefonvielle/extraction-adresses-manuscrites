import math
import io
from typing import List, Union, Tuple
from PIL import Image
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import os
import s3fs
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import json
import base64
import requests
from code.utils.second_step_scores.scores_config import JSON_PATTERN, REQUIRED_KEYS

# --- Constantes ---
PATCH_FACTOR = 32
DEFAULT_MODEL_SEQ_LEN = 90000

load_dotenv(override=True)

CSV_OUTPUT = "/home/sfonvielle-stagiai01/projets/new_extraction_adresse/local/fichiers_traite.csv"


API_IP = os.getenv("PIA_IP")
API_PORT = os.getenv("PIA_PORT")
API_KEY = os.getenv("PIA_API_KEY")

FIXED_MIN_DPI = 150
FIXED_MAX_DPI = 200

# Définition précise des champs avec instructions pour le prompt
FIELDS_CONFIG = [
    {"key": "nom", "nom_human": "Nom de famille", "desc": "Le nom de famille du demandeur. En MAJUSCULES."},
    {"key": "prenoms", "nom_human": "Prénoms", "desc": "Tous les prénoms mentionnés."},
    {"key": "date_naissance", "nom_human": "Date de naissance", "desc": "Format JJ/MM/AAAA."},
    {"key": "commune_naissance", "nom_human": "Commune de naissance", "desc": "Le lieu ou la ville de naissance."},
    {"key": "rue", "nom_human": "Numéro et nom de rue", "desc": "Le numéro de voirie et le nom de la rue."},
    {"key": "complement_adresse", "nom_human": "Complément d'adresse", "desc": "Étage, bâtiment, résidence, CS, BP."},
    {"key": "ville", "nom_human": "Nom de commune", "desc": "La ville de résidence actuelle."},
    {"key": "code_postal", "nom_human": "Code postal", "desc": "5 chiffres."},
    {"key": "email", "nom_human": "Courriel", "desc": "L'adresse email."},
    {"key": "telephone", "nom_human": "Téléphone", "desc": "Numéro de téléphone fixe ou mobile."}
]

# Initialisation S3
s3 = s3fs.S3FileSystem(
    anon=False,
    client_kwargs={
        "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    },
)


# config_no_proxy.py
import os

# -------------------------------------------------------------------------
# Configuration du proxy – équivalent de la magic %env du notebook
# -------------------------------------------------------------------------
os.environ["no_proxy"] = (
    "100.70.1.199,forge.dgfip.finances.rie.gouv.fr,"
    "pia-exp-back.dev.dgfip,pia-exp-front.dev.dgfip,"
    "10.156.253.10,huggingface.co,10.156.253.13,"
    "10.156.226.144,10.156.226.145"
)

def calculate_page_tokens(width_pt: float, height_pt: float, dpi: int) -> int:
    """Calcule le nombre de tokens pour une page selon le DPI."""
    width_px = (width_pt / 72.0) * dpi
    height_px = (height_pt / 72.0) * dpi

    patches_w = math.ceil(width_px / PATCH_FACTOR)
    patches_h = math.ceil(height_px / PATCH_FACTOR)

    return patches_w * patches_h

def get_pdf_native_resolution(pdf_bytes: bytes, default_dpi: int = 300) -> int:
    """Analyse le PDF pour trouver la résolution maximale des images intégrées."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        max_detected_dpi = 0
        for i in range(min(len(reader.pages), 3)):
            page = reader.pages[i]
            if '/Resources' in page and '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        width_px = xObject[obj]['/Width']
                        mb = page.mediabox
                        width_pt = float(mb.width)
                        if width_pt > 0:
                            dpi_w = (width_px / width_pt) * 72
                            max_detected_dpi = max(max_detected_dpi, dpi_w)
        return int(round(max_detected_dpi)) if max_detected_dpi > 0 else default_dpi
    except Exception:
        return default_dpi

def optimize_pdf_params(
    page_sizes_pt: List[Tuple[float, float]],
    token_budget: int,
    min_dpi: int,
    max_dpi: int
) -> Tuple[int, int]:
    """Détermine le meilleur DPI et le nombre de pages à traiter selon le budget tokens."""

    # 1. Test au DPI maximum
    total_tokens_max = sum(calculate_page_tokens(w, h, max_dpi) for w, h in page_sizes_pt)
    if total_tokens_max <= token_budget:
        return max_dpi, len(page_sizes_pt)

    # 2. Test au DPI minimum
    total_tokens_min = sum(calculate_page_tokens(w, h, min_dpi) for w, h in page_sizes_pt)
    if total_tokens_min <= token_budget:
        # Recherche binaire de la meilleure qualité entre min et max
        low, high = min_dpi, max_dpi - 1
        best_dpi = min_dpi
        while low <= high:
            mid = (low + high) // 2
            current_tokens = sum(calculate_page_tokens(w, h, mid) for w, h in page_sizes_pt)
            if current_tokens <= token_budget:
                best_dpi = mid
                low = mid + 1
            else:
                high = mid - 1
        return best_dpi, len(page_sizes_pt)

    # 3. Tronquage des pages au DPI minimum
    current_tokens = 0
    pages_to_keep = 0
    for w, h in page_sizes_pt:
        t = calculate_page_tokens(w, h, min_dpi)
        if current_tokens + t <= token_budget:
            current_tokens += t
            pages_to_keep += 1
        else:
            break
    return min_dpi, pages_to_keep

def smart_prepare_media(
    file_input: Union[str, bytes, io.BytesIO],
    text_token_buffer: int = 2000,
    min_dpi: int = 150,
    max_dpi: int = 300
) -> Tuple[List[Image.Image], int, int, int]:
    """
    Traite un PDF ou une Image en optimisant le DPI pour respecter le budget de tokens.

    Returns:
        (images, target_dpi, num_pages_to_process, native_dpi)
    """
    # Conversion de l'entrée en bytes
    if isinstance(file_input, str):
        with open(file_input, "rb") as f: file_bytes = f.read()
    elif isinstance(file_input, io.BytesIO):
        file_bytes = file_input.getvalue()
    else:
        file_bytes = file_input

    # Détection PDF / Image (fallback PDF pour traitement unifié)
    if not file_bytes.startswith(b'%PDF'):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != "RGB": img = img.convert("RGB")
            pdf_io = io.BytesIO()
            img.save(pdf_io, format="PDF")
            file_bytes = pdf_io.getvalue()
        except Exception:
            return[], 0, 0, 0

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        native_dpi = get_pdf_native_resolution(file_bytes, default_dpi=max_dpi)
        effective_max_dpi = min(max_dpi, native_dpi)

        page_sizes_pt =[]
        for p in reader.pages:
            try:
                mb = p.mediabox
                page_sizes_pt.append((float(mb.width), float(mb.height)))
            except:
                page_sizes_pt.append((595.0, 842.0)) # A4 par défaut

        tokens_budget = max(DEFAULT_MODEL_SEQ_LEN - text_token_buffer, 0)
        target_dpi, num_pages = optimize_pdf_params(page_sizes_pt, tokens_budget, min_dpi, effective_max_dpi)

        print(f"Resizing en DPI : natif {native_dpi} -> cible {target_dpi}")
        print(f"Pages conservées : {num_pages} / {len(page_sizes_pt)}")

        if num_pages == 0:
            return[], target_dpi, 0, native_dpi

        pil_images = convert_from_bytes(
            file_bytes,
            dpi=target_dpi,
            first_page=1,
            last_page=num_pages
        )

        return list(pil_images), target_dpi, num_pages, native_dpi

    except Exception as e:
        print(f"Erreur process: {e}")
        return[], 0, 0, 0

def generer_instructions_prompt(x):
    
    target_nom= x["ppd_lnnom"] 
    target_prenom= x["ppd_lipren"] 
    target_dob= x["ppd_dnnai"] 

    """
    Génère un prompt ciblant une personne spécifique.
    """
    instructions = "Tu es un expert en lecture de documents administratifs.\n"
    instructions += f"IMPORTANT : Tu dois extraire les informations uniquement pour la personne suivante :\n"
    instructions += f"- NOM : {target_nom}\n"
    instructions += f"- PRÉNOM : {target_prenom}\n"
    instructions += f"- DATE DE NAISSANCE : {target_dob}\n\n"

    instructions +=f"""A partir du texte, identifie la bonne personne et détermine son adresse de domicile, c'est-à-dire l'adresse où il demeure. Ta réponse sera un json dont les clés : valeurs seront les suivantes :
    "NOM_PRENOM" : prénoms et noms de la personne identifiée dans le texte, il est possible qu'ils soient mal orthographiés ou que le nom de famille soit différent de celui de la personne P bien que ça soit la même personne
    "COMMUNE_NAISSANCE" : commune de naissance souvent précédé du préfixe: "NE A"
    "PAYS_NAISSANCE" : pays de naissance
    "COMMUNE_DOMICILE" : commune de l'adresse, souvent précédée du mot "DOMICILE" ou "DEMEURANT", il s'agit du lieu où la personne P demeure
    "CODE_POSTAL_DOMICILE" : code postal de l'adresse (composé de chiffres) s'il est contenu dans le texte, "" sinon
    "ARRONDISSEMENT_DOMICILE" : défini seulement dans le cas où la COMMUNE_DOMICILE est Lyon, Marseille ou Paris, seulement le chiffre de l'arrondissement doit être écrit
    "DEPARTEMENT_DOMICILE" : département de l'adresse du domicile
    "REGION_DOMICILE" : région de l'adresse du domicile
    "PAYS_DOMICILE" : pays de l'adresse du domicile, écris en toutes lettres et en français (états-unis pour usa, royaume-uni pour uk ...)
    "NOM_VOIE_DOMICILE" : nom de la voie lorsqu'il est indiqué
    "NUMERO_VOIE_DOMICILE" : numéro de la voie lorsqu'il est indiqué
    "BONNE_PERSONNE" : "OUI" si l'adresse extraite est bien celle de la personne indiquée, "NON" sinon
    
    RÈGLES SUPPLÉMENTAIRES :
1. Si une information est absente pour cette personne précise, retourne null pour la clé.
2. Ne pas confondre avec d'autres personnes éventuellement citées dans le document.
3. Retourne UNIQUEMENT l'objet JSON, sans texte superflu.
Ta réponse sera sans commentaire ni explication et sera uniquement un json.
    """
   # instructions += "Extrais les informations sous forme de JSON plat :\n"
   # for field in FIELDS_CONFIG:
       # instructions += f"- {field['nom_human']} (clé JSON: '{field['key']}'): {field['desc']}\n"

   # instructions += """

    x["prompt"]= instructions 

    return instructions


# imports requis (ajoutez‑les en haut du fichier si ce n’est pas déjà fait)
import io, base64, json, requests
from json import JSONDecoder  # uniquement si vous l’utilisez ailleurs
# ...

def get_predictions(x) :
    x["NOM_PRENOM"]=x["vlm_json"]["NOM_PRENOM"]
    x["COMMUNE_NAISSANCE"]=x["vlm_json"]["COMMUNE_NAISSANCE"]
    x["PAYS_NAISSANCE"]=x["vlm_json"]["PAYS_NAISSANCE"]
    x["COMMUNE_DOMICILE"]=x["vlm_json"]["COMMUNE_DOMICILE"]
    x["CODE_POSTAL_DOMICILE"]=x["vlm_json"]["CODE_POSTAL_DOMICILE"]
    x["ARRONDISSEMENT_DOMICILE"]=x["vlm_json"]["ARRONDISSEMENT_DOMICILE"]
    x["DEPARTEMENT_DOMICILE"]=x["vlm_json"]["DEPARTEMENT_DOMICILE"]
    x["REGION_DOMICILE"]=x["vlm_json"]["REGION_DOMICILE"]
    x["DEPARTEMENT_DOMICILE"]=x["vlm_json"]["DEPARTEMENT_DOMICILE"]
    x["PAYS_DOMICILE"]=x["vlm_json"]["PAYS_DOMICILE"]
    x["NOM_VOIE_DOMICILE"]=x["vlm_json"]["NOM_VOIE_DOMICILE"]
    x["BONNE_PERSONNE"]=x["vlm_json"]["BONNE_PERSONNE"]
    
    return x


def appel_vlm(x):
    prompt_dynamique = generer_instructions_prompt(x)
    imgs, _, _, _ = smart_prepare_media(
        x["pdf_path"], min_dpi=FIXED_MIN_DPI, max_dpi=FIXED_MAX_DPI
            )
    b64_imgs = []
    for img in imgs:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        b64_imgs.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    url = f"http://{API_IP}:{API_PORT}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        }
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in b64_imgs
        ]
    
    content.append({"type": "text", "text": prompt_dynamique})
    
    payload = {
        "model": "Qwen3-VL-32B-Instruct-FP8",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        }
    
    resp = requests.post(url, json=payload, headers=headers, timeout=240)
    resp.raise_for_status()
    
    resultat_json = json.loads(
        resp.json()["choices"][0]["message"]["content"]
        )
        
    x["vlm_json"] = resultat_json
    
    return x


def get_pred_full_vlm(df):
    
    df_output=df.copy()

    df_output=df_output.apply(appel_vlm, axis=1)

    df_output=df_output.apply(get_predictions, axis=1)

    return df_output