import re
import unicodedata
from rapidfuzz import fuzz, process
import regex
from datetime import date

# Import des constantes et expressions régulières définies dans le module de configuration
from code.utils.second_step_scores.scores_config import (
    MONTH_MAPPING,
    CONVDATE1_PATTERN,
    CONVDATE2_PATTERN,
    KEYWORD_PATTERN,
    DEMEUR_PATTERN,
    BAD_WORDS_PATTERN,
    DIST_DEMEURA,
    DATENAISS_PATTERN,
    CODEPOS_PATTERN,
    DIST_PRENOMS,
    DIST_DATENAI,
    DIST_ADRESSE,
)

# ----------------------------------------------------------------------
# I-   Mise en forme du texte résultant de l'OCR
# ----------------------------------------------------------------------
def clean_text(input_str, upper=False):
    """
    Nettoie un texte en supprimant les accents et cédilles, remplace les tirets,
    double sauts de ligne par des espaces, convertit le texte en majuscules si demandé.
    
    Args:
        input_str (str): Le texte à nettoyer.
        upper (bool): Si ``True``, le texte est converti en majuscules.
    
    Returns:
        str: Le texte nettoyé.
    """
    # Normalise le texte en forme NFKD → décompose les caractères accentués
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    # Retire les diacritiques (accents) en ne gardant que les caractères non combinants
    res = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Optionnel : conversion en majuscules
    if upper:
        res = res.upper()

    # Remplace les tirets par des espaces pour éviter la concaténation de mots
    res = re.sub("-", " ", res)

    # Remplace les doubles sauts de ligne (ou plus) par un espace unique
    res = re.sub("\n\n", " ", res)

    # Normalise les abréviations « st » en « saint » (ex. « St Pierre » → « Saint Pierre »)
    res = re.sub(r"(?:^|\b)(?:st)(\b|E\b)", r"saint\1", res, flags=re.IGNORECASE)

    return res


# ----------------------------------------------------------------------
# II-  Conversion des dates au format ISO (YYYY‑MM‑DD)
# ----------------------------------------------------------------------
def convert_dates(text):
    """
    Remplace les dates présentes dans ``text`` par le format standard ISO.
    
    La fonction gère plusieurs formats courants, par exemple :
    - ``1er Novembre 1992``
    - ``01/02/2004``
    - ``03.10.1965``
    
    Args:
        text (str): Le texte à traiter.
    
    Returns:
        str: Le texte où les dates ont été reformattées.
    """
    # ------------------------------------------------------------------
    # Helper 1 – Dates du type « 1er Novembre 1992 » (mot‑month‑year)
    # ------------------------------------------------------------------
    def replace_date_1(match):
        day, er, month_text, year = match.groups()
        # Recherche la version canonique du mois (ex. « NOVEMBRE » → « 11 ») parmi le mapping
        month_text = process.extractOne(month_text.upper(), MONTH_MAPPING.keys())[0]
        month = MONTH_MAPPING[month_text]
        # Retourne la date au format ISO, le jour étant z‑filled à deux caractères
        return f"{year}-{month}-{day.zfill(2)}"

    # ------------------------------------------------------------------
    # Helper 2 – Dates du type « 01/02/2004 » ou « 03.10.1965 » (d/m/y)
    # ------------------------------------------------------------------
    def replace_date_2(match):
        d, m, y = match.groups()
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Applique les deux patterns sur le texte
    text = CONVDATE1_PATTERN.sub(replace_date_1, text)
    text = CONVDATE2_PATTERN.sub(replace_date_2, text)

    return text


# ----------------------------------------------------------------------
# III- Détermination des paragraphes candidats à l'extraction
# ----------------------------------------------------------------------
def get_indexes_of_keywords(text):
    """
    Retourne les positions (indices) où apparaissent les mots‑clés définis dans
    ``KEYWORD_PATTERN``. Ces mots‑clés correspondent généralement aux titres de
    sections civiles (ex. « NOM », « PRÉNOM », etc.).
    
    Args:
        text (str): Le texte à analyser.
    
    Returns:
        list[int]: Liste des indices de début de chaque occurrence.
    """
    idx = [m.start() for m in KEYWORD_PATTERN.finditer(text)]
    return idx


# ----------------------------------------------------------------------
# IV- Attribution de scores afin de déterminer le meilleur candidat
# ----------------------------------------------------------------------
# 1. Recherche d’une adresse (ou d’un indice de localisation)
def match_demeur(text):
    """
    Recherche les mots clés « DEMEURANT » ou « DOMICILIE » dans ``text`` à l’aide du
    fuzzy matching de RapidFuzz.
    
    Args:
        text (str): Le texte à analyser.
    
    Returns:
        tuple[float, int]: (score entre 0‑1, indice du mot trouvé) ;
                           retourne (0, 0) si aucun mot n’est détecté.
    """
    all_match = [(m.group(0), m.start()) for m in DEMEUR_PATTERN.finditer(text)]
    if all_match:
        # Conserve le plus long parmi les correspondances (probablement le plus pertinent)
        match = max(all_match, key=lambda x: len(x[0]))
        # Score fuzzy contre les deux formes possibles
        score = max(fuzz.ratio(match[0], "DEMEURANT"),
                    fuzz.ratio(match[0], "DOMICILIE")) / 100
        return score, match[1]
    return 0, 0


def match_code_postal(text):
    """
    Recherche les codes postaux dans ``text``. Le score décroît linéairement
    avec la distance (indice) à laquelle le code a été trouvé.
    
    Args:
        text (str): Le texte à analyser.
    
    Returns:
        float: Score entre 0 et 1 (0 → non trouvé, 1 → trouvé très tôt).
    """
    match = [m.start() for m in CODEPOS_PATTERN.finditer(text)]
    if match:
        best_match = min(match)               # indice le plus petit (première occurrence)
        return 1 - best_match / 750           # normalisation empirique
    else:
        return 0


# 2. Vérification que le texte concerne bien la personne ciblée
def match_bad_words(text):
    """
    Détecte les mots indésirables (ex. « décès », « pension », etc.) qui, s’ils
    apparaissent, réduisent le score global. Le score est négatif et dépend de
    la position du premier mot trouvé.
    
    Args:
        text (str): Le texte à analyser.
    
    Returns:
        float: Valeur négative croissante avec l’indice du premier mot trouvé,
               ou 0 si aucun mot n’est présent.
    """
    all_match = [m.start() for m in BAD_WORDS_PATTERN.finditer(text)]
    if all_match:
        return -1 + all_match[0] / DIST_DEMEURA
    return 0


def match_prenoms_nom(text, data):
    """
    Vérifie la présence (avec tolérance d’une erreur) des prénoms et du nom
    fournis dans le dictionnaire ``data``.
    
    Args:
        text (str): Le texte à analyser.
        data (dict): Contient les clés ``PRENOM_0``, ``PRENOM_1``, ``PRENOM_2`` et ``NOM``.
    
    Returns:
        list[bool]: [prenom0_present, prenom1_present, prenom2_present, nom_present]
    """
    match_prenom_0 = bool(regex.search(' (' + data["PRENOM_0"] + '){e<=1}( |,|\\))', text)) if data["PRENOM_0"] else False
    match_prenom_1 = bool(regex.search(' (' + data["PRENOM_1"] + '){e<=1}( |,|\\))', text)) if data["PRENOM_1"] else False
    match_prenom_2 = bool(regex.search(' (' + data["PRENOM_2"] + '){e<=1}( |,|\\))', text)) if data["PRENOM_2"] else False
    match_nom      = bool(regex.search(' (' + data["NOM"] + '){e<=1}( |,|\\))', text)) if data["NOM"] else False
    return [match_prenom_0, match_prenom_1, match_prenom_2, match_nom]


def match_date_naiss(text, data):
    """
    Recherche une date de naissance proche de celle fournie dans ``data``.
    Le score diminue avec la distance de la position et, le cas échéant,
    avec l’écart de jours entre les deux dates.
    
    Args:
        text (str): Le texte à analyser.
        data (dict): Doit contenir la clé ``DATE_NAISSANCE`` au format ``YYYY-MM-DD``.
    
    Returns:
        float: Score >0 si une date plausible est trouvée, sinon 0.
    """
    real_date = data["DATE_NAISSANCE"].split('-')
    try:
        real_date = date(int(real_date[0]), int(real_date[1]), int(real_date[2]))
    except ValueError:
        return 0

    # Extraction de toutes les dates potentielles détectées par le pattern
    matches = [( (m.group(1), m.group(2), m.group(3)), m.start() )
               for m in DATENAISS_PATTERN.finditer(text)]

    for (elt, idx) in matches:
        # Comparaison fuzzy du format texte avec la date attendue
        fuzz_score = fuzz.ratio('-'.join([elt[0], elt[1], elt[2]]), data["DATE_NAISSANCE"])
        if fuzz_score >= 90:                     # tolérance d’une erreur sur les chiffres
            return 1.2 - idx / 1000 - (1 - fuzz_score / 100)

        # Vérification exacte de la date (si le format est correct)
        try:
            match_date = date(int(elt[0]), int(elt[1]), int(elt[2]))
        except ValueError:
            continue

        diff = abs(real_date - match_date).days
        if diff <= 3:                            # différence de moins de 3 jours acceptée
            return 1.2 - idx / 1000 - diff / 30

    return 0


# ----------------------------------------------------------------------
# V- Calcul des scores pour chaque segment de texte identifié
# ----------------------------------------------------------------------
def rate_indexes_on_text(text, indexes, data):
    """
    Calcule, pour chaque indice fourni, une série de scores basés sur les
    différentes fonctions de matching définies ci‑dessus.
    
    Args:
        text (str):   Le texte complet.
        indexes (list[int]): Indices (début de chaque segment) à évaluer.
        data (dict):   Dictionnaire contenant les informations de la personne
                       (prénoms, nom, date de naissance, etc.).
    
    Returns:
        list[list[float]]: Tableau où chaque sous‑liste contient les scores
                           associés à un indice donné.
    """
    scores = []
    for idx in indexes:
        # Portion de texte autour du nom/prénoms (largeur = longueur totale des champs + marge)
        text_prenoms_nom = text[
            idx : idx + len(data["PRENOM_0"]) + len(data["PRENOM_1"]) +
                  len(data["PRENOM_2"]) + len(data["NOM"]) + DIST_PRENOMS
        ]
        # Vérification des prénoms et du nom
        match = match_prenoms_nom(text_prenoms_nom, data)
        s = [1 if m else 0 for m in match]     # 1 = trouvé, 0 = absent

        # Extraction d’un fragment dédié à la localisation (adresse/demeure)
        text_demeur = text[idx: idx + DIST_DEMEURA]
        sco, idx_demeur = match_demeur(text_demeur)
        s.append(sco)                           # Score fuzzy « demeurant »
        s.append(match_bad_words(text_demeur))  # Penalité éventuelle via mots « bad »

        # Recherche de la date de naissance proche de l’indice
        text_date_naiss = text[idx: idx + DIST_DATENAI]
        s.append(match_date_naiss(text_date_naiss, data))

        # Recherche du code postal dans un voisinage de l’adresse
        text_code_postal = text[idx: idx + DIST_ADRESSE]
        s.append(match_code_postal(text_code_postal))

        scores.append(s)

    return scores