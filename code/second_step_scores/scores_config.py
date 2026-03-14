import re
import regex

#On définit ce qu'est un titre de civilité et un mois avec équivalent numérique du mois. 

TITRE_CIVILITE = ["MONSIEUR", "MADAME", "MADEMOISELLE", "COMTESSE", "PRINCESSE", "DOCTEUR", "NOM DE NAISSANCE", 
                  "MME ", "SOUSSIGNE", "CEDANT", "UNE PART", "AUTRE PART", "PROPRIETAIRE", "CONJOINT", "FILS", 
                  "FILLE", "VENDEUR", "ACQUEREUR", r"NOM\s+PRENOM", r" M\.", "MLLE", " MR"]

MONTH_MAPPING = {
    "JANVIER": "01", "FEVRIER": "02", "MARS": "03", "AVRIL": "04", "MAI": "05", 
    "JUIN": "06", "JUILLET": "07", "AOUT": "08", "SEPTEMBRE": "09", 
    "OCTOBRE": "10", "NOVEMBRE": "11", "DECEMBRE": "12"
}

#Tolérance des distances pour reconnaissance des champs
DIST_PRENOMS = 70
DIST_DEMEURA = 400
DIST_ADRESSE = 500
DIST_DATENAI = 500

#Motif regex pour trouver les motifs intéressants
DEMEUR_PATTERN = re.compile(r"(DEMEURANT|DEMEUR|DEMEU|DOMICIL)")
BAD_WORDS_PATTERN = re.compile(r"REPRESENT")
DATENAISS_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
CODEPOS_PATTERN = re.compile(r"(\d{5})")

#Regex pour civilité 
KEYWORD_PATTERN = regex.compile('(' + '|'.join([elt for elt in TITRE_CIVILITE if len(elt) >= 6]) +
                             '){e<=1}' + '|(' + '|'.join([elt for elt in TITRE_CIVILITE if len(elt) < 6]) + ')')

#Regex pour dates
CONVDATE1_PATTERN = regex.compile(r'(\d{1,2})(|ER)\s(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre){e<=3}\s(\d{4})', regex.IGNORECASE) # 1ER JUILLET 1992
CONVDATE2_PATTERN = re.compile(r'(\d{1,2})\s*(?:\/|-|\.)\s*(\d{1,2})\s*(?:\/|-|\.)\s*(\d{4})') # 01/07/1992

#Capture d'un JSON avec les champs avec ???
JSON_PATTERN = re.compile(r"{[^\"{}]*(?:\"[^\"]*\"[^\"{}]*[^\"{}]*)*}")
REQUIRED_KEYS = ["CODE_POSTAL_DOMICILE", "PAYS_DOMICILE", "COMMUNE_DOMICILE", "DEPARTEMENT_DOMICILE", "ARRONDISSEMENT_DOMICILE", "BONNE_PERSONNE"]