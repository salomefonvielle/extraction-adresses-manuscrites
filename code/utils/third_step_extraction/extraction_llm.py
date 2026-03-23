import concurrent.futures
from code.utils.second_step_scores.scores_config import JSON_PATTERN, REQUIRED_KEYS
from openai import OpenAI
import os
import pandas as pd 
from json import JSONDecoder
from dotenv import load_dotenv


 
## Client PIA 
load_dotenv(override=True) 

no_proxy = os.getenv("NO_PROXY", "")
if no_proxy:
    os.environ["no_proxy"] = no_proxy


API_KEY = os.getenv("PIA_API_KEY")
serveur_url = os.getenv("serveur_url")
timeout = 90
NOM_MODEL = os.getenv("PIA_LLM_MODEL")

client_PIA = OpenAI(
    base_url=serveur_url,
    api_key=API_KEY,
    timeout=timeout,
)


#Les fonctions d'appel au LLM n'ont pas été modifiée

def llama_prompt(nom_prenom, texte):
    """
    Créé un prompt LLM (adapté pour llama3) pour extraire l'adresse d'une personne à partir d'un paragraphe
    
    Args:
        nom_prenom (str): Le nom et prénom de la personne.
        texte (str): Le paragraphe contenant l'adresse.
    
    Returns:
        str: Le prompt
    """
    
    aide_pr_1 = "Il s'agit d'extraire d'un texte l'adresse du domicile d'une personne P dont on connait le nom et le prénom. Un code postal est obligatoirement formé de 5 chiffres."
    prompt = f"""{aide_pr_1}

    A partir du texte, identifie la bonne personne et détermine son adresse de domicile, c'est-à-dire l'adresse où il demeure. Ta réponse sera un json dont les clés : valeurs seront les suivantes :
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
    
    Essaye de déduire le pays de l'adresse du domicile lorsque celui-ci n'est pas indiqué.
    Voici le nom et prénom de la personne P : {nom_prenom}
    Voici le texte : {texte}
    Ta réponse sera sans commentaire ni explication et sera uniquement un json.
    """
    
    return prompt

def fetch_completion(prompt):
    try:
        response = client_PIA.chat.completions.create(
            model=NOM_MODEL,
            messages=[
        {"role": "system", "content": "you are an assistant"},
        {"role": "user", "content": prompt}
        ]
        )
        return response.model_dump()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"
    

def get_llama_ans(prompts):
    """
    Pareil que celle au dessus mais test avec une liste d'inférence.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map each prompt to the fetch_completion function in parallel
        results = list(executor.map(fetch_completion, prompts))
        print("results :  ------")
        print(results)
    return results


def extract_json_from_text(text, decoder=JSONDecoder()):
    """
    Extrait le dernier JSON contenu dans un texte et s'assure qu'une liste de clé y figure en leur donnant une valeur par défaut.
    Voir la liste de clés REQUIRED_KEYS dans extraction_config.py
    
    Args:
        text (str): Le texte à parser.
    
    Returns:
        dict: Le JSON sous forme de dictionnaire
    """
    match = text.rfind('{') #last occurence
    dic = {}
    if match != -1:
        try:
            dic, _ = decoder.raw_decode(text[match:])
        except ValueError:
            print("Parsing error: ", text)
            pass
        
    for k in REQUIRED_KEYS:
        if k not in dic.keys():
            dic[k] = ""
    return dic



def get_prompts(x):
    nom_prenom = x["ppd_lnnom"] + " " + x["ppd_lipren"]
    x["prompt"] = llama_prompt(nom_prenom, x["text"])
    return x
    
def get_fetch_completion(x):
    x["llma_ans"] = fetch_completion(x["prompt"])
    return x
    
def get_extract_json_from_text(x) : 
    x["llma_json"] = extract_json_from_text(x["llma_ans"])
    return x
    
def get_predictions(x) :
    x["NOM_PRENOM"]=x["llma_json"]["NOM_PRENOM"]
    x["COMMUNE_NAISSANCE"]=x["llma_json"]["COMMUNE_NAISSANCE"]
    x["PAYS_NAISSANCE"]=x["llma_json"]["PAYS_NAISSANCE"]
    x["COMMUNE_DOMICILE"]=x["llma_json"]["COMMUNE_DOMICILE"]
    x["CODE_POSTAL_DOMICILE"]=x["llma_json"]["CODE_POSTAL_DOMICILE"]
    x["ARRONDISSEMENT_DOMICILE"]=x["llma_json"]["ARRONDISSEMENT_DOMICILE"]
    x["DEPARTEMENT_DOMICILE"]=x["llma_json"]["DEPARTEMENT_DOMICILE"]
    x["REGION_DOMICILE"]=x["llma_json"]["REGION_DOMICILE"]
    x["DEPARTEMENT_DOMICILE"]=x["llma_json"]["DEPARTEMENT_DOMICILE"]
    x["PAYS_DOMICILE"]=x["llma_json"]["PAYS_DOMICILE"]
    x["NOM_VOIE_DOMICILE"]=x["llma_json"]["NOM_VOIE_DOMICILE"]
    x["BONNE_PERSONNE"]=x["llma_json"]["BONNE_PERSONNE"]
    
    return x

#Extraire les données de ce paragraphe
def get_pred(df, df_in, extraction_output_path, append=False):
    """
    Utilise un LLM pour extraire l'adresse de chaque paragraphe ayant le plus haut score

    Args:
        df (pandas.DataFrame): le DataFrame contenant les plus haut scores
        df_in (pandas.DataFrame): le DataFrame source (lignes originales du CSV)
        extraction_output_path: chemin du CSV de sortie
        append (bool): si True, écrit en mode ajout (pour le traitement par batch)

    Returns:
        pandas.DataFrame: le DataFrame contenant les prédictions LLM
    """
    from pathlib import Path

    #Pour chaque personne on formule un prompt
    df=df.apply(get_prompts, axis=1)
    df=df.apply(get_fetch_completion, axis=1)
    df=df.apply(get_extract_json_from_text, axis=1)
    df=df.apply(get_predictions, axis=1)

    df_out=df_in.copy()
    df_final=pd.merge(df_out, df[["doc_idatlas","NOM_PRENOM","COMMUNE_NAISSANCE", "PAYS_NAISSANCE", "COMMUNE_DOMICILE","CODE_POSTAL_DOMICILE", "ARRONDISSEMENT_DOMICILE","DEPARTEMENT_DOMICILE","REGION_DOMICILE","DEPARTEMENT_DOMICILE","PAYS_DOMICILE","NOM_VOIE_DOMICILE","BONNE_PERSONNE", "tot_score", "text"]],
                      on="doc_idatlas",
                      how="left" )

    output_path = Path(extraction_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not output_path.is_file()
    df_final.to_csv(output_path, mode="a" if append else "w", index=False, header=write_header)

    return df_final
