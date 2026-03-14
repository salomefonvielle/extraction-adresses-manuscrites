from tqdm import tqdm
from code.utils.second_step_scores.scores_calcul import clean_text, convert_dates, get_indexes_of_keywords, rate_indexes_on_text
from code.utils.second_step_scores.scores_config import DIST_ADRESSE
import pandas as pd
import numpy as np


#On récupère le score de chauqe paragraphe océrisé en utilisant le csv résultant de l'OCR et le csv initial avec les information de la personne d'intérêt 

def get_all_scores(df, df_text):
    """
    Identifie tous les paragaphes importants des textes d'un DataFrame et leur associe un score. 
    Plus le score est haut et plus il y a de chance que le paragraphe contienne l'adresse à extraire.
    
    Args:
        df (pandas.DataFrame): le DataFrame contenant les anomalies
        df_text (pandas.DataFrame): le DataFrame contenant les textes océrisés associés. Dans ce dataFrame une ligne corespond à une page.On peut avoir les pages de plusieurs doc atlas.
    
    Returns:
        pandas.DataFrame: le DataFrame contenant tous les scores
    """
    all_values = []

  #On itère sur chaque id atlas
   
    for i, row in tqdm(df.iterrows(), total=len(df)):

       #Dans le csv des anomalies on récupère les données personnelles et on nettoie 
        
        idatlas = row["doc_idatlas"]
        prenoms = clean_text(row["ppd_lipren"], upper=True).split()[:3] + [""]*3
        personal_data = {
            "NOM": clean_text(row["ppd_lnnom"], upper=True),
            "PRENOM_0": prenoms[0],
            "PRENOM_1": prenoms[1],
            "PRENOM_2": prenoms[2],
            "DATE_NAISSANCE": row["ppd_dnnai"]
        }
        
       #On extrait du texte océrisé les pages qui correspondent à l'Id Atlas étudié 
        text_rows = df_text.groupby("doc_idatlas")[['doc_idatlas', 'max_page_number', 'text']].apply(lambda g: g[g['doc_idatlas'] == idatlas]) # get all the pages for this specific idatlas
        if text_rows.empty:
            continue
        max_page_number = text_rows["max_page_number"].values[0]
        
       #identifier la position des paragraphes  
        text = ""
        cur_idx = 0
        pages_indexes = []
        for _, text_row in text_rows.iterrows():
            text += text_row["text"] + " "
            cur_idx += len(text_row["text"]) + 1
            pages_indexes.append(cur_idx)

        pages_indexes = np.array(pages_indexes) 
        
        #On nettoie 
        
        text = convert_dates(text)

        text_upper = text.upper()
        
        #On récupère les indices pour identifier les paragraphes candidats   
        
        indexes = get_indexes_of_keywords(text_upper)
        
        #On récupère les scores pour tous les candidats  
        scores = rate_indexes_on_text(text_upper, indexes, personal_data) 
        
        #On récupère les scores pour tous les candidats  
        for j, index in enumerate(indexes):
            #On rend un résultat pour chaque paragraphe identifié comme candidat
            #recherche du numéro de page du candidat  
            page_number = np.searchsorted(pages_indexes, index)

            # Pour cahque candidat on construit un tableau avec tous les infos associés à la personn ( ligne csv anomalies) et on ajoute les scores 
            cur_values = [row["index"], row["doc_idatlas"],row["doc_ndo_conat"],row["ppd_lipren"],row["ppd_lnnom"],
                          row["ppd_dnnai"],row["ppd_cosages"],row["ppd_refdoc"],row["ppd_nuspi"],row["ppd_lncommu"],row["ppd_adr_copos"],
                          row["ppd_adr_lnpays"],row["ppd_adr_copays"],row["ppd_adr_codep"],row["ppd_adr_cocommu"],
                          text[index: index + DIST_ADRESSE], page_number, max_page_number] + scores[j]
            
            all_values.append(cur_values)
                          
                          
    df_out = pd.DataFrame(all_values, columns=["index", "doc_idatlas", "doc_ndo_conat", "ppd_lipren",
                                               "ppd_lnnom", "ppd_dnnai", "ppd_cosages", "ppd_refdoc",
                                               "ppd_nuspi", "ppd_lncommu", "ppd_adr_copos", "ppd_adr_lnpays", "ppd_adr_copays",
                                               "ppd_adr_codep", "ppd_adr_cocommu", "text", "page_number", "max_page_number",
                                               "score_lipren_0", "score_lipren_1", "score_lipren_2",
                                               "score_lnnom", "score_demeur", "score_bad_words", "score_dnnai", "score_copos"])
    
    return df_out

#On ne sélecionne que le passage avec le meilleure score par personne/ id atalas

def get_best_scores(df):
    """
    Garde uniquement un paragraphe par document ayant le score le plus élevé
    
    Args:
        df (pandas.DataFrame): le DataFrame contenant tous les scores
    
    Returns:
        pandas.DataFrame: le DataFrame contenant un score par document
    """
    def compute_score(x):
        return 1.3 * x["score_lipren_0"] + 1.1 * x["score_lipren_1"] + \
            x["score_lipren_2"] + x["score_lnnom"] + x["score_demeur"] + \
            x["score_dnnai"] + x["score_copos"] + x["score_bad_words"]
    
    df_out = df.copy()
    df_out["best_score"]=df_out.apply(compute_score, axis=1)
    df_out=df_out.loc[df_out.groupby("doc_idatlas")["best_score"].idxmax()]

    return df_out 


#Pour executer en une fois les deux get_all_scores et get_best_scores
def compute_best_scores(df_src: pd.DataFrame, df_ocr: pd.DataFrame)-> pd.DataFrame:
    from code.utils.second_step_scores.scores_attribution import get_all_scores, get_best_scores
    df_scores = get_all_scores(df_src, df_ocr)
    df_best_scores=get_best_scores(df_scores)
    return df_best_scores

