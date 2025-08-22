""" 
Prétraitement du dataset Netflix 
Nettoie les valeurs manquantes, convertit les formats de date et enrichit les données à l'aide de l'API TMDb
"""

import pandas as pd 
import numpy as np 
import requests 
from datetime import datetime 
import time 
import os
from dotenv import load_dotenv # type: ignore
from tqdm import tqdm # type: ignore

# ----------
# Configuration
# ----------

load_dotenv() # Charge le fichier .env
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_RELEASE_URL = "https://api.themoviedb.org/3/movie/{id}/release_dates"
TMDB_CREDITS_URL = "https://api.themoviedb.org/3/movie/{movie_id}/credits"

# ----------
# Fonctions 
# ----------

def convert_date_format(date_str):
    """ Convertit date du format '18 Feb 2009' -> '2009-02-18' """
    try:
        date_str = date_str.strip().title() # '17 jul 2020' -> '17 Jul 2020'
        return datetime.strptime(date_str, "%d %b %Y").date()
    except:
        return None 
    
def query_tmdb(title):
    """ Recherche la date de sortie d'un film via l'API TMDb"""
    try:
        # Step 1 : recherche du film
        response = requests.get(TMDB_SEARCH_URL, params={"api_key":TMDB_API_KEY, "query": title}).json()
        results = response.get("results")
        if not results:
            return None, None, None
        movie_id = results[0]['id']

        # Step 2 : date de sortie
        release_response = requests.get(TMDB_RELEASE_URL.format(id=movie_id), params={"api_key": TMDB_API_KEY}).json()
        releases = release_response.get("results", [])
        release_date = None

        for country_data in releases:
            if country_data['iso_3166_1'] in ['US', 'FR', 'GB']: # On regarde sortie US puis FR puis GB
                for entry in country_data['release_dates']:
                    release_date = entry['release_date'][:10]
                    break
            if release_date:
                break
        
        # Step 3 : Réalisateur et acteurs
        credits_data = requests.get(TMDB_CREDITS_URL.format(movie_id=movie_id), params={"api_key": TMDB_API_KEY}).json()
        director = next((p['name'] for p in credits_data.get("crew", []) if p['job']=="Director"), None)
        actors_list = [actor['name'] for actor in credits_data.get("cast", [])[:3]]
        actors = ", ".join(actors_list) if actors_list else None

        return release_date, director, actors
    
    except Exception as e:
        print(f"Erreur TMDb pour '{title}' : {e}")
        return None, None, None
    
# ----------
# Traitement
# ----------

def clean_netflix_data(df):
    """Nettoyage des colonnes et remplissage des valeurs manquantes"""

    df['Boxoffice_cleaned'] = (
        df['Boxoffice']
        .replace({'Unknown': np.nan, r'^\$0$': np.nan}, regex=True)
        .str.replace(r'[\$,]', '', regex=True)
        .astype(float)
    )

    df.fillna({
        'Hidden Gem Score': df['Hidden Gem Score'].mean(),
        'Genre': "Unknown",
        'Tags': "Unknown",
        'Languages': "Unknown",
        'Country Availability': "Unknown",
        'Runtime': "30-60 mins", # 1 seule valeur manquante : film W1A
        'Director': "Unknown",
        'Writer': "Unknown",
        'Actors': "Unknown",
        'View Rating': "Unknown",
        'IMDb Score': df['IMDb Score'].mean(),
        'Rotten Tomatoes Score': df['Rotten Tomatoes Score'].mean(),
        'Metacritic Score': df['Metacritic Score'].mean(),
        'Awards Received': 0,
        'Awards Nominated For': 0,
        'Boxoffice_cleaned': df['Boxoffice_cleaned'].mean(),
        'Production House': "Unknown",
        'IMDb Votes': df['IMDb Votes'].mean(),
        'Trailer Site': "Unknown",
        'Release Date': pd.to_datetime("1900-01-01").date(),
        'Summary': "Unknown"
    }, inplace=True)

    # Conversion de date
    df['Release Date'] = df['Release Date'].apply(convert_date_format)

    # Suppression de colonnes
    df.drop(columns=['Netflix Link', 'IMDb Link', 'TMDb Trailer', 'Boxoffice'], inplace=True)

    return df

def enrich_missing_data(df):
    """Appel TMDb pour compléter les dates manquantes"""
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enrichissement TMDb"):
        missing_release = pd.isna(row['Release Date'])
        missing_director = pd.isna(row['Director']) or row['Director'] == "Unknown"
        missing_actors = pd.isna(row['Actors']) or row['Actors'] == "Unknown"

        if missing_release or missing_director or missing_actors:
            release, director, actors = query_tmdb(row['Title'])
        
            if missing_release and release:
                df.at[idx, 'Release Date'] = release
            if missing_director and director:
                df.at[idx, 'Director'] = director 
            if missing_actors and actors:
                df.at[idx, 'Actors'] = actors 

            time.sleep(0.2)  # éviter surcharge API

    return df

# ----------
# Main
# ----------

if __name__=="__main__":
    start_time = time.time()

    input_file = "../../data/netflix-rotten-tomatoes-metacritic-imdb.csv"
    output_file = "../../data/netflix_data.csv"

    df = pd.read_csv(input_file)
    df = enrich_missing_data(df)
    df = clean_netflix_data(df)


    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    end_time = time.time() # 37.06 minutes
    print("\n Dataset nettoyé et sauvegardé sous:", output_file)
    print(df.info())
    print(f"\nTemps d'exécution total : {(end_time - start_time)/60:.2f} minutes")
