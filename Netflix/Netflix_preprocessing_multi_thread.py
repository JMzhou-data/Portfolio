import pandas as pd 
import numpy as np 
import requests 
from datetime import datetime 
import time 
import os
from dotenv import load_dotenv # type: ignore
from tqdm import tqdm # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        return datetime.strptime(date_str, "%d %b %Y").strftime("%Y-%m-%d")
    except:
        return date_str 
    
def query_tmdb(title):
    """ Recherche la date de sortie d'un film via l'API TMDb"""
    try:
        response = requests.get(TMDB_SEARCH_URL, params={"api_key":TMDB_API_KEY, "query": title}).json()
        results = response.get("results")
        if not results:
            return None, None, None
        movie_id = results[0]['id']

        release_response = requests.get(TMDB_RELEASE_URL.format(id=movie_id), params={"api_key": TMDB_API_KEY}).json()
        releases = release_response.get("results", [])
        release_date = None

        for country_data in releases:
            if country_data['iso_3166_1'] in ['US', 'FR', 'GB']:
                for entry in country_data['release_dates']:
                    release_date = entry['release_date'][:10]
                    break
            if release_date:
                break
        
        credits_data = requests.get(TMDB_CREDITS_URL.format(movie_id=movie_id), params={"api_key": TMDB_API_KEY}).json()
        director = next((p['name'] for p in credits_data.get("crew", []) if p['job']=="Director"), None)
        actors_list = [actor['name'] for actor in credits_data.get("cast", [])[:3]]
        actors = ", ".join(actors_list) if actors_list else None

        return release_date, director, actors
    
    except Exception as e:
        print(f"Erreur TMDb pour '{title}' : {e}")
        return None, None, None
    
def clean_netflix_data(df):
    """Nettoyage des colonnes et remplissage des valeurs manquantes"""

    df['Boxoffice_cleaned'] = (
        df['Boxoffice']
        .replace({'Unknown': np.nan, r'^\$0$': np.nan}, regex=True)
        .str.replace(r'[\$,]', '', regex=True)
        .astype(float)
    )

    df.fillna({
        'Hidden Gem Score': -1,
        'Country Availability': "Unknown",
        'Writer': "Unknown",
        'Actors': "Unknown",
        'View Rating': "Unknown",
        'IMDb Score': df['IMDb Score'].mean(),
        'Rotten Tomatoes Score': df['Rotten Tomatoes Score'].mean(),
        'Metacritic Score': df['Metacritic Score'].mean(),
        'Awards Received': 0,
        'Awards Nominated For': 0,
        'Boxoffice': "$0",
        'Production House': "Unknown",
        'IMDb Votes': df['IMDb Votes'].mean(),
        'Trailer Site': "Unknown"
    }, inplace=True)

    df['Release Date'] = df['Release Date'].apply(convert_date_format)

    return df

def fetch_tmdb_info(row):
    """Fonction utilitaire pour multi-threading"""
    missing_release = pd.isna(row['Release Date'])
    missing_director = pd.isna(row['Director']) or row['Director'] == "Unknown"
    missing_actors = pd.isna(row['Actors']) or row['Actors'] == "Unknown"
    
    if missing_release or missing_director or missing_actors:
        release, director, actors = query_tmdb(row['Title'])
        return (row.name, release, director, actors)
    else:
        return (row.name, None, None, None)

def enrich_missing_data(df, max_workers=5):
    """Multi-threaded enrichment avec barre de progression"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_tmdb_info, row): idx for idx, row in df.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Enrichissement TMDb (multi-thread)"):
            idx, release, director, actors = future.result()
            if idx is None:
                continue

            if pd.isna(df.at[idx, 'Release Date']) and release:
                df.at[idx, 'Release Date'] = release
            if (pd.isna(df.at[idx, 'Director']) or df.at[idx, 'Director'] == "Unknown") and director:
                df.at[idx, 'Director'] = director
            if (pd.isna(df.at[idx, 'Actors']) or df.at[idx, 'Actors'] == "Unknown") and actors:
                df.at[idx, 'Actors'] = actors

            time.sleep(0.2)  # Respecter limite API

    return df

# ----------
# Main
# ----------

if __name__=="__main__":
    start_time = time.time()

    input_file = "../../data/netflix-rotten-tomatoes-metacritic-imdb.csv"
    output_file = "../../data/netflix_data.csv"

    df = pd.read_csv(input_file)
    df = clean_netflix_data(df)
    df = enrich_missing_data(df, max_workers=5)

    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    end_time = time.time() # 52.79 minutes
    print("\n Dataset nettoyé et sauvegardé sous:", output_file)
    print(df.info())
    print(f"\nTemps d'exécution total : {(end_time - start_time)/60:.2f} minutes")
