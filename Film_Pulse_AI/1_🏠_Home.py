import streamlit as st
import pandas as pd
import gzip
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Configuration de la page
st.set_page_config(page_title="Film Pulse AI", page_icon=":movie_camera:", layout="wide", initial_sidebar_state="expanded")
with st.sidebar:
  st.image("/Users/wilders/Desktop/Film_Pulse_AI/logo_film_pulse_ai.png", width=280)

st.header("Film Pulse AI")

# Base lien pour les posters de films de la base TMDB
base_poster = "https://image.tmdb.org/t/p/w600_and_h900_bestv2"

# Lien de notre dataset de films
link = r"C:\Users\wilders\Desktop\Film_Pulse_AI\df_film_pulse_AI.csv"

# Importation du dataset

@st.cache_data
def load_data():
    # Chargez votre DataFrame ici
    df_film_pulse_AI = pd.read_csv(link, header=0, low_memory=False)
    return df_film_pulse_AI
 
df_film_pulse_AI = load_data()

colonne_titre = st.selectbox('Sélectionnez le titre de votre film :', df_film_pulse_AI["Titre"])
df_filtre_film = df_film_pulse_AI[df_film_pulse_AI["Titre"] == colonne_titre]

#colonne_param = st.selectbox('Sélectionnez la colonne à filtrer', df_film_pulse_AI.select_dtypes("number").columns)
#valeur_min, valeur_max = st.slider('Sélectionnez la plage de valeurs', min(df_film_pulse_AI[colonne_param]), max(df_film_pulse_AI[colonne_param]), (min(df_film_pulse_AI[colonne_param]), max(df_film_pulse_AI[colonne_param])))
#df_filtre_param = df_film_pulse_AI[(df_film_pulse_AI[colonne_param] >= valeur_min) & (df_film_pulse_AI[colonne_param] <= valeur_max)]

# st.dataframe(df_filtre_film[["Titre", "Annee_de_sortie", "Duree", "Genre", "Realisateur", "Note","Nombre_de_votants", "Acteur"]], height=300)


import streamlit as st

# Assuming df_filtre_film is already defined and populated

for index, row in df_filtre_film.iterrows():
    image_path = base_poster + row['Affiche']
    
    # Créer deux colonnes
    col_affiche, col_details = st.columns([1, 2])  # Une colonne pour l'affiche, deux pour les détails
    
    # Afficher l'affiche dans la première colonne
    with col_affiche:
        st.image(image_path, caption=row['Titre'], width=300)
    
    # Afficher les détails dans la deuxième colonne
    with col_details:
        st.write("Titre:", row['Titre'])
        st.write("Année de sortie:", row['Annee_de_sortie'])
        st.write("Durée:", row['Duree'])
        st.write("Genre:", row['Genre'])
        st.write("Réalisateur:", row['Realisateur'])
        st.write("Acteurs:", row['Acteur'])
        st.write("Note:", row['Note'])
        st.write("Nombre de votants:", row['Nombre_de_votants'])

# Partie ML

from sklearn.preprocessing import StandardScaler

df_film_pulse_AI['genre_fct'] = df_film_pulse_AI['Genre'].factorize()[0]
df_film_pulse_AI["realisateur_fct"] =  df_film_pulse_AI['Realisateur'].factorize()[0]

# Supposons X contient vos données avec les colonnes pertinentes
X = df_film_pulse_AI[['genre_fct', 'realisateur_fct', "Note", "Nombre_de_votants"]]

# Normalisation des données
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Créer un objet NearestNeighbors avec la distance cosinus et les données normalisées
distanceKNN = NearestNeighbors(n_neighbors=4, metric='cosine').fit(X_normalized)

@st.cache_resource
def find(name):
    # Extraire les caractéristiques du film donné pour la recherche de voisins
    film_features = df_film_pulse_AI.loc[df_film_pulse_AI['Titre'] == name, ['genre_fct', 'realisateur_fct', "Note", "Nombre_de_votants"]]
    
    # Normaliser les caractéristiques du film donné
    film_features_normalized = scaler.transform(film_features)
    
    # Trouver les voisins du film donné en utilisant la distance cosinus et les données normalisées
    _, neighbors = distanceKNN.kneighbors(film_features_normalized)
    
    # Exclure le film lui-même des voisins
    neighbors = neighbors[0][1:]
    
    # Initialiser une liste de films prédits
    predicted_movies = []
    
    # Ajouter les trois films prédits à la liste
    for neighbor_index in neighbors[:3]:  # Sélectionner les trois premiers voisins
        neighbor = df_film_pulse_AI.iloc[neighbor_index]
        predicted_movies.append(neighbor)
    
    # Retourner la liste des trois films prédits
    return predicted_movies

# Appel de la fonction pour trouver les trois films prédits pour un film donné
predicted_films = find(colonne_titre)

# Stockage des trois films prédits dans des variables distinctes (film1, film2, film3)
if len(predicted_films) >= 3:
    film1, film2, film3 = predicted_films[:3]

else:
    print("Il n'y a pas assez de films prédits.")
    
# Créer 2 colonnes pour chaque film
col_affiche1,col_details1,col_affiche2,col_details2,col_affiche3,col_details3= st.columns([1, 2, 1, 2, 1, 2])
    
# Afficher l'affiche dans la première colonne
with col_affiche1:
    st.image(base_poster + film1[8], caption=film1[0], width=150)
    
# Afficher les détails dans la deuxième colonne
with col_details1:
    st.write("Titre:", film1[0])
    st.write("Année de sortie:", film1[1])
    st.write("Durée:", film1[2])
    st.write("Genre:", film1[3])
    st.write("Réalisateur:", film1[4])
    st.write("Acteurs:", film1[7])
    st.write("Note:", film1[5])
    st.write("Nombre de votants:", film1[6])

# Afficher l'affiche dans la première colonne
with col_affiche2:
    st.image(base_poster + film2[8], caption=film2[0], width=150)
    
# Afficher les détails dans la deuxième colonne
with col_details2:
    st.write("Titre:", film2[0])
    st.write("Année de sortie:", film2[1])
    st.write("Durée:", film2[2])
    st.write("Genre:", film2[3])
    st.write("Réalisateur:", film2[4])
    st.write("Acteurs:", film2[7])
    st.write("Note:", film2[5])
    st.write("Nombre de votants:", film2[6])

# Afficher l'affiche dans la première colonne
with col_affiche3:
    st.image(base_poster + film3[8], caption=film3[0], width=150)
    
# Afficher les détails dans la deuxième colonne
with col_details3:
    st.write("Titre:", film3[0])
    st.write("Année de sortie:", film3[1])
    st.write("Durée:", film3[2])
    st.write("Genre:", film3[3])
    st.write("Réalisateur:", film3[4])
    st.write("Acteurs:", film3[7])
    st.write("Note:", film3[5])
    st.write("Nombre de votants:", film3[6])




