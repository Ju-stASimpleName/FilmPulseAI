import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import base64

# Configuration de la page
st.set_page_config(page_title="Film Pulse AI - KPIs BDD", page_icon=":brain:", layout="wide", initial_sidebar_state="expanded")

# Affichage du logo 

col1, col2, col3 = st.columns([2,1,2])
with col1:
    st.write("")
with col2:
    image = Image.open('Film_Pulse_AI_KPI/Logo_film_Pulse_AI.PNG')
    st.image(image, width=300, use_column_width=False)
with col3:
    st.write("")

# Titre de la page
st.markdown("<div style='text-align: center;'><h1>KPIs : compréhension de votre base de données IMDB/TMBD filtrée</h1></div>", unsafe_allow_html=True)

# Importation des data sets
link_df_FilmPulseAI = r"Film_Pulse_AI_KPI/df_Pulse_AI.csv"
link_df_acteurs = r"Film_Pulse_AI_KPI/df_acteurs.csv"

# Lecture du data set
df_FilmPulseAI = pd.read_csv(link_df_FilmPulseAI)
df_acteurs = pd.read_csv(link_df_acteurs)

# Supression des colonnes superflues 
df_FilmPulseAI.drop(['Affiche'], axis=1, inplace=True)
# df_FilmPulseAI.drop(['Unnamed: 0'], axis=1, inplace=True)

# Filtrage sur la base de données :

# Sélectionner la colonne pour le filtre
selected_column = st.selectbox("Sélectionner la colonne", df_FilmPulseAI.columns)

# Vérifier si la colonne sélectionnée contient des valeurs numériques (sauf 'Annee_de_sortie')
if pd.api.types.is_numeric_dtype(df_FilmPulseAI[selected_column]) and selected_column != 'Annee_de_sortie':
    # Si c'est une colonne numérique (et différente de 'Annee_de_sortie'), utiliser un slider pour définir une plage de valeurs
    min_value, max_value = st.slider(f"Plage de valeurs pour {selected_column}", float(df_FilmPulseAI[selected_column].min()), float(df_FilmPulseAI[selected_column].max()), (float(df_FilmPulseAI[selected_column].min()), float(df_FilmPulseAI[selected_column].max())))

    # Filtrer le DataFrame en fonction de la plage de valeurs sélectionnée
    filtered_df = df_FilmPulseAI[(df_FilmPulseAI[selected_column] >= min_value) & (df_FilmPulseAI[selected_column] <= max_value)]
else:
    # Si ce n'est pas une colonne numérique (ou si c'est 'Annee_de_sortie'), utiliser la recherche de texte
    search_term = st.text_input("Rechercher dans la colonne {}".format(selected_column))
    filtered_df = df_FilmPulseAI[df_FilmPulseAI[selected_column].astype(str).str.contains(search_term, case=False)]

# Afficher le DataFrame filtré
st.write(filtered_df)

# KPI Top 20 recettes par acteurs/actrices :

st.title("Top 20 recettes par acteur/actrice :")

df_acteurs_filtre = df_acteurs[~df_acteurs['Acteurs'].str.contains('Non renseigné')]

# Groupement par acteurs et somme des recettes et compte du nombre de films
acteurs_grouped = df_acteurs.groupby('Acteurs').agg({'Recettes': 'sum', 'Titre': 'count'}).nlargest(20, 'Recettes')

# Création du graphique à barres empilées avec deux échelles y et une ligne
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=acteurs_grouped.index, y=acteurs_grouped['Recettes'], name='Bénéfices ($)', marker_color='midnightblue'),secondary_y=False)
fig.add_trace(go.Scatter(x=acteurs_grouped.index, y=acteurs_grouped['Titre'], mode='lines', line=dict(color='crimson'), name='Nombre de Films (ligne)'), 
	secondary_y=True)
fig.update_layout(
    xaxis=dict(tickangle=45, tickmode='array'),
    yaxis=dict(title='Bénéfices ($)', side='left', showgrid=False),
    yaxis2=dict(title='Nombre de Films', overlaying='y', side='right', showgrid=False),
    height=600,
    width=1000)
st.plotly_chart(fig)

# Définition des couleurs par genre

couleurs_par_genre = {'Comedy': 'crimson', 'Crime': 'mediumaquamarine', 'Thriller': 'midnightblue',}

# KPI Top 20 recettes par réalisateurs/réalisatrices :

st.title("Top 20 recettes par réalisateur/réalisatrice :")

df_realisateurs_filtre = df_FilmPulseAI[~df_FilmPulseAI['Realisateur'].str.contains('Non renseigné')]

# Groupement par réalisateurs et somme des recettes et compte du nombre de films
realisateurs_grouped = df_realisateurs_filtre.groupby('Realisateur').agg({'Recettes': 'sum', 'Titre': 'count'}).nlargest(20, 'Recettes')

# Création du graphique à barres empilées avec deux échelles y et une ligne
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=realisateurs_grouped.index, y=realisateurs_grouped['Recettes'], name='Bénéfices ($)', marker_color='midnightblue'), secondary_y=False,)
fig.add_trace(go.Scatter(x=realisateurs_grouped.index, y=realisateurs_grouped['Titre'], mode='lines', line=dict(color='crimson'), name='Nombre de Films (ligne)'),
	secondary_y=True)
fig.update_layout(
    xaxis=dict(tickangle=45, tickmode='array'),
    yaxis=dict(title='Bénéfices ($)', side='left', showgrid=False),
    yaxis2=dict(title='Nombre de films', overlaying='y', side='right', showgrid=False),
    height=600,
    width=1000)
st.plotly_chart(fig)

# KPI acteurs les plus représentés : 

st.title("Acteurs/actrices les plus représenté(e)s :")

top_acteurs = df_acteurs_filtre['Acteurs'].value_counts().nlargest(20)
fig_top_acteurs = px.bar(top_acteurs, x=top_acteurs.index, y=top_acteurs.values, labels={"y": "Nombre de films"}, color_discrete_sequence=['midnightblue'])
st.plotly_chart(fig_top_acteurs)

# KPI réalisateurs les plus représentés : 

st.title("Réalisateurs/réalisatrices les plus représenté(e)s :")

top_realisateurs = df_realisateurs_filtre['Realisateur'].value_counts().nlargest(20)
fig_top_acteurs = px.bar(top_realisateurs, x=top_realisateurs.index, y=top_realisateurs.values, labels={"y": "Nombre de films"}, color_discrete_sequence=['midnightblue'])
st.plotly_chart(fig_top_acteurs)

# KPI genre X durée

st.title("Durée moyenne des films par genre :")

fig = px.violin(data_frame=df_FilmPulseAI, x="Genre", y="Duree", box=True, color="Genre")
st.plotly_chart(fig)

# KPI recettes X genre

st.title("Recettes moyennes des films par genre :")

average_revenue_by_genre = df_FilmPulseAI.groupby('Genre')['Recettes'].mean().reset_index()

couleurs_par_genre = {'Comedy': 'midnightblue', 'Crime': 'crimson', 'Thriller': 'mediumaquamarine',}

fig = px.bar(average_revenue_by_genre, x='Genre', y='Recettes', color='Genre', color_discrete_map=couleurs_par_genre)
st.plotly_chart(fig)

# KPI films par genre

st.title("Nombre de films par genre :")

fig = px.histogram(df_FilmPulseAI, x="Genre", color="Genre", color_discrete_map=couleurs_par_genre)
st.plotly_chart(fig)
