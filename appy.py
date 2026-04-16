import streamlit as st
import pandas as pd
import sqlite3
import re
import os
import logging
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import asyncio
from requests import get
from bs4 import BeautifulSoup as bs
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
from typing import TypedDict, Annotated, List, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

import datetime



from langgraph.prebuilt import  create_react_agent
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATIONS & SILENCE WARNINGS ---
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- 2. FONCTIONS DE CALCUL (HORS TOOLS POUR LE CACHE) ---

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform 


import joblib # N'oublie pas d'importer joblib en haut de ton fichier

def train_model_logic():
    conn = sqlite3.connect("annonces_vehicules.db")
    df = pd.read_sql_query("SELECT * FROM voitures ORDER BY id DESC LIMIT 5000", conn)
    conn.close()

    if len(df) < 10:
        return None, None, None, 0, 0, {}

    df['price']       = pd.to_numeric(df['price'], errors='coerce')
    df['kilometrage'] = pd.to_numeric(df['kilometrage'], errors='coerce')
    df['annee']       = pd.to_numeric(df['annee'], errors='coerce')
    df = df.dropna(subset=['price', 'kilometrage', 'annee'])

    # ✅ Filtres réalistes
    df = df[df['price'].between(500_000, 100_000_000)]
    df = df[df['kilometrage'].between(0, 500_000)]
    df = df[df['annee'].between(1990, 2025)]

    # ✅ Outliers
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df['price'].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]

    # ✅ Feature engineering
    df['log_price'] = np.log1p(df['price'])
    df['age'] = 2025 - df['annee']
    df['km_par_an'] = df['kilometrage'] / (df['age'] + 1)

    # ✅ Encodage
    le_carb  = LabelEncoder()
    le_boite = LabelEncoder()
    df['carb_enc']  = le_carb.fit_transform(df['carburant'].fillna('Inconnu').astype(str))
    df['boite_enc'] = le_boite.fit_transform(df['boite'].fillna('Inconnu').astype(str))

    X = df[['annee', 'age', 'kilometrage', 'km_par_an', 'carb_enc', 'boite_enc']].values
    y = df['log_price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Entraînement
    param_distributions = {
        'n_estimators':     randint(200, 500),
        'max_depth':        randint(3, 8),
        'learning_rate':    uniform(0.01, 0.15),
        'subsample':        uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 10),
        'gamma':            uniform(0, 0.5),
    }

    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, verbosity=0),
        param_distributions,
        n_iter=40,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        random_state=42,
        refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # ✅ Métriques
    y_pred_log  = best_model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)
    r2  = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)

    # ✅ SAUVEGARDE SUR DISQUE
    # On packe tout dans un seul fichier pour simplifier le chargement futur
    artifacts = {
        'model': best_model,
        'le_carb': le_carb,
        'le_boite': le_boite,
        'r2': r2,
        'mae': mae,
        'best_params': search.best_params_
    }
    joblib.dump(artifacts, "expert_vehicule_model.pkl")
    print("📢 Modèle sauvegardé avec succès dans 'expert_vehicule_model.pkl'")

    return best_model, le_carb, le_boite, r2, mae, search.best_params_

# --- 3. OUTILS (TOOLS) ---

@tool
def scraping_voitures_tool(pages: int = 1) -> list:
    """Scrape les voitures depuis dakar-auto.com"""
    all_data = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for ind in range(1, pages + 1):
        try:
            url = f'https://dakar-auto.com/senegal/voitures-4?&page={ind}'
            res = get(url, headers=headers, timeout=10)
            soup = bs(res.content, 'html.parser')
            containers = soup.find_all('div', 'listings-cards__list-item mb-md-3 mb-3')
            for container in containers:
                try:
                    link = container.find('a')['href']
                    url_item = 'https://dakar-auto.com/' + link
                    res_item = get(url_item, headers=headers, timeout=5)
                    s = bs(res_item.content, 'html.parser')
                    title = s.find('h1', 'listing-item__title').text.strip()
                    price = int(re.sub(r'\D', '', s.find('h4', 'listing-item__price').text))
                    km = int(re.sub(r'\D', '', s.find('li', 'listing-item__attribute').text))
                    bloc = s.find('ul', 'listing-item__attribute-list').text.strip().split('\n')
                    all_data.append({
                        "title": title, "price": price, "kilometrage": km,
                        "carburant": bloc[-1].strip(), "boite": bloc[-4].strip(),
                        "annee": int(re.sub(r'\D', '', bloc[3])), "source": url_item
                    })
                except: continue
        except: continue
    return all_data

@tool
def manage_car_db(action: str, cars: Optional[List[dict]] = None):
    """Gère la base de données SQLite."""
    conn = sqlite3.connect("annonces_vehicules.db")
    cursor = conn.cursor()
    if action == "init":
        cursor.execute('''CREATE TABLE IF NOT EXISTS voitures (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       title TEXT, price INTEGER, kilometrage INTEGER, carburant TEXT, boite TEXT, annee INTEGER, source TEXT)''')
        res = "Base de données initialisée."
    elif action == "insert" and cars:
        cursor.executemany("INSERT INTO voitures (title, price, kilometrage, carburant, boite, annee, source) VALUES (?,?,?,?,?,?,?)",
                           [(c['title'], c['price'], c['kilometrage'], c['carburant'], c['boite'], c['annee'], c['source']) for c in cars])
        res = f"{len(cars)} véhicules insérés."
    elif action == "read":
        cursor.execute("SELECT * FROM voitures ORDER BY id DESC LIMIT 100")
        res = cursor.fetchall()
    else: res = "Action inconnue."
    conn.commit()
    conn.close()
    return res

@tool
def expert_ml_prediction_tool(annee: int, km: int, carburant: str, boite: str) -> dict:
    """
    Tool expert pour prédire le prix d'un véhicule.
    Vérifie l'existence du modèle et l'entraîne si nécessaire.
    """
    import os
    import joblib
    import numpy as np

    try:
        # 1. Vérification et chargement du modèle
        model_path = "expert_vehicule_model.pkl"
        
        if not os.path.exists(model_path):
            # Si le fichier .pkl n'existe pas, on lance l'entraînement
            print("🚀 Modèle absent, lancement d'un entraînement automatique...")
            result = train_model_logic()
            if result[0] is None: # Si train_model_logic renvoie None (pas assez de données)
                return {"error": "Impossible d'entraîner le modèle : base de données trop petite (< 10 annonces)."}
        
        # Chargement sécurisé des artefacts
        artifacts = joblib.load(model_path)
        model = artifacts['model']
        le_carb = artifacts['le_carb']
        le_boite = artifacts['le_boite']
        r2 = artifacts['r2']
        mae = artifacts['mae']

        # 2. Préparation des caractéristiques (Feature Engineering)
        # L'ordre doit être identique à celui de l'entraînement :
        # [annee, age, kilometrage, km_par_an, carb_enc, boite_enc]
        
        annee = int(annee)
        km = int(km)
        age = 2025 - annee
        age = max(0, age)  # Évite les années futures
        km_par_an = km / (age + 1)

        # 3. Encodage sécurisé des variables catégorielles
        try:
            c_enc = le_carb.transform([str(carburant).strip().capitalize()])[0]
        except:
            c_enc = 0 # Valeur par défaut si nouveau carburant
            
        try:
            b_enc = le_boite.transform([str(boite).strip().capitalize()])[0]
        except:
            b_enc = 0 # Valeur par défaut si nouvelle boîte

        # 4. Création du vecteur d'entrée (Shape 1, 6)
        input_data = np.array([[
            float(annee), 
            float(age), 
            float(km), 
            float(km_par_an), 
            float(c_enc), 
            float(b_enc)
        ]])

        # 5. Prédiction (sur log) et reconversion
        log_pred = model.predict(input_data)
        price_pred = np.expm1(log_pred)[0]

        # 6. Retour structuré pour l'agent
        return {
            "prediction_prix": f"{int(price_pred):,}".replace(",", " ") + " FCFA",
            "indice_confiance": f"{round(r2 * 100, 1)}%",
            "marge_erreur": f"± {int(mae):,}".replace(",", " ") + " FCFA",
            "details": {
                "age_du_vehicule": f"{age} ans",
                "utilisation_annuelle": f"{int(km_par_an)} km/an",
                "statut_modele": "Chargé/Auto-généré avec succès"
            }
        }

    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}


    
    
@tool
def full_eda_and_viz_tool():
    """Analyse EDA complète avec visualisation."""
    conn = sqlite3.connect("annonces_vehicules.db")
    df = pd.read_sql_query("SELECT * FROM voitures", conn)
    conn.close()
    if df.empty: return "Base vide."
    df['price_clean'] = winsorize(df['price'], limits=[0.05, 0.05])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.kdeplot(df['price'], ax=axes[0,0], fill=True, color="red")
    sns.boxplot(data=[df['price'], df['price_clean']], ax=axes[0,1])
    axes[0,1].set_xticks([0,1]); axes[0,1].set_xticklabels(['Brut', 'Nettoyé'])
    sns.regplot(x='annee', y='price_clean', data=df, ax=axes[1,0])
    sns.countplot(x='boite', data=df, ax=axes[1,1], hue='boite', legend=False)
    plt.tight_layout()
    plt.savefig("dashboard_eda_final.png")
    plt.close()
    return "Dashboard généré avec succès sous le nom 'dashboard_eda_final.png'."

# --- 4. AGENT & GRAPH ---


def get_agent():
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        groq_api_key="VOTRE_CLE_API_GROQ" # Ou configurée en variable d'environnement
    )
    
    
    # llama3.1 est indispensable pour que LangGraph puisse appeler les outils
    #llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    
    # Tes 4 outils
    tools = [
        scraping_voitures_tool, 
        manage_car_db, 
        full_eda_and_viz_tool, 
        expert_ml_prediction_tool
    ]

    # En LangGraph, l'instruction système remplace le template ReAct manuel
    system_instruction = (
    "Tu es l'expert automobile n°1 au Sénégal. "
    "Lorsqu'on te demande un prix : \n"
    "1. Si des informations manquent (km, carburant, boîte), utilise des valeurs logiques par défaut "
    "(ex: 15 000 km/an, Essence, Manuelle) et précise-le dans ta réponse.\n"
    "2. Appelle TOUJOURS l'outil 'expert_ml_prediction_tool' pour donner un chiffre en FCFA.\n"
    "3. Ne pose pas de questions d'abord, prédis PUIS suggère d'affiner avec plus de détails."
)

    # Création de l'agent LangGraph
    # Note : 'state_modifier' est le nom correct de l'argument dans les versions récentes
    return create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_instruction  # 
)


agent_executor = get_agent()



# --- CONFIGURATION ET STYLE ---
st.set_page_config(
    page_title="IA Dakar Auto - Expert Automobile",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; transition: 0.3s; }
    .stButton>button:hover { border: 1px solid #ff4b4b; color: #ff4b4b; }
    .agent-card { padding: 15px; border-radius: 10px; background-color: #ffffff; border: 1px solid #e0e0e0; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (PANNEAU DE CONTRÔLE) ---
with st.sidebar:
    # Logo et Titre
    st.markdown("<h1 style='text-align: center;'>🚗 IA DAKAR AUTO</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("🌐 Acquisition de Données")
    # Ajout du bouton Scraper
    nb_pages = st.slider("Nombre de pages à scanner", 1, 20, 5)
    if st.button("🔍 Lancer le Scraper Web"):
        with st.status("Agent Scraper en cours...", expanded=True) as status:
            st.write("🌐 Connexion à Dakar-Auto.com...")
            # On passe l'argument nb_pages au tool
            res = scraping_voitures_tool.invoke({"pages": nb_pages})
            st.write(f"✅ {res}")
            status.update(label="Collecte terminée !", state="complete", expanded=False)
        st.toast("Base de données mise à jour !")

    st.divider()
    st.subheader("🛠️ Intelligence Artificielle")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Init DB"):
            manage_car_db.invoke({"action":"init"})
            st.toast("Base réinitialisée")
    with col2:
        if st.button("🧹 Train ML"):
            with st.spinner("Optimisation..."):
                # Appelle ta fonction qui génère le expert_vehicule_model.pkl
                train_model_logic()
                st.success("IA Entraînée")

    st.divider()
    st.subheader("🕵️ Système Multi-Agent")
    st.info("""
    **Agents Actifs :**
    - **Scraper**: Collecte Web.
    - **Data**: Gestion SQLite.
    - **Analyste**: Rapports EDA.
    - **Expert ML**: Prédiction XGBoost.
    """)

# --- CORPS PRINCIPAL ---
tabs = st.tabs(["💬 Assistant Expert", "📊 Dashboard Analytique", "📁 Explorateur de Données"])

# --- TAB 1 : CHATBOT (L'AGENT RE-ACT) ---
with tabs[0]:
    st.subheader("💬 Échangez avec l'Expert IA")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis l'expert du marché automobile sénégalais. Posez-moi vos questions sur les prix ou les modèles."}]

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: Estime le prix d'une Ford Focus 2017 Diesel"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.status("🧠 L'agent réfléchit...", expanded=True) as status:
                st.write("🔍 Analyse de la demande...")
                # Appel de l'agent LangGraph
                response = agent_executor.invoke({"messages": [HumanMessage(content=prompt)]})
                st.write("⚙️ Appel des outils de prédiction...")
                status.update(label="Analyse terminée", state="complete", expanded=False)
            
            ans = response["messages"][-1].content
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

# --- TAB 2 : ANALYSE VISUELLE (EDA) ---
with tabs[1]:
    st.subheader("📊 Analyse du Marché en Temps Réel")
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("🚀 Re-générer EDA"):
            with st.spinner("Calcul des stats..."):
                full_eda_and_viz_tool.invoke({})
    
    if os.path.exists("dashboard_eda_final.png"):
        st.image("dashboard_eda_final.png", use_container_width=True)
    else:
        st.info("Cliquez sur le bouton pour générer le premier rapport visuel.")

# --- TAB 3 : EXPLORATEUR DE DONNÉES ---
with tabs[2]:
    st.subheader("📁 Données Brutes de la Base")
    data = manage_car_db.invoke({"action": "read"})
    if data:
        df_display = pd.DataFrame(data, columns=["ID","Titre","Prix","KM","Carburant","Boîte","Année","Source"])
        
        # Filtre interactif
        carb_filter = st.multiselect("Filtrer par Carburant", options=df_display['Carburant'].unique())
        if carb_filter:
            df_display = df_display[df_display['Carburant'].isin(carb_filter)]
            
        st.dataframe(df_display, use_container_width=True, height=500)
        st.download_button("📥 Télécharger CSV", df_display.to_csv(index=False), "cars_data.csv", "text/csv")
    else:
        st.warning("Aucune donnée disponible. Lancez le scraper !")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: grey;'>Système Multi-Agent IA Dakar Auto © {datetime.datetime.now().year} - Projet de Soutenance</p>", unsafe_allow_html=True)

        
