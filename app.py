import streamlit as st
import pandas as pd

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg
from pycaret.regression import pull as pull_reg

from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class
from pycaret.classification import pull as pull_class

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

url = "https://www.linkedin.com/in/josu%C3%A9-afouda/"

st.title("JoshAutoML")
st.sidebar.write("[Auteur: Josué AFOUDA ](%s)" % url)
st.sidebar.markdown(
    "**Cette application web est un outil No-Code pour la construction de modèles d'apprentissage automatique pour les tâches de Régression et de Classification.**\n"            
        "1. Chargez votre jeu de données (fichier CSV);\n"            
        "2. Choisissez votre variable cible;\n"
        "3. Choisissez la tâche d'apprentissage automatique (Regression ou Classification);\n"
        "5. Cliquez sur *Run Modelling* pour commencer le processus d'entraînement.\n"
        "Une fois que le modèle est construit, vous pouvez visualiser les résultats comme le pipeline modèle, le graphe de résidus, la courbe ROC, la matrice de confusion, l'importance des caractéristiques, etc.\n"
        "\n6. Téléchargez le modèle formé sur votre ordinateur en local."
)

#st.set_option('deprecation.showfileUploaderEncoding', False)

file = st.file_uploader("Téléchargez votre jeu de données au format CSV", type=["csv"])

if file is not None:
    st.write("Veuillez patienter jusqu'à l'appartion du bouton 'Download the model'")
    data = load_data(file)
    st.dataframe(data.head())

    target = st.selectbox("Sélectionnez la variable cible", data.columns)

    data = data.dropna(subset=[target]) # au cas où il y aura des valeurs manquantes dans la variable cible

    task = st.selectbox("Sélectionnez une tâche", ["Régression", "Classification"])

    if task == "Régression":
        if st.button('Run Modelling'):
            setup_reg(data, target = target)
            setup_reg_df = pull_reg()
            st.write("Mise en place de l'environnement d'entraînement")
            st.dataframe(setup_reg_df)
            model_reg = compare_models_reg(sort="RMSE")
            compare_reg_df = pull_reg()
            st.write("Résultats de la modélisation")
            st.dataframe(compare_reg_df)
            save_model_reg(model_reg, 'meilleur_modele_reg')
            st.success("Modèle de régression construit avec succès !")

            # Résultats
            st.write("Graphique des résidus")
            plot_model_reg(model_reg, plot = 'residuals', save=True)
            st.image("Residuals.png")
                
            st.write("Importance des caractéristiques")
            plot_model_reg(model_reg, plot = 'feature', save=True)
            st.image("Feature Importance.png")

            with open('meilleur_modele_reg.pkl', 'rb') as f: 
                st.download_button('Télécharger le modèle', f, file_name="meilleur_modele_reg.pkl")
                    
        
    if task == "Classification":
        if st.button('Run Modelling'):
            setup_class(data, target = target)
            setup_class_df = pull_class()
            st.write("Mise en place de l'environnement d'entraînement")
            st.dataframe(setup_class_df)
            model_class = compare_models_class(sort="AUC")
            compare_class_df = pull_class()
            st.write("Résultats de la modélisation")
            st.dataframe(compare_class_df)
            save_model_class(model_class, 'meilleur_modele_class')
            st.success("Modèle de classification construit avec succès !")

            # Résultats
            col5, col6 = st.columns(2)
            with col5:
                st.write("Courbe ROC")
                plot_model_class(model_class, save=True)
                st.image("AUC.png")
                
            with col6:
                st.write("Rapport de classification")
                plot_model_class(model_class, plot = 'class_report', save=True)
                st.image("Class Report.png")
                
            col7, col8 = st.columns(2)
            with col7:
                st.write("Matrice de confusion")
                plot_model_class(model_class, plot = 'confusion_matrix', save=True)
                st.image("Confusion Matrix.png")
                
            with col8:
                st.write("Importance des caractéristiques")
                plot_model_class(model_class, plot = 'feature', save=True)
                st.image("Feature Importance.png")

            # Télécharger le modèle
            with open('meilleur_modele_class.pkl', 'rb') as f: 
                st.download_button('Download the model', f, file_name="meilleur_modele_class.pkl")

else:
    st.image("https://github.com/JosueAfouda/JoshAutoML/raw/main/home-image.png")
