import streamlit as st

def run():
   
  
   st.markdown("<h1 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%; margin: 0 auto;'>Conclusion</h1>", unsafe_allow_html=True)


   choix = ['Résumé du projet', 'Recommandations', 'Perspectives']
   option = st.selectbox('Paramètres de conclusion', choix)
   
   if option == 'Résumé du projet' :
    
        st.write("Afin de résumer rapidement l'ensemble des étapes du projet, nous pouvons découper ce dernier selon ces étapes :")
        st.write("- Prise en main du dataset")
        st.write("- Exploration et analyse du jeu de données")
        st.write("- Analyse de dépendances statistiques des données")
        st.write("- Visualisation des données")
        st.write("- Clustering")
        st.write("- Nettoyage des données")
        st.write("- Mise en place du processus de machine learning")
        st.write("- Test de plusieurs modèles puis optimisation des meilleurs")
        st.write("- Interprétabilité des résultats")
   
   elif option == 'Recommandations' :
        st.write("L'interprétation des données nous permet d'effectuer des recommandations au marketing :")
        st.write("- Augmenter la durée d'appel")
        st.write("- Cibler les clients sans crédit immobilier")
        st.write("- Cibler les clients n'ayant pas souscrit au prêt lors de la campagne précédente")
        st.write("- Privilégier les cadres")
        st.write("Il peut également être utile de donner aux équipes métier une représentation visuelle de leur clientèle avec quelques plots bien choisis afin de leur permettre d'appréhender au mieux les tendances qui s'en dégagent. Une visualisation de la distribution de plusieurs variables ou de clusters de clients est également une part non négligeable du service que nous pouvons rendre au service marketing.")
   
   elif option == 'Perspectives' :
        st.write("Pour aller plus loin, nous pouvons nous poser quelques questions. Il réside des axes d'analyse qu'il est possible d'ouvrir. Afin d'avoir un premier aperçu des directions et du potentiel du travail que nous pourrions accomplir en plus, nous allons exposer ces perspectives.")
        st.write("Certaines relations, comme la relation entre la durée d'appel et la souscription, peuvent être plus complexes que les résultats donnés par l'interprétabilité.")
        st.write("Se comparer avec les résultats d'autres banques possédant les mêmes types de produits, le but étant d'améliorer nos canaux de communication en fonction de ceux des autres banques.")
        st.write("Améliorer la diversification des moyens de prospection, ce qui nous permettrait également d'enrichir le jeu de données afin d'améliorer nos résultats.")
        st.write("Travailler avec les équipes Data Ingénieurs et métier en amont du jeu de données afin de disposer de données de meilleure qualité.")
