import streamlit as st 

def run():
    st.markdown("<h1 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%; margin: 0 auto;'>Introduction</h1>", unsafe_allow_html=True)

    st.write(" ")
    #st.markdown("<h1 style='text-align: center;'>Prédiction du succés d'une campagne de marketing d'une banque</h1>",unsafe_allow_html=True)

    #st.markdown("<h2 style='text-align: center;'>Avril à Novembre 2023</h2>", unsafe_allow_html=True)

    st.write("Dans le cadre de notre formation **Data Analyst avec Datascientest**, nous avons entrepris un projet majeur visant à explorer toutes les étapes du développement d'une solution basée sur les données. Notre mission nous a conduits à nous pencher sur un ensemble de données essentiel pour le projet **Prédiction du succès d’une campagne de Marketing d’une banque**. L'objectif principal était de comprendre les facteurs déterminants qui influencent le succès des campagnes marketing d'une institution bancaire.")
    st.write("")
    if st.checkbox("Voir la suite"):
        st.write("Ce jeu de données contient une variété d'informations précieuses, notamment l'âge des clients, leur profession, leur situation matrimoniale, leur niveau d'éducation, leur historique de crédit, leur solde bancaire, la possession d'un prêt immobilier, d'un prêt personnel, le mode de contact, le jour et le mois du dernier contact, la durée de ce contact, le nombre de contacts effectués lors de cette campagne, les jours écoulés depuis le dernier contact précédent, le nombre de contacts précédents, et le résultat de la campagne marketing. La colonne 'deposit' indique si les clients ont souscrit à un produit bancaire.")
        st.write("Notre projet est d'une grande importance, car il vise à prédire le succès des campagnes marketing de la banque en s'appuyant sur l'analyse de ces données. Grâce aux compétences acquises au cours de notre formation, nous sommes en mesure d'extraire des informations significatives, d'identifier des tendances et de mettre en lumière les éléments clés qui déterminent le succès des campagnes marketing, ce qui permettra à la banque de prendre des décisions éclairées pour optimiser ses futures campagnes.")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%; margin: 0 auto;'>Contexte du Projet</h2>", unsafe_allow_html=True)
    st.write(" ")
    st.write ("Nous allons nous imaginer data analyst pour une banque qui souhaite améliorer sa stratégie de marketing pour un produit appelé dépôts à terme.")
             
    st.write("Les clients ayant souscrit à cette offre acceptent de placer une quantité d'argent sur un compte spécifique. En contrepartie, la banque s'engage à verser des intérêts attractifs à la fin du contrat.")

    st.write("L'objectif global de la banque est d optimiser les revenus et de renforcer la relation avec les clients.")

    st.write("Notre objectif est donc d identifier les facteurs qui influencent la décision des clients de souscrire à un dépôt à terme afin de cibler plus efficacement les campagnes de marketing et d'augmenter le taux de souscription.")


   
    #st.image("img/finance image 2.PNG", use_column_width=True, caption="Légende de l'image")


    
    
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 45%; margin: 0 auto;'>Methodologie</h2>", unsafe_allow_html=True)

    st.write("")
    

    st.write("Pour mener à bien notre projet de prédiction du succès d’une campagne de Marketing d’une banque, nous suivrons une méthodologie rigoureuse et structurée, qui comprendra les étapes suivantes :")
    st.write("")
    
    
    st.write(" - **Présentation du Dataset afin de comprendre les différentes variables:** Nous approfondirons en regardant la qualité des données fournies.")
    st.write(" - **Réalisations des premières visualisations afin de voir la répartitions des données**:Analyse univariée et multivarié ")
    st.write(" - **Nettoyage du dataset et modélisation,machine learning**: test de plusieurs modèle et selection des meilleurs")
    st.write(" - **interprétabilité des résultats et reponse à la problèmatique** ")
    if st.checkbox("Voir la methodoligie Detaillée"):

        st.write(" **- Collecte des données :** Nous recueillerons les données relatives aux clients de la banque à partir de sources authentiques et fiables. Il est essentiel que ces données couvrent une période suffisamment représentative pour une analyse pertinente. Nous veillerons également à ce que les données soient complètes, cohérentes et de haute qualité")
        st.write(" **- Nettoyage et préparation des données :** Une fois les données collectées, nous entreprendrons une phase de nettoyage et de préparation méticuleuse. Cela impliquera la correction d'erreurs, la gestion des incohérences et le traitement des valeurs manquantes. De plus, nous normaliserons les données pour faciliter les analyses ultérieures.")
        st.write(" **- Analyse exploratoire des données :** Nous réaliserons une analyse exploratoire des données pour identifier les tendances, les variations et les relations entre les variables. Des techniques de visualisation des données, telles que des graphiques et des tableaux, seront utilisées pour rendre les résultats plus compréhensibles.")
        st.write(" **- Modélisation et prediction :** Nous développerons des modèles statistiques et de machine learning pour analyser les facteurs qui influencent le succès des campagnes de marketing de la banque. Ces modèles nous permettront de prévoir les résultats futurs. Nous évaluerons la performance de ces modèles à l'aide d'indicateurs tels que Rappel (Recall),le F1 Score, la précision (accuracy) .")
        st.write(" **- Interprétation et communication des résultats :** Nous interpréterons les résultats obtenus à travers nos modèles pour mettre en évidence les facteurs qui ont le plus d'impact sur le succès des campagnes marketing de la banque. Nous communiquerons nos conclusions de manière claire et accessible, en utilisant des supports visuels et des explications simples.")
        st.write("")
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%; margin: 0 auto;'>Résultats attendus</h2>", unsafe_allow_html=True)
    st.write("")
    if st.checkbox("Voir les Resultats Attendus"):

        st.write(" Pour évaluer l'efficacité de notre approche dans le contexte de la prédiction du succès des campagnes de marketing de la banque, il est essentiel d'identifier les critères de performance clés. Cela repose à la fois sur les performances intrinsèques de nos modèles (scores) et sur notre capacité à les traduire en résultats concrets pour l'entreprise.")
        st.write("**1. Accuracy (Exactitude) :** L'accuracy élevée est cruciale pour prédire avec précision le succès d'une campagne de marketing. Elle garantit que le modèle classifie correctement la majorité des clients, minimisant ainsi les erreurs de classification.")
        st.write("**2. Précision :** En plus de l'accuracy, une précision élevée est recherchée pour minimiser les erreurs de classification positives, ce qui permet de cibler efficacement les clients susceptibles de souscrire au produit")
        st.write("**3. Rappel :** Un rappel élevé permet d'identifier efficacement les clients réellement intéressés par le produit, minimisant ainsi les opportunités de conversion manquées.")
        st.write("**4. F1 Score :** Cette mesure harmonieuse de la précision et du rappel est essentielle pour évaluer la performance globale du modèle.")
        st.write("")
        st.write("5. En plus des métriques de performance, notre objectif est de rendre nos modèles interprétables, en identifiant les facteurs clés influençant la décision de souscrire au produit de dépôt à terme. Cette interprétabilité permettra à la banque d'améliorer ses campagnes de marketing.")
        st.write("")
        st.write("Enfin, notre objectif central est d'aligner notre travail sur les besoins et les objectifs métier de la banque, en évaluant comment l'application de nos modèles de prédiction peut apporter une réelle valeur ajoutée à l'entreprise. C'est en tenant compte de ces trois critères que nous serons en mesure de proposer des modèles et des solutions pertinents et performants à la banque")
    st.write("")
    st.write("")
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 45%; margin: 0 auto;'>Remerciments</h2>", unsafe_allow_html=True)
    st.write("")
    if st.checkbox("Voir les Remerciments"):
        st.write("")
        

        #-----------------------------------------------------------------------------------------
        st.write("Tout d'abord, nous exprimons notre gratitude envers l'équipe de DataScientest, notre chef de cohorte, **CHRISTOPHE**, et surtout notre Mentor projet, **MANON**, pour leur soutien constant et leur encadrement pédagogique exceptionnel. Leur précieuse assistance a joué un rôle déterminant dans notre réussite et la réalisation fructueuse de notre projet au cours de cette formation.")
    