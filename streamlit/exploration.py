import streamlit as st 
import pandas as pd

def run():
    @st.cache_data()
    def load_data():
        df = pd.read_csv('./datasets/bank.csv')
        return df
    # Load the data
    df = load_data()
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%; margin: 0 auto;'>Périmètre du projet</h2>", unsafe_allow_html=True)
    st.write("")
  
    st.write("Dans ce projet, nous allons travailler sur un jeux de données contenant les informations clients d'une banque. L'étude des variables disponibles nous permetra de prédire la souscription au dépôt à terme.")
    st.write("Nous allons concentrer notre analyse sur les 17 variables de notre dataset. Grace à elles nous allons comprendre les facteurs qui influencent la souscription aux dépôts à terme et à formuler des recommandations pour améliorer l'efficacité de la campagne marketing:")
    st.write("")
    st.write("")
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 90%; margin: 0 auto;'>Présentation des variables du Dataset</h2>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<span style='font-size:1.25em'><b>Age</b></span> : l'âge de l'individu, généralement mesuré en années.", unsafe_allow_html=True)

    st.markdown("<span style='font-size:1.25em'><b>Job</b></span> : la catégorie de métier ou la profession de l'individu.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Marital</b></span> : l'état matrimonial de l'individu, tel que marié, célibataire, divorcé, etc.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Education</b></span> : le niveau d'éducation atteint par l'individu, comme l'école primaire, le collège, le lycée, l'université, etc.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Default</b></span> : indique si l'individu a déjà fait défaut sur un prêt ou une dette.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Balance</b></span> : le solde du compte bancaire de l'individu.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Housing</b></span> : indique si l'individu possède un prêt hypothécaire ou non.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Loan</b></span> : indique si l'individu a un prêt personnel ou non.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Contact</b></span> : le mode de contact utilisé pour communiquer avec l'individu, comme le téléphone, le courrier électronique, etc.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Day</b></span> : le jour du mois où le dernier contact a été établi avec l'individu.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Month</b></span> : le mois de l'année où le dernier contact a été établi avec l'individu.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Duration</b></span> : la durée en secondes du dernier contact avec l'individu.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Campaign</b></span> : le nombre de contacts effectués lors de la campagne publicitaire ou marketing.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Pdays</b></span> : le nombre de jours écoulés depuis le dernier contact avant la campagne actuelle (-1 signifie que le client n'a pas été contacté auparavant).", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Previous</b></span> : le nombre de contacts effectués avant la campagne actuelle.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Poutcome</b></span> : le résultat de la campagne marketing précédente.", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.25em'><b>Deposit</b></span> : indique si l'individu a souscrit à un dépôt à terme ou non.", unsafe_allow_html=True)

    st.write("La variable Deposit sera notre variable cible, il s'agit d'une variable binaire")
    st.write("")   
    st.write("")
    st.write("")   
    st.write("")

    st.markdown("<h4 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Visualisation des 5 premières lignes du Dataframe</h4>", unsafe_allow_html=True)

    st.write("") 

    st.dataframe(df.head())
    
    st.write("")   
    st.write("")
    st.write("")   
    st.write("")
    
    st.markdown("<h4 style='text-align: center; color: #87CEEB;'>Listes des premières informations issues de notre jeux de données</h4>", unsafe_allow_html=True)

    
    st.write("")   
    st.write("")

    if st.checkbox("Afficher la dimension du Dataframe"):
                st.write((df.shape))
                st.write(" Le dataset contient 17 variables et 11162 lignes.")
    if st.checkbox("Afficher les types de variables"):
        st.write(df.dtypes)
                
                
                
    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())
        st.write("**Nous constatons que le dataset ne comporte pas de valeurs manquantes**")
        
        
        
    if st.checkbox("Afficher le nombre de lignes duppliquées"):
        st.write(df.duplicated().sum())  
        st.write("**Nous constatons que le dataset ne comporte pas de lignes dupliquées**")
        
    if st.checkbox("Afficher les modalités de chaque variable"):
        for col in df.columns:
            
            
            st.write(f"{col} :{df[col].unique()}\n")
    if st.checkbox("Afficher le nombre de modalités de chaque variable"):
        for col in df.columns:
            st.write(f"{col} :{df[col].nunique()}\n")
    
    
    if st.checkbox("Afficher le mini et maxi des variables continues"):
       for col in df.select_dtypes(include=(['int','float'])).columns:
        st.write(f"{col}")
        st.write(f"Valeur MIN : {df[col].min()}")
        st.write(f"Valeur MAX : {df[col].max()}")
   
    
    if st.checkbox("Afficher le mode de chaque variables catégorielles"):
       for col in df.select_dtypes(include=('object')).columns:
           st.write(f" {col}")
           st.write(f"Mode : {df[col].mode()[0]}")
    
    
    if st.checkbox("Afficher la fréquence des modalités des variables qualitatives"):
       for col in list(df.select_dtypes(include='object')):
        st.write("----------------La Variable *"+col+"* comporte ",df[col].nunique()," Modalites distincts---------------")
        st.write("les frequences des  modalites sont:  ")
        st.write(round(df[col].value_counts(normalize=True)*100,2))
        st.write("\n")
    
    
    if st.checkbox("Resumé statistiques des variables numériques"):    
        st.write(df.describe().T)
       
        
    st.write("")   
    st.write("")
    st.write("**Dans cette partie nous avons pu constater que notre dataset est de bonne qualité,il ne contient ni valeurs manquante, ni lignes dupliquées.**")
    st.write("**Nous allons pouvoir passer à la prochaine étape avec la visualisation des données.**")
    
    st.write("")   
    st.write("")
    
    #st.image("img/finance image 1.PNG")