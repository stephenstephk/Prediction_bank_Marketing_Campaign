import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run():
    
# Les résultats des 7 modèles
    resultats_models = {
    'Logistic Regression': {'Précision': 0.845142, 'F1-Score': 0.817425, 'Rappel': 0.791469, 'Score_Entraînement': 0.809507, 'Score_Test(Accuracy)': 0.829525},
    'Decision Tree': {'Précision': 0.784785, 'F1-Score': 0.763389, 'Rappel': 0.743128, 'Score_Entraînement': 1.000000, 'Score_Test(Accuracy)': 0.777879},
    'Random Forest': {'Précision': 0.834559, 'F1-Score': 0.847410, 'Rappel': 0.860664, 'Score_Entraînement': 1.000000, 'Score_Test(Accuracy)': 0.850548},
    'SVM': {'Précision': 0.843870, 'F1-Score': 0.839447, 'Rappel': 0.835071, 'Score_Entraînement': 0.846532, 'Score_Test(Accuracy)': 0.845978},
    'KNN': {'Précision': 0.781014, 'F1-Score': 0.730575, 'Rappel': 0.686256, 'Score_Entraînement': 0.823563, 'Score_Test(Accuracy)': 0.755941},
    'Gradient Boosting': {'Précision': 0.832418, 'F1-Score': 0.846763, 'Rappel': 0.861611, 'Score_Entraînement': 0.848817, 'Score_Test(Accuracy)': 0.849634},
    'AdaBoost': {'Précision': 0.836130, 'F1-Score': 0.819149, 'Rappel': 0.802844, 'Score_Entraînement': 0.819106, 'Score_Test(Accuracy)': 0.829068}
  }

    # Les résultats des 3 modèles optimisés
    results_models_optimise = {
        'Logistic Regression': {'Score Train ': 0.8129282777523984, 'Accuracy': 0.8295},
        'SVM': {'Score Train ': 0.8573549566011878, 'Accuracy': 0.8514625228519196},
        'Gradient Boosting': {'Score Train ': 0.9090909090909091, 'Accuracy': 0.8506167199634537}
    }

    # Afficher les résultats des 7 modèles
    st.markdown("<h1 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Résultats des 7 Modèles</h1>", unsafe_allow_html=True)
    st.write("")
    st.write('Les résultats initiaux des 7 modèles de classification :')
    st.write("")
    df_models = pd.DataFrame(resultats_models).T
    st.table(df_models)

    # Afficher les résultats des 3 modèles optimisés
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Résultats des 3 Modèles Optimisés</h2>", unsafe_allow_html=True)
    st.write("")
    st.write('Les résultats après optimisation des 3 modèles sélectionnés :')
    st.write("")
    df_models_optimise = pd.DataFrame(results_models_optimise).T
    st.table(df_models_optimise)

    

    
    st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 100%; margin: 0 auto;'>Tester tous les models sur lequels nous avons travaillé</h2>", unsafe_allow_html=True)
    st.write("")
    st.write(' Nous pouvons  tester différents modèles de classification binaire et afficher les métriques de classification.')
    st.write('Vous pouvez choisir le modèle, les variables explicatives et la taille de l\'ensemble de test  ainsi que les paramettres du model .')
    @st.cache_data()
    def load_data():
        df = pd.read_csv('./datasets/bank_modif.csv')
        return df
    #Charger le jeux de donnees
    df = load_data()
    
    # charger les models
    regl = joblib.load('LogisticRegression.pkl')
    dcl = joblib.load('DecisionTreeClassifier.pkl')
    rfcl = joblib.load('RandomForestClassifier.pkl')
    svm = joblib.load('SVM.pkl')
    knn = joblib.load('KNeighborsClassifier.pkl')
    gbc = joblib.load('GradientBoostingClassifier.pkl')
    abc = joblib.load('AdaBoostClassifier.pkl')

    

    # Definir le slidebar
    if st.checkbox("**Afficher les Donnees brutes**"):
        st.table(df.sample(10))
    st.write("")  
    st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choix des Variables et du Model de prediction</h3>", unsafe_allow_html=True)
    st.write("")
    st.warning('Veuillez sélectionner  une ou plusieurs Variable pour la Prediction, Par defaut toutes les variables sont prise en compte')
    st.write("")
    st.write("")    

    model_choice = st.selectbox("**Choisissez un modèle**", ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "Super Vector Machine(SVM)", "KNeighborsClassifier(Knn)", "GradientBoostingClassifier", "AdaBoostClassifier"])
    st.write("")
    st.write("")    

    features_choice = st.multiselect("**Choisir les variables explicatives**", list(df.drop('deposit',axis=1).columns))
    st.write("")
    st.write("")    


    #features_choice = st.multiselect('Choisissez les variables explicatives', df.columns)
    test_size = st.slider('Taille de l\'ensemble de test', 0.1, 0.5, 0.2, 0.05)
    st.write("")  
    st.write("")    
  

    if not features_choice:
        features_choice = list(df.drop('deposit',axis=1).columns)
    st.write("")   
    st.write("")    
 
    
    # Separation des donnees 

    feats = df[features_choice]
    target = df['deposit']
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=test_size, random_state=42)

    imputer = SimpleImputer(strategy="most_frequent")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    cat_cols = feats.select_dtypes(include="object").columns
    ohe = OneHotEncoder(drop='first', sparse=False)
    X_train_encode = pd.DataFrame(ohe.fit_transform(X_train[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    X_test_encode = pd.DataFrame(ohe.transform(X_test[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop (columns=cat_cols)
    X_train = pd.concat([X_train, X_train_encode], axis=1)
    X_test = pd.concat([X_test, X_test_encode], axis=1)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    num_cols = feats.select_dtypes(include=["int64", "float64"]).columns
    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Definir le model
    
    if model_choice == 'LogisticRegression':
        model = regl
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)
        C = st.selectbox(label='**C**', options=(0.001, 0.01, 0.1, 1, 10), index=3)
        model.set_params(C=C)
    elif model_choice == 'DecisionTreeClassifier':
        model = dcl
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)
        max_depth = st.selectbox(label='**Profondeur Maxi**', options=(100, 300, 400, 500,700,800,900), index=4)
        criterion = st.selectbox("**Critère de division**", ["gini", "entropy"],index=1)
        model.set_params(max_depth=max_depth, criterion=criterion)
    elif model_choice == 'RandomForestClassifier':
        model = rfcl
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)
        n_estimators = st.selectbox("**Nombre d'estimateurs**", [100, 300, 400, 500, 700, 800], index=3)
        max_depth = st.selectbox(label='**Profondeur Maxi**', options=(10, 50, 100, 250,300,400,500), index=2)
        model.set_params(n_estimators=n_estimators, max_depth=max_depth)
    
    elif model_choice == 'Super Vector Machine(SVM)':
        model = svm
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)
        C = st.selectbox(label='**C**', options=(0.001, 0.01, 0.1, 1, 10), index=3)
        kernel = st.selectbox("**Noyau**", ["linear", "rbf", "poly"],index=2)
        model.set_params(kernel=kernel, C=C)
    elif model_choice == 'KNeighborsClassifier(Knn)':
        model = knn
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)
        n_neighbors = st.selectbox(label='**Nombre de voisins**', options=(1,10, 50, 100, 250), index=0)
        p = st.selectbox(label='**P**', options=[1, 2, 3, 4, 5], index=0)
        model.set_params(n_neighbors=n_neighbors, p=p)
    elif model_choice == 'GradientBoostingClassifier':
        model = gbc
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)
        n_estimators = st.selectbox("**Nombre d'estimateurs**", [ 300,360,400], index=1)
        learning_rate = st.selectbox("**Taux d'apprentissage**", [0.001, 0.01,0.1,1], index=1)
        max_depth = st.selectbox(label='**Profondeur Maxi**', options=( 7,8,10), index=1)
        model.set_params(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    else:
        model = abc
        st.markdown("<h3 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 80%; margin: 0 auto;'>Choisir les Hyperparametre du Model</h3>", unsafe_allow_html=True)

        n_estimators = st.selectbox("**Nombre d'estimateurs**", [100, 300, 400, 500, 700, 800,900,1000], index=5)
        learning_rate = st.selectbox("**Taux d'apprentissage**", [0.01, 0.1,1,10], index=1)
        model.set_params(learning_rate=learning_rate, n_estimators=n_estimators)

    # Entrainer le model
    model.fit(X_train, y_train)

    # Faire les  predictions
    y_pred = model.predict(X_test)

    # Calculer les  metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    confusion = pd.DataFrame(confusion, columns=['Prédit 0', 'Prédit 1'], index=['Réel 0', 'Réel 1'])
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    df_classification = pd.DataFrame(classification_rep).transpose()
    st.write("")
    st.write("")



    # Afficher les resultats
    if st.button("Afficher les resultats"):
        st.write("")
        st.write("")


        st.markdown("<h2 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 40%; '>Resultat</h2>", unsafe_allow_html=True)
        st.write("")

        #st.sub(f'Model :  {model_choice}')
        st.write(f'<h5 style=" color: #87CEEB;">Model : {model_choice}</h5>', unsafe_allow_html=True)
        #st.write('Modèle choisi:', model_choice)
        st.write('Variables explicatives choisies:', features_choice)
        st.markdown("<h6 style='color: #87CEEB;'>Taille de l\'ensemble de test:</h6>", unsafe_allow_html=True)
        st.write(test_size)
        st.markdown("<h6 style='color: #87CEEB;'>Score Accuracy:</h6>", unsafe_allow_html=True)
        st.write(accuracy.round(3))
        st.markdown("<h6 style='color: #87CEEB;'>Matrice de Confusion:</h6>", unsafe_allow_html=True)
        st.write(confusion)
        st.markdown("<h6 style='color: #87CEEB;'>Rapport de classification:</h6>", unsafe_allow_html=True)
        st.write(df_classification)
