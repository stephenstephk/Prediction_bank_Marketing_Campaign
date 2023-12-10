import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def run():
    @st.cache_data()
    def load_data():
        df = pd.read_csv('./datasets/bank_modif.csv')
        return df

    # Load the data
    df = load_data()
  
    st.markdown("<h1 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%; margin: 0 auto;'>Interpretation</h1>", unsafe_allow_html=True)

    feats = df.drop("deposit", axis=1)
    target = df["deposit"]
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)
    # Intanciation de l'imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    cat_cols = feats.select_dtypes(include="object").columns
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    X_train_encode = pd.DataFrame(ohe.fit_transform(X_train[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    X_test_encode = pd.DataFrame(ohe.transform(X_test[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    X_train = X_train.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)
    X_train = pd.concat([X_train, X_train_encode], axis=1)
    X_test = pd.concat([X_test, X_test_encode], axis=1)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    num_cols = feats.select_dtypes(include=["int", "float"]).columns
    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    reglog = LogisticRegression()
    reglog.fit(X_train, y_train)
    joblib.dump(reglog, "LogisticRegression")
    logis_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],  # Inverse de la force de régularisation
        "solver": ["lbfgs", "liblinear", "sag", "newton-cg"],
        'penalty': ['l2'],  # Type de pénalité
    }
    logis_grid_search = GridSearchCV(estimator=reglog, param_grid=logis_param_grid, scoring='accuracy', cv=5)
    logis_grid_search.fit(X_train, y_train)
    logis_best_params = logis_grid_search.best_params_
    logis_best_model = logis_grid_search.best_estimator_
    y_pred = logis_best_model.predict(X_test)
    logis_accuracy = accuracy_score(y_test, y_pred)
    reglog = LogisticRegression(**logis_best_params)
    reglog.fit(X_train, y_train)

    explainer = shap.LinearExplainer(reglog, X_train)
    shap_values_model_2 = explainer.shap_values(X_test)
    explainer = shap.Explainer(reglog, X_train)
    shap_values_model_2 = explainer(X_test)

    st.write('Les paramètres ayant le plus de poids sur la décision finale selon le modèle de régression logistique :')
    fig = plt.figure()
    shap.summary_plot(shap_values_model_2, X_test, plot_type="bar")
    st.pyplot(fig)
    st.write('Une autre représentation de cette distribution est le diagramme dit de waterfall :')
    fig = plt.figure()
    shap.plots.waterfall(shap_values_model_2[0])
    st.pyplot(fig)
