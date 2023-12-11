import streamlit as st 
#from scripts import modelisation
#from scripts import model 
import introduction
import exploration
import visualization
import modelisation
import conclusion 
import interpretation



st.markdown("<h1 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 100%; margin: 0 auto;'>Projet de Prédiction du succès d'une campagne marketing d'une banque</h1>", unsafe_allow_html=True)

st.write(" ")
st.markdown("<h2 style='text-align: center;'>Avril à Novembre 2023</h2>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")
st.write(" ")
st.image("bank.jpg", use_column_width=True, caption="Campagne marketing")

st.sidebar.markdown("<h1 style='color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 50%;'>Sommaire</h1>", unsafe_allow_html=True)
pages = ["**Introduction**", "**Exploration**", "**Visualization**", "**Modélisation**", "**Interpretation**","**Conclusion**"]
page = st.sidebar.radio("Menu", pages)
if page == pages[0]:
    introduction.run()

elif page == pages[1]:
    exploration.run()

elif page == pages[2]:
    visualization.run()

elif page == pages[3]:
    modelisation.run()

elif page == pages[4]:
    interpretation.run()
else:
    conclusion.run()
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.markdown("<h1 style='text-align: center; color: #87CEEB; background-color: #EAEAEA; border-radius: 10px; padding: 5px; width: 100%; margin: 0 auto;'>Membres du groupe</h1>", unsafe_allow_html=True)
st.sidebar.write("")

# Ajout des liens LinkedIn et GitHub pour chaque personne dans la barre latérale
st.sidebar.info("Stephane KOFFI - stephane.koffi@10000codeurs.com \n"
                "[LinkedIn](https://www.linkedin.com/in/stephane-koffi/) \n"
                "[GitHub](https://github.com/stephenstephk)")
                

st.sidebar.info("Ludovic DURAND - ludovic.durand@itech.fr \n"
                "[LinkedIn](https://www.linkedin.com/in/ludovic-durand/) \n"
               ) #"[GitHub](https://github.com/LudovicDurand)"

st.sidebar.info("Simon Martinez - simonmartinez4@gmail.com \n"
                "[LinkedIn](https://www.linkedin.com/in/simon-martinez-da/) \n"
                "[GitHub](https://github.com/SimonMartinez4)")

st.sidebar.info("Becam Nour - nour.becam@vertuoconseil.com \n"
                "[LinkedIn](https://www.linkedin.com/in/nour-becam-954672157/) \n"
                "[GitHub](https://github.com/NourBecam)")
