#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 2024

@author: Gruppe 9 (Allegra Trepte, Claudia Rutz, Ewgenie Kunkel, Mohamed Latri)
"""

###### Imports ######

import streamlit as st
import pands as pd


###### Config ######

st.set_page_config(
        page_title = "Auto Wiederverkaufswert-Rechner",
        page_icon = "🚗",
        layout = "wide", 
        initial_sidebar_state = "collapsed", #Sidebar initial eingeklappt
    )


###### Definitionen ######
@st.cache_data
def load_data():
    data = pd.read_csv('./Data/sample_data_100k.csv')
    return data


###### Sidebar ######

with st.sidebar: #Sidebar mit Informationen zur App und zu uns
    st.header("About this App")
    st.caption("Das Ziel dieser Data-App ist es, den Wiederverkaufswert eines Autos vorherzugsgen. Die Vorhersage basiert auf einem Machine Learning Algorithums, welcher mit einem umfassenden Datensatz mit Informatioenen über Gebrauchtwagen in den USA trainiert wurde. Die berechneten Werte stellen eine ungefähre Schätzung dar.")
    st.caption("")
    st.subheader("About us")
    st.caption("Wir sind Allegra, Claudia, Ewgenie und Mo und haben diese App im Rahmen eines Projekts der Uni St. Gallen programmiert.")



###### Seiteninhalt ######
st.title("Auto Wiederverkaufswert-Rechner 🚗 ") #Titel
st.markdown("Finde mehr über den Wiederverkaufswert deines Autos heraus.") #Text

tab1, tab2 = st.tabs(["Explorative Datenanalyse", "Wiederverkaufswert-Rechner"]) #Tabs



###### Tab 1 - Explorative Datenanalyse ######
