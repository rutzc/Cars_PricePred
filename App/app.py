#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 2024

@author: Gruppe 9 (Allegra Trepte, Claudia Rutz, Ewgenie Kunkel, Mohamed Latri)
"""

###### Imports ######

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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



###### Daten laden ######
data = load_data()



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

with tab1:
    #Subheader
    st.header("Erkunde hier den Einfluss verschiedener Variablen auf den Preis", dvider = True)
    
    #Selection für x-Variable
    selected_variable = st.selectbox("Wähle eine Variable", list(data.drop("price").columns))
    
    #Tickbox für Anzeige Raw Data
    if st.checkbox("Rohdatensatz anzeigen", False):
        st.subheader("Rohdaten")
        st.write(data)
    
    #Numerische und kategorielle Variablen trennen
    numeric_variables = ["age","average_fuel_economy", "horsepower", "mileage"]
    categorical_variables = ["body_type", "engine_type", "fuel_type", "make_name", "model_name", "transmission", "wheel_system_display"]
    
    #Numerische Analyse
    if selected_variable in numeric_variables:
        st.subheader(f"Analyse numerischer Variable {selected_variable}")
        
        #Zwei Spalten
        col1, col2 = st.columns([1, 1])
        
        #Histogramm
        col1.write(f"Histogramm von {selected_variable}")
        fig1, ax1 = plt.subplots(figsize=(8,3.7))
        ax1.hist(data[selected_variable], color = "#fc8d62")
        ax1.set_xlabel(selected_variable)
        ax1.set_ylabel("Häufigkeit")
        col1.pyplot(fig1, use_container_width=True)
        
        #Scatterplot mit Preis
        col2.write(f"Scatterplot Preis vs. {selected_variable}")
        fig2, ax2 = plt.subplots(figsize=(8, 3.7))
        sns.regplot(x=selected_variable, y="price", data=data, ax=ax2, scatter_kws={'color': '#66c2a5'}, line_kws={'color': '#fc8d62'})
        ax2.set_xlabel(selected_variable)
        ax2.set_ylabel("Preis")
        col2.pyplot(fig2, use_container_width=True)
        
        
##weiter hier:
        st.subheader("Kategorische Variablen")
    categorical_columns = ['model', 'brand']  # Beispielhafte kategorische Variablen
    selected_categorical = st.selectbox("Wähle eine kategorische Variable", categorical_columns)

    if selected_categorical:
        st.write(f"Boxplot von {selected_categorical} gegen Preis")
        fig4, ax4 = plt.subplots(figsize=(8, 3.7))
        sns.boxplot(x=selected_categorical, y='price', data=data, ax=ax4, palette="Set3")
        plt.xticks(rotation=90)
        ax4.set_xlabel(selected_categorical)
        ax4.set_ylabel('Preis')
        st.pyplot(fig4, use_container_width=True)

        st.write(f"Barplot von {selected_categorical} gegen Preis")
        fig5, ax5 = plt.subplots(figsize=(8, 3.7))
        avg_price_by_category = data.groupby(selected_categorical)['price'].mean().reset_index()
        sns.barplot(x=selected_categorical, y='price', data=avg_price_by_category, ax=ax5, palette="Set3")
        plt.xticks(rotation=90)
        ax5.set_xlabel(selected_categorical)
        ax5.set_ylabel('Durchschnittspreis')
        st.pyplot(fig5, use_container_width=True)




