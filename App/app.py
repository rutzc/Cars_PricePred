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
    data = pd.read_csv("./App/df_clean.csv")
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
    st.header("Erkunde hier den Einfluss verschiedener Variablen auf den Preis", divider = "red")
    
    #Selection für x-Variable
    selected_variable = st.selectbox("Wähle eine Variable", list(data.drop("price", axis=1).columns))
    
    #Tickbox für Anzeige Raw Data
    if st.checkbox("Rohdatensatz anzeigen", False):
        st.subheader("Rohdaten")
        st.write(data)
    
    #Numerische und kategorielle Variablen trennen
    numeric_variables = ["age","average_fuel_economy", "horsepower", "mileage"]
    categorical_variables = ["body_type", "engine_type", "fuel_type", "make_name", "model_name", "transmission", "wheel_system_display"]
    
    #Numerische Analyse
    if selected_variable in numeric_variables:
        
        #Titel
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
        
    #Kategorische Analyse
    elif selected_variable in categorical_variables:
        
        #Titel
        st.subheader(f"Analyse kategorischer Variable {selected_variable}")
        
        #Zwei Spalten
        col1, col2 = st.columns([1, 1])
        
        #Barplot
        col1.write(f"Barplot von {selected_variable} und Preis")
        fig3, ax3 = plt.subplots(figsize=(8,3.7))
        avg_price_by_category = data.groupby(selected_variable)["price"].mean().reset_index()        
        sns.barplot(x=selected_variable, y="price", data=avg_price_by_category, ax=ax3, palette="Set2")
        plt.xticks(rotation=90)
        ax3.set_xlabel(selected_variable)
        ax3.set_ylabel("Durchschnittlicher Preis")
        col1.pyplot(fig3, use_container_width=True)
        
        #Boxplot
        col2.write(f"Boxplot von {selected_variable} und Preis")
        fig4, ax4 = plt.subplots(figsize=(8, 3.7))
        sns.boxplot(x=selected_variable, y="price", data=data, ax=ax4, palette="Set2")
        plt.xticks(rotation=90)
        ax4.set_xlabel(selected_variable)
        ax4.set_ylabel("Preis")
        col2.pyplot(fig4, use_container_width=True)
        
        st.write("Test")



###### Tab 2 - Explorative Datenanalyse ######

with tab2:
    #Subheader
    st.header("Willkommen beim Auto Wiederverkaufswert-Rechner", divider = "red")
    st.subheader("Mit dem Widerverkaufswert-Rechner erhälst du eine ungefähre Vorhersage für den Preis, den heute oder in x Jahren für den Verkauf deines Autos erhälst")
    
    #Anweisungen an den User
    st.markdown("Wir bitten dich deshalb, einige Angaben über die Daten deines Fahrzeuges zu machen. Zudem solltest du uns, für eine möglichst exakte Berechnung angeben, Informationen zu deinen Fahrgewohnheiten angeben und wie lange du das Auto noch fahren möchtest.")
    
    #Eingabewerte abfragen für das Training des Modells        
   
    #Marke
    make_name = st.multiselect("Automarke", options=sorted(data["make_name"].unique()), default=[], max_selections=1)
    
    #Modellname
    model_name = st.multiselect("Automodell", options=sorted(data["model_name"].unique()), default=[], max_selections=1)
                                    
    #Karosserietyp
    body_type = st.multiselect("Karosserietyp", options=sorted(data["body_type"].unique()), default=[], max_selections=1)
    
    #Motortyp
    engine_type = st.multiselect("Motortyp", options=sorted(data["engine_type"].unique()), default=[], max_selections=1)
    
    #Motorleistung
    horsepower = st.slider("Motorleistung (in PS)", min_value=int(data["horsepower"].min()), max_value=int(data["horsepower"].max()), step=10, value=int(data["horsepower"].median()))
    
    #Kraftstoffart
    fuel_type = st.radio("Kraftstoffart", options=data["fuel_type"].unique())
    
    #Durchschnittlicher Verbrauch
    average_fuel_economy = st.slider("Durchschnittlicher Verbrauch (in km pro Liter)", min_value=float(data["average_fuel_economy"].min()), max_value=float(data["average_fuel_economy"].max()), step=float(1), value=float(data["average_fuel_economy"].median()))
    
    #Antriebssystem
    wheel_system_display = st.radio("Antriebssystem", options=sorted(data["wheel_system_display"].unique()))
    
    #Manuell oder automatisch
    manual = st.radio("Schaltgetriebe oder Automatikgetriebe", options=["Schaltung", "Automatik"])
    
    #Alter
    age = st.slider("Alter des Fahrzeugs", min_value=int(data["age"].min()), max_value=int(data["age"].max()), step=1, value=0)

    #KM-Stand
    mileage = st.slider("Kilometerstand", min_value=float(data["mileage"].min()), max_value=float(data["mileage"].max()), step=float(100), value=float(0))
    
    
    #Abfrage über Zeitpunkt des Wiederverkaufs
    jahre = st.number_input("In wie vielen Jahren möchtest du dein Auto gerne verkaufen", min_value=0, value=0, step=1)
    
    #Abfrage über jährlich gefahrene Kilometer
    km_jahrlich = st.slider("Wie viele Kilometer fährst du ungefähr jährlich", min_value=0, max_value=60000, value=15000)
    
    

    





