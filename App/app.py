#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 2024

@author: Gruppe 9 (Allegra Trepte, Claudia Rutz, Ewgenie Kunkel, Mohamed Latri)

Quellen: Code Vorlesung, Streamlit Documentation (https://docs.streamlit.io), ChatGPT (https://chatgpt.com/share/46393070-1539-4237-9ace-b206d99801bb)
"""

###### Imports ######

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np


###### Config ######

st.set_page_config(
        page_title = "Auto Wiederverkaufswert-Rechner",
        page_icon = "🚗",
        layout = "wide", 
        initial_sidebar_state = "collapsed", #Sidebar initial eingeklappt
    )



###### Definitionen der Load-Functions ######

#Definition für das Laden der Daten 
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("./App/clean_data.csv")
    return data

#Definition für das Laden des Modells
@st.cache(allow_output_mutation=True)
def load_model():
    filename = "./App/model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)



###### Daten und Modell laden ######
data = load_data()
model = load_model()


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

tab1, tab2 = st.tabs(["Wiederverkaufswert-Rechner", "Explorative Datenanalyse"]) #Tabs



###### Tab 1 - Rechner ######

with tab1:
    #Subheader
    st.header("Willkommen beim Auto Wiederverkaufswert-Rechner", divider = "red")
    st.subheader("Mit dem Widerverkaufswert-Rechner erhälst du eine ungefähre Vorhersage für den Preis, den heute oder in der Zukunft für den Verkauf deines Autos erhälst")
    
    #Anweisungen an den User
    st.markdown("Wir bitten dich deshalb, einige Angaben über die Daten deines Fahrzeuges zu machen. Zudem solltest du uns, für eine möglichst exakte Berechnung angeben, Informationen zu deinen Fahrgewohnheiten angeben und wie lange du das Auto noch fahren möchtest.")
    st.divider()
    
    #Eingabewerte abfragen für das Training des Modells
    #Grid Row 1 - Dropdowns für Automarke, -modell, Karosserietyp
    row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])
    #Marke
    make_name = row1_col1.selectbox("Automarke", options=[" "] + sorted(data["make_name"].unique()), index=0) #Unique values jeweils sortiert ausgeben; bei Optionen einen leeren Wert hinzugefügt, damit per Default noch kein Wert ausgewählt wird
    #Modellname
    model_name = row1_col2.selectbox("Automodell", options=[" "] + sorted(data[data["make_name"]==make_name]["model_name"].unique()), index=0) #Filter basierend auf Marke                    
    #Karosserietyp
    body_type = row1_col3.selectbox("Karosserietyp", options=[" "] + sorted(data[data["model_name"]==model_name]["body_type"].unique()), index=0) #Filter basierend auf Modell
    st.divider() #Für Übersichtlichkeit
    
    #Grid Row 2 - Slider für Motorleistung und durchschnittl. Verbrauch
    row2_col1, row2_col2 = st.columns([1,1])
    #Motorleistung
    horsepower = row2_col1.slider("Motorleistung (in PS)", min_value=int(data["horsepower"].min()), max_value=int(data["horsepower"].max()), step=1, value=int(data["horsepower"].min())) #Min, Max basierend auf den Werten aus dem Dataset, für Default = Median aus den Daten für dieses Modell
    #Durchschnittlicher Verbrauch
    average_fuel_economy = row2_col2.slider("Durchschnittlicher Verbrauch (in km pro Liter)", min_value=float(data["average_fuel_economy"].min()), max_value=float(data["average_fuel_economy"].max()), step=float(1), value=float(data["average_fuel_economy"].min()))
    st.divider()
    
    #Grid Row 3
    row3_col1, row3_col2, row3_col3 = st.columns([1,1,1])
    #Kraftstoffart
    fuel_type = row3_col1.radio("Kraftstoffart", options=data["fuel_type"].unique()) 
    #Antriebssystem
    wheel_system_display = row3_col2.radio("Antriebssystem", options=sorted(data["wheel_system_display"].unique())) 
    #Manuell oder automatisch
    manual = row3_col3.radio("Schaltgetriebe oder Automatikgetriebe", options=["Schaltung", "Automatik"]) #Optionen Schaltung und Automatik, nicht 1, 0
    if manual == "Schaltung": #je nach Auswahl, Wert von manual auf 0 oder 1
        manual = 1
    else:
        manual = 0
    st.divider()
    
    #Grid Row 4 - Slider und Number Input für Alter und Mileage
    row4_col1, row4_col2 = st.columns([1,1])
    #Alter
    age = row4_col1.slider("Alter des Fahrzeugs", min_value=int(data["age"].min()), max_value=int(data["age"].max()), step=1, value=0)
    #Kilometerstand
    mileage = row4_col2.number_input("Kilometerstand", value=None, placeholder="Gib eine Zahl ein...", step=1000) #Number Input mit Textfeld
    st.divider()
    
    #Grid Row 5 - Slider für Haltungsjahre und durchschnittliche km jährlich
    row5_col1, row5_col2 = st.columns([1,1])
    #Abfrage über Zeitpunkt des Wiederverkaufs
    jahre = row5_col1.slider("In wie vielen Jahren möchtest du dein Auto gerne verkaufen?", min_value=0, max_value=10, value=0, step=1) #nicht weiter als 20 Jahr in Zukunft
    #Abfrage über jährlich gefahrene Kilometer
    km_jahrlich = row5_col2.slider("Wie viele Kilometer fährst du ungefähr jährlich?", min_value=0, max_value=60000, value=15000, step=1000)
    st.divider()
    
    
    #Bestätigung
    on = st.toggle("Ich bestätige hiermit, dass ich die Werte vollständig und korrekt erfasst habe") #Schalter für Bestätigung
    
    
    #Ausgabe momentaner Wert des Autos
    if on: 
        #Alle User Inputs in ein DataFrame für spätere Vorhersage
        user_input = pd.DataFrame({"make_name": [make_name], 
                              "model_name": [model_name], 
                              "body_type": [body_type], 
                              "horsepower": [horsepower], 
                              "average_fuel_economy": [average_fuel_economy], 
                              "fuel_type": [fuel_type], 
                              "wheel_system_display": [wheel_system_display], 
                              "manual": [manual], 
                              "age": [age], 
                              "mileage": [mileage]})
        #Konvertierung Datentypen
        user_input = user_input.astype({"make_name": "object", 
                              "model_name": "object", 
                              "body_type": "object", 
                              "horsepower": "int", 
                              "average_fuel_economy": "float", 
                              "fuel_type": "object", 
                              "wheel_system_display": "object", 
                              "manual": "int", 
                              "age": "int", 
                              "mileage": "float"})
    
        #Dummy-Variablen erstellen
        user_input = pd.get_dummies(user_input, drop_first = True)
    
        #Alle Dummy-Spalten ergänzen und mit 0 füllen 
        dummy_columns = pd.get_dummies(data.drop(columns=["price"]), drop_first = True).columns #Dummy-Columns genau gleich wie in ursprünglichem Dataset
        user_input = user_input.reindex(columns=dummy_columns, fill_value=0) #Dummy-Columns übernehmen und mit 0 füllen

        #Titel Preis-Vorhersage (heute)
        st.divider()
        st.subheader("Vorhersage für den aktuellen Preis deines Autos basierend auf deinen Angaben")      

        #Preis-Vorhersage (heute), sobald alle User Inputs eingegeben
        usd_chf = 0.91 #USD-CHF-Kurs am 19.05.2024 für Annäherung an CHF-Preis des Autos (Quelle: https://www.finanzen.ch/devisen/dollarkurs)
        if not user_input.empty:
            price_now_usd = model.predict(user_input) #Berechnung des aktuellen Preises über Modell
            price_now_chf = price_now_usd * usd_chf #in CHF umrechnen
            price_now_formatted = f"{price_now_chf[0]:,.0f}".replace(",", "'") #Tiefkomma mit Hochkamma ersetzen
            st.markdown(f"Der aktuelle Preis für dein Auto liegt bei :red-background[**{price_now_formatted}** CHF]") #Ausgabe mit optischer Hervorhebugn des Preises
            st.markdown(" ") #Zeilenumbruch 
    
        #Vorhersage in x Jahren
        if st.button("Berechne Wiederverkaufswert"): #Button für Auslösen der Berechnung
            #Berechnung zukünftiges Alter und Kilometerstand
            age_verkauf = age + jahre
            km_verkauf = mileage + (jahre * km_jahrlich)
        
            #User Inputs in DataFrame updaten
            user_input_updates = user_input.copy()
            user_input_updates["age"] = age_verkauf
            user_input_updates["mileage"] = km_verkauf

            #Titel Verkaufswert-Vorhersage
            st.divider()
            st.subheader("Vorhersage für den Wiederverkaufswert deines Autos basierend auf deinen Angaben")
        
            #Berechnung
            usd_chf = 0.91 #USD-CHF-Kurs am 19.05.2024 für Annäherung an CHF-Preis des Autos (Quelle: https://www.finanzen.ch/devisen/dollarkurs)
            if not user_input.empty:
                price_usd = model.predict(user_input_updates) #Berechnung des künftigen Preises über Modell
                price_chf = price_usd * usd_chf #in CHF umrechnen
                price_formatted = f"{price_chf[0]:,.0f}".replace(",", "'") #Tiefkomma mit Hochkamma ersetzen
                st.markdown(f"Der Wiederverkaufswert deines Autos in {jahre} Jahren liegt bei :red-background[**{price_formatted}** CHF]") #Ausgabe mit optischer Hervorhebugn des Preises
                
                
                #Anzeige eines Plots, der einem die Preise über die Zeit zeigt von heute bis in gewünschtes Verkaufsjahr + 5
                #Variablen initialisieren
                jahre_plus = jahre + 5 #Gewünschtes Verkaufsjahr + 5
                jahr_range = np.arange(0, jahre_plus+1) #1+ wegen Range
                prices = [] #Leere Liste, in die Preise hinzugefügt werden können
        
                #Für jedes Jahr DataFrame erstellen -> mittels Modell Preis-Vorhersage erstellen -> Preis zur Liste hinzufügen
                for jahr in jahr_range: #Für jedes Jahr in der Range der Jahr durchgehen
                    #Alter und neue Mileage ausrechnen
                    age_verkauf = age + jahr
                    km_verkauf = mileage + (jahr * km_jahrlich)
            
                    #Alle User Inputs in ein DataFrame für spätere Vorhersage
                    auto_user = pd.DataFrame({"make_name": [make_name], 
                                      "model_name": [model_name], 
                                      "body_type": [body_type], 
                                      "horsepower": [horsepower], 
                                      "average_fuel_economy": [average_fuel_economy], 
                                      "fuel_type": [fuel_type], 
                                      "wheel_system_display": [wheel_system_display], 
                                      "manual": [manual], 
                                      "age": [age_verkauf], 
                                      "mileage": [km_verkauf]})
                    #Konvertierung Datentypen
                    auto_user = auto_user.astype({"make_name": "object", 
                                      "model_name": "object", 
                                      "body_type": "object", 
                                      "horsepower": "int", 
                                      "average_fuel_economy": "float", 
                                      "fuel_type": "object", 
                                      "wheel_system_display": "object", 
                                      "manual": "int", 
                                      "age": "int", 
                                      "mileage": "float"})
                
                    #Dummy-Variablen erstellen
                    auto_user = pd.get_dummies(auto_user, drop_first = True)
            
                    #Alle Dummy-Spalten ergänzen und mit 0 füllen 
                    dummy_columns = pd.get_dummies(data.drop(columns=["price"]), drop_first = True).columns
                    auto_user = auto_user.reindex(columns=dummy_columns, fill_value=0) 
                
                    #Preis zur Preisliste hinzufügen
                    if not auto_user.empty:
                        price = model.predict(auto_user)
                        price = price * usd_chf
                        prices.append(price[0]) #Preis zu Liste hinzufügen über append Funktion (Quelle: https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.DataFrame.append.html)
                
                #Plot erstellen mit Jahren und Preisen
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(jahr_range, prices, marker="o")
                ax.set_title("Entwicklung Wiederverkaufswert")
                ax.set_xlabel("Jahre ab heute")
                ax.set_ylabel("Wiederverkaufswert (CHF)")
                st.pyplot(fig, use_container_width=True)
                st.divider()
    


###### Tab 2 - Explorative Datenanalyse ######

with tab2:
    #Subheader
    st.header("Erkunde hier den Einfluss verschiedener Variablen auf den Preis", divider = "red")
    
    #Selection für x-Variable
    #Dictionary für die Anzeige der Variablennamen
    variable_names = {"age": "Alter des Fahrzeugs", 
                      "average_fuel_economy": "Durchschnittlicher Verbrauch", 
                      "body_type": "Karosserietyp",
                      "fuel_type": "Kraftstoffart", 
                      "horsepower": "Motorleistung (PS)", 
                      "make_name": "Automarke",
                      "mileage": "Kilometerstand",
                      #"model_name": "Automodell", #Weglassen wegen Darstellungsproblemen beim Plot (zu viele unique-values)
                      "manual": "Getriebeart",
                      "wheel_system_display": "Antriebssystem"}
    #Reverse Dictionary, um von Auswahl auf die ursprüngliche Variable zu schliessen
    reverse_variable_names = {v: k for k, v in variable_names.items()}
    #Dorpdown für Auswahl, sortiert
    selected_variable = st.selectbox("Wähle eine Variable", list(sorted(variable_names.values())))
    #Auswahl im reverse dict
    selected_variable =reverse_variable_names[selected_variable]
    
    #Numerische und kategorielle Variablen trennen in je einer Liste
    numeric_variables = ["age","average_fuel_economy", "horsepower", "mileage"]
    categorical_variables = ["body_type", "fuel_type", "make_name", "manual", "wheel_system_display"]
    
    #Divider für besser Darstellung
    st.divider()
    
    
    #Numerische Analyse, wenn die ausgewählte Variable in der Liste für numerische Variablen ist
    if selected_variable in numeric_variables:
        
        #Titel
        st.subheader(f"Analyse numerischer Variable {selected_variable} (N = {len(data)})")
        
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
        #Scatterplot lädt sehr lange, deshalb nur mit einem Sample der Daten
        sample_data = data.sample(frac=0.3)
        preis_chf = sample_data["price"] * usd_chf
        col2.write(f"Scatterplot Preis vs. {selected_variable} (n = {len(sample_data)})")
        fig2, ax2 = plt.subplots(figsize=(8, 3.7))
        sns.regplot(x=selected_variable, y=preis_chf, data=sample_data, ax=ax2, scatter_kws={'color': '#66c2a5'}, line_kws={'color': '#fc8d62'})
        ax2.set_xlabel(selected_variable)
        ax2.set_ylabel("Preis (CHF)")
        col2.pyplot(fig2, use_container_width=True)
        
        
    #Kategorische Analyse, wenn eine Variable gewählt wird, die in Liste für kategorische Variablen ist
    elif selected_variable in categorical_variables:
        
        #Titel
        st.subheader(f"Analyse kategorischer Variable {selected_variable} (N = {len(data)})")
        
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

        
            
    #Tickbox für Anzeige Raw Data
    if st.checkbox("Rohdatensatz anzeigen", False):
        st.subheader("Rohdaten")
        st.write(data)
    


