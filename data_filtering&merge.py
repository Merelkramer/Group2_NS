#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:58:12 2025

@author: Laure
"""

#%% Modules importation

import pandas as pd 
from datetime import datetime, date, timedelta, time


#%% Data importation

# /!\ Change the paths here !
ns_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\prog en realisatie ophalen 2.csv"
disrup_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\Verstoringen.csv"
weather_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\Weather.csv"
output_path1 = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\complete_dataset_clean.csv"
output_path2 = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\complete_dataset_clean_delaysonly.csv"


df = pd.read_csv(ns_path)
disrup = pd.read_csv(disrup_path)
weather = pd.read_csv(weather_path)

# Station list
stations = [
    "Dvnk", "Dt", "Dtcp", "Gvc", "Gv", "Laa", "Gvm", "Gvmw", 
    "Hfd", "Ledn", "Nvp", "Rsw", "Rtd", "Ssh", "Sdm", "Shl", "Vst"
]


#%% Filtering the database on the study area
# Split TRAJECT into two parts
df[["station1", "station2"]] = df["TRAJECT"].str.split("_", expand=True)

# Filter rows where BOTH stations are in the list
filtered = df[df["station1"].isin(stations) & df["station2"].isin(stations)]


#%% Add new features
filtered["Cancelled"] = (
    filtered["PLANTIJD_VERTREK"].notna() & filtered["UITVOERTIJD_VERTREK"].isna()
)

filtered["ExtraTrain"] = (
    filtered["PLANTIJD_VERTREK"].isna() & filtered["UITVOERTIJD_VERTREK"].notna()
)

#%% New feature : delay at departure

filtered["delay"]= pd.to_timedelta(
    filtered["UITVOERTIJD_VERTREK"]
    .str.split('.')
    .str[0])- pd.to_timedelta(
    filtered["PLANTIJD_VERTREK"]
    .str.split('.')
    .str[0])
   
    
def timedelta_to_time(td):
    """
    It transforms a (positive) timedelta value to a time value
    Args : 
        td : time in timedelta format
    Retunrs : 
        time in time format
    """
    # Round the early trains (delay at departure <0), and the trains with less than 1 minute delay to 0
    if td < pd.Timedelta("1 min"):
        td = pd.Timedelta(0)
    if pd.isna(td) : 
        return td
    total_seconds = int(td.total_seconds())
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return time(hour=hours, minute=minutes, second=seconds)

filtered["delay"] = filtered["delay"].apply(timedelta_to_time)


# Convertir a tipo datetime
filtered['UITVOERTIJD_VERTREK'] = pd.to_datetime(filtered['UITVOERTIJD_VERTREK'], errors='coerce')

# Extraer solo la hora (número entero)
filtered['hour'] = filtered['UITVOERTIJD_VERTREK'].dt.hour

# Merge disruption data
merged_disrup = filtered.merge(
    disrup[['Dagnr', "BEWEGINGNUMMER", "KLANTHINDERINMINUTEN", "TOELICHTING"]],
    left_on=['DAGNR', "BEWEGINGNUMMER"],
    right_on=['Dagnr', "BEWEGINGNUMMER"],
    how="left"
)

# Keep only disruption info
merged_disrup.rename(columns={"KLANTHINDERINMINUTEN": "Disruption (minutes)"}, inplace=True)
merged_disrup.rename(columns={"TOELICHTING": "Disruption description"}, inplace=True)

#%% Merge weather data
weather_cols = ['Day', 'HH', 'Rain', 'Gusts', 'Storms']
merged_weather = merged_disrup.merge(
    weather[weather_cols],
    left_on=['DAGNR', 'hour'],
    right_on=['Day', 'HH'],
    how='left'
)

merged_weather.drop(columns=['Dagnr', "Day", "hour", "HH"], inplace=True)


# Cleaning / feature prep

# (a) delay: HH:MM:SS -> minutes
if "delay" in merged_weather.columns:
    merged_weather["delay"] = pd.to_timedelta(merged_weather["delay"].astype(str), errors="coerce").dt.total_seconds() / 60
    merged_weather["delay_missing"] = merged_weather["delay"].isna().astype("int8")
    merged_weather["delay"] = merged_weather["delay"].fillna(0)

# (b) drop rows missing the target
merged_weather = merged_weather.dropna(subset=["REALISATIE"]).copy()

# (c) ensure TREINSERIEBASIS is string (not generic object / mixed)
if "TREINSERIEBASIS" in merged_weather.columns:
    merged_weather["TREINSERIEBASIS"] = merged_weather["TREINSERIEBASIS"].astype("string").fillna("Unknown")

# (d) fill other categoricals we use
for col in ["DAGDEELTREIN"]:
    if col in merged_weather.columns:
        merged_weather[col] = merged_weather[col].astype("string").fillna("Unknown")

# (e) operator forecast missing (extra trains)
if "PROGNOSE_REIZEN" in merged_weather.columns:
    merged_weather["prognose_missing"] = merged_weather["PROGNOSE_REIZEN"].isna().astype("int8")
    merged_weather["PROGNOSE_REIZEN"] = merged_weather["PROGNOSE_REIZEN"].fillna(0)

# (e) disruptions missing 
if "Disruption (minutes)" in merged_weather.columns:
    merged_weather["disruption_missing"] = merged_weather["Disruption (minutes)"].isna().astype("int8")
    merged_weather["Disruption (minutes)"] = merged_weather["Disruption (minutes)"].fillna(0)

# (f) Weather columns -> one-hot/binary flags (no leakage; simple mapping)
#   Rain: values -> {NaN, "Rain", "Heavy Rain"}
#   Gusts: {NaN, "Heavy Wind"}
#   Storms: {NaN, "Thunderstorm"}
if "Rain" in merged_weather.columns:
    merged_weather["Rain_flag"] = merged_weather["Rain"].notna().astype("int8")
    merged_weather["Heavy_Rain_flag"] = (merged_weather["Rain"] == "Heavy Rain").astype("int8")
if "Gusts" in merged_weather.columns:
    merged_weather["Gusts_flag"] = merged_weather["Gusts"].notna().astype("int8")  # 1 if Heavy Wind present
if "Storms" in df.columns:
    merged_weather["Storms_flag"] = merged_weather["Storms"].notna().astype("int8")  # 1 if Thunderstorm present

# (g) (optional) simple keyword flags from Disruption description
if "Disruption description" in merged_weather.columns:
    s = merged_weather["Disruption description"].fillna("").str.lower()
    merged_weather["disrupt_any"]   = (s.str.len() > 0).astype("int8")
    merged_weather["disrupt_signal"] = s.str.contains("signal", na=False).astype("int8")
    merged_weather["disrupt_track"]  = s.str.contains("track|rail", na=False).astype("int8")
    merged_weather["disrupt_power"]  = s.str.contains("overhead|power|line", na=False).astype("int8")

# (h) ensure boolean columns are numeric ints for XGBoost
for col in ["Cancelled", "ExtraTrain"]:
    if col in merged_weather.columns:
        merged_weather[col] = merged_weather[col].astype("int8")

# (i) map Week_Dag_nr -> Weekday/Weekend
if "WEEK_DAG_NR" in merged_weather.columns:
    merged_weather["WEEK_DAG_NR"] = merged_weather["WEEK_DAG_NR"].apply(lambda x: "Weekend" if x in [6, 7] else "Weekday")


# (j) reset index after drops (tidy)
merged_weather = merged_weather.reset_index(drop=True)

merged_weather.drop(columns=['prognose_missing', "Disruption description", "disruption_missing", "delay_missing", "Rain", "Gusts", "Storms"], inplace=True)

delays_only = merged_weather[merged_weather["delay"] != 0]

#%% Export in .csv

merged_weather.to_csv(
    output_path1,
    index=False,
    sep=";",  
    quoting=3,  
    encoding="utf-8"
)

delays_only.to_csv(
    output_path2,
    index=False,
    sep=";",  
    quoting=3,  
    encoding="utf-8"
)

