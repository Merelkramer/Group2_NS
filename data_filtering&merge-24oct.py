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
weather_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\Weather-temp.csv"
output_path1 = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\complete_dataset_clean_24oct_SE.csv"
output_path2 = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\complete_dataset_clean_delaysonly_24oct_SE.csv"


df = pd.read_csv(ns_path)
disrup = pd.read_csv(disrup_path)
weather = pd.read_csv(weather_path)

# Station list
stations = [
    "Dvnk", "Dt", "Dtcp", "Gvc", "Gv", "Laa", "Gvm", "Gvmw", 
    "Hfd", "Ledn", "Nvp", "Rsw", "Rtd", "Ssh", "Sdm", "Shl", "Vst"
]

df.loc[
    (df["TRAJECT"].isin(["Shl_Rtd", "Rtd_Shl"])),
    "train_type"
] = "TGV"

# We also assign the IC based on their trajectory (it doesn't cover exceptions ->>> data cleaning)
mask = (df["TRAJECT"].isin(["Dt_Gv", "Gv_Dt", "Ledn_Shl", "Shl_Ledn", 
                                             "Ledn_Gvc", "Gvc_Ledn","Rtd_Gv", "Gv_Rtd","Dt_Rtd", "Rtd_Dt",
                                             "Dt_Sdm", "Sdm_Dt", "Ledn_Gv", "Gv_Ledn",
                                              "Ledn_Laa", "Laa_Ledn"])    
        & df["train_type"].isna()
)
bewegings_to_update = df.loc[mask, "BEWEGINGNUMMER"].unique()
df.loc[
    df["BEWEGINGNUMMER"].isin(bewegings_to_update),
    "train_type"
] = "IC"

# We assume the rest of them are sprinters
df.loc[
    pd.isna(df["train_type"]) ,
    "train_type"
] = "Sprinter"

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
weather_cols = ['Day', 'HH', 'Rain', 'Gusts', 'Storms', 'Temp', 'Sunshine']
merged_weather = merged_disrup.merge(
    weather[weather_cols],
    left_on=['DAGNR', 'hour'],
    right_on=['Day', 'HH'],
    how='left'
)

merged_weather.drop(columns=['Dagnr', "Day", "hour", "HH"], inplace=True)


# Cleaning / feature prep

special_event_days = [6, 7, 8, 9, 13, 14, 15, 16, 17, 27, 28, 29, 30]

merged_weather["Special Events"] = merged_weather["DAGNR"].isin(special_event_days).astype("int8")

# (a) delay: HH:MM:SS -> minutes
if "delay" in merged_weather.columns:
    merged_weather["delay"] = pd.to_timedelta(merged_weather["delay"].astype(str), errors="coerce").dt.total_seconds() / 60
    merged_weather["delay_missing"] = merged_weather["delay"].isna().astype("int8")
    merged_weather["delay"] = merged_weather["delay"].fillna(0)

    # Categorize delay values into bins
    bins = [0, 1, 5, 10, 30, float("inf")]
    labels = ["No delay (0-1 min)", "Small (1-5 min)", "Medium (5-10 min)", "Large (10-30 min)", "Very Large (+30min)"]

    merged_weather["delay_category"] = pd.cut(
        merged_weather["delay"], bins=bins, labels=labels, right=False
    )

# (c) ensure TREINSERIEBASIS is string (not generic object / mixed)
if "TREINSERIEBASIS" in merged_weather.columns:
    merged_weather["TREINSERIEBASIS"] = merged_weather["TREINSERIEBASIS"].astype("string").fillna("Unknown")


# (e) operator forecast missing (extra trains)
if "PROGNOSE_REIZEN" in merged_weather.columns:
    merged_weather["prognose_missing"] = merged_weather["PROGNOSE_REIZEN"].isna().astype("int8")
    merged_weather["PROGNOSE_REIZEN"] = merged_weather["PROGNOSE_REIZEN"].fillna(0)

# (e) disruptions missing 
if "Disruption (minutes)" in merged_weather.columns:
    merged_weather["disruption_missing"] = merged_weather["Disruption (minutes)"].isna().astype("int8")
    merged_weather["Disruption (minutes)"] = merged_weather["Disruption (minutes)"].fillna(0)

        # Categorize delay values into bins
    bins = [0, 0.1, 60, 120, float("inf")]
    labels = ["No Disruption","Small (0-1 h)", "Medium (1-2 h)", "Large (+2 h)"]

    merged_weather["disruption_category"] = pd.cut(
        merged_weather["Disruption (minutes)"], bins=bins, labels=labels, right=False
    )

# (f) Weather columns -> one-hot/binary flags (no leakage; simple mapping)
#   Rain: values -> {NaN, "Rain", "Heavy Rain"}
#   Gusts: {NaN, "Heavy Wind"}
#   Storms: {NaN, "Thunderstorm"}
if "Rain" in merged_weather.columns:
    merged_weather["Rain_flag"] = merged_weather["Rain"].notna().astype("int8")
    merged_weather["Heavy_Rain_flag"] = (merged_weather["Rain"] == "Heavy Rain").astype("int8")
if "Gusts" in merged_weather.columns:
    merged_weather["Gusts_flag"] = merged_weather["Gusts"].notna().astype("int8")  # 1 if Heavy Wind present
if "Storms" in merged_weather.columns:
    merged_weather["Storms_flag"] = merged_weather["Storms"].notna().astype("int8")  # 1 if Thunderstorm present
if "Temp" in merged_weather.columns:
    merged_weather["Warm_flag"] = (merged_weather["Temp"] == "Warm").astype("int8")  # 1 if Warm present
    merged_weather["Cold_flag"] = (merged_weather["Temp"] == "Cold").astype("int8")  # 1 if Cold present
if "Sunshine" in merged_weather.columns:
    merged_weather["Sunny_flag"] = merged_weather["Sunshine"].notna().astype("int8")  # 1 if Sunny present


# (h) ensure boolean columns are numeric ints for XGBoost
for col in ["Cancelled", "ExtraTrain","Special Events"]:
    if col in merged_weather.columns:
        merged_weather[col] = merged_weather[col].astype("int8")

# (h2) New feature: Previous train canceled
merged_weather = merged_weather.sort_values(
    by=["TRAJECT", "DAGNR", "PLANTIJD_VERTREK"], ascending=[True, True, True]
)

# For each TRAJECT + DAGNR, check if the previous train was canceled
merged_weather["Previous train canceled"] = (
    merged_weather.groupby(["TRAJECT", "DAGNR"])["Cancelled"]
    .shift(1)  # look at the previous train
    .fillna(0)  # first train of the day has no previous
    .astype("int8")
)

# For each TRAJECT + DAGNR, check if the previous train was delayed (min)
merged_weather["Previous train delayed (min)"] = (
    merged_weather.groupby(["TRAJECT", "DAGNR"])["delay"]
    .shift(1)  # look at the previous train
    .fillna(0)  # first train of the day has no previous
    .astype("int8")
)

if "Previous train delayed (min)" in merged_weather.columns:
    # Categorize delay values into bins
    bins = [0, 1, 5, 10, 30, float("inf")]
    labels = ["No delay (0-1 min)", "Small (1-5 min)", "Medium (5-10 min)", "Large (10-30 min)", "Very Large (+30min)"]

    merged_weather["Previous train delayed (cat)"] = pd.cut(
        merged_weather["Previous train delayed (min)"], bins=bins, labels=labels, right=False
    )

# (h2) New feature: Extra train
merged_weather = merged_weather.sort_values(
    by=["TRAJECT", "DAGNR", "UITVOERTIJD_AANKOMST"], ascending=[True, True, True]
)

merged_weather["Extra train added before departure"] = (
    merged_weather.groupby(["TRAJECT", "DAGNR"])["ExtraTrain"]
    .shift(1)  # look at the previous train
    .fillna(0)  # first train of the day has no previous
    .astype("int8")
)

# (b) drop rows missing the target
merged_weather = merged_weather.dropna(subset=["REALISATIE"]).copy()


# (d) fill other categoricals we use
for col in ["DAGDEELTREIN"]:
    if col in merged_weather.columns:
        merged_weather[col] = merged_weather[col].astype("string").ffill()

# (j) reset index after drops (tidy)
merged_weather = merged_weather.reset_index(drop=True)

merged_weather.drop(columns=['prognose_missing', "Disruption description", "disruption_missing", "delay_missing", "Rain", "Gusts", "Storms", "Temp", "Sunshine"], inplace=True)

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

