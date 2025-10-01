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
input_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\Verstoringen.csv"
output_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\Verstoringen_filtered.csv"


df = pd.read_csv(input_path)
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


#%% Export in .csv

filtered.to_csv(
    output_path,
    index=False,
    sep=";",  
    quoting=3,  
    encoding="utf-8"
)