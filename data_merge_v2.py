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
ns_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\data_NS_filtered.csv"
ve_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\Verstoringen.csv"
output_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\data_NS_filtered_merged_v2.csv"

# Read CSVs
ns_df = pd.read_csv(ns_path, sep=";")
ve_df = pd.read_csv(ve_path)

# Merge on all three keys
merged_df = ns_df.merge(
    ve_df[['Dagnr', "BEWEGINGNUMMER", "KLANTHINDERINMINUTEN", "TOELICHTING"]],
    left_on=['DAGNR', "BEWEGINGNUMMER"],
    right_on=['Dagnr', "BEWEGINGNUMMER"],
    how="left"
)

# Keep only disruption info
merged_df.rename(columns={"KLANTHINDERINMINUTEN": "Disruption (minutes)"}, inplace=True)
merged_df.rename(columns={"TOELICHTING": "Disruption description"}, inplace=True)

# Drop duplicate join columns from ve_df (since we already have station1/station2 in ns_df)
merged_df.drop(columns=['Dagnr', "DIENSTREGELPUNT_VAN", "DIENSTREGELPUNT_NAAR"], inplace=True)

#%% Export in .csv

merged_df.to_csv(
    output_path,
    index=False,
    sep=";",  
    quoting=3,  
    encoding="utf-8"
)
# %%
