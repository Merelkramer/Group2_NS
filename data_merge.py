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
output_path = r"C:\Users\Alejandro Fiatt\Documents\GitHub\Group2_NS\data_NS_filtered_merged.csv"

# Read CSVs
ns_df = pd.read_csv(ns_path)
ve_df = pd.read_csv(ve_path)

print("NS columns:", ns_df.columns.tolist())
print("VE columns:", ve_df.columns.tolist())

# Merge on the 3 keys
merged_df = ns_df.merge(
    ve_df[["DIENSTREGELPUNT_VAN", "DIENSTREGELPUNT_NAAR", "BEWEGINGNUMMER", "KLANTHINDERINMINUTEN"]],
    left_on=["station1", "station2", "BEWEGINGNUMMER"],
    right_on=["DIENSTREGELPUNT_VAN", "DIENSTREGELPUNT_NAAR", "BEWEGINGNUMMER"],
    how="left"
)
#%% Export in .csv

merged_df.to_csv(
    output_path,
    index=False,
    sep=";",  
    quoting=3,  
    encoding="utf-8"
)
# %%
