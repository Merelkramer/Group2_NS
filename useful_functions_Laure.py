#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:07:51 2025

@author: Laure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point
import branca
import folium
from folium import plugins
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta, time

def convert_time_columns(df) : 
    df["UITVOERTIJD_VERTREK"] = pd.to_timedelta(df["UITVOERTIJD_VERTREK"].astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})')[0], errors='coerce')
    df['PLANTIJD_VERTREK'] = pd.to_timedelta(df['PLANTIJD_VERTREK'].str.split('.').str[0], errors='coerce')
    df['UITVOERTIJD_AANKOMST'] = pd.to_timedelta(df['UITVOERTIJD_AANKOMST'].str.split('.').str[0], errors='coerce')
    return df



def attribute_train_type(df) :
    """
    It creates a columns train_type in a dataframe, based on the trajectory on the train. 
    This function doesn't cover errors such as exceptionnal stops encountered by IC on stations usually
    used by Sprinter
    Args : 
        df (DataFrame) 
    Returns : 
        df (DataFrame)
    """
    # ----TGV---- 
    df.loc[
        (df["TRAJECT"].isin(["Shl_Rtd", "Rtd_Shl"])),
        "train_type"
    ] = "TGV"
    
    # ----IC----
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
    
    # ----Sprinters----
    df.loc[
        pd.isna(df["train_type"]) ,
        "train_type"
    ] = "Sprinter"
    return df 


def get_stations_geometries (stations_codes,gdf_points,df) :
    """
    It associates to each station that can be found in the main dataframe it's geometry POINT
    Args : 
        stations_codes (DataFrame) : Codes of stations associated to their commercial name
        gdf_points (GeoDataFrame) : Geometries of stations associated to their commercial name
        df (DataFrame) : 
    Returns : 
        stations_geometrie (DataFrame) : POINT Geometries of all the stations 
    """
    # The list of the stations we use is the list of the uniques elements of column "station1".
    stations = df["station1"].unique().tolist()
    
    # Associate the stations of the list with their full commercial name
    stations_geometries = stations_codes.loc[stations_codes["Code"].isin(stations)]
    
    # Merge the list of stations with their geometry stocked in gdf_points 
    stations_geometries = gpd.GeoDataFrame(stations_geometries.merge(
        gdf_points[["name","geometry"]], 
        left_on="Commercial Name", 
        right_on="name"), geometry='geometry')
    return stations_geometries




def split_in_intermediate_sections(df,infra,infra_dict) : 
    """
    It creates a table that will keep the same trajects, but splitted into the infrastructure they go through
    Args : 
        df (DataFrame) : Original dataframe
        infra (list) : List of all the infrastructures defined by the succession of stations
        infra_dict (dict) : Dictionnary that associate each station to it's name
    Returns : 
        intermediate_sections (DataFrame) : Table with the trajects split on the infrastructure
    """
    intermediate_sections = df.copy() 
    
    # Create a list that will store all the rows corresponding to some infrastructure sub-sections of 
    # the already existing TRAJECTS
    new_rows_all = [] 
    
    # This list will store all the rows that were divied into subsections. Some of them already fit the 
    # infrastructure (ex : traject between Dt and Dtcp), so it won't be split.
    rows_to_delete = []
    
    for i, row in intermediate_sections.iterrows():
        found = False
        for infrastructure in infra: # Find the infrastructure that contains this traject
            if found == False and row["station1"] in infrastructure and row["station2"] in infrastructure[infrastructure.index(row["station1"])+2:]:
                intermediate_stops = infrastructure[infrastructure.index(row["station1"]): infrastructure.index(row["station2"])+1]
                for j in range(1, len(intermediate_stops)):
                    section = f"{intermediate_stops[j-1]}_{intermediate_stops[j]}"
                    #Each inserted row corresponds to a infrastructure section crossed by the train
                    inserted_row = row.copy()
                    inserted_row["TRAJECT_infra"] = section
                    inserted_row["station1"] = intermediate_stops[j-1]
                    inserted_row["station2"] = intermediate_stops[j]
                    new_rows_all.append(inserted_row)
                    df.loc[i, "global_infra_name"] = [k for k, v in infra_dict.items() 
                                                 if v == infrastructure][0] # Name of the infra 
                rows_to_delete.append(i)
                found = True
            if found==False and row["station1"] in infrastructure :  
                if row["station2"] in infrastructure[infrastructure.index(row["station1"])+1:]:
                    df.loc[i, "global_infra_name"] = [k for k, v in infra_dict.items() if v == infrastructure][0]
                    found = True 
    
    intermediate_sections=intermediate_sections.drop(rows_to_delete)
    intermediate_sections = pd.concat([intermediate_sections, pd.DataFrame(new_rows_all)], ignore_index=True)
    intermediate_sections.loc[pd.isna(intermediate_sections["TRAJECT_infra"]), "TRAJECT_infra"] = intermediate_sections["TRAJECT"]
    return intermediate_sections




def attribute_n_of_tracks (intermediate_sections) : 
    """
    It adds the attribute "number_of_tracks" to the sections of intermediate_sections. Based on observations on 
    openrailwaymap.org, we define the list of sections that are composed of only 1 track per direction
    Args : 
        intermediate_sections (DataFrame) : 
    Returns : 
        intermediate_sections (DataFrame) : 
    """
    number_of_tracks = pd.DataFrame(intermediate_sections["TRAJECT_infra"].unique(),columns=["TRAJECT_infra"])
    
    # Based on observations on openrailwaymap.org, all the sections have 2 tracks, 
    # except a few of them listed below
    number_of_tracks["n_of_tracks"] = 2
    one_track_sections = ["Dtcp_Sdm", "Sdm_Dtcp","Ledn_Ssh", "Ssh_Ledn", "Ssh_Nvp", "Nvp_Ssh","Hfd_Rtd","Hfd_Rtd"]
    number_of_tracks.loc[number_of_tracks["TRAJECT_infra"].isin(one_track_sections), "n_of_tracks"] = 1
    
    # Merge with intermediate_sections to create the attribute
    intermediate_sections= pd.merge(
        number_of_tracks, intermediate_sections, 
        on="TRAJECT_infra"
    )
    return intermediate_sections




def geom_intermediate_sections(intermediate_sections,stations_geometries) : 
    """
    It merges the sections kept in intermediate sections with the geometries of both their departure and arrival stations
    Args : 
        intermediate_sections (DataFrame) : 
        stations_geometries (DataFrame) : POINT geometries of all the stations on the scope of study
    Returns : 
        gdf_intermediate_sections (DataFrame) : 
    """
    gdf_intermediate_sections = intermediate_sections.merge(
            stations_geometries, 
            left_on="station1",
            right_on = "Code",
            ).rename(columns = {
                "Commercial Name" : "Origin_com_name", 
                "Code" : "Origin_code",
                "geometry" : "geometry_origin"
                })
        
    gdf_intermediate_sections = gdf_intermediate_sections.merge(
            stations_geometries, 
            left_on="station2",
            right_on = "Code",
            ).rename(columns = {
                "Commercial Name" : "Destin_com_name", 
                "Code" : "Destin_code",
                "geometry" : "geometry_destination"
                })
    return gdf_intermediate_sections



def line_geometries(gdf_intermediate_sections) :
    """
    It takes all of the unique segments kept in intermediate_sections, and calculate their geometries
    Args : 
        gdf_intermediate_sections (GeoDataFrame) : 
    Returns : 
        gdf_segments (GeoDataFrame) : gdf with the geometries of all segments that will be represented on the map
    """
    gdf_segments = gdf_intermediate_sections[["TRAJECT_infra","geometry_origin", 
                                          "geometry_destination","station1","station2"]].drop_duplicates()
    
    # Based on the geometries of arrival and departure stations, we create a line geometry
    gdf_segments['geometry'] = gdf_segments.apply(
        lambda row: LineString([row['geometry_origin'], row['geometry_destination']]), 
        axis=1
    )
    gdf_segments = gpd.GeoDataFrame(gdf_segments, geometry='geometry')
    return gdf_segments


## -- Functions to display the maps -- 

def merge_with_geometries(tab,gdf_segments) : 
    """
    It merges a table tab, containing information about sections, with the table containing 
    the geometries. 
    Args : 
        tab (DataFrame) : Dataframe with informations about the intermediate sections
    Returns : 
        new_gdf (GeoDataFrame) : Same dataframe but with each section associated to it's geometry
    """
    new_gdf = gdf_segments.merge(
    tab,
    left_on="TRAJECT_infra",
    right_on="TRAJECT_infra",
    how="left"
    )
    new_gdf = new_gdf.set_crs("EPSG:4326", allow_override=True)
    return new_gdf

def display_map(gdf_segments, attribute,alias, valuemin, valuemax) : 
    """
    It displays the map using folium
    Code from the 1st workshop on September 9 
    Args : 
        gdf_segments (GeoDataFrame) : Dataframe with informations and geometries of the intermediate sections 
        attribute (str) : the name of the attribute (as it appears in the dataframe) that will be displayed
        alias (str) : How the attribute should be display in the legend of the ToolTip
        valuemin (int) : Minimum value of the attribute (for the colormap)
        valuemax (int) : Maximum value of the attribute (for the colormap)
    Returns : 
        poly_map (folium.Map) 
    """
    # create learn colormap interpolating 3 colors
    colors = branca.colormap.LinearColormap(
        ['green', 'yellow', 'red'], vmin=valuemin, vmax=valuemax)
        
    # define style function
    def raster_choropleth(row):
        return {
            "color": colors(row['properties'][attribute]),
            "weight" : 5,
            "fillOpacity": 0.75,
        }
    
    # create base map    
    poly_map = folium.Map(
            location=[52.10, 4.30], # The map is centered on Leiden 
            zoom_start=10
        )
    
    # overlay choropleth
    gjson = folium.features.GeoJson(
        gdf_segments[['TRAJECT_infra', attribute, 'geometry']],
        style_function=raster_choropleth,
        ).add_to(poly_map)

    # Draw the triangle arrow
    for idx, row in gdf_segments.iterrows():
        dest_point = row["geometry"].coords[-1]
        origin_point= row["geometry"].coords[0]    
        line_coords = list(row['geometry'].coords)
        if len(line_coords) >= 2:
            p1, p2 = line_coords[-2], line_coords[-1]
            import math
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))             
            folium.RegularPolygonMarker(
                location=[dest_point[1], dest_point[0]],
                color=colors(row[attribute]),
                fill_opacity=1,
                fill_color = colors(row[attribute]),
                number_of_sides=3,  
                radius=5,
                rotation=-angle,
                weight = 1,
                popup=row['TRAJECT_infra']
            ).add_to(poly_map)
    
    # add Tooltip
    folium.features.GeoJsonTooltip(
        fields=['TRAJECT_infra',attribute],
        aliases=['Segment', alias]
    ).add_to(gjson)
    
    poly_map.add_child(colors)      
    return poly_map

def simple_offset_perpendicular(point, offset, direction_vector):
    """ 
    It adds an offset to a linestring. This function is used in the function apply_offset_perpendicular
    Args : 
        point (shapely.geometry.point.Point) : geometry of the station 
        offset (float) : offset applied in meters
        direction_vector (tupe) : directional vectore of the line (dx, dy)
    Returns : 
        point (shapely.geometry.point.Point) : geometry of the station with the offset 
    """
    dx, dy = direction_vector
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return point
    # normalized perpendicular vector
    px = -dy / length
    py = dx / length

    # approximative conversion from meters to degrees
    dlat = py * offset / 111000
    dlon = px * offset / (111000 * np.cos(np.radians(point.y)))
    
    return Point(point.x + dlon, point.y + dlat)

def apply_offset_perpendicular(gdf_segments, offset=200):
    """ 
    It adds an offset to parrallel line (so that they are not displayed with an overlap on the map). 
    Args : 
        gdf_segments (GeoDataFrame) : Dataframe with the geometries of the sections 
        offset (float) : offset that will be applied 
    Returns : 
        gdf_segments (GeoDataFrame) : Dataframe with the geometries of the sections, some 
        of them with an offset 
    """
    new_geometries = []
    
    for i, row in gdf_segments.iterrows():
        line = row['geometry']
        coords = list(line.coords)
        
        # direction of the line for the first to the end point
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]

        # for the lines in the opposite directions, we take the reverse value
        applied_offset = offset if row.get("direction", "Forward") == "Forward" else -offset

        # create the new stations with an offset
        new_coords = [simple_offset_perpendicular(Point(x, y), applied_offset, (dx, dy)) for x, y in coords]
        new_geometries.append(LineString([(p.x, p.y) for p in new_coords]))
    
    gdf_segments['geometry'] = new_geometries
    return gdf_segments



def convert_delays(df) :
    """
    It creates the columns delay and delay_seconds of the data
    Args : 
        df (DataFrame) : NS data
    Retunrs : 
        df (DataFrame) : NS data with delay in datetime format, and delay_seconds in total seconds
    """
    df["delay"] = datetime(2025, 1,1)+pd.to_timedelta(df['delay'], unit='m')
    df["delay_seconds"] = (
        df["delay"].dt.hour * 3600
        + df["delay"].dt.minute * 60
        + df["delay"].dt.second
    )
    return df




def make_intervals(t,interval=10, mode="minute"):
    """
    It rounds a time of date format 
    Args : 
        t : time in datetime format
        interval (int) : interval for the rounding (exemple : 10-minutes intervals)
        mode (string) : it gives the unit of the interval. Either "minute" or "second"
    Returns : 
        rounded time in date format
    """
    if t is pd.NaT or t is None:
        return None
    dt = datetime.combine(datetime.today(), t)
    if mode == "minute" : 
        total = dt.hour * 60 + dt.minute
    elif mode == "second" : 
        total = dt.hour * 3600 + dt.minute * 60 + dt.second
    rounded = round(total / interval) * interval
    if mode == "minute" : 
        dt_rounded = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=rounded)
    elif mode == "second" : 
        dt_rounded = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=rounded)
    return dt_rounded.time()




def round_to_interval(df,minute) : 
    """
    It adds a new column "round_UITVOERTIJD_AANKOMST" in the data with the 10min-rounded UITVOERTIJD_AANKOMST 
    Args : 
        df (DataFrame) : 
        minute (int) : lenght of intervals
    Returns : 
        group_in_intervals (DataFrame) : new dataframe with the column of 10 min groups
    """
    df_round_to_interval = df.loc[(df["Cancelled"]==False) & (df["ExtraTrain"]== False)].copy()
    
    # convert to datetime
    df_round_to_interval["round_UITVOERTIJD_AANKOMST"] = df_round_to_interval["UITVOERTIJD_AANKOMST"].apply(
        lambda td: (datetime.min + td).time() 
    )
    
    # separate the data in 10 minutes groups using the function make_intervals
    df_round_to_interval["round_UITVOERTIJD_AANKOMST"] = df_round_to_interval["round_UITVOERTIJD_AANKOMST"].apply(lambda t : make_intervals(t,minute,"minute"))
    
    # This column will serve only for the plot (matplotlib cannot put date format on an x-axis)
    df_round_to_interval['minutes'] = df_round_to_interval['round_UITVOERTIJD_AANKOMST'].apply(
        lambda t: t.hour * 60 + t.minute
    )
    return df_round_to_interval

def plot_average_deviation_per_route(route_stats) : 
    """
    It plots the bar chart about average deviation per route
    Args : 
        route_stats (DataFrame) : data with the mean deviation grouped by traject
    Returns : 
    """
    plt.figure(figsize=(14,3))
    plt.bar(route_stats["TRAJECT"], route_stats["AFWIJKING"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg Deviation")
    plt.title("Average Deviation per Route")
    plt.grid(axis="y")
    plt.show()
    
    
def plot_quality_of_previsions_through_the_day (quality_of_previsions) :  
    """
        It plots the line chart of evolution of crowding prevision through the day
    Args : 
        quality_of_previsions (DataFrame) : Contains the average PROGNOSE_REIZEN and REALISATIE per 10 min intervals
    Returns : 
    """
    plt.figure(figsize=(6, 3))
    plt.plot(quality_of_previsions['minutes'], quality_of_previsions['REALISATIE'], label='Realisation')
    plt.plot(quality_of_previsions['minutes'], quality_of_previsions['PROGNOSE_REIZEN'], label='Prevision')
    plt.xticks(range(0, 24*60+1, 60), [f"{h}" for h in range(25)])  # ticks every hours
    plt.xlabel("Hour")
    plt.ylabel("Passengers number")
    plt.title("Comparison between the average number of expected passengers and the average number measured")
    plt.legend()
    plt.show()



def get_1_delay_per_train(df) :
    """
        For a train that has multiple values of delays on it's journey, it keeps only one value that will be the 
        official delay of the train. For example, a train has a 11min38 delay from Shl to Hfd, 11min43 delay from 
        Hfd to Ledn, and a 11min35 delay from Ledn to Gvc. Then, the global delay of the tran should be the last 
        one (to discuss), so 11 min 35
    Args : 
        df (DataFrame) : NS Data
    Returns : 
        df_delay_per_train (DataFrame) : Filtered data keeping only the lines with the selected delay for the train
    """
    df_delay_per_train= df.loc[(df["Cancelled"]==False) & (df["ExtraTrain"]== False)].copy()
    
    df_delay_per_train = convert_delays(df_delay_per_train)
    
    #FIRST APPROACH : 
    #idx = df_delay_per_train.groupby(["DAGNR", "BEWEGINGNUMMER"])["delay_seconds"].idxmax()    
    #SECOND APPROACH : 
    idx = df_delay_per_train.groupby(["DAGNR", "BEWEGINGNUMMER"]).tail(1).index
       
    df_delay_per_train = df_delay_per_train.loc[idx].reset_index(drop=True)
    return df_delay_per_train



def mean_delays_per_period(period, df) :
    """
        It gives the average delay for trains running on the selected period
    Args : 
        df (DataFrame) : NS Data, with 1 value of delay selected per train with function get_1_delay_per_train
    Returns : 
        (datetime.time) : Mean delay of all trains running during this period
    """
    df_selection = df.loc[df["DAGDEELTREIN"]==period]
    mean = df_selection["delay_seconds"].mean()
    return (datetime.min + timedelta(seconds=mean)).time()




def select_following_trains(i,small_delays, number=20) : 
    """
    It selects the trains 20 next trains that run on the same infrastructure after train of index i
    Args : 
        i (int) : index of the train we want to consider
        number (float) : number of trains after train i we want to show. Default value is 20
    Returns : 
        select (DataFrame) : attributes of the 20 next trains that run on the same infrastructure 
        after train of index i
    """
    select = small_delays.loc[
                    (small_delays["ORDER_ON_INFRA"] >= small_delays.loc[i, "ORDER_ON_INFRA"]) &
                    (small_delays["ORDER_ON_INFRA"] < small_delays.loc[i, "ORDER_ON_INFRA"] + number) & 
                    (small_delays["station1"] ==  small_delays.loc[i, "station1"]) & 
                    (small_delays["station2"] == small_delays.loc[i, "station2"]) & 
                    (small_delays["DAGNR"] == small_delays.loc[i, "DAGNR"]) &
                    (~ pd.isna(small_delays["delay"]))].reset_index()
    
    
    select.sort_values(by='ORDER_ON_INFRA', inplace=True)
    select.reset_index(drop=True, inplace=True)
    return select


def plot_propagation_of_delays_unique(i, select_20_following_trains,small_delays) : 
    """
        It plots the propagation of delays on the 20 next trains running after train of index i 
    Args : 
        i (int) : index of initally delayed train to consider
        select_20_following_trains (DataFrame) : data of the 20 next trains running on the same infra as train i
        small_delays (DataFrame) : NS data 
    Returns : 
    """
    plt.figure(figsize=(4, 2))
    plt.bar(select_20_following_trains['ORDER_ON_INFRA'], select_20_following_trains['delay_seconds'])
    plt.xlabel(f"Trains running after train {small_delays.loc[i, "BEWEGINGNUMMER"]} on the infrastructure between {small_delays.loc[i, "station1"]} and {small_delays.loc[i, "station2"]} " )
    plt.ylabel('Delay (seconds)')
    plt.title(f'Delays of trains running after train {small_delays.loc[i, "BEWEGINGNUMMER"]} on the infrastructure between {small_delays.loc[i, "station1"]} and {small_delays.loc[i, "station2"]} ')
    plt.show()
    


def identify_initial_delays(small_delays) : 
    """
        It identifies lines of data that should be considered as initial delays.
        Rules for being an initial delay : 
        - The last 3 trains on the same infra were delayed of 0 minutes
        - The train is 3 times more delayed than any last 3 train
    Args : 
        small_delays (DataFrame) : NS data
    Returns : 
        small_delays (DataFrame) : Data with an extra colum "initial_delay", which is True if the line represents an
        initial delay, False in the other case
    """
    small_delays = small_delays.sort_values(
        by=['DAGNR', 'station1', 'station2', 'ORDER_ON_INFRA']
    ).reset_index()

    small_delays["initial_delay"]= False
    for i, row in small_delays.iterrows() : 
        if row["delay_seconds"] >0 : 
            if row["ORDER_ON_INFRA"] >3 : 
                max_previous_delay  = max ([small_delays.loc[i-1, "delay_seconds"],
                small_delays.loc[i-2, "delay_seconds"],
                small_delays.loc[i-3, "delay_seconds"]])
                if max_previous_delay ==0 : small_delays.loc[i,"initial_delay"]= True
                if row["delay_seconds"] >= 3*max_previous_delay : small_delays.loc[i,"initial_delay"]= True
            if row["ORDER_ON_INFRA"] ==3 : 
                max_previous_delay  = max ([small_delays.loc[i-1, "delay_seconds"],
                small_delays.loc[i-2, "delay_seconds"]])
                if max_previous_delay ==0 : small_delays.loc[i,"initial_delay"]= True
            if row["ORDER_ON_INFRA"] ==2 : 
                if small_delays.loc[i-1, "delay_seconds"] ==0 :  row["initial_delay"]= True
            if row["ORDER_ON_INFRA"] ==1 :  small_delays.loc[i,"initial_delay"]= True
    return small_delays

def select_following_trains_bis(minute, small_delays,initially_delayed_trains, max_number=20):
    """
        It selects, for each train that has been identified an a initial delay during the selected interval, the 
        max_number next trains running after on the same infrastructure
    Args : 
        minute (int) : amount of time the selected trains are initially delayed
        small_delays (DataFrame) : NS data
        initially_delayed_trains (DataFrame) : trains that have been identified an a initial delay during 
            the selected interval
        max_number (int) : maximum number of following traisn to consider
    Returns : 
        merged (DataFrame) : for each train following an initially delayed train, there is a line with the 
            corresponding initially delayed train, and the relative position of this train after 
            the initially delayed train
    """
    
    # Merge initially delayed trains with all trains on the same infrastructure and day
    merged = initially_delayed_trains.merge(
        small_delays,
        on=["station1", "station2", "DAGNR"],
        suffixes=("_base", "_other")
    )

    # Compute the relative position (order) of the "other" train after the initially delayed train
    merged["order_of_apparition"] = (
        merged["ORDER_ON_INFRA_other"] - merged["ORDER_ON_INFRA_base"]
    )

    # Keep only trains that:
    # - are after or at the same order as the initially delayed train (>=0)
    # - are within the maximum number of following trains (<= max_number)
    # - have a delay > 0
    merged = merged[(merged["order_of_apparition"] >= 0) & 
                    (merged["order_of_apparition"] <= max_number)& 
                    (merged["delay_seconds_other"] >0)]

    # Initialize a list to store rows that need to be deleted
    rows_to_delete = []

    # Loop through the merged DataFrame to remove trains that:
    # - are themselves initially delayed AND
    # - appear after the first initially delayed train (order_of_apparition > 0)
    for i, row in merged.iterrows():
        if (row["initial_delay_other"] == True)& (row["order_of_apparition"]>0):
            to_delete = merged.loc[
                (merged["station1"] == row["station1"]) &
                (merged["station2"] == row["station2"]) &
                (merged["DAGNR"] == row["DAGNR"]) &
                (merged["order_of_apparition"] >= row["order_of_apparition"])
            ]
            rows_to_delete.extend(to_delete.index)  

    # Remove the identified rows to avoid counting trains that are themselves initially delayed
    merged = merged.drop(index=rows_to_delete).reset_index(drop=True)

    # Recalculate the order_of_apparition:
    # - sort by day, station pair, and scheduled departure time of the following train
    # - group by the initially delayed train (index_base) and station/day
    # - count sequentially to get the updated order
    merged["order_of_apparition"] = (
        merged
        .sort_values(by=['DAGNR', 'station1', 'station2', 'UITVOERTIJD_VERTREK_dt_other'])
        .groupby(['index_base', 'DAGNR', 'station1', 'station2'])
        .cumcount() + 1     
        )
    return merged



def plot_propagation_of_delays (minute, df_agg) : 
    """
        It plots the mean propagation of delays for trains running after trains that were initially delayed 
        of approximately "minute"(arg) minutes.
    Args : 
        minute (int) : amount of time the selected trains are initially delayed
        df_agg (DataFrame) : data with the average delays for the 1st, 2nd, 3rd... trains running after all
        the initially delayed trains that were selected, and the size of the sample used for these calculations
    Returns : 
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Left y-axis: average delay
    bars1 = ax1.bar(
        df_agg["Order of apparition on infra"],
        df_agg["mean_delay_seconds"],
        color="steelblue",
        label="Mean delay (s)",
        width=0.6,
    )
    ax1.set_xlabel("Trains running on same infra after the 1st delayed train (ordered)")
    ax1.set_ylabel("Average delay (seconds)", color="steelblue")
    ax1.tick_params(axis='y', labelcolor="steelblue")

    # Right y-axis: number of data points
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        df_agg["Order of apparition on infra"],
        df_agg["n_obs"],
        color="orange",
        alpha=0.4,
        width=0.4,
        label="Number of trains",
    )
    ax2.set_ylabel("Number of trains (sample size)", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    # Title and layout
    plt.title(f"Propagation of delays for trains around {minute} min delay")
    fig.tight_layout()
    plt.show()



def agg_propagation_of_delays(minute, small_delays, limit=8):
    """
    Aggregates the propagation of delays along a given infrastructure.For trains that have an initial delay 
    within a selected interval, this function examines the following trains on the same route and day,
    computes their relative order, and aggregates the mean delay and count per relative position.

    Args:
        minute (int): amount of time the selected trains are approximately initially delayed 
        small_delays (DataFrame): NS train data including delay information
        limit (int): minimum number of observations required to compute an avergae

    Returns:
        df_agg (DataFrame): aggregated statistics for each delta_order, including mean delay (seconds) and 
        number of observations
    """
    interval = [60 * minute - 60, 60 * minute + 60]

    # Select trains that have an initial delay within the interval
    base_trains = small_delays.loc[
        (small_delays["delay_seconds"] >= interval[0]) &
        (small_delays["delay_seconds"] < interval[1]) &
        (small_delays["initial_delay"] == True)
    ]

    # Merge the initially delayed trains with all trains on the same infrastructure and day
    merged = base_trains.merge(
        small_delays,
        on=["station1", "station2", "DAGNR"],
        suffixes=("_base", "_other")
    )

    # Compute the relative order of the "other" train after the initially delayed train
    merged["delta_order"] = (
        merged["ORDER_ON_INFRA_other"] - merged["ORDER_ON_INFRA_base"]
    )

   # Keep only trains that:
    # - appear after or at the same position as the initially delayed train (>=0)
    # - are within 20 following trains
    # - have a delay > 0
    merged = merged[(merged["delta_order"] >= 0) & 
                    (merged["delta_order"] <= 20)& 
                    (merged["delay_seconds_other"] >0)]

    # Identify trains that are themselves initially delayed and appear after the base train
    rows_to_delete = []
    for i, row in merged.iterrows():
        if (row["initial_delay_other"] == True)& (row["delta_order"]>0):
            to_delete = merged.loc[
                (merged["station1"] == row["station1"]) &
                (merged["station2"] == row["station2"]) &
                (merged["DAGNR"] == row["DAGNR"]) &
                (merged["delta_order"] >= row["delta_order"])
            ]
            rows_to_delete.extend(to_delete.index)  
    
    # Drop the identified rows to avoid counting trains that are themselves initially delayed
    merged = merged.drop(index=rows_to_delete).reset_index(drop=True)

    # Recalculate delta_order based on sorted departure times
    # - sort by day, station pair, and departure time of the following train
    # - group by initially delayed train (index_base) and station/day
    # - count sequentially to assign updated delta_order
    merged["delta_order"] = (
        merged
        .sort_values(by=['DAGNR', 'station1', 'station2', 'UITVOERTIJD_VERTREK_dt_other'])
        .groupby(['index_base', 'DAGNR', 'station1', 'station2'])
        .cumcount() + 1     
        )
               
    # Aggregate the delay information by relative order (delta_order)
    # Compute both mean delay and count of observations
    df_agg = (
        merged.groupby("delta_order")["delay_seconds_other"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            "delta_order": "ORDER_ON_INFRA_INDEX",
            "mean": "mean_delay_seconds",
            "count": "n_obs"
        })
    )

    # Keep only delta_order positions with sufficient observations
    df_agg = df_agg.loc[df_agg["n_obs"] >limit]
    return df_agg



def list_index_first_delays (minute, df_small_delays) : 
    """
        It selects all the trains that are delayed BEFORE their terminus of a certain amount of minutes 
        (between minute-30 and minute+30)
    Args : 
        minute (int) : delay we want to consider
    Returns : 
        list_index (list) : indexes of all the trains delayed before their terminus
    """
    interval = [60 * minute - 30, 60 * minute + 30]
    list_index = df_small_delays.loc[
        (df_small_delays["delay_seconds"] >= interval[0]) &
        (df_small_delays["delay_seconds"] < interval[1])&
        (df_small_delays["IS_TERMINUS"] == False) &
        (~ pd.isna(df_small_delays["delay_seconds"]))].index.tolist()
    return list_index



def evolution_journey(df_small_delays, index_first_delays) :
    """
        It selects the lines representing the rest of the journeys of trains which indexes are in index_first_delays.
    Args : 
        df_small_delays (DataFrame) : NS data
        index_first_delays (list) : list of indexes of initially delayed trains
    Returns : 
        df_result (DataFrame) : Filtered data with only bthe lines representing the rest of the journeys of 
        trains which indexes are in index_first_delays.
    """
    df_selected = df_small_delays.loc[index_first_delays]
    final_indices = set(index_first_delays) 
    
    for index in index_first_delays :
        row = df_small_delays.loc[index]
        mask = (
            (df_small_delays.index > index) &
            (df_small_delays['BEWEGINGNUMMER'] == row['BEWEGINGNUMMER']) &
            (df_small_delays['DAGNR'] == row['DAGNR'])
        )
        final_indices.update(df_small_delays.index[mask])
    
    df_result = df_small_delays.loc[sorted(final_indices)]
    return df_result 


def plot_propagation_of_deviation(df_arg,minute):
    """
        It plots, on one side, the deviation in prediction for the sections run by the same train after 
        the apparition of the delay
    Args : 
        df_arg (DataFrame) : df with all the sections run by a train after it is delayed 
        minute (int) : amount of time trains are (approximately) initially delayed
    Returns : 
    """
    df_mean = (
                df_arg.groupby("ORDER_ON_SERVICE")["AFWIJKING"]
                .agg(mean_deviation=lambda x: x.abs().mean(), 
                     counted_data="count")
                .reset_index()
                .rename(columns={"ORDER_ON_SERVICE": "order_on_service"})
                )

    x = np.arange(len(df_mean)) # Positions of bars on X-axis
    width = 0.4
    fig, ax1 = plt.subplots(figsize=(4, 3))
    
    # Principal axis : mean_deviation
    bars1 = ax1.bar(x - width/2, df_mean["mean_deviation"], width, label="Mean deviation", color="purple")
    ax1.set_xlabel("Next sections after the apparition of the delay")
    ax1.set_ylabel("Mean deviation", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")
    ax1.set_ylim(0, df_mean["mean_deviation"].max() * 1.5)
    
    # Secondary axis : counted data
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, df_mean["counted_data"], width, label="Counted data", color="orange", alpha = 0.4)
    ax2.set_ylabel("Counted data", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim(0, df_mean["counted_data"].max() * 1.5)
    
    # X-Axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_mean["order_on_service"])
    
    plt.title(f"Propagation of deviation for trains delayed \n of {minute} minutes before their terminus")
    fig.tight_layout()
    bars = [bars1, bars2]
    labels = [b.get_label() for b in bars]
    ax1.legend(bars, labels, loc="upper left")   
    #plt.savefig("propagationdeviation.jpeg")
    

def select_trains_leaving_during_delays(considered_delayed_trains, df_is_terminus,minute) : 
    """
        For trains that are delayed before their terminus between station A and B, it selects all the trains leaving
        from station B during the delay of the train
    Args : 
        considered_delayed_trains (list) : indexes of all the trains delayed before their terminus
        df_is_terminus (DataFrame) : NS data (with a column saying whether a train is at his terminus)
    Returns : 
        trains_leaving_during_delays (DataFrame) : filtered data with only the trains that are leaving from 
        a station where passengers are waiting for another delayed train
    """
    final_indices_replacement =set() 
        
    for idx in considered_delayed_trains :
            row = df_is_terminus.loc[idx]
            mask = (
                (df_is_terminus['UITVOERTIJD_VERTREK'] > row['PLANTIJD_VERTREK']) &
                (df_is_terminus['UITVOERTIJD_VERTREK'] < row['PLANTIJD_VERTREK']+pd.to_timedelta(f"{minute} minutes")) &
                (df_is_terminus['DAGNR'] == row['DAGNR'])& 
                (df_is_terminus['station1'] == row['station2'])&
                (~ pd.isna(df_is_terminus["delay_seconds"]))
            )
            final_indices_replacement.update(df_is_terminus.index[mask])
        
    trains_leaving_during_delays = df_is_terminus.loc[sorted(final_indices_replacement)]
    return trains_leaving_during_delays
    
    
def plot_rain_deviation_wind(df_meteo_graph) : 
    """
        It plots a bar chart to compare, for each day, the quantity of rain, the wind speed and the deviation
        of the prediction
    Args : 
        df_meteo_graph (DataFrame) : df with aggregated data on DAGNR
    Returns :      
    """
    x = np.arange(len(df_meteo_graph.index))         
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    
    bars1 = ax1.bar(x - width, df_meteo_graph["RH"], width, color='skyblue', label="Rain (0.1 mm)")
    ax1.set_ylabel("Rain (0.1 mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_meteo_graph["Weekday"], rotation=45, ha='right')
    
    ax2 = ax1.twinx()  
    bars2 = ax2.bar(x, df_meteo_graph["AFWIJKING"], width, color='orange', label="Mean deviation", alpha=0.7)
    ax2.set_ylabel("Mean deviation")
    
    ax3 = ax1.twinx()  
    ax3.spines["right"].set_position(("outward", 60))
    bars3 = ax3.bar(x + width , df_meteo_graph["FH"], width, color='purple', label="Average wind speed")
    ax3.set_ylabel("Average wind speed (0.1 m/s)")
    
    bars = bars1 + bars2 + bars3 
    labels = [b.get_label() for b in bars]
    ax1.legend([bars1, bars2, bars3], ["Rain (0.1 mm)", "Mean deviation","Average wind speed (0.1 m/s)"], loc='upper right')
    
    plt.title("Comparison of rain, deviation and wind speed on each day of the month")
    plt.tight_layout()
    plt.show()
    #plt.savefig("deviationweather.jpeg")
    
    
    
def plot_rain_crowding_wind(df_meteo_graph) : 
    """
        It plots a bar chart to compare, for each day, the quantity of rain, the wind speed and the 
        number of people boarding on trains
    Args : 
        df_meteo_graph (DataFrame) : df with aggregated data on DAGNR
    Returns :      
    """
    x = np.arange(len(df_meteo_graph.index))         
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    
    bars1 = ax1.bar(x - width, df_meteo_graph["RH"], width, color='skyblue', label="Rain (0.1 mm)")
    ax1.set_ylabel("Rain (0.1 mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_meteo_graph["Weekday"], rotation=45, ha='right')
    
    ax2 = ax1.twinx()  
    bars2 = ax2.bar(x, df_meteo_graph["REALISATIE"], width, color='salmon', label="Crowding", alpha=0.7)
    ax2.set_ylabel("Crowding")
    
    ax3 = ax1.twinx()  
    ax3.spines["right"].set_position(("outward", 60))
    bars3 = ax3.bar(x + width , df_meteo_graph["FH"], width, color='purple', label="Average wind speed")
    ax3.set_ylabel("Average wind speed (0.1 m/s)")
    
    bars = bars1 + bars2 + bars3 
    labels = [b.get_label() for b in bars]
    ax1.legend([bars1, bars2, bars3], ["Rain (0.1 mm)", "Crowding", "Average wind speed (0.1 m/s)"], loc='upper right')
    
    plt.title("Comparison of rain, crowding and wind speed on each day of the month")
    plt.tight_layout()
    #plt.savefig("crowdingmeteo.jpeg")
    plt.show()
    


def plot_rain_deviation_wind_specific_day(df_june_10,day) : 
    """
        It plots a bar chart to compare, for each hour of a specific day, the quantity of rain, the wind speed 
        and the deviation of the prediction
    Args : 
        df_june_10 (DataFrame) : df with aggregated data on Hours, for a specific day (default June 10)
    Returns :      
    """
    x = np.arange(len(df_june_10.index))         
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    
    bars1 = ax1.bar(x - width, df_june_10["RH"], width, color='skyblue', label="Rain (0.1 mm)")
    ax1.set_ylabel("Rain (0.1 mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_june_10.index)
    
    ax2 = ax1.twinx()  
    bars2 = ax2.bar(x, df_june_10["AFWIJKING"], width, color='salmon', label="Mean deviation", alpha=0.7)
    ax2.set_ylabel("Deviation")
    
    ax3 = ax1.twinx()  
    ax3.spines["right"].set_position(("outward", 60))
    bars3 = ax3.bar(x + width , df_june_10["FH"], width, color='purple', label="Average wind speed")
    ax3.set_ylabel("Average wind speed (0.1 m/s)")
    
    bars = bars1 + bars2 + bars3 
    labels = [b.get_label() for b in bars]
    ax1.legend([bars1, bars2, bars3], ["Rain (0.1 mm)", "Deviation", "Average wind speed (0.1 m/s)"], loc='upper right')
    ax1.set_xlabel("Hour")
    plt.title(f"Comparison of rain, deviation and wind speed on June {day}")
    plt.show()
    

def plot_disruptions_patterns (df_disruptions_plot_2,df_meteo_graph) : 
    """
        It plots a bar chart and a line chart superposed to compare the number of predictions on a day and the 
        amount of time they lasted
    Args : 
        df_disruptions_plot_2 (DataFrame) : df with aggregated data about the number and the duration of disruptions
        df_meteo_graph (DataFrame) : data with aggregation on days of the month
    Returns :      
    """
    
    # Bars for the number of disruptions
    ax = df_disruptions_plot_2["KLANTHINDERNAAM"].plot(kind='bar',figsize=(10,6), color = "lightblue", label= "Number of disruptions")
    ax.set_xticklabels( df_disruptions_plot_2["Weekday"], rotation=45, ha='right')
    
    
    # Scatter for the time amount of disruptions
    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(), df_disruptions_plot_2['KLANTHINDERINMINUTEN'], color='brown', marker='o', label='Total disturbance minutes')
    
    # Scatter for the deviation
    ax3 = ax.twinx()
    ax3.plot (ax.get_xticks(), df_meteo_graph["AFWIJKING"], color="green", marker = "x", label = "Mean absolute deviation")
    ax3.spines["right"].set_position(("outward", 60))
    
    # Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(handles1 + handles2 + handles3, labels1 + labels2 +labels3,  loc='upper center')
    
    ax.set_xlabel("Dagnr (Day)")
    ax.set_ylabel("Number of disruptions", color ="blue")
    ax2.set_ylabel("Total disturbance minutes",color ="brown")
    ax3.set_ylabel("Mean absolute deviation", color ="green")
    plt.title("Number and duration of disruptions, compared to mean absolute deviation")
    plt.tight_layout()
    #plt.savefig("crowdingmeteo.jpeg")
    plt.show()
    



def plot_model_comparison (df_agg, col_x, title, label_x, marksize, line_width) : 
    """
        It plots a multi line-chart to compare the models' deviation on predictions
    Args : 
        df_agg (DataFrame) : df with aggregated data about the deviation of prediction
        col_x (DataFrame) : column for the aggregation (day or hour)
        title (str) : title of the graph
        label_x : title of the x_axis
        marksize (int) : size of the marker
        line_width : width of the line of the plot
    Returns :      
    """
    plt.figure(figsize=(8,5))
    plt.plot(df_agg[col_x],df_agg['deviation_OPERATOR_PRED'],marker="x", color = "red", label = "Operator prediction",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['deviation_LINEAR_PRED'],marker="o", color = "orange", label = "Linear Regression",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['deviation_MLP_PRED'],marker="^", color = "purple", label = "MLP Model",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['deviation_XGBOOST_PRED'],marker="s", color = "lightblue", label = "XGB Model",markersize = marksize,linewidth=line_width)
    plt.axhline(y=0, color="green", linestyle="--", linewidth=2, label="Realisation (0)")
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel("Sum of absolute difference |Predicted - Realisation| for each service")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"modelcomparison-{title}.jpeg")
    plt.show()
    

def plot_model_comparison_occupancy (df_agg, col_x, title, label_x, marksize, line_width) : 
    """
        It plots a multi line-chart to compare the models' predictions on total occupancy
    Args : 
        df_agg (DataFrame) : df with aggregated data about the deviation of prediction
        col_x (DataFrame) : column for the aggregation (day or hour)
        title (str) : title of the graph
        label_x : title of the x_axis
        marksize (int) : size of the marker
        line_width : width of the line of the plot
    Returns :      
    """
    plt.figure(figsize=(8,5))
    plt.plot(df_agg[col_x],df_agg['LINEAR_PRED'],marker="o", color = "orange", label = "Linear Regression",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['MLP_PRED'],marker="^", color = "purple", label = "MLP Model",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['XGBOOST_PRED'],marker="s", color = "lightblue", label = "XGB Model",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['OPERATOR_PRED'],marker="x", color = "red", label = "Operator prediction",markersize = marksize,linewidth=line_width)
    plt.plot(df_agg[col_x],df_agg['REALISATIE'],marker="*", color = "green", label = "Realisation",markersize = marksize,linewidth=line_width)
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel("Total occupancy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    #plt.savefig("modelcomparison.jpeg")
    plt.show()

