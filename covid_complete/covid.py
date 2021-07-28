# COVID-19 South Korea Propagation Scheme Visualization Code in Python
# By Kyunghyun Ryu and Seongwon Yoon

import sys
import os

import numpy as np
import numba as nb
import pandas as pd
import seaborn as sb
import datetime

import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import TimestampedGeoJson, HeatMap

# Manually input variable
# nation
# 's', 'sf', 'r', etc...
# date

str = 'confirmed'
nation = 'South_Korea'
startdate = '2020-03-01'
date = '2020-02-01'
city_name = 'Jeju-do'
city_list = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon', 'Ulsan', 'Sejong', 'Gyeonggi-do', 'Gangwon-do', 'Chungcheongbuk-do', 'Chungcheongnam-do', 'Jeollabuk-do', 'Jeollanam-do', 'Gyeongsangbuk-do', 'Gyeongsangnam-do', 'Jeju-do']
population_list = [9662000, 3373000, 2432000, 2944000, 1494000, 1509000, 1147000, 331000, 13238000, 1517000, 1626000, 2188000, 1803000, 1773000, 2665000, 3350000, 660000]
latitude_list = [37.566953 ,35.179884 ,35.87215 ,37.456188 ,35.160467 ,36.350621 ,35.539797 ,36.480132 ,37.275119 ,37.885369 ,36.63568 ,36.658976 ,35.820308 ,34.816095 ,36.576032 ,35.238294, 33.488936]
longitude_list = [126.977977, 129.074796, 128.601783, 126.70592, 126.851392, 127.384744, 129.311538, 127.289021, 127.009466, 127.729868, 127.491384, 126.673318, 127.108791, 126.463021, 128.505599, 128.692397, 126.500423]
city_population_list = {'province' : city_list,
                       'population' : population_list,
                        'latitude' : latitude_list,
                        'longitude' : longitude_list
                       }
df_city_population_list = pd.DataFrame(city_population_list, columns = ['province', 'population', 'latitude', 'longitude'])
d = 150 #total simulation days
day = np.array(range(d))
# transmission rate and other coefficient
b = 0.1389
b_f = b * 0.6
T = 0.001
delta = 0.02
k = 1 / 4
a = 1 / 4
g = 1 / 14

# Dataset
geo_data = '/home/tonyyoon/week5/TL_SCCO_CTPRVN.json'
df_region_info = pd.read_csv('/home/tonyyoon/week5/Region.csv')
df_time = pd.read_csv('/home/tonyyoon/week5/Time.csv')
df_time_province = pd.read_csv('/home/tonyyoon/week5/TimeProvince.csv')

# Map Features
radius_min = 10
radius_max = 40
fill_color_test = '#f4a582'
fill_color_s = '#EFEFE8FF'  # s denotes confirmed patients
fill_color_r = '#OA5E2AFF'
fill_color_d = '#E80018'
fill_opacity = 0.5
line_opacity = 0.2
weight = 1

color_scale = np.array(['#053061','#2166ac','#4393c3','#92c5de','#d1e5f0','#fddbc7','#f4a582','#d6604d','#b2182b','#67001f'])
sb.palplot(sb.color_palette(color_scale))


def main():
    
    for i in city_list:
        solve_ode(d, i, startdate)
    
    # Draw Simulation Results
    df_simulation_results = pd.DataFrame()
    df_simulation_results = integrate_csv_files(df_simulation_results, city_list)
    # add_lat_long_to_df(df_simulation_results, df_region_info)
    df_simulation_results = df_simulation_results.sort_values('date')
    df_simulation_results.to_csv('/home/tonyyoon/week5/covidproject/odeData/simulationresults.csv')
    
    simulation_timestamped_features = load_features(df_simulation_results)
    draw_timestampedgeojson(simulation_timestamped_features)
    
    # Draw Actual Data
    add_lat_long_to_df(df_time_province, df_region_info)
    df_date_to_datetime(df_time_province)
    actual_timestamped_features = load_features(df_time_province)
    #draw_timestampedgeojson(actual_timestamped_features)
    
    
def solve_ode(d, city_name, startdate):
    
    # Simulation parameters
    N = 0 # population
    S = np.zeros(d) # Susceptible
    S_f = np.zeros(d)# people wearing mask
    E = np.zeros(d) # Exposed
    I = np.zeros(d) # Infected
    Q = np.zeros(d) # Quarantined
    R = np.zeros(d) # Recoverd
    
    dS = np.zeros(d)
    dS_f = np.zeros(d)
    dE = np.zeros(d)
    dI = np.zeros(d)
    dQ = np.zeros(d)
    dR = np.zeros(d)
    
    # --Set Simulation Parameters--
    startday = datetime.datetime.strptime(startdate, '%Y-%m-%d')
    enddate = startday + datetime.timedelta(days=d - 1)
    yesterday = startday + datetime.timedelta(days=-1)
    yesterday = yesterday.strftime('%Y-%m-%d')

    day = np.array(range(d))

    df_startdate = df_time_province[df_time_province['date'] == startdate]
    df_yesterday = df_time_province[df_time_province['date'] == yesterday]

    startdate_tmp = df_startdate[df_startdate['province'].isin([city_name])]
    startdate_confirmed = startdate_tmp.loc[:, ['confirmed']]
    startdate_released = startdate_tmp.loc[:, ['released']]

    yesterday_tmp = df_yesterday[df_yesterday['province'].isin([city_name])]
    yesterday_confirmed = yesterday_tmp.loc[:, ['confirmed']]

    pop_tmp = df_city_population_list[df_city_population_list['province'].isin([city_name])]
    pop_tmp = pop_tmp.loc[:, ['population']]

    
    S_f[0] = 0
    N = pop_tmp.iat[0, 0] # population of province
    I[0] = startdate_confirmed.iat[0, 0] - yesterday_confirmed.iat[0, 0] # Infected
    E[0] = I[0] * 7 # Exposed   
    Q[0] = startdate_confirmed.iat[0, 0] - startdate_released.iat[0, 0] # Quarantined
    R[0] = startdate_released.iat[0, 0] # Recoverd
    S[0] = N - E[0] - I[0] - Q[0] - R[0] - S_f[0]  # Susceptible
    
    
    for i in range(d - 1):
        dS[i] = -b * S[i] / N - b_f * (1 - np.exp(-T * Q[i])) * S[i]
        dS_f[i] = b_f * (1 - np.exp(-T * Q[i])) * S[i] - delta * b * S_f[i] / N
        dE[i] = b * S[i] / N + delta * b * S_f[i] / N - k * E[i]
        dI[i] = k * E[i] - a * I[i]
        dQ[i] = a * I[i] - g * Q[i]
        dR[i] = g * Q[i]
        S[i + 1] = S[i] + dS[i]
        S_f[i + 1] = S_f[i] + dS_f[i]
        E[i + 1] = E[i] + dE[i]
        I[i + 1] = I[i] + dI[i]
        Q[i + 1] = Q[i] + dQ[i]
        R[i + 1] = R[i] + dR[i]

    # Save data
    startdate = datetime.datetime.strptime(startdate, '%Y-%m-%d')
    enddate = startdate + datetime.timedelta(days=d - 1)
    yesterday = startdate + datetime.timedelta(days=-1)
    dt_index = pd.date_range(start=startdate, end=enddate)

    lat_tmp = df_city_population_list[df_city_population_list['province'].isin([city_name])]
    lat_tmp = lat_tmp.loc[:, ['latitude']]
    
    long_tmp = df_city_population_list[df_city_population_list['province'].isin([city_name])]
    long_tmp = long_tmp.loc[:, ['longitude']]
    
    city = []
    lat = []
    long = []
    for i in range(d):
        city.append(city_name)
        lat.append(lat_tmp.iat[0, 0])
        long.append(long_tmp.iat[0, 0])

        result = {'date': dt_index,
                  'province': city,
                  'latitude':lat,
                  'longitude':long,
                  'confirmed' : Q+R,
                  'S': S,
                  'S_f': S_f,
                  'E': E,
                  'I': I,
                  'Q': Q,
                  'R': R}

    result = pd.DataFrame(result, columns=['date', 'province', 'latitude', 'longitude', 'confirmed', 'S', 'S_f', 'E', 'I', 'Q', 'R'])
    result.to_csv('/home/tonyyoon/week5/covidproject/odeData/' + city_name + '.csv')
    
    
    
    
        
    
def df_date_to_datetime(df):
    df['date'] = pd.to_datetime(df['date'])    

    
    
def RadiusCheck(r):
    if r !=0:
        if r < radius_min:
            r = radius_min
        elif r > radius_max:
            r = radius_max
    return r
    

def export_data_in_specific_date(df, date):
    # df should contain date - place - number
    dates = df.loc[:, "date"]
    daily = df[df['date'] == date]

    return daily
    
    
    
def locate_map(nation):
    if nation == 'South_Korea':
        center = [36.241, 127.986]
        m = folium.Map(location=center, zoom_start=6)
    return m



def add_lat_long_to_df(df_target, df_region):
    # Assume that there is only city name in the df_target file but no latitude or longitude
    # This function adds latitude and longitude from df_region to df_target
    
    # Add latitude and longitude to df_target DataFrame
    df_target['latitude'] = 0.0
    df_target['longitude'] = 0.0

    for row in df_target.itertuples():
        # Extract latitude and longitude of specific city
        # df_target 'province' = df_region 'city'
        df_region_info_tmp = df_region[df_region['city'].isin([row.province])]
        df_region_info_tmp = df_region_info_tmp.loc[:, ['latitude']]
        df_target.loc[row.Index, ['latitude']] = df_region_info_tmp.iat[0,0]
    
        df_region_info_tmp = df_region[df_region['city'].isin([row.province])]
        df_region_info_tmp = df_region_info_tmp.loc[:, ['longitude']]
        df_target.loc[row.Index, ['longitude']] = df_region_info_tmp.iat[0,0]

        
        
def integrate_csv_files(df, city_list):
    
    for i in city_list:
        df_tmp = pd.read_csv('/home/tonyyoon/week5/covidproject/odeData/'+ i + '.csv')
        df = pd.concat([df, df_tmp])
    
    return df
        

    
def load_features(df):
    
    features = []
    feature = []

    for _,row in df.iterrows():
        radius = np.sqrt(row['confirmed'])
        radius = RadiusCheck(radius)

        feature = {
            'type': 'Feature',
            'geometry': {
                'type' : 'Point',
                'coordinates' : [row['longitude'],row['latitude']]
            },
            'properties': {
                'time': row['date'].__str__(),
                'style': {'color':fill_color_test},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': fill_color_test,
                    'fillOpacity': fill_opacity,
                    'stroke':'true',
                    'radius':radius,
                    'weight':weight
                }
            }
        }
        features.append(feature)

    return features



def draw_choropleth_in_specific_date(df, date, str):

    # Load DataFrame of specific date
    daily = export_data_in_specific_date(df_time_province, date)

    # Load map
    m = locate_map(nation)

    Choropleth(
        geo_data=geo_data,
        name='choropleth',
        data=daily,
        columns=['province', str],
        key_on='feature.properties.CTP_ENG_NM',

        fill_color='BuPu',
        # fill_opacity=0.7,
        # line_opacity=0.2,
        legend_name='infected'
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save('/home/tonyyoon/week5/folium_maps/' + date + '.html')    


    
def draw_timestampedgeojson(features):

    # Load map
    m = locate_map(nation)
    
    Choropleth(
        geo_data=geo_data,
        name='choropleth',
        key_on='feature.properties.CTP_ENG_NM',
        fill_color='#d1e5f0',
        fill_opacity=0.5,
        line_opacity=0.5
    ).add_to(m)
    
    TimestampedGeoJson(
        {'type':'FeatureCollection', 'features':features},
        period='P1D',
        duration='P1D',
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='MM/DD/YYYY',
        time_slider_drag_update=True,
        transition_time = 1000
    ).add_to(m)

    m.save('/home/tonyyoon/week5/folium_maps/' + nation + '_timestamped'+'.html')

    
    
def printUsage():
    print("Usage:")
    print("python diff.py -n <val> -t <val> [-m <val> -i <val> -r <val> -q -h]")
    print("    -n, --num:     number of grid points in each dimension.")
    print("    -m, --mode:    mode of the Laplacian operation implementation (1, 2 or 3).")
    print("    -t, --tend:    end simulation time.")
    print("    -i, --input:   input restart file name.")
    print("    -r, --restart: restart time period.")
    print("    -q, --quiet:   quiet run.")
    print("    -h, --help:    print usage.")

if __name__ == '__main__':
    main()