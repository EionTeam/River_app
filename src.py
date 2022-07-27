#!/usr/bin/env python
# coding: utf-8

# SRC File for River Runner 

# In[1]:
import os 
import requests
import json
import numpy as np
import pandas as pd 
import geopandas as gpd
import shapely as sh
from shapely.geometry import LineString
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import requests
import urllib.parse
import plotly.graph_objects as go
import matplotlib as mpl 
import plotly.express as px

from shapely.geometry import LineString






def get_json(url):
    response = requests.get(url)
    return  response.json()
       
def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color='b', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)


# In[ ]:


def take_input_coords(coords):
    """Input function with validations for coordinates
    """
    while True:
            try:
                # x  = input('Enter your coordinates: ')
                # coords = [float(a) for a in x.split(",")]
                x,y = float(coords[0]), float(coords[1])    
                assert len(coords) == 2 
            except AssertionError:
                print("Please enter two coordinates: latitude and longitude")
                continue
            except ValueError:
                print('Please enter two coordinates: latitude and longitude')
                continue
            else:
                break
    if len(coords) != 2:
        print("Please enter coords as (lat,long) ")
    else:
        print('Coords are acceptable - thanks!')
    return coords


# In[ ]:


def find_downstream_route(coords):
    """ Retrun the json of the flowlines for the downstream route 
    """
    coords = take_input_coords(coords)
    
    url = 'https://labs.waterdata.usgs.gov/api/nldi/linked-data/hydrolocation?coords=POINT%28{}%20{}%29'.format(coords[0], coords[1])
    djson = get_json(url)
    navurl = djson['features'][0]['properties']['navigation']

    navjson = get_json(navurl)

    ds_main = navjson['downstreamMain']    
    downstream_main = get_json(ds_main)
    ds_flow = downstream_main[0]['features']
    
    with_distance = ds_flow + '?distance=5500'
    flowlines = get_json(with_distance)
    print('Num of features = {}'.format( len(flowlines['features'])) )
    
    return flowlines


# In[ ]:


def print_downstream():
    """ Print map of downstream route 
    """
    
    flowlines = find_downstream_route()
    
    fig, ax = plt.subplots() 

    for x in flowlines['features']:
        coords = x['geometry']['coordinates']
        plot_line(ax, LineString(coords) )


def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color='b', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

def find_overlapping_stations(data, buffer_rad = 0.01 ):
    #Extract coords and convert into geo-df
    coords = [data['features'][i]['geometry']['coordinates'] for i in range(len(data['features'])) ]
    dict = {key: value for (key, value) in zip([i for i in range(len(data['features'])) ] , [LineString(coords[i]) for i in range(len(data['features'])) ] ) }
    river_df = pd.DataFrame(dict, index=['geometry']).T
    river_gdf = gpd.GeoDataFrame(river_df , crs='EPSG:4326', geometry=river_df['geometry'] )
    river_gdf.to_crs('EPSG:4326')
    
    #Create buffer geo-df
    buffer_river = river_gdf.buffer(buffer_rad)
    buffer_river.to_crs('EPSG:4326')
    buffer_gdf = gpd.GeoDataFrame(buffer_river , crs='EPSG:4326', geometry=buffer_river ) 
    
    #Find overalapping stations
    locations = create_filtered_locations()
    loc_gdf = gpd.GeoDataFrame(locations , crs='EPSG:4326', geometry=locations['geometry'] )  
    overlap_station =  buffer_gdf.sjoin(loc_gdf, how='inner')[[ 'Latitude', 'Longitude', 'pH', 'dDICdTA' ]]
    overlap_station = overlap_station.drop_duplicates()
    overlap_station  = overlap_station.reset_index()
    overlap_station = overlap_station
    overlap_station.pH = overlap_station.pH.round(3)
    overlap_station['index'] = [ i+1 for i in range(len(overlap_station)) ]
    return overlap_station.dropna()

def create_CRI(overlap_station):
    CRI = overlap_station[['index', 'dDICdTA']]
    ocean = pd.DataFrame( columns =  ('index','dDICdTA' ))
    ocean_index = len(CRI)+2
    ocean.loc[0]= [ocean_index , 0.85]

    field = pd.DataFrame( columns =  ('index','dDICdTA' ))
    field.loc[0]= [0 , 1]

    chart = pd.concat([CRI, field, ocean], axis =0 )
    return chart.sort_values('index') , ocean_index

def plot_CRI(CRI, ocean_indx):
    fig,ax = plt.subplots()
    ax.plot( CRI['index'], CRI['dDICdTA'])
    ax.set_title('Carbon Retention Index (CRI) by Station',
        fontsize='large',
        loc='center',
        fontweight='bold',
        style='normal',
        family='monospace')
    # ax.set_suptitle('Index 0 = Field. Final Index = Ocean')
    ax.set_xlabel('Station Index (Field = 0. Ocean = {})'.format(ocean_indx) )
    ax.set_ylabel('CRI')
    #add ocean CRI line 
    x = np.linspace(0, ocean_indx+.2)
    y = [0.85 for i in range(len(x))]
    ax.text(0, 0.855, 'Ocean CRI')
    ax.plot(x,y , color = 'r', linestyle = 'dashed')

    ax.grid('on')
    
    return fig,ax 


def load_stations():
    path =  os.path.join(os.path.dirname(__file__),"data/sampling_locations.csv") 
    loc = pd.read_csv(path)
    usloc = loc[loc['Country']=='USA']
    return gpd.GeoDataFrame(usloc, geometry=gpd.points_from_xy(usloc['Longitude'], usloc['Latitude']), crs='EPSG:4326')
    


def load_chem( locations):
    path = os.path.join(os.path.dirname(__file__),"data/uschem_pyco2sys.csv")
    uschem = pd.read_csv(path)
    uschem = uschem.rename(columns={'Alkalinity': 'TA', 'Temp_water': 'T'})
    uschem.loc[:,'RESULT_DATETIME'] = pd.to_datetime(uschem['RESULT_DATETIME'])
    uschem['Q'] = (uschem['RESULT_DATETIME'].dt.month-1)//3
    uschem['Y'] = (uschem['RESULT_DATETIME'].dt.year)
    # stations = us_geostat[us_geostat['STAT_ID'].isin(catchments['STAT_ID'])]
    chem = uschem[uschem['STAT_ID'].isin(locations['STAT_ID'])]

    return chem[['Y','Q','STAT_ID','RESULT_DATETIME', 'TA', 'T', 'pCO2', 'pH', 'dDICdTA']]

     
def create_filtered_locations():

  locations = load_stations()
  chem = load_chem( locations)

  pH = chem[['STAT_ID','pH']].groupby(['STAT_ID']).mean()
  CRI = chem[['STAT_ID','dDICdTA']].groupby(['STAT_ID']).mean()
  locations = locations.set_index('STAT_ID')
  station_pH = locations.merge(pH, left_index=True, right_index=True)
  stat_ph_CRI = station_pH.merge(CRI, left_index=True, right_index=True)

  # 1. stations with very few data points
  station_qa1 = chem[chem['dDICdTA']>0].groupby(['STAT_ID']).count()
  qa1 = station_qa1[station_qa1['dDICdTA']<5].index.tolist()
  chem = chem[~chem['STAT_ID'].isin(qa1)]

  # 2. years with only one station 
  station_qa2 = chem[chem['dDICdTA']>0].groupby(['Y','Q']).count()
  qa2 = station_qa2[station_qa2['dDICdTA']==1].index.tolist()
  for Y, Q in qa2:
      chem = chem[~((chem['Y']==Y) & (chem['Q']==Q))]

  # stations with only one year 
  station_qa3 = chem[chem['dDICdTA']>0].groupby(['Y','Q','STAT_ID']).count()
  station_qa3 = station_qa3.groupby(['STAT_ID']).count()
  qa3 = station_qa3[station_qa3['dDICdTA']==1].index.tolist()
  chem = chem[~chem['STAT_ID'].isin(qa3)]

  return stat_ph_CRI[stat_ph_CRI.index.isin(chem['STAT_ID'])]



def get_coords(address):
  url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
  response = requests.get(url).json()
  return ( float(response[0]["lon"]), float(response[0]["lat"]))

