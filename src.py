#!/usr/bin/env python
# coding: utf-8

# SRC File for River Runner 

# In[1]:

import os
import urllib.parse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString
import shapely as sh
import random
from matplotlib.pyplot import cm
import pyproj


def get_json(url):
    response = requests.get(url)
    return  response.json()
       
def plot_line_col(ax, ob, c):
    x, y = ob.xy
    ax.plot(x, y, color=c, alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

def plot_line(ax, ob, ):
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
    dict_data = {key: value for (key, value) in zip([i for i in range(len(data['features'])) ] , [LineString(coords[i]) for i in range(len(data['features'])) ] ) }
    river_df = pd.DataFrame(dict_data, index=['geometry']).T
    river_gdf = gpd.GeoDataFrame(river_df , crs='EPSG:4326', geometry=river_df['geometry'] )
    river_gdf.to_crs('EPSG:4326')
    
    #Create buffer geo-df
    buffer_river = river_gdf.buffer(buffer_rad)
    buffer_river.to_crs('EPSG:4326')
    buffer_gdf = gpd.GeoDataFrame(buffer_river , crs='EPSG:4326', geometry=buffer_river ) 
    
    #Find overalapping stations
    locations = create_filtered_locations()
    loc_gdf = gpd.GeoDataFrame(locations , crs='EPSG:4326', geometry=locations['geometry'] )  
    loc_gdf.reset_index(inplace=True)
    overlap_station =  buffer_gdf.sjoin(loc_gdf, how='inner')[[ 'STAT_ID', 'Latitude', 'Longitude', 'pH', 'dDICdTA' ]]
    overlap_station = overlap_station.drop_duplicates().dropna()
    overlap_station  = overlap_station.reset_index()
    overlap_station = overlap_station
    overlap_station.pH = overlap_station.pH.round(3)
    overlap_station['index'] = [ i+1 for i in range(len(overlap_station)) ]
    return overlap_station

def find_searoute(): 
  address  = input('Enter your address: ')
  coords = get_coords(address)
  return  find_downstream_route(coords)




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


def find_oean_point(data):
    coords = [data['features'][i]['geometry']['coordinates'] for i in range(len(data['features'])) ]
    dict_data = {key: value for (key, value) in zip([i for i in range(len(data['features'])) ] , [LineString(coords[i]) for i in range(len(data['features'])) ] ) }
    river_df = pd.DataFrame(dict_data, index=['geometry']).T
    river_gdf = gpd.GeoDataFrame(river_df , crs='EPSG:4326', geometry=river_df['geometry'] )
    river_gdf.to_crs('EPSG:4326')
    line = river_gdf.iloc[-1]['geometry']
    f,l = line.boundary.geoms
    return l.xy

def random_point_mis_basin():
    minx, miny, maxx, maxy = -113.938141, 37.5, -77.83937, 49.73911
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    return sh.geometry.Point(x,y)

def generate_field_point():
    """Generates random point in Missipi basin, filters using shapefile 
    """
    while True:
        point= random_point_mis_basin()
        p_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries( point) ).set_crs( crs = 'EPSG:4326')
        p_df =  p_df.to_crs( crs = 'EPSG:4326')

        basin_sh =  os.path.join(os.path.dirname(__file__),'data/Miss_RiverBasin/Miss_RiverBasin.shp')
        basin = gpd.read_file(basin_sh)
        basin =  basin.to_crs('EPSG:4326')
        buff_basin = basin.buffer(-1)
        buff_basin.to_crs('EPSG:4326')
        buffer_gdf = gpd.GeoDataFrame(buff_basin , crs='EPSG:4326', geometry=buff_basin ) 


        overlap = buffer_gdf.sjoin(p_df)
        if len(overlap) != 0:
            break
        
    return point.x,  point.y



def get_river_df_utm(data):
    utm = pyproj.CRS('EPSG:26907')
    coords = [data['features'][i]['geometry']['coordinates'] for i in range(len(data['features'])) ]
    dict_data = {key: value for (key, value) in zip([i for i in range(len(data['features'])) ] , [LineString(coords[i]) for i in range(len(data['features'])) ] ) }
    river_df = pd.DataFrame(dict_data, index=['geometry']).T
    river_gdf = gpd.GeoDataFrame(river_df , crs='EPSG:4326', geometry=river_df['geometry'] )
    return river_gdf.to_crs(utm)

def snap_points(data,  offset = 1000):
    """Generate the snapped points to the river for stations, uses https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
    """

    utm = pyproj.CRS('EPSG:26907')

    lines = get_river_df_utm(data)

    locations = create_filtered_locations()
    locations = locations.to_crs(utm)
    points = gpd.GeoDataFrame(locations , crs=utm, geometry=locations['geometry'] )  
    print(points.crs, lines.crs)

    bbox = points.geometry.bounds + [-offset, -offset, offset, offset]
    hits = bbox.apply(lambda row: [x for x in lines.sindex.intersection(row)], axis=1)
    tmp = pd.DataFrame({ "pt_idx": np.repeat(hits.index, hits.apply(len)),   "line_i": np.concatenate(hits.values)})
    
    lines.reset_index(drop=True)
    lines['line_i'] =[int(x) for x in lines.index]
    
    # Join back to the lines on line_i; we use reset_index() to 
    # give us the ordinal position of each line
    tmp = tmp.join(lines, on='line_i', lsuffix='_left', rsuffix='_right')
                        
        
    # Join back to the original points to get their geometry
    # rename the point geometry as "point"
    tmp = tmp.join(points.geometry.rename("point"), on="pt_idx" )
    
    # Convert back to a GeoDataFrame, so we can do spatial ops
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=points.crs)
    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
    tolerance = offset
    
    # Discard any lines that are greater than tolerance from points
    tmp = tmp.loc[tmp.snap_dist <= tolerance]
    # Sort on ascending snap distance, so that closest goes to top
    tmp = tmp.sort_values(by=["snap_dist"])

    
     # group by the index of the points and take the first, which is the
    # closest line 
    closest = tmp.groupby("pt_idx").first()
    # construct a GeoDataFrame of the closest lines
    closest = gpd.GeoDataFrame(closest, geometry="geometry")
    closest['STAT_ID'] = [int(x) for x in closest.index ]
    
    
    # Position of nearest point from start of the line
    pos = closest.geometry.project(gpd.GeoSeries(closest.point))
    # Get new point location geometry
    new_pts = closest.geometry.interpolate(pos)

    #Identify the columns we want to copy from the closest line to the point, such as a line ID.
    line_columns = ['STAT_ID', 'line_i_right']
    # Create a new GeoDataFrame from the columns from the closest line and new point geometries (which will be called "geometries")
    snapped = gpd.GeoDataFrame( closest[line_columns],geometry=new_pts, crs=utm)

    # Join back to the original points: on index which is station id
    # print(points.crs )
    # print(snapped.crs )
    updated_points = points.drop(columns=["geometry"]).join(snapped)
    updated_points= gpd.GeoDataFrame( updated_points, geometry =updated_points['geometry'], crs=utm)
    # updated_points = updated_points.set_crs(utm)
    # You may want to drop any that didn't snap, if so:
    updated_points = updated_points.dropna(subset=['pH', 'dDICdTA', "geometry"])
    updated_points= updated_points.to_crs('EPSG:4326')
    updated_points['index'] = [i+1 for i in range(len(updated_points)) ]
   
    overlap = updated_points.loc[:, ['index', 'geometry', 'pH', 'dDICdTA']]
    overlap['Longitude']= [point.x for point in overlap['geometry']]
    overlap['Latitude']=[point.y for point in overlap['geometry'] ]
    overlap.drop(columns= 'geometry', inplace=True)
    
    return updated_points, overlap

def find_CRI_years():
    locations = load_stations()
    valid_ids = locations.index.unique().tolist()
    chem=load_chem(locations)
    q_CRI = chem[['STAT_ID','Y', 'dDICdTA']].groupby(['STAT_ID',  'Y']).mean().dropna()
    return q_CRI.reset_index(level=[1])


def create_multi_CRI(data):
    loc, _= snap_points(data)
    loc['index_plot']= [i +1 for i in range(len(loc))]
    
    q_CRI = find_CRI_years()
    join = loc.join(q_CRI, lsuffix='l')
    
    year = join.groupby('Y').count()
    populated = year[year['pH'] >3].index.tolist()
    
    fig, ax = plt.subplots(figsize=(7,5) )
    ax.set_title('Carbon Retention Index (CRI) by Station',
        fontsize='medium',
        loc='center',
        # fontweight='normal',
        style='normal',
        family='monospace')
    # ax.set_suptitle('Index 0 = Field. Final Index = Ocean')
    
    ocean_indx= len(loc)+1
    ax.set_xlabel('Station Index (Field = 0. Ocean = {})'.format(ocean_indx) )
    ax.set_ylabel('CRI')
    num_drops=0
    for y in populated:
        data= join[join.Y == y]
        a,*b =  data.index_plot.values.tolist()
        c,*d =  data.dDICdTA.values.tolist()
        ax.plot([0,a,*b, ocean_indx], [1, c, *d, 0.85], label=y, alpha=0.5)
        min = data.dDICdTA.values.min()
        if min < 0.85:
            num_drops+=1 
 
    leg = ax.legend(bbox_to_anchor=(1.05, 1.05))
    
    return fig , num_drops