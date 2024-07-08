#!/usr/bin/env python
# coding: utf-8

# SRC File for River Runner 

# In[1]:

import os
import urllib.parse
import streamlit as st

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString
import shapely as sh
import random
from matplotlib.pyplot import cm
import pyproj
from shapely.geometry import Point, LineString
utm = pyproj.CRS('EPSG:26907')

def get_json(url):
    response = requests.get(url)
    return  response.json()
       
def plot_line_col(ax, ob, c):
    """Plot Lines with chosen colours, for shapely Linestring objects 
    Inputs:
    ax: axes of fig, ax 
    ob: linestring
    c: colour to plot 
    """
    x, y = ob.xy
    ax.plot(x, y, color=c, alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

def plot_line(ax, ob ):
    """Plot Lines, for shapely Linestring objects 
    Inputs:
    ax: axes of fig, ax 
    ob: linestring
     
    """
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
    """ Retrun the json of the flowlines for the downstream route using the USGS API river; 
    
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



def print_downstream():
    """ Print map of downstream route 
    """
    
    flowlines = find_downstream_route()
    
    fig, ax = plt.subplots() 

    for x in flowlines['features']:
        coords = x['geometry']['coordinates']
        plot_line(ax, LineString(coords) )


# def plot_line(ax, ob):
#     x, y = ob.xy
#     ax.plot(x, y, color='b', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

def find_overlapping_stations(data, buffer_rad = 0.01 ):
    """ FInd stations that overlap with JSON of river lines; using a radius in degrees set by the buffer rad; 
    Deprecated in favour of using snap points 
    Inputs:
    data: json / dict form of river flow lines 
    """
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

# @st.cache(allow_output_mutation=True)
def load_stations():
    """
    Load sampling stations; from Glorich global chem database
    """
    path =  os.path.join(os.path.dirname(__file__),"data/sampling_locations.csv") 
    loc = pd.read_csv(path)
    usloc = loc[loc['Country']=='USA']
    return gpd.GeoDataFrame(usloc, geometry=gpd.points_from_xy(usloc['Longitude'], usloc['Latitude']), crs='EPSG:4326')
    

# @st.cache(allow_output_mutation=True)
def load_chem( locations):
    """Load processed file with the dDICdTA info calc by Adam 
    """
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
    """Filter list of locations based on filters of clean data 
    TODO update this to just be a file? 
    """
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
    """use Open streem map to get lat lon coords from address that user will give on stremalit app 
    """
    address = address.replace(", ", "+")
    address = address.lower()
    # url = 'https://nominatim.openstreetmap.org/search' + urllib.parse.quote(address) +'?format=json'
    # https://nominatim.openstreetmap.org/search?q=vicksburg+mississippi+us&format=jsonv2&limit=1
    
    url = f"https://nominatim.openstreetmap.org/search?q={address}+us&format=jsonv2&limit=1"
    headers = {
        'User-Agent': 'YourAppName/1.0 (your-email@example.com)'
    }
    
    response = requests.get(url, headers=headers)
    output = response.json()
    lat = output[0]['lat']
    lon = output[0]['lon']
    coords = (float(lon), float(lat))
    return coords


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

# @st.cache(allow_output_mutation=True)
def open_missipi_sh_file():
    basin_sh =  os.path.join(os.path.dirname(__file__),'data/Miss_RiverBasin/Miss_RiverBasin.shp')
    basin = gpd.read_file(basin_sh)
    basin =  basin.to_crs('EPSG:4326')
    buff_basin = basin.buffer(-1)
    buff_basin.to_crs('EPSG:4326')
    buffer_gdf = gpd.GeoDataFrame(buff_basin , crs='EPSG:4326', geometry=buff_basin ) 
    return buffer_gdf

def load_rand_points():
    return pd.read_csv(os.path.join(os.path.dirname(__file__),'data/interesting_points.csv'))
  
def choose_field_point():
    """ Choose random field point out of pre created list 
    """
    df_points = load_rand_points()
    i= random.randint(0,61)
    return df_points.iloc[i, :]['lon'], df_points.iloc[i, :]['lat']

def generate_field_point():
    """Generates random point in Missipi basin, filters using shapefile 
    """
    while True:
        point= random_point_mis_basin()
        p_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries( point) ).set_crs( crs = 'EPSG:4326')
        p_df =  p_df.to_crs( crs = 'EPSG:4326')
        buffer_gdf = open_missipi_sh_file()

        overlap = buffer_gdf.sjoin(p_df)
        if len(overlap) != 0:
            break
        
    return point.x,  point.y



def get_river_df_utm(data):
    """Process the JSON version of river flowlines data into nice GeoPandas dataframe; reused multiple times 
    """
    utm = pyproj.CRS('EPSG:26907')
    coords = [data['features'][i]['geometry']['coordinates'] for i in range(len(data['features'])) ]
    dict_data = {key: value for (key, value) in zip([i for i in range(len(data['features'])) ] , [LineString(coords[i]) for i in range(len(data['features'])) ] ) }
    river_df = pd.DataFrame(dict_data, index=['geometry']).T
    river_gdf = gpd.GeoDataFrame(river_df , crs='EPSG:4326', geometry=river_df['geometry'] )
    return river_gdf.to_crs(utm)

def snap_points(data,  offset = 1000):
    """Generate the snapped points to the river for stations, uses https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
    Converts into UTM projected coords and then takes a radius of 1km 
    """

    utm = pyproj.CRS('EPSG:26907')

    lines = get_river_df_utm(data)

    # get time to ocean based on all lines and speed of 1.2MPH / 1.93 KMPH
    len_lines = [x.length for x in lines.geometry]
    lines['len_lines'] = len_lines
    speed_per_week_km = 1.93 * 24 * 7
    len_to_ocean_km = lines.len_lines.sum()/1000
    time_to_ocean = len_to_ocean_km/speed_per_week_km
    
    #get locations 
    locations = create_filtered_locations()
    locations = locations.to_crs(utm)
    points = gpd.GeoDataFrame(locations , crs=utm, geometry=locations['geometry'] )  
    # print(points.crs, lines.crs)

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
    updated_points = updated_points.dropna(subset=['STATION_NAME','STATION_ID_ORIG','pH', 'dDICdTA', "geometry"])
    updated_points= updated_points.to_crs('EPSG:4326')
    updated_points['index'] = [i+1 for i in range(len(updated_points)) ]
   
    overlap = updated_points.loc[:, ['index','STATION_NAME','STATION_ID_ORIG', 'geometry', 'pH', 'dDICdTA']]
    overlap['Longitude']= [point.x for point in overlap['geometry']]
    overlap['Latitude']=[point.y for point in overlap['geometry'] ]
    overlap.drop(columns= 'geometry', inplace=True)
    
    return updated_points, overlap, time_to_ocean

def find_CRI_years():
    """ Takes valid station ids of sampling locations, and returns the yearly mean values of CRI for these stations
    __ this could also be replaced with singular file 
    """
    locations = load_stations()
    valid_ids = locations.index.unique().tolist()
    chem= load_chem(locations)
    q_CRI = chem[['STAT_ID','Y', 'dDICdTA']].groupby(['STAT_ID',  'Y']).mean().dropna()
    return q_CRI.reset_index(level=[1])


def create_multi_CRI(json_flow, CRI_ocean):
    """ Create the mutli-CRI plots to use in streamlit app from the json form of the river flowlines 
    Filter to at least three data points to include a year. It also checks how many years the CRI falls below the ocean CRI 
    Returns: the fig to plot and the num of years it dropped 
    Inputs:
    data: json form of flowlines river data
    cri_ocean: the ocean carbonate values for the ocean stations for all years 
    """
    #Get nearby sampling stations
    loc, _, _= snap_points(json_flow)
    loc['index_plot']= [i +1 for i in range(len(loc))]

    # Get the yearly values of CRI for these statiosn 
    q_CRI = find_CRI_years()
    join = loc.join(q_CRI, lsuffix='l')

    year = join.groupby('Y').count()
    #filter to 3 data points 
    populated = year[year['pH'] >2].index.tolist()
    if len(populated) !=0 :
        
        fig, ax = plt.subplots(figsize= (10,6))

        cmap = plt.cm.twilight_shifted
        norm = mpl.colors.Normalize(vmin=int(populated[0]), vmax=int(populated[-1]) )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)


        fig.suptitle('DIC Retention Index (DRI) by Station',
            fontsize='large',
            # loc='center',
            fontweight='bold',
            style='normal',
            family='monospace'
            )


        ocean_indx= len(loc)+1
        oc, *oc2 = [x for x in range( ocean_indx, ocean_indx+6)]

        ax.set_xlabel('Station Index (Field = 0)'.format(ocean_indx) )
        ax.set_ylabel('DRI')
        
        miny = 1982
        maxy = 2010
        
        ocean_years = [i for i in range(miny, maxy+1) ]
        for y in ocean_years:
            ocean_data= CRI_ocean[CRI_ocean.Y == y]
            oc_c, * oc_cri = ocean_data.dDICdTA.values.tolist()
            ax.plot([oc, *oc2], [  oc_c, * oc_cri], label=y, alpha=0.8)
        
        num_drops=0
        for y in populated:
            data= join[join.Y == y]
            # color=cmap(norm(y))
            
            a,*b =  data.index_plot.values.tolist()
            c,*d =  data.dDICdTA.values.tolist()
            ax.plot([0,a,*b], [1, c, *d,  ], label=y, alpha=0.8 )
            min = data.dDICdTA.values.min()
            if min < 0.88:
                num_drops+=1 

        # leg = ax.legend(bbox_to_anchor=(1.15, 1.05))
        
        plt.text(  ocean_indx- (len(loc)/3), 1, 'Line represent distinct years between 1980-2020', bbox=dict(facecolor='thistle', alpha=0.3))
        plt.text( ocean_indx-(len(loc)/3),0.98, 'RIVER', horizontalalignment='center')
        plt.text(ocean_indx+2, (ocean_data.dDICdTA.max() +0.01), 'OCEAN', horizontalalignment='center')
    else:
        print('Not enough data population from sampling stations')
        fig, num_drops = None, None 
    return fig , num_drops


def process_river(data): 
    """ 
    Process the json output from UGSG api into clean geodataframe in 4326 crs 
    """
    coords = [data['features'][i]['geometry']['coordinates'] for i in range(len(data['features'])) ]
    dict_data = {key: value for (key, value) in zip([i for i in range(len(data['features'])) ] , [LineString(coords[i]) for i in range(len(data['features'])) ] ) }
    river_df = pd.DataFrame(dict_data, index=['geometry']).T
    river_gdf = gpd.GeoDataFrame(river_df , crs='EPSG:4326', geometry=river_df['geometry'] )
    river_gdf.to_crs('EPSG:4326')
    return river_gdf
    
    
def getExtrapoledLine(p1,p2):
    """
    Creates a line extrapoled in p1->p2 direction. input points need to be in utm coords. this creates line 500 km out
    """
    dist = p1.distance(p2) 
    EXTRAPOL_RATIO = 500000 / dist 
    a = p2
    b = (p2.x+EXTRAPOL_RATIO*(p2.x-p1.x), p2.y+EXTRAPOL_RATIO*(p2.y-p1.y) )
    return LineString([a,b])



def get_ocean_nodes(data, df_loc):
    """ 
    Extrapolate from last station and mouth of river to get 5 stations each 100km out into the ocean 
    """

    river_gdf = process_river(data)
    river_utm = river_gdf.to_crs(utm)
    _ ,ocean = river_utm.iloc[-1].geometry.boundary.geoms
    
    df_loc['geometry'] = [Point(lon, lat) for lon, lat in zip(df_loc['Longitude'],df_loc['Latitude']  )]
    df_utm = df_loc.to_crs(utm)
    final_station =  df_utm.iloc[-1].geometry
    print(final_station)
    extend = getExtrapoledLine(final_station,ocean)
    km = 1000
    one, two, three, four,five = extend.interpolate(100*km), extend.interpolate(200 *km ), extend.interpolate(300*km), extend.interpolate(400*km) , extend.interpolate(500*km)                                                                                                                              
    points = [ocean, one, two, three, four,five]
    new_nodes = gpd.GeoDataFrame(geometry = points, crs = utm)
    new_nodes = new_nodes.to_crs("EPSG:4326")  

    return new_nodes



def load_ocean_grid_carbonate_data():
    path = os.path.join(os.path.dirname(__file__),"data/ocean_grid/ocean_carbonate_grid_data.shp") 
    gdf = gpd.read_file(path)
    return gdf


def get_CRI_ocean(json_flow, overlap_station):
    """
    Pull the geogrid data for ocean sampling and ocean ph for the 5 ocean stations + mouth of river
    """
    #Load processed Grid Cabonate data
    gdf = load_ocean_grid_carbonate_data()

    #Find extended 5 stations 
    ocean_stations = get_ocean_nodes(json_flow, overlap_station)

    CRI_ocean = pd.DataFrame(columns = ['Y', 'dDICdTA', ])
                        
    for point in ocean_stations.geometry:
        
        gdf['dist'] = [x.distance(point) for x in gdf.geometry ]
        test = gdf[gdf['dist'] == gdf['dist'].min()][['Y', 'dDICdTA', 'ph_total']]
        test['geometry'] = point
        
        CRI_ocean= pd.concat([CRI_ocean,test],axis =0)    
    CRI_ocean['Latitude'] = [x.y for x in CRI_ocean.geometry]
    CRI_ocean['Longitude'] = [x.x for x in CRI_ocean.geometry]
    
    ocean_ph = CRI_ocean[[ 'Latitude','Longitude', 'ph_total']].groupby([ 'Longitude', 'Latitude']).mean().reset_index()
    ocean_ph['hover_text'] = ['Ocean Sample, pH: {}'.format(ph) for ph in ocean_ph['ph_total'].round(1) ]
    ocean_ph.rename(columns={ 'ph_total': 'pH'}, inplace=True )

    return CRI_ocean,  ocean_ph 