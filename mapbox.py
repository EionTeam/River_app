import streamlit as st
import json
import geopandas as gpd
import pyproj
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import zipfile
import os
import io

from src import choose_field_point, get_CRI_ocean, get_ocean_nodes, find_overlapping_stations, load_stations, generate_field_point,  get_coords, find_downstream_route, find_oean_point
from src import create_multi_CRI , snap_points

buffer = 0.009

#title and format app 
st.title('Eion Carbon Removal Verification')
# st.subheader()
def find_map(coords):

    json_flow = find_downstream_route(coords)
    _, overlap_station, time_to_ocean = snap_points(json_flow)
    
    # mapbox token
    #coords for X on map and ocean point
    cross_lon, cross_lat = coords[0], coords[1]
    mapboxt = 'MapBox Token'
    layout = go.Layout(title_text ='Sampling locations', title_x =0.5,  
            width=950, height=700,mapbox = dict(center= dict(lat=37,  
            lon=-95),accesstoken= mapboxt, zoom=4,style='stamen-terrain' ))
    
    field_loc = go.Scattermapbox(lat=[cross_lat], lon=[cross_lon],mode='markers+text', name = '', 
            below='False', opacity =1, marker=go.scattermapbox.Marker(
                autocolorscale=False,
                # showscale=True,
                size=10,
                opacity=1,
                color='red' ,  
            ))

    if len(overlap_station)== 0:
        #No sampling stations - just plot field  
        fig_cri_multi, num_drops, url_df, time_to_ocean = None, None, None, None
        # assign Graph Objects figure
        layer1 = [field_loc]
        
        
    else:
        #Add sampling stations and ph etc. 
        name_df = overlap_station.reset_index()
        name_df['hover_text']  =['Name: {}, Station ID: {},  USGS ID : {}, Index : {}'.format(a,b,c,d) for a,b,c,d in zip(name_df['STATION_NAME'], name_df['STATION_ID_ORIG'], name_df['STAT_ID'], name_df['index'])]
        name_df['url'] = ['https://waterdata.usgs.gov/monitoring-location/{}/'.format(('0'+ str(x)) ) for x in name_df['STATION_ID_ORIG']]
        url_df = name_df[['index', 'STATION_ID_ORIG', 'url', 'STATION_NAME']]
        data_plot = name_df[['Latitude', 'Longitude', 'pH', 'hover_text']]
        
        CRI_ocean,  ocean_ph = get_CRI_ocean(json_flow, overlap_station)
        df_plot = pd.concat([data_plot, ocean_ph], axis =0)

        fig_cri_multi, num_drops = create_multi_CRI(json_flow, CRI_ocean)

        o_coords = find_oean_point(json_flow)
        o_lon, o_lat = o_coords[0][0], o_coords[1][0]


        # define layers and plot map
        scatt = go.Scattermapbox(lat=df_plot['Latitude'], lon=df_plot['Longitude'],mode='markers+text', name ='pH' ,
            below='False', hovertext = df_plot['hover_text'],  marker=go.scattermapbox.Marker(
                autocolorscale=False,
                showscale=True,
                size=10,
                opacity=1,
                color=df_plot['pH'],
                colorscale='viridis_r', 
            )) #  
        

        # # streamlit multiselect widget
        # layer1 = st.multiselect('Layer Selection', [field_loc, scatt], 
        #     format_func=lambda x: 'Field' if x==field_loc else 'Stations')

        layer1 = [field_loc, scatt]

    # assign Graph Objects figure
    fig = go.Figure(data=layer1, layout=layout )
        
    #update with the river layer
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0}, mapbox=go.layout.Mapbox(style= "open-street-map", zoom=4, 
    center_lat = coords[1] -5 ,
        center_lon = coords[0]+7 ,
        layers=[{
            'sourcetype': 'geojson',
            'source': json_flow,
            'type': 'line',
            'color': 'cornflowerblue',
            
            'below' : 1000
        }]
    )
    )
    # fig.update_traces(name=  'pH', selector=dict(type='scattermapbox'))
    return fig, fig_cri_multi, num_drops, url_df, time_to_ocean, CRI_ocean




if 'num' not in st.session_state:
    st.session_state.num = 1
# if 'data' not in st.session_state:
#     st.session_state.data = []


default_address = 'Vicksburg, Mississippi'
address = st.text_input("Enter field address to see path of dissolved Carbon from field to ocean (City, State)", default_address )
go_b = st.button('Go', key='go')
rand_b = st.button('Take me to an interesting point', key='rand')



def main(coords):
        fig, fig_cri, num_drops, url_df, time_to_ocean, CRI_ocean  = find_map(coords)
        print(CRI_ocean)
        print(CRI_ocean.columns)

        if fig_cri is None:
            st.write("### No sampling stations downstream or not enough data available - please choose another location or click 'Take me to an interesting point")
            st.plotly_chart(fig)
        else:
        # display streamlit map
            tab1, tab2, tab3  = st.tabs(["Map", "Carbon Retention", "Station Links"])

            with tab1:
                
                st.plotly_chart(fig)

            with tab2: 
                with st.container():
                    # st.markdown('----')
                    st.markdown('#### DIC trapped in water from this point risks escape to the atmosphere *{}* times. Estimated travel time to ocean after reaching waterway is {} weeks.'.format(num_drops, time_to_ocean.round(1)))
                    # Create a download button
                    gdf = gpd.GeoDataFrame(CRI_ocean, crs="EPSG:4326")

                    # Save GeoDataFrame to a shapefile in a temporary directory
                    shapefile_name = "CRI_ocean"
                    gdf.to_file("CRI_ocean.shp")

                    # Create a zip file of the shapefile components
                    shapefile_components = [f"{shapefile_name}.shp", f"{shapefile_name}.shx", f"{shapefile_name}.dbf", f"{shapefile_name}.prj"]

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for component in shapefile_components:
                            zip_file.write(component, os.path.basename(component))

                    # Reset buffer position to the beginning
                    zip_buffer.seek(0)

                    # Create a download button for the zip file
                    st.download_button(
                        label="Download Ocean CRI",
                        data=zip_buffer,
                        file_name='CRI_ocean.zip',
                        mime='application/zip',
                    )
                    st.pyplot(fig_cri)
                    # st.markdown('----')

            with tab3:
                st.header("Sampling stations")
                with st.container():
                    for index, id, url , name in zip(url_df['index'], url_df['STATION_ID_ORIG'], url_df['url'] , url_df['STATION_NAME']):
                        st.write('{} {} ({})'.format(index, name, url))

            st.write("Sources: River data is sourced from USGS's [NLDI API](https://waterdata.usgs.gov/blog/nldi-intro/) and the [GLORICH](https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glorich.html) Global River Chemistry Database. Ocean data is sourced from [NOAA.](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0220059)")
        
        
        

while True:    
    num = st.session_state.num

    if go_b:
        coords = get_coords(address)
        if coords is not None:
            main(coords)
        else:
            st.write("### Address is not within USA or is inaccurate -  please try again or click 'Take me to an interesting point'")
        st.session_state.num += 2
        
        break
    elif rand_b:
        coords = choose_field_point()
        print(coords)
        main(coords)
 
        st.session_state.num += 2
        
        break
    else:        
        st.stop()


    