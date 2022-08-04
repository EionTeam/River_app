
import streamlit as st
import json
import geopandas as gpd
import pyproj
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from src import find_overlapping_stations, load_stations, generate_field_point,  get_coords, find_downstream_route, create_CRI, plot_CRI, find_oean_point
from src import create_multi_CRI , snap_points

buffer = 0.009

#title and format app 
st.title('Eion Carbon Removal Verification')
# st.subheader()
def find_map(coords):
    json_flow = find_downstream_route(coords)
    overlap_station = find_overlapping_stations( json_flow, buffer_rad =buffer)
    _, overlap_station = snap_points(json_flow)
    # CRI , ocean_index =create_CRI(overlap_station)
    
    fig_cri_multi, num_drops = create_multi_CRI(json_flow, )
    #coords for X on map and ocean point
    cross_lon, cross_lat = coords[0], coords[1]

    o_coords = find_oean_point(json_flow)
    o_lon, o_lat = o_coords[0][0], o_coords[1][0]
    
    # mapbox token
    mapboxt = 'MapBox Token'

    # define layers and plot map
    scatt = go.Scattermapbox(lat=overlap_station['Latitude'], lon=overlap_station['Longitude'],mode='markers+text',  
        below='False', hovertext = overlap_station['index'],  marker=go.scattermapbox.Marker(
            autocolorscale=False,
            showscale=True,
            size=10,
            opacity=1,
            color=overlap_station['pH'],
            colorscale='viridis_r', 
        )) #  

    field_loc = go.Scattermapbox(lat=[cross_lat, o_lat], lon=[cross_lon, o_lon],mode='markers+text',    
        below='False', opacity =1, marker=go.scattermapbox.Marker(
            autocolorscale=False,
            # showscale=True,
            size=10,
            opacity=1,
            color='red' 
            
        ))

    layout = go.Layout(title_text ='Sampling locations', title_x =0.5,  
        width=950, height=700,mapbox = dict(center= dict(lat=37,  
        lon=-95),accesstoken= mapboxt, zoom=4,style='stamen-terrain' ))


    # # streamlit multiselect widget
    # layer1 = st.multiselect('Layer Selection', [field_loc, scatt], 
    #     format_func=lambda x: 'Field' if x==field_loc else 'Stations')

    layer1 = [field_loc, scatt]

    # assign Graph Objects figure
    fig = go.Figure(data=layer1, layout=layout )

    #update with the river layer
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0}, mapbox=go.layout.Mapbox(style= "open-street-map", zoom=4, 
    center_lat = coords[1],
        center_lon = coords[0],
        layers=[{
            'sourcetype': 'geojson',
            'source': json_flow,
            'type': 'line',
            'color': 'cornflowerblue',
            
            'below' : 1000
        }]
    )
    )
    return fig, fig_cri_multi, num_drops



   
# def find_map(coords):
#     json_flow = find_downstream_route(coords)
#     # overlap_station = find_overlapping_stations( json_flow, buffer_rad =buffer)
#     _, overlap_station = snap_points(json_flow)
#     # CRI , ocean_index =create_CRI(overlap_station)
#     # fig_cri , ax_cri  =  plot_CRI(CRI , ocean_index)
#     fig_cri_multi, num_drops = create_multi_CRI(json_flow, )
#     #coords for X on map and ocean point
#     cross_lon, cross_lat = coords[0], coords[1]

#     o_coords = find_oean_point(json_flow)
#     o_lon, o_lat = o_coords[0][0], o_coords[1][0]
    
#     # mapbox token
#     mapboxt = 'MapBox Token'

#     # define layers and plot map
#     scatt = go.Scattermapbox(lat=overlap_station['Longitude'], lon=overlap_station['Latitude'],mode='markers+text',  
#         below='False', hovertext = overlap_station['index'],  marker=go.scattermapbox.Marker(
#             autocolorscale=False,
#             showscale=True,
#             size=10,
#             opacity=1,
#             color=overlap_station['pH'],
#             colorscale='viridis_r', 
#         )) #  

#     field_loc = go.Scattermapbox(lat=[cross_lat, o_lat], lon=[cross_lon, o_lon],mode='markers+text',    
#         below='False', opacity =1, marker=go.scattermapbox.Marker(
#             autocolorscale=False,
#             # showscale=True,
#             size=10,
#             opacity=1,
#             color='red' 
            
#         ))

#     layout = go.Layout(title_text ='Sampling locations', title_x =0.5,  
#         width=950, height=700,mapbox = dict(center= dict(lat=37,  
#         lon=-95),accesstoken= mapboxt, zoom=4,style='stamen-terrain' ))


#     # # streamlit multiselect widget
#     # layer1 = st.multiselect('Layer Selection', [field_loc, scatt], 
#     #     format_func=lambda x: 'Field' if x==field_loc else 'Stations')

#     layer1 = [field_loc, scatt]

#     # assign Graph Objects figure
#     fig = go.Figure(data=layer1, layout=layout )

#     #update with the river layer
#     fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0}, mapbox=go.layout.Mapbox(style= "open-street-map", zoom=4, 
#     center_lat = coords[1],
#         center_lon = coords[0],
#         layers=[{
#             'sourcetype': 'geojson',
#             'source': json_flow,
#             'type': 'line',
#             'color': 'cornflowerblue',
            
#             'below' : 1000
#         }]
#     )
#     )
#     return fig, fig_cri_multi, num_drops



   


# if 'num' not in st.session_state:
#     st.session_state.num = 1
# if 'data' not in st.session_state:
#     st.session_state.data = []


# class AddresForm:
#     def __init__(self):
#         st.title(f"Enter address")
#         self.address = st.text_input("Address", 'Gilette Wyoming')
#         # self.randomPoint = st.button('Take me to a random point'):
    

if 'num' not in st.session_state:
    st.session_state.num = 1
# if 'data' not in st.session_state:
#     st.session_state.data = []


default_address = 'Gilette Wyoming'
address = st.text_input("Enter field address to see path of dissolved Carbon from field to ocean", default_address )
go_b = st.button('Go', key='go')
rand_b = st.button('Take me to an interesting point', key='rand')



def main(fig, fig_cri, num_drops):
    # display streamlit map
        st.plotly_chart(fig)
        with st.container():
            st.markdown('----')
            st.markdown('### DIC trapped in water from this point risks escape to the atmosphere *{}* times.'.format(num_drops))
            st.pyplot(fig_cri)
            st.markdown('----')
            st.write("Sources: Data is sourced from USGS's [NLDI API](https://waterdata.usgs.gov/blog/nldi-intro/) and the [GLORICH](https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glorich.html) Global River Chemistry Database.")


while True:    
    num = st.session_state.num

    if go_b:
        coords = get_coords(address)
        print(coords)
        fig, fig_cri, num_drops  = find_map(coords)
        
        main(fig, fig_cri, num_drops)
        # fig_line, ax_line = plt.subplots()
        # ax_line.plot( CRI['index'], CRI['dDICdTA'])
        
        # # display streamlit map
        # st.plotly_chart(fig)
        # with st.container():
        #     st.markdown('----')
        #     st.markdown('### DIC trapped in water from this point risks escape to the atmosphere *{}* times.'.format(num_drops))
        #     st.pyplot(fig_cri)
        #     st.markdown('----')
        st.session_state.num += 2
        
        break
    elif rand_b:
        coords = generate_field_point()
        print(coords)

        fig, fig_cri, num_drops  = find_map(coords)
        
        # fig_line, ax_line = plt.subplots()
        # ax_line.plot( CRI['index'], CRI['dDICdTA'])
        main(fig, fig_cri, num_drops)
        # display streamlit map
        # st.plotly_chart(fig)
        # with st.container():
        #     st.markdown('----')
        #     st.markdown('### DIC trapped in water from this point risks escape to the atmosphere *{}* times.'.format(num_drops))
        #     st.pyplot(fig_cri)
        #     st.markdown('----')
        st.session_state.num += 2
        
        break
    else:        
        st.stop()


    