
import streamlit as st
import json
import geopandas as gpd
import pyproj
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from src import find_overlapping_stations, load_stations, get_coords, find_downstream_route, create_CRI, plot_CRI
base_dir = '/Users/gracecolverd/EionTeam/Eion'
buffer = 0.009

#title and format app 
st.title('Eion Carbon Removal Verification')
st.subheader('Enter field address to see path of dissolved Carbon from field to ocean.')


#### Ask for address, generate lat/lon and find river and stations
address = st.text_input("Address", 'Gilette Wyoming')
coords = get_coords(address)
json_flow = find_downstream_route(coords)
overlap_station = find_overlapping_stations( json_flow, base_dir, buffer_rad =buffer)
CRI , ocean_index =create_CRI(overlap_station)
fig_cri , ax_cri  =  plot_CRI(CRI , ocean_index)

#coords for X on map 
cross_lat = coords[1]
cross_lon = coords[0]

fig_line, ax_line = plt.subplots()
ax_line.plot( CRI['index'], CRI['dDICdTA'])



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

field_loc = go.Scattermapbox(lat=[cross_lat], lon=[cross_lon],mode='markers+text',    
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


# streamlit multiselect widget
layer1 = st.multiselect('Layer Selection', [field_loc, scatt], 
         format_func=lambda x: 'Field' if x==field_loc else 'Stations')


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


# display streamlit map
st.plotly_chart(fig)

st.pyplot(fig_cri)

# st.line_chart(chart_data)
