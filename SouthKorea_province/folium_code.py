import folium
import pandas as pd

# Data import
geo_data = 'TL_SCCO_SIG.json'
df = pd.read_csv('Region.csv')
df.head()

# Plot target data
str = 'elderly_population_ratio'

# Plot Choropleth
center = [37.541, 126.986]
m = folium.Map(location=center, zoom_start=6)

folium.Choropleth(
    geo_data=geo_data,
    name='choropleth',
    data=df,
    columns=['city', str],
    key_on='feature.properties.SIG_ENG_NM',
    
    fill_color='BuPu',
    #fill_opacity=0.7,
    #line_opacity=0.2,
    legend_name=str
    
).add_to(m)

folium.LayerControl().add_to(m)

m