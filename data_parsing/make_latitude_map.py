import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path to data folder CHANGE AS NEEDED
df = pd.read_csv("results_and_data/all_sites_PSO_parameter_data/all_sites_with_coordinates.csv")

# Convert the text POLYGON ((...)) strings into actual Shapely geometry objects
df['geometry'] = df['geometry'].apply(wkt.loads)

# Create a GeoDataFrame, explicitly telling it that the coordinates are standard GPS (EPSG:4326)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# Assign colors based on the 40-degree latitude cutoff
gdf['color'] = gdf['latitude'].apply(lambda x: 'blue' if x >= 40 else 'red')

# -------------------------
# Fetch US State Boundaries
# -------------------------

# We pull the 2021 low-resolution state shapefiles directly from the US Census Bureau
print("Downloading state boundaries from US Census...")
states_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip"
states = gpd.read_file(states_url)

# Define the Midwest states to filter the map
midwest_names = [
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Michigan', 'Minnesota',
    'Missouri', 'Nebraska', 'North Dakota', 'Ohio', 'South Dakota', 'Wisconsin'
]
midwest = states[states['NAME'].isin(midwest_names)]

# ---------------------------
# Reproject for aesthetic map
# ---------------------------

# EPSG:5070 is the "CONUS Albers" projection. It prevents the map from 
# looking stretched out horizontally, which happens if you just plot raw Lat/Lon.
midwest = midwest.to_crs("EPSG:5070")
gdf = gdf.to_crs("EPSG:5070")

# Create a horizontal line across the map at exactly 40 degrees Latitude
line_40_deg = gpd.GeoDataFrame(
    geometry=[LineString([(-105, 40), (-80, 40)])], 
    crs="EPSG:4326"
).to_crs("EPSG:5070")

# ------------
# Draw the Map
# ------------

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title("Midwest Bat Hibernacula (Divided at 40° Latitude)", fontsize=16, fontweight='bold')

# Plot the state boundaries
midwest.plot(ax=ax, facecolor='#f4f4f4', edgecolor='black', linewidth=1)

# Add state abbreviations as labels in the center of each state
for idx, row in midwest.iterrows():
    # Get the center point of the state
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    ax.annotate(text=row['STUSPS'], xy=(x, y), 
                horizontalalignment='center', color='#555555', 
                fontsize=12, fontweight='bold')

# Plot the 40-degree dividing line
line_40_deg.plot(ax=ax, color='gray', linestyle='--', linewidth=1.5, zorder=1)
ax.text(line_40_deg.geometry.bounds.minx.iloc[0], line_40_deg.geometry.bounds.miny.iloc[0], 
        ' 40° N Latitude', color='gray', verticalalignment='bottom')

# Plot your 10x10km polygons!
gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', linewidth=0.5, alpha=0.8, zorder=5)

# ------------------
# Clean up + display
# ------------------

# Turn off the messy coordinate axes
ax.axis('off')

# Add a custom legend
blue_patch = mpatches.Patch(color='blue', alpha=0.8, label='Upper Midwest (≥ 40° N)')
red_patch = mpatches.Patch(color='red', alpha=0.8, label='Lower Midwest (< 40° N)')
plt.legend(handles=[blue_patch, red_patch], loc='lower right', fontsize=12)

# Save to your computer and display
plt.savefig("results_and_data/all_sites_PSO_parameter_data/midwest_polygons_map.png", dpi=300, bbox_inches='tight')
print("Map saved to 'midwest_polygons_map.png'")
plt.show()