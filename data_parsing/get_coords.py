import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import wkb

# Path to data folder CHANGE AS NEEDED
df = pd.read_csv("results_and_data/all_sites_PSO_parameter_data/all_sites_all_data.csv")

# Convert the hex string to a spatial geometry object
def decode_hex_to_geom(hex_string):
    try:
        # Convert hex string to bytes, then load as Shapely geometry
        return wkb.loads(bytes.fromhex(hex_string))
    except Exception:
        return None

# Create a geometry column
df['geometry'] = df['grts_geometry'].apply(decode_hex_to_geom)

# Extract the Latitude and Longitude from the polygon's centroid
df['longitude'] = df['geometry'].apply(lambda geom: geom.centroid.x if geom else None)
df['latitude'] = df['geometry'].apply(lambda geom: geom.centroid.y if geom else None)

# save
df.to_csv("results_and_data/all_sites_PSO_parameter_data/all_sites_with_coordinates.csv", index=False)