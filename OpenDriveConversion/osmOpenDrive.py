import os
import glob
import sys

try:
    sys.path.append('~/carla/PythonAPI/carla/dist/carla-*3.10-linux-x86_64.egg')
except IndexError:
    pass

import carla

# Read the .osm data
# file_name = "./OpenDriveConversion/map.osm"
file_name = "map.osm"
f = open(file_name, 'r')
osm_data = f.read()
f.close()

# Define the desired settings. In this case, default values.
settings = carla.Osm2OdrSettings()
# Set OSM road types to export to OpenDRIVE
settings.set_osm_way_types(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"])
# Convert to .xodr
xodr_data = carla.Osm2Odr.convert(osm_data, settings)

# save opendrive file
f = open(file_name, 'w')
f.write(xodr_data)
f.close()