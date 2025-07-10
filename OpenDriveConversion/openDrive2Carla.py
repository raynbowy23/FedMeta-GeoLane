import glob
import os
import sys
import xml.etree.ElementTree as ET
import argparse

# Add CARLA egg to sys.path
carla_egg = glob.glob(os.path.expanduser('~/carla/PythonAPI/carla/dist/carla-*3.10-linux-x86_64.egg'))

if carla_egg:
    sys.path.append(carla_egg[0])
else:
    raise ImportError("CARLA .egg file not found. Please check the path.")


import carla

parser = argparse.ArgumentParser()
parser.add_argument('--map_file', default='./results/test2/test2', help='The path to the xodr file')
args = parser.parse_args()

myFile = ET.parse(args.map_file + ".xodr")
root = myFile.getroot()
xodr_data = ET.tostring(root, encoding='unicode', method="xml")

actor_list = []

# try:
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

print('load opendrive map.')
vertex_distance = 2.0  # in meters
max_road_length = 500.0 # in meters
wall_height = 1.0      # in meters
extra_width = 0.6      # in meters

world = client.generate_opendrive_world(
    xodr_data, carla.OpendriveGenerationParameters(
                vertex_distance=vertex_distance,
                max_road_length=max_road_length,
                wall_height=wall_height,
                additional_width=extra_width,
                smooth_junctions=True,
                enable_mesh_visibility=True))
# except:
    # print("Error in generating world from xodr file")