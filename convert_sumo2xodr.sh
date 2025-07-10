#!/bin/bash

netconvert --opendrive-output ./results/511video/$1/sumo/$1.xodr --sumo-net-file ./results/511video/$1/sumo/$1.net.xml --junctions.scurve-stretch 0.1
python OpenDriveConversion/openDrive2Carla.py --map_file results/511video/$1/sumo/$1
