# 511 video to OpenDrive Conversion

## Procedure

Basic procedure is following:

Preprocess

- OSM -> SUMO
- Required files: .osm
1. (Manual) Download 511 video and make it into a good length video.
<!-- 2. (Manual) Find the same location from OpenStreetMap (OSM) with `osmWebWizard.py`. -->
2. (Manual) Manually select target area from OpenStreetMap (OSM) from (https://www.openstreetmap.org/#map=17/43.034678/-89.426753).

Optional
3. (Manual) Open it in SUMO.
4. (Manual) Trim it to have only target road (Remove unnecessary part).

Place map.osm in the working folder and run, `netconvert --osm-files map.osm --o osm.net.xml`, and you'll get osmnet.net.xml.

**NOTE** Remove offset from netfile so Carla coordinates will match.
GeoReference in OpenDrive will look like this. I added lat_0 and lon_0 by calculated with mean of origBoundary.
```
<![CDATA[
 +lat_0=43.0354385 +lon_0=-89.429378 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs
]]>
```

OpenDrive Conversion to Carla Sync in one shell file.

- SUMO -> XODR with netconvert
- Required files: .net.xml

- XODR -> Carla
- Required files: .xodr
5. Launch Carla server.
6. Run `bash convert_sumo2xodr.sh {path to camera location}` to get .xodr file.
(Option: `python openDrive2Carla.py --map_file ../results/511video/{camera location}/sumo/{camera location}`)

7. .xodr can be viewed at odrviewer.io.
<!-- 8. (Auto) On the different console, convert .xodr to run in Carla by running `openDrive2Carla.py`. -->


Trajectory

- Detection CSV -> SUMO -> Carla
- Required files: .csv, .osm, .net.xml, .rou.xml, .view.xml, .sumocfg
- Copy our .view.xml and .sumocfg as needed. .rou.xml file is updated for each location and captured trajectories.

8. Run detected trajectory in SUMO with `python det2sumo_sync.py --camera_loc {camera location}`. Make sure to have trajectory in csv file and place it in the right location. This generates vehicles.rou.xml. You can check vehicles are running in SUMO.
8.5. On the different console, make sure to change map in Carla by running `python openDrive2Carla.py --map_file osmnet`. If you already run Step 6, ignore this.
9. Change camera location in osm.sumocfg. Co-simulation SUMO and Carla with `python run_synchronization.py osm.sumocfg --sumo-gui`.


## Make it intelligent!

[ ] Location can be automatically cripped such as finding the same location bounding box easily by codes. Then extract all nodes with api/0.6/map. Or auto-select bounding box at osmWebWizard.
[ ] Auto-select (auto-remove) target road.
[ ] Use Video Positioning System (VPS) to automatically select the scene.
[ ] Calibrate or evaluate with lane detection from clustered trajectories.
[ ] Auto-alignment to original image (e.g. image reconstruction, morphing, or doing like SPADE?)

## Questions

- Google map can recognize lanes while OSM totally relies on user inputs, which may lead some mistakes. Thus, we need to justify the number of lanes with acuumulated trajectories.
- Also, we should estimate lane widths with some way.