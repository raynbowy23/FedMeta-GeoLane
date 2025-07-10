# OSM extraction

Get Open Street Map data with SUMO.

1. `python osmWebWizard.py` to get OSM information. Clip as accurate as possible compared to camera range. Note: Vanishing points can be ignored for better accuracy.
2. Unpack oxm.net.xml.gz.
3. Generate net file with netconvert. `netconvert --sumo-net-file osm.net.xml --plain-output-prefix=osm`.
4. When running python train.py, add args.osm_save_date generated_folder_name_within_osm_extraction.