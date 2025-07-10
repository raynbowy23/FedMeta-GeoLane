"""
Traffic video -> OSM map -> SUMO network -------------- |
        |                                               |     
        |--> Detected, reasoned, and percepted features --> OpenDrive Map

- Generate traffic network with reference scene.
- Can also generate with LLM without any reference scene (just like MapLLM by NVIDIA)
- Can also match with sampling from Waymo Dataset -> so we can make drivable area for multi-agent simulation

"""