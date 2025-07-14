<div id="top" align="center">
<p align="center">
  <strong>
    <h1 align="center">Geo-ORBIT</h1>
    <!-- <h3 align="center"><a href="">Website</a> | <a href="#">Paper</a></h3> -->
  </strong>
</p>
</div>

> Geo-ORBIT: A Federated Digital Twin Framework for Scene-Adaptive Lane Geometry Detection (Paper be uploaded)
>
> **Authors**: [Rei Tamaru](https://tamaru.com), [Pei Li](https://scholar.google.com/citations?user=0QzhzL0AAAAJ), and [Bin Ran](https://scholar.google.com/citations?user=Kg5OoCUAAAAJ)
>
> **Affiliation**: University of Wisconsin–Madison

---

**Geo-ORBIT** (Geometrical Operational Roadway Blueprint with Integrated Twin) is a unified framework that integrates real-time lane detection, federated learning, and digital twin synchronization. It is designed to support active traffic management, infrastructure monitoring, and real-time scenario testing without relying on centralized data collection.

At the core of Geo-ORBIT is **FedMeta-GeoLane**, a federated meta-learning-based lane detection model that adapts to scene-specific geometry using only vehicle trajectory data. By preserving privacy and reducing bandwidth, this system enables scalable deployment across diverse roadside camera environments.

<figure style="text-align: center;">
  <img src="figs/qualitative_result.png" width="100%">
  <figcaption><b>Figure:</b> Qualitative comparison across multiple locations. The camera at Park is treated as an unseen location for Meta-GeoLane and FedMeta-GeoLane. Blue lines represent trajectory contours, and each lane is colored accordingly in the same lane group.</figcaption>
</figure>
  <!-- <figcaption><b>Figure:</b> Architecture of the federated meta-learning framework. The framework detects roadway geometry at local entities with local GeoLane models. The central server collects parameters from local entities with federated learning. The DT synchronizes road geometry and trajectories in a simulated environment.</figcaption> -->



## System Architecture

Geo-ORBIT is composed of three modular and interconnected processes:

- **Detection Process**  
  Roadside cameras capture traffic video, from which vehicle trajectories are extracted and projected to GPS space.

- **Service Process**  
  The **FedMeta-GeoLane** model infers lane geometries from trajectories using adaptive parameters, refined through meta-learning and weak supervision (e.g., OpenStreetMap).

- **Simulation Process**  
  Detected lanes are synchronized with **SUMO** and **CARLA** to create a high-fidelity, real-time **Digital Twin** that supports traffic flow rendering and scenario replay.

## Installation
Geo-ORBIT works with Python 3.10+ and Pytorch 2.5.1+.

Clone the repository
```bash
git clone https://github.com/raynbowy23/FedMeta-GeoLane.git
cd FedMeta-GeoLane
```

Create conda environment with Python >= 3.10. 
```bash
conda create -n lane_detection Python3.10
conda activate lane_detection
```

Install dependencies
```bash
pip install -r requirements.txt
```

Set up SUMO
```bash
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
```

Prepare data directories
```bash
mkdir -p dataset/511video dataset/511calibration
mkdir -p results logs
```

## How to Use
### Quickstart

Run the complete federated learning pipeline:
```bash
bash run.sh
```
This executes the defualt configuration with federated meta-learning on historical data.

### Basic Usage
1. Prepare camera calibration data.
- Place GPS calibration points in `dataset/511calibration`.
- Format: `camera_name.csv` with columns: `pixel_x, pixel_y, latitude, longitude`.

2. Add camera locations
- List camera names in `dataset/camera_location_list.txt`.
- One camera name per line.

3. Map Data Selection
- Extract corresponding OpenStreetMap data using `python osmWebWizard.py` in `./LaneDetection/osm_extraction`. Alternatively, download it online. (e.g. https://www.openstreetmap.org/#map=17/43.034678/-89.426753)
- Change extracted folder name to camera_name. (e.g. US12_Park)
- Extract `osm.net.xml.gz`.
- Run `netconvert -s osm.net.xml --plain-output-prefix osm`, and convert to plainXML `osm.nod.xml` and `osm.edg.xml`.

4. Map Data Preprocess
- Open `osm.net.xml` in local SUMO.
- Trim it to have only target road (Remove unnecessary part).

5. Run Lane Detection
```bash
python main.py --T 60 --is_save --model federated
```

### Advanced Configuration
**Federated Learning**
```bash
python main.py --model federated --T 60 --is_save --skip_continuous_learning --use_historical_data
```

Note: include `--skip_continuous_learning` and `--use_historical_data` if you want to skip video detection part for test, which reduce a lot of time.

**Meta Learning (Training on Single Camera)**
```bash
python main.py --model meta --T 60 --is_save --skip_continuous_learning --use_historical_data
```

**Baseline (Fixed Parameters)**
```bash
python main.py --model federated --T 60 --is_save --skip_continuous_learning --use_historical_data
```

### Key Parameters
- `--T`: Time interval for data collection (seconds)
- `--model`: Learning approach (federated, meta, baseline)
- `--is_save`: Save intermediate results and visualizations
- `--use_historical_data`: Use pre-processed trajectory data
- `--skip_continuous_learning`: Skip real-time detection
- `--lambda_thres`: Vehicle count threshold for learning cycles

### Output Structure
```bash
results/
└── 511video/
    └── model_name/
        ├── camera_location/
        |   ├── figures/                            # Visualizations
        |   ├── preprocess/                         # Processed data
        |   ├── sumo/                               # SUMO network files
        |   ├── pixel/                              # Mid opeartion visualization
        |   └── federated_trajectory_clustering.csv # Only for federated learning
        └── training_results/                       # Final results
```

## Simulation Integration

Look at [the detailed process](./OpenDriveConversion/README.md) to create digital twin integration.

### SUMO Integration
1. Generate SUMO network
```bash
bash convert_sumo2xodr.sh camera_name
```

2. Run trajectory synchroniztaion
```bash
python OpenDriveConversion/det2sumo_sync.py \
    --camera_loc camera_name \
    --dataset_path ./dataset/
```

### CARLA Integration

1. Start CARLA server
```bash
# In CARLA directory
make launch
# OR
./CarlaUE4.sh
```

2. Load generated map
```bash
python OpenDriveConversion/openDrive2Carla.py \
    --map_file results/511video/camera_name/sumo/camera_name
```

3. Run co-simulation
```bash
python OpenDriveConversion/run_synchronization.py osm.sumocfg --sumo-gui
```

## FedMeta-GeoLane: Federated Meta-Learning Lane Detection

FedMeta-GeoLane treats each roadside camera deployment as a unique task. A shared meta-learner predicts optimal detection parameters using context features like vehicle speed and trajectory distribution. Key highlights include:

- **Black-box meta-learning**: No need for gradient flow through detection pipeline  
- **Federated optimization**: Local training with privacy-preserving aggregation  
- **Scene adaptation**: Immediate configuration for unseen locations

<figure style="text-align: center;">
  <img src="figs/geo_lane.png" width="100%">
  <figcaption><b>Figure:</b> Overview of Knowledge-Based Lane Detection Algorithm. (a) Video detection and trajectory projection to GPS coordinates. (b) Lane center estimation using histogram analysis. (c) Lane-based trajectory clustering with KMeans. (d) Lane geometry estimation and boundary generation.</figcaption>
</figure>


_Compared to baseline and centralized models, FedMeta-GeoLane reduces geometric error by over 50% in unseen locations while achieving a 98% reduction in communication cost._


## Performance Summary

### Lane Detection Accuracy
**Table: Validation Loss Component Comparisons of Each Model on Seen and Unseen Locations**
| **Model**            | **Consistency Loss (m) &#8595;** | **Geometry Loss (m) &#8595;** | **Centerline Error (m) &#8595;** | **Lane Count Error &#8595;** |**Total Loss &#8595;** |
|------------------|---------------------|-------------------------|--------------------|-----------------------|-----------------------|
| _Seen_             |                   |                    |                       |                  |                     |
| Baseline         | 5.45              | 15.12              | 6.78                  | 5.00             | 77.84               |
| Meta-GeoLane     | 7.04              | 11.76             | 4.73                 | **2.67**            | 12.16               |
| **FedMeta-GeoLane** | 0.0            | **2.65**         | **3.16**             | **2.67**            | **6.94**           |
| _Unseen_             |                   |                    |                       |                  |                   |
| Meta-GeoLane     | 18.51             | 105.35             | 34.60                 | 12.00            | 69.61               |
| **FedMeta-GeoLane** | 0.0            | **12.82**         | **21.39**             | 12.00            | **32.38**           |

<figure style="text-align: center;">
  <img src="figs/qualitative_result.png" width="100%">
  <figcaption><b>Figure:</b> Qualitative comparison across multiple locations. The camera at Park is treated as an unseen location for Meta-GeoLane and FedMeta-GeoLane. Blue lines represent trajectory contours, and each lane is colored accordingly in the same lane group.</figcaption>
</figure>



### Transmission Cost Analysis

**Table: Bit Per Second Performance Comparison for All Clients**

| Parameters         | Baseline | Meta | Federated Meta |
|--------------------|----------|------|----------------|
| Model size (MB)    | 0        | 0    | 0.2            |
| Clients            | 4        | 4    | 4              |
| Rounds             | 1        | 20   | 20             |
| Model Upload (MB)  | 0        | 0    | 0.01           |
| File Upload (MB)   | 427.3    | 427.3| 5.6            |
| Download (MB)      | 0        | 0    | 0.018          |
| BPS (Mbps)         | 3418     | 3418 | **47.2**       |
 

## Digital Twin Integration

Geo-ORBIT connects real-world observations to virtual testbeds using a synchronized SUMO–CARLA pipeline:

- GPS-aligned trajectories enable accurate replay in simulation  
- Supports scene-level validation, vehicle re-routing, and visual analytics  
- (Will be implemented) Extendable to multi-scenario environments with dynamic overlays (e.g., vegetation, accidents, road closures)

<figure style="text-align: center;">
  <img src="figs/DTSynchro.png" width="100%">
  <figcaption><b>Figure:</b> Digital twin synchronization with SUMO and CARLA at multiple locations employing real-time vehicle trajectory.</figcaption>
</figure>


<!-- ### Citation
If you use this work in your research, please cite:

```bibtex
@article{tamaru2025,
  title={Geo-ORBIT: A Federated Digital Twin Framework for Scene-Adaptive Lane Geometry Detection},
  author={Rei Tamaru, Pei Li, and Bin Ran},
  journal={[Journal Name]},
  year={2025}
}
``` -->