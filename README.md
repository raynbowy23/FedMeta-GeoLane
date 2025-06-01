# Geo-ORBIT: A Federated Digital Twin Framework for Scene-Adaptive Lane Geometry Detection

**Authors**: Rei Tamaru, Pei Li, Bin Ran  
**Affiliation**: University of Wisconsin–Madison

---

## 🧠 Overview

**Geo-ORBIT** (Geometrical Operational Roadway Blueprint with Integrated Twin) is a unified framework that integrates real-time lane detection, federated learning, and digital twin synchronization. It is designed to support active traffic management, infrastructure monitoring, and real-time scenario testing without relying on centralized data collection.

At the core of Geo-ORBIT is **FedMeta-GeoLane**, a federated meta-learning-based lane detection model that adapts to scene-specific geometry using only vehicle trajectory data. By preserving privacy and reducing bandwidth, this system enables scalable deployment across diverse roadside camera environments.

---

## 🏗️ System Architecture

Geo-ORBIT is composed of three modular and interconnected processes:

- **🕵️ Detection Process**  
  Roadside cameras capture traffic video, from which vehicle trajectories are extracted and projected to GPS space.

- **🧠 Service Process**  
  The **FedMeta-GeoLane** model infers lane geometries from trajectories using adaptive parameters, refined through meta-learning and weak supervision (e.g., OpenStreetMap).

- **🧪 Simulation Process**  
  Detected lanes are synchronized with **SUMO** and **CARLA** to create a high-fidelity, real-time **Digital Twin** that supports traffic flow rendering and scenario replay.

---

## 🔍 FedMeta-GeoLane: Federated Meta-Learning Lane Detection

FedMeta-GeoLane treats each roadside camera deployment as a unique task. A shared meta-learner predicts optimal detection parameters using context features like vehicle speed and trajectory distribution. Key highlights include:

- **Black-box meta-learning**: No need for gradient flow through detection pipeline  
- **Federated optimization**: Local training with privacy-preserving aggregation  
- **Scene adaptation**: Immediate configuration for unseen locations

📈 _Compared to baseline and centralized models, FedMeta-GeoLane reduces geometric error by over 50% in unseen locations while achieving a 98% reduction in communication cost._

---

## 📊 Performance Summary

| Model            | Geometry Loss (m) | Centerline Error (m) | Lane Count Error | Total Loss (Unseen) |
|------------------|-------------------|-----------------------|------------------|---------------------|
| Baseline         | 15.12              | 6.78                  | 5.00             | 77.84               |
| Meta-GeoLane     | 105.35             | 34.60                 | 12.00            | 69.61               |
| **FedMeta-GeoLane** | **12.82**         | **21.39**             | 12.00            | **32.38**           |

---

## 🌐 Digital Twin Integration

Geo-ORBIT connects real-world observations to virtual testbeds using a synchronized SUMO–CARLA pipeline:

- 📍 GPS-aligned trajectories enable accurate replay in simulation  
- 🚦 Supports scene-level validation, vehicle re-routing, and visual analytics  
- 🧩 Extendable to multi-scenario environments with dynamic overlays (e.g., vegetation, accidents, road closures)

---

## 📂 Repository Contents (coming soon)

- `geo_orbit/` – Main framework scripts for detection, simulation, and synchronization  
- `fedmeta_geolane/` – Federated meta-learning modules and parameter optimization  
- `configs/` – Task-specific scene and camera configuration  
- `data/` – Example trajectory datasets and calibration files  
- `eval/` – Metric computation: trajectory discrepancy, lane alignment, and communication cost  
- `scripts/` – CLI tools for running detection, training, and visualization

---
