import cv2
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.set_loglevel(level = 'warning')
import logging
logger = logging.getLogger(__name__)

import sumolib
import seaborn as sns
import polars as pl

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from collections import Counter
from shapely.geometry import MultiPoint, Point, LineString

from pathlib import Path
import xml.etree.ElementTree as ET

from .utils import *


class OSMConnection:
    """
    Extract OSM data, form it in SUMO, validate points, and store it to new OSM as raw OSM has unnecesary data.
    """
    def __init__(self, args, saving_file_path):
        self.args = args
        self.is_save = args.is_save

        self.dataset_path = args.dataset_path
        self.filepath = saving_file_path
        self.sumo_filepath = args.osm_path
        self.sumo_netfile = 'osm.net.xml'
        self.sumo_savefile = 'out.net.xml'

        self.lane_class = 5
        self.latitude = 43.0356 # approximate

        self.colors = [
            (1.00, 0.00, 0.00),  # vivid red
            (0.00, 0.45, 0.74),  # strong blue
            (0.47, 0.67, 0.19),  # vivid green
            (0.93, 0.69, 0.13),  # strong orange
            (0.49, 0.18, 0.56),  # purple
            (0.00, 0.75, 0.75),  # cyan
            (0.85, 0.33, 0.10),  # deep orange
            (0.00, 0.50, 0.00),  # strong forest green
            (0.75, 0.00, 0.75),  # magenta
            (0.25, 0.25, 0.25),  # dark gray
            (0.13, 0.70, 0.67),  # teal
            (0.55, 0.71, 0.00),  # lime green
            (0.64, 0.08, 0.18),  # wine red
            (0.80, 0.40, 0.00),  # burnt orange
            (0.12, 0.47, 0.70),  # deep sky blue
            (0.58, 0.00, 0.83),  # violet
            (0.93, 0.17, 0.31),  # cherry red
            (0.20, 0.63, 0.79),  # bright turquoise
            (0.56, 0.93, 0.56),  # pastel green
            (1.00, 0.60, 0.00),  # vivid amber
        ]

    def get_net(self, camera_loc):
        tree = ET.parse(Path(self.sumo_filepath, camera_loc, self.sumo_netfile))
        root = tree.getroot()

        # Count the number of lanes in OSM
        # You have to check whther its on the same section or not. OR, you can calibrate with trajectory extraction results.
        # Get calibrated results
        lane_edge_geometries = {}
        grouped_lane_edge_geometries = {}
        lane_geometries = {}
        lane_shape = {}
        for i, edge in enumerate(root.findall('edge')):
            lane_id_list = []
            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                shape = lane.get('shape')
                # length = lane.get('length')
                length = self.compute_polyline_length(shape, camera_loc)
                # TODO: Update the width of the lane by lane type, default is 3.2m
                lane_shape[lane_id] = (length, 3.2)
                # Convert shape to list of coordinates
                points = [tuple(map(np.float64, p.split(','))) for p in shape.split()]
                lane_edge_geometries[lane_id] = points
                lane_id_list.append(lane_id)
            grouped_lane_edge_geometries[i] = lane_id_list

        for junction in root.findall('junction'):
            junction_id = junction.get('id')
            x = np.float64(junction.get('x'))
            y = np.float64(junction.get('y'))
            # Convert shape to list of coordinates
            # points = [tuple(map(float, p.split(','))) for p in shape.split()]
            lane_geometries[junction_id] = ('junction', (x, y))
            lane_id_list.append(junction_id)

        return lane_geometries, lane_edge_geometries, grouped_lane_edge_geometries, lane_id_list, lane_shape
    
    def compute_polyline_length(self, shape_str, camera_loc):
        """Return the lane length complying with the GPS coordinates
        """
        # Parse shape string into Nx2 array
        points = np.array([
            [float(x), float(y)]
            for pair in shape_str.strip().split()
            for x, y in [pair.split(",")]
        ])
        points = self.sumo_to_global(points, camera_loc)
        points_cartesian = gps_to_cartesian(torch.tensor(points))
        # Compute segment lengths and sum
        deltas = np.diff(points_cartesian, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        return segment_lengths.sum()

    def global_to_sumo(self, nodes, lane_geometries, lane_id_list):
        """
        Transform GPS coordinates to SUMO coordinates using homography transformation.

        Args:
            nodes (_type_): _description_
            lane_geometries (_type_): _description_
            lane_id_list (_type_): _description_

        Returns:
            hom: _description_
            sumo_coordinate: _description_
        """
        sumo_points_list = []
        gps_points_list = []

        # SUMO nodes (GPS)
        for node_id, (y, x) in nodes.items():
            gps_points = [x, y]
            gps_points_list.append(gps_points)

        # SUMO nodes (sumo coordinates)
        for node_id, lanes in lane_geometries.items():
            for lane_id in lane_id_list:
                if len(node_id.split('_')) > 1:
                    node_id = node_id.split('_')[0]
                    if len(node_id.split(':')) > 1:
                        node_id = node_id.split(':')[1]
                if len(lane_id.split('#')) > 1:
                    lane_id = lane_id.split('#')[0]

                if lane_id == node_id:
                    sumo_x, sumo_y = lanes[1]
                    sumo_points_list.append([sumo_x, sumo_y])
            
        gps_points_array = np.array(gps_points_list, dtype=np.float64)
        sumo_points_array = np.round(np.array(sumo_points_list, dtype=np.float64), 6)
        np.set_printoptions(precision=6, suppress=True)
        logger.debug(f'GPS POINTS LIST: {gps_points_array}')
        logger.debug(f'SUMO POINTS LIST: {sumo_points_array}')

        hom, _ = cv2.findHomography(gps_points_array, sumo_points_array, cv2.RANSAC, 5)
        sumo_coordinate = apply_homography_transform(gps_points_array, hom)

        return hom, sumo_coordinate

    def gps_to_sumo(self, gps_points, camera_loc) -> np.array:
        """
        Convert GPS coordinates to SUMO coordinates using the inverse homography transformation.

        Args:
            net (sumolib.net.Net): SUMO network object
            sumo_points (list or np.array): List of SUMO coordinates [[x, y], [x, y], ...]

        Returns:
            gps_coordinate (np.array): Transformed GPS coordinates
        """
        net = sumolib.net.readNet(Path(self.sumo_filepath, camera_loc, self.sumo_netfile))
        lane_points = np.array([
            net.convertLonLat2XY(lon, lat)
            for lat, lon in zip(gps_points[:, 0], gps_points[:, 1])
        ])

        return lane_points

    def sumo_to_global(
            self,
            sumo_points,
            camera_loc
        ) -> np.array:
        """
        Convert SUMO coordinates back to GPS coordinates using the inverse homography transformation.

        Args:
            net (sumolib.net.Net): SUMO network object
            sumo_points (list or np.array): List of SUMO coordinates [[x, y], [x, y], ...]

        Returns:
            gps_coordinate (np.array): Transformed GPS coordinates
        """
        sumo_points = np.array(sumo_points)
        net = sumolib.net.readNet(Path(self.sumo_filepath, camera_loc, self.sumo_netfile))
        lane_points = np.array([
            net.convertXY2LonLat(lat, lon)[::-1]
            for lat, lon in zip(sumo_points[:, 0], sumo_points[:, 1])
        ])

        return lane_points

    def global_to_pixel(self, global_points, hom):
        """
        Convert GPS coordinates to pixel coordinates using inverse homography.

        Args:
            global_points (list or np.array): List of (latitude, longitude) GPS coordinates.
            hom (np.array): 3x3 homography matrix (Pixel → GPS).

        Returns:
            np.array: Transformed pixel coordinates (x, y).
        """
        global_points = np.array(global_points, dtype=np.float64)

        hom_inv = np.linalg.pinv(hom)
        pixel_points = apply_homography_transform(global_points, hom_inv)

        return pixel_points

    def pixel_to_sumo(self, pixel_df, sumo_hom, root_path, camera_loc):
        """Wrapper function of trajectory_calibration"""
        pixel_points = pixel_df.select(pl.col('x', 'y')).to_numpy().astype(np.float64)
        calb_points, pixel_hom, pixel_min, pixel_max = trajectory_calibration(pixel_points, root_path, camera_loc)
        sumo_points = apply_homography_transform(calb_points, sumo_hom)
        pixel_df = pixel_df.with_columns(
            pl.Series("x_sumo", sumo_points[:, 0]),
            pl.Series("y_sumo", sumo_points[:, 1])
        )

        return pixel_df, pixel_hom, pixel_min, pixel_max

    def visualize_gps_network(self, camera_loc, traj_df, detected_cnts):
        """
        Visualizes trajectory points in GPS coordinates alongside the transformed SUMO network.
        
        Parameters:
        trajectory_points (np.array): Array of trajectory points in pixel coordinates
        """

        root_filepath = Path(self.dataset_path, '511calibration')
        pixel_filepath = Path(self.filepath, camera_loc)

        plt.figure(figsize=(10, 20))

        THRESHOLD_DISTANCE = 10 # meters

        # Get SUMO data using get_net()
        lane_geometries, lane_edge_geometries, grouped_lane_edge_geometries, lane_id_list, lane_shape = self.get_net(camera_loc)

        gps_df = traj_df.with_columns(
            pl.Series("x_gps", np.full(traj_df.shape[0], np.nan)),
            pl.Series("y_gps", np.full(traj_df.shape[0], np.nan)),
            pl.Series("lane_id", np.full(traj_df.shape[0], -1)) # For later use to assign sumo lane ids for each trajectory point
        )
        # if "pseudo_lane_id" not in gps_df.columns:
        #     gps_df = gps_df.with_columns(pl.Series("pseudo_lane_id", np.full(gps_df.shape[0], "-1", dtype=str))) # Initialize with "-1"

        # Transform SUMO lane geometries to pixel coordinates
        gps_lane_geometries = {}
        all_sumo_points = []
        sumo_lane_ids = []

        for group_id, lane_ids in grouped_lane_edge_geometries.items():
            for lane_id in lane_ids:
                if lane_id in lane_edge_geometries:
                    # Get SUMO points
                    sumo_points = np.array(lane_edge_geometries[lane_id])
                    interpolated_sumo_points = interpolate_edge(sumo_points, num_points=30)
                    interpolated_sumo_points = np.array(interpolated_sumo_points, dtype="float64")

                    # Convert to global (GPS) coordinates
                    gps_points = self.sumo_to_global(interpolated_sumo_points, camera_loc)
                    
                    # Store pixel coordinates
                    gps_lane_geometries[lane_id] = gps_points

                    all_sumo_points.extend(gps_points)
                    sumo_lane_ids.extend([lane_id] * len(gps_points))

        all_sumo_points = np.array(all_sumo_points)
        sumo_lane_ids = np.array(sumo_lane_ids)
        sumo_kdtree = cKDTree(all_sumo_points)

        # Identify each lane group and create pseudo_lane_ids for each groups [0, 1, 2, 3] for group 0.
        # Find the closest lane groups from the trajectory points and create target area
        sumo_kdtree = cKDTree(all_sumo_points) # all_sumo_points is (N,2) array of SUMO lane points
        trajectory_points = gps_df.select(pl.col('x', 'y')).to_numpy().astype(np.float64)
        x_gps = gps_df["x_gps"].to_numpy().copy()
        y_gps = gps_df["y_gps"].to_numpy().copy()

        # Dictionary to store the lane groups and their associated lane points
        lane_group_dict = {}
        for lane_group, (cnt, color) in enumerate(zip(detected_cnts, self.colors)):
            # color = tuple(float(c) for c in color)
            # Convert contour coordinates to GPS coordinates
            mask = np.array([cv2.pointPolygonTest(cnt, (pos[0], pos[1]), False) > 0 for pos in trajectory_points])
            masked_positions = trajectory_points[mask]

            if masked_positions.shape[0] > 0:
                gps_traj_points, pixel_hom = trajectory_calibration(masked_positions, root_filepath, camera_loc)

                x_gps[mask] = gps_traj_points[:, 0]
                y_gps[mask] = gps_traj_points[:, 1]

                distances, nearest_idx = sumo_kdtree.query(gps_traj_points) # masked_positions = trajectory points inside detected contours
                distances = distance_conversion(distances, self.latitude) # GPS to meters

                closest_lane_ids = sumo_lane_ids[nearest_idx]

                assigned_lanes = np.where(distances < THRESHOLD_DISTANCE, closest_lane_ids, "-1")

                valid_indices = distances < THRESHOLD_DISTANCE
                filtered_lane_ids = closest_lane_ids[valid_indices]

                target_lane_id_full = np.full(len(gps_df), "-1", dtype=object)
                target_lane_id_full[mask] = assigned_lanes

                gps_df = gps_df.with_columns(
                    pl.Series("x_gps", x_gps),
                    pl.Series("y_gps", y_gps),
                    pl.Series("target_lane_id", target_lane_id_full)
                )

                closest_lane_ids = sumo_lane_ids[nearest_idx]

                valid_indices = distances < THRESHOLD_DISTANCE
                filtered_lane_ids = closest_lane_ids[valid_indices]

                lane_counts = Counter(filtered_lane_ids)

                min_points_threshold = 30

                if len(lane_counts) > 0:
                    target_lanes = {
                        lane if count >= min_points_threshold else np.str_(-1)
                        for lane, count in lane_counts.items()
                    }
                else:
                    target_lanes = np.str_(-1)

                # Ensure lane_group exists in the dictionary
                if lane_group not in lane_group_dict:
                    lane_group_dict[lane_group] = {}

                # Filter out any invalid extrapolation results
                valid_points = ~np.isnan(gps_traj_points).any(axis=1) & ~np.isinf(gps_traj_points).any(axis=1)
                gps_traj_points = gps_traj_points[valid_points]

                # Store lane points in a structured dictionary
                for lane_id, points in gps_lane_geometries.items():
                    if target_lanes == "-1":
                        lane_group_dict[lane_group] = {-1: np.zeros((10, 2))}
                    if lane_id in target_lanes: # Only keep lanes with enough nearby trajectory points

                        # We want to regulate points to only proximity, otherwise SUMO covers larger area so we cannot compare fairly
                        # points are SUMO, gps_traj_points are GPS

                        # Convert to shapely geometries
                        pts = gps_lane_geometries[lane_id]

                        lane_line = LineString(pts)
                        traj_multipoint = MultiPoint(gps_traj_points)

                        # Get bounding box coordinates of the trajectory points
                        minx, miny, maxx, maxy = traj_multipoint.bounds

                        # Create Points for bottom-left and top-right of the bounding box
                        bottom_left = Point(minx, miny)
                        top_right = Point(maxx, maxy)

                        # Project these bounding box points onto the SUMO lane line
                        start_proj = lane_line.project(bottom_left)
                        end_proj = lane_line.project(top_right)

                        # Interpolate to get the actual points on the lane
                        point_start = lane_line.interpolate(start_proj)
                        point_end = lane_line.interpolate(end_proj)
                        segment = LineString([point_start, point_end])

                        interpolated_points = [segment.interpolate(i / 29, normalized=True) for i in range(30)]
                        interpolated_coords = np.array([[pt.x, pt.y] for pt in interpolated_points])

                        x_vals = [pt.x for pt in interpolated_points]
                        y_vals = [pt.y for pt in interpolated_points]

                        # Plot the interpolated segment
                        plt.plot(x_vals, y_vals, 'o-', color=color, alpha=0.7, linewidth=2,
                                label=f'Interpolated Lane {lane_id}' if lane_id == lane_ids[0] else "")

                        if lane_id not in lane_group_dict[lane_group]:
                            # lane_group_dict[lane_group][lane_id] = []
                            lane_group_dict[lane_group][lane_id] = interpolated_coords


                if self.is_save:
                    try:
                        plt.scatter(
                            gps_traj_points[:, 0],
                            gps_traj_points[:, 1],
                            s=5, alpha=0.7, color=color,
                            label=f'Lane {lane_id}'
                        )
                    except:
                        print("Got some error")
                        print(color)
            
                # Print the selected target lanes
                logger.info(f"Target lanes selected based on trajectory density for lane group {lane_group}: {target_lanes}")

        sumo_node_matrix, sumo_edge_matrix = self.create_sumo_graph(lane_group_dict)

        if self.is_save:
            # Add annotations for lane IDs
            for lane_id, points in gps_lane_geometries.items():
                if len(points) > 0:
                    midpoint = points[len(points) // 2] # Use middle point for label
                    plt.annotate(f"{lane_id}", xy=(midpoint[0], midpoint[1]), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                                                    fc="white", ec="gray", alpha=0.7))
            
            # Customize plot
            plt.title('Trajectory Points and SUMO Network in Pixel Coordinates')
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            
            # Create legend with unique entries only
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
            
            # Save figure if required
            if self.args.is_save:
                plt.savefig(Path(pixel_filepath, 'gps_trajectory_visualization.png'), dpi=300)
            
            # plt.show()
            plt.close('all')
        return gps_df, lane_group_dict, pixel_hom, (sumo_node_matrix, sumo_edge_matrix)

    def create_sumo_graph(self, lane_group_dict):

        sumo_group_node_list = []
        sumo_group_edge_list = []
        for lane_group in lane_group_dict.values():
            sumo_node_list = []
            sumo_edge_list = []
            node_idx = {}
            for lane_id, lane_points in lane_group.items():
                idx = 0
                sumo_nodes = []
                sumo_edges = []
                prev_idx = None
                for point in lane_points:
                    pt = tuple(point.tolist())
                    if pt not in node_idx:
                        node_idx[tuple(point)] = idx
                        sumo_nodes.append(point)
                        current_idx = idx
                        idx += 1
                    else:
                        current_idx = node_idx[pt]
                        continue

                    if prev_idx is not None:
                        sumo_edges.append((prev_idx, node_idx[tuple(point)])) 

                    prev_idx = current_idx
                sumo_node_list.append(sumo_nodes)
                sumo_edge_list.append(sumo_edges)
            sumo_group_node_list.append(sumo_node_list)
            sumo_group_edge_list.append(sumo_edge_list)

        return sumo_group_node_list, sumo_group_edge_list # Inhomogenous

    def get_lane_id_sumo(self, lane_edge_geometries, grouped_lane_edge_geometries):
        '''
        Return:
            Number of lane, lane id, road cluster

        '''
        threshold = 50 # Distance threshold for closeness
        theta = 0.2 # Moving distance at each iteration

        sumo_edge_points = []
        lane_list = []

        for lane_id, points in lane_edge_geometries.items():
            # for l in lane_id:
            #     lane_list.append(l.split('_')[0])
            # Convert the dictionary to a DataFrame for better display
            lane_list.append(lane_id)

        processed_data = []
        points_list = []
        for lane_id, points in lane_edge_geometries.items():
            for gid, glane_id in enumerate(grouped_lane_edge_geometries):
                if lane_id in grouped_lane_edge_geometries[glane_id]:
                    lid = gid
            # Split on underscore and isolate the ID and lane number
            parts = lane_id.split('_')
            main_id = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            lane_number = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
            processed_data.append((lid, main_id, lane_number))
            points_list.append(points)
        # Convert processed data to Polars DataFrame
        df = pl.DataFrame(
            processed_data,
            schema=["Group ID", "ID", "Lane"],
            orient="row"
        )
        df = df.with_columns(
            pl.Series(name="Coordinates", values=points_list)
        )

        # Group by ID and calculate the number of unique lanes
        # lane_count_df = df.groupby("ID").agg(pl.col("Lane").n_unique().alias("Lane Count"), pl.col("Coordinates").first())
        df = df.with_columns(
            pl.arange(0, df.height).alias("index")
        )

        return df

    def get_sumo_centers(self, lane_group_dict):
        """ Get SUMO centers for lane similarity

        Args:
            detected_points [List(Set)]: detected center points
        """
        lane_id_mapping = {}

        for lg_key, lane_group in lane_group_dict.items():

            lane_id_mapping[lg_key] = {}

            for lane_class in range(self.lane_class):
                if int(lane_class) not in lane_id_mapping:
                    lane_id_mapping[lg_key][lane_class] = []

                lane_list = list(lane_group.keys())

                # Get points belonging to this lane class
                if lane_class < len(lane_list):
                    lane_id_mapping[lg_key][lane_class] = lane_group[lane_list[lane_class]]
                else:
                    lane_id_mapping[lg_key][lane_class] = np.zeros((10, 2))

        return lane_id_mapping

    def extract_osm(self, detected_points, sumo_df, camera_loc):
        """Detected points are center points detected from LaneContinousLearning

        Args:
            detected_points [List(Set)]: detected center points (not lines)
            sumo_lane_geometries [Dict]: lane geometries from SUMO
            sumo_df [DataFrame]: lane id dataframe from SUMO
            camera_loc [str]: camera location
        """
        detected_points = self.gps_to_sumo(detected_points, camera_loc)

        sumo_edges = [np.array(coord) for coord in sumo_df.get_column('Coordinates')]
        group_id = sumo_df.get_column('Group ID')

        # Uniform interpolation (e.g., 100 points per edge)
        interpolated_coords = [interpolate_edge(edge, num_points=30) for edge in sumo_edges]
        interpolated_coords = np.array(interpolated_coords, dtype="float64")

        # print("interpolated_edges", interpolated_edges)
        group_dict = {g: sumo_df.filter(sumo_df["Group ID"] == g)["index"].to_list() for g in group_id.unique()}

        # Cluster red points using DBSCAN
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # Detected points should be differed from each other
        dbscan = DBSCAN(eps=2, min_samples=1)
        detected_clusters = dbscan.fit_predict(detected_points)

        # Get the cluster to edge assignments
        logger.info(f"Detected clusters: {np.unique(detected_clusters)}")
        cluster_to_edge_map = assign_cluster_to_edges(detected_points, detected_clusters, interpolated_coords, group_dict)
        logger.info(f"Cluster to edge mapping: {cluster_to_edge_map}")

        # Visualization
        plt.figure(figsize=(24, 6))

        colors = sns.color_palette("husl", 120)

        fig, ax = plt.subplots()
        if self.args.is_save:
            # Plot blue edges
            for i, edge in enumerate(interpolated_coords):
                ax.plot(edge[:, 0], edge[:, 1], label=f'Edge {i}', linestyle='--', color=colors[i])
                ax.scatter(edge[:, 0], edge[:, 1], label=f'Edge {i}',color='b')
                ax.text(edge[0, 0], edge[1, 1], f'Edge {i}')

            # Plot detected points and edges
            for i, cluster_id in enumerate(np.unique(detected_clusters)):
                cluster_points = detected_points[detected_clusters == cluster_id]
                if cluster_id == -1:
                    label = 'Noise'
                    color = 'gray'
                else:
                    label = f'Cluster {cluster_id} -> Edge {cluster_to_edge_map.get(cluster_id, "N/A")[1]}'
                    color = plt.cm.tab10(cluster_id / 10.0)
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=label, color=colors[cluster_to_edge_map.get(cluster_id, "N/A")[1]])
                ax.text(cluster_points[0, 0], cluster_points[0, 1], label)

            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Cluster Detected Points and Assign to Edges')
            fig.tight_layout()
            fig.savefig(Path(self.filepath, camera_loc, 'sumo', 'cluster_detected_points.png'))
            plt.close(fig)

        return interpolated_coords, cluster_to_edge_map

    def get_sumo_data(self, detected_centers, camera_loc, trial):
        """Validate the lane information extracted from the OSM data.
        - Compare and adjust center lane points in SUMO. -> If loss is small enough, adjust the points as we think they are trustable
        - Take care of missing lanes in SUMO. -> If the detected lane is not in SUMO, add the lane to SUMO for any loss

        Also, convert pixel coordinates to SUMO coordinates of some variables to compute loss
        """
        logger.info(f"Extract lane from OSM at {camera_loc}...")
        self.trial = trial

        lane_geometries, lane_edge_geometries, grouped_lane_edge_geometries, lane_id_list, lane_shape = self.get_net(camera_loc)
        
        sumo_df = self.get_lane_id_sumo(lane_edge_geometries, grouped_lane_edge_geometries)
        sumo_points, cluster_to_edge_map = self.extract_osm(
            detected_centers, sumo_df, camera_loc
        )
        sumo_gps_points = [self.sumo_to_global(sp, camera_loc) for sp in sumo_points]
        return sumo_gps_points, cluster_to_edge_map, lane_shape

def main():
    parser = argparse.ArgumentParser(description='Extract OSM data')
    parser.add_argument('--osm_save_date', type=str, default="2024-12-04-23-05-50", help='The date of the OSM file to extract data from')
    args = parser.parse_args()

    # with gzip.open(Path(args.osm_save_date, 'osm.net.xml.gz'), 'rb') as f:
    #     file_content = f.read()
        # with open(Path(args.osm_save_date, 'osm.net.xml'), 'wb') as f:
        #     f.write(file_content)

    tree = ET.parse(Path(args.osm_save_date, 'osm.net.xml'))
    root = tree.getroot()

    # Count the number of lanes in OSM
    # You have to check whther its on the same section or not. OR, you can calibrate with trajectory extraction results.
    # Get calibrated results
    lane_geometries = {}
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            shape = lane.get('shape')
            # Convert shape to list of coordinates
            points = [tuple(map(float, p.split(','))) for p in shape.split()]
            lane_geometries[lane_id] = points

    # print(lane_geometries)

if __name__ == '__main__':
    main()

