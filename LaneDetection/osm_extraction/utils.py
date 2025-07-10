import cv2
import math
import numpy as np
import polars as pl
import torch

from shapely.geometry import LineString, MultiPoint, Point
from pathlib import Path
from scipy.interpolate import splprep, splev

def trajectory_calibration(points, root_path, camera_loc):
    """
    Convert pixel coordinates to GPS coordinates using homography transformation.
    As the data point from OSM will not be changed, we can make homography matrix for each lane.

    Args:
        points (list or np.array): List of (x, y) pixel coordinates.

    Returns:
        np.array: Transformed GPS coordinates (latitude, longitude).
        np.array: Homography matrix.
    """
    # Convert input points to numpy format
    points = np.array(points, dtype=np.float64)

    # Get GPS and Pixel calibration points
    gps_points, pixel_points = pixel_to_global(root_path, camera_loc)

    hom, _ = cv2.findHomography(pixel_points, gps_points, cv2.RANSAC)

    gps_points = apply_homography_transform(points, hom)
    return gps_points, hom


def apply_homography_transform(points, hom):
    """
    Apply homography transformation with high precision (float64) manually.

    Args:
        points (np.array): Nx2 array of coordinates (lat, lon or x, y)
        hom (np.array): 3x3 homography matrix

    Returns:
        np.array: Transformed coordinates
    """
    # Convert to homogeneous coordinates (x, y, 1)
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])

    # Apply transformation manually
    transformed_points = hom @ homogeneous_points.T
    transformed_points /= transformed_points[2] # Normalize with last row

    return transformed_points[:2].T # Extract transformed (x, y)

def pixel_to_global(root_path, camera_loc):

    # Get gps, pixel points from text file in each camera location

    # Load as (pixel_x, pixel_y, lat, lon)
    csv_path = Path(root_path) / f"{camera_loc}.csv"
    df = pl.read_csv(csv_path)

    # Extract arrays
    pixel_points = df.select(["pixel_x", "pixel_y"]).to_numpy().astype(np.float64)
    gps_points = df.select(["latitude", "longitude"]).to_numpy().astype(np.float64)
    # gps_points = df.select(["longitude", "latitude"]).to_numpy().astype(np.float64)

    return gps_points, pixel_points

# Interpolate edges
def interpolate_edge(edge_points, num_points=100):
    """
    Interpolate an edge defined by discrete nodes using spline interpolation.

    Args:
        edge_points (np.ndarray): Array of shape (N, 2) representing the edge nodes.
        num_points (int): Number of interpolated points to generate.

    Returns:
        np.ndarray: Interpolated points of shape (num_points, 2).
    """
    edge_points = np.array([np.array(p, dtype=np.float64) for p in edge_points], dtype=np.float64)
        
    if len(edge_points) < 2:
        return edge_points

    # Adjust the spline degree for small edge point sets
    k = min(3, len(edge_points) - 1)

    # Calculate cumulative arc length
    distances = np.linalg.norm(np.diff(edge_points, axis=0), axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Normalize cumulative distances
    normalized_param = cumulative_distances / cumulative_distances[-1]

    # Spline parameterization
    tck, u = splprep(edge_points.T, u=normalized_param, s=0, k=k)  # `s=0` ensures the spline passes through all points
    u_fine = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_fine, tck)
    return np.vstack((x_new, y_new)).T

# Step 2: Assign clusters to edges
def assign_cluster_to_edges(detected_points, detected_clusters, sumo_edges, group_dict):
    _sumo_edges = sumo_edges.copy()
    cluster_to_edge_map = {}
    unique_clusters = set(detected_clusters)
    assigned_edge_list = []
    last_edge_index = None

    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1: # Ignore noise points
            continue

        # Get points in this cluster
        cluster_points = detected_points[detected_clusters == cluster_id]

        # Calculate the average distance from this cluster to each edge
        min_distance = float('inf')
        assigned_edge = None
        for edge_index, edge_points in enumerate(_sumo_edges):
            # group_id = items[edge_index][0]
            if edge_index in assigned_edge_list:
                continue
            distances = np.linalg.norm(cluster_points[:, None] - edge_points, axis=2).mean(axis=0)
            # distances = np.linalg.norm(edge_points - detected_points, axis=2).mean(axis=0)
            avg_distance = np.min(distances)

            if avg_distance < min_distance:
                min_distance = avg_distance
                assigned_edge = edge_index
                last_edge_index = edge_index
                for gid, lanes in group_dict.items():
                    if edge_index in lanes:
                        group_id = gid
                        break
        
        if assigned_edge is None:
            assigned_edge = last_edge_index
        # _sumo_edges.pop(assigned_edge) # Not gonna assign same edge
        assigned_edge_list.append(assigned_edge)
        cluster_to_edge_map[int(cluster_id)] = (group_id, assigned_edge)
        
    return cluster_to_edge_map

def distance_conversion(distances_in_degrees, latitude):
    # Conversion factors
    meters_per_degree_lat = 111132 # 1 degree latitude ≈ 111,132 meters
    meters_per_degree_lon = 111320 * np.cos(np.radians(latitude)) # Longitude varies with latitude

    # Convert degrees to meters
    meters_per_degree = np.sqrt(meters_per_degree_lat**2 + meters_per_degree_lon**2)

    return distances_in_degrees * meters_per_degree

def gps_point_distance_m(lat1, lon1, lat2, lon2):
    avg_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111_320 * math.cos(avg_lat)
    dy = (lat2 - lat1) * 111_320
    return math.sqrt(dx**2 + dy**2)

def compute_lane_width_from_gps(left, right):
    """
    left, right: np.ndarray of shape (N, 2), where columns are [lon, lat]
    """
    assert left.shape == right.shape
    widths = []
    for (lat1, lon1), (lat2, lon2) in zip(left, right):
        width = gps_point_distance_m(lat1, lon1, lat2, lon2)
        widths.append(width)
    widths = np.array(widths)
    return widths, widths.mean()

def meters_to_gps_offset(meters, ref_lat):
    lat_scale = 1 / 111_320
    lon_scale = 1 / (111_320 * math.cos(math.radians(ref_lat)))
    delta_lat = meters * lat_scale
    delta_lon = meters * lon_scale
    return delta_lat, delta_lon


def interpolate_lane_segment_to_detected_area(
    sumo_lane_coords,  # (N, 2)
    detected_gps_coords,  # (M, 2) in lat/lon
    gps_to_sumo_func,  # callable: gps_to_sumo(gps_points, camera_loc)
    camera_loc,
    num_interp_points=30,
):
    """
    Trims a SUMO lane line to the region near detected points and interpolates evenly.

    Returns:
        interpolated_coords: (num_interp_points, 2)
    """
    # Convert detected GPS points to SUMO coordinates
    detected_sumo_coords = gps_to_sumo_func(detected_gps_coords, camera_loc)

    # Skip if not enough points
    if len(detected_sumo_coords) < 2:
        return None

    # Convert to shapely geometries
    lane_line = LineString(sumo_lane_coords)
    traj_multipoint = MultiPoint(detected_sumo_coords)

    # Get bounding box of detections
    minx, miny, maxx, maxy = traj_multipoint.bounds
    bottom_left = Point(minx, miny)
    top_right = Point(maxx, maxy)

    # Project start and end of detection bounding box onto lane
    start_proj = lane_line.project(bottom_left)
    end_proj = lane_line.project(top_right)

    # Ensure order (start < end)
    if start_proj > end_proj:
        start_proj, end_proj = end_proj, start_proj

    # Create trimmed segment
    segment = lane_line.segmentize(start_proj, end_proj)
    segment = LineString([lane_line.interpolate(start_proj), lane_line.interpolate(end_proj)])

    # Interpolate fixed number of points
    interpolated_points = [segment.interpolate(i / (num_interp_points - 1), normalized=True) for i in range(num_interp_points)]
    interpolated_coords = np.array([[pt.x, pt.y] for pt in interpolated_points])

    return interpolated_coords

def gps_to_cartesian(gps_tensor, origin=None):
    """
    Converts GPS [lon, lat] to Cartesian x, y in meters using equirectangular projection.

    Args:
        gps_tensor: torch.Tensor of shape (N, 2) with [lon, lat]
        origin: [lon0, lat0] to use as projection origin. If None, use first point.

    Returns:
        xy_tensor: torch.Tensor of shape (N, 2), units in meters
    """
    lon = gps_tensor[:, 0]
    lat = gps_tensor[:, 1]

    if origin is None:
        lon0 = lon[0]
        lat0 = lat[0]
    else:
        lon0, lat0 = origin

    lat_avg = math.radians((lat0 + lat.mean().item()) / 2)

    # Convert to Cartesian meters
    dx = (lon - lon0) * 111_320 * math.cos(lat_avg)
    dy = (lat - lat0) * 111_320

    return torch.stack([dx, dy], dim=1)
