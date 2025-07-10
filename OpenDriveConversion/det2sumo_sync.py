import traci
import cv2
import numpy as np
import sumolib
import argparse
import polars as pl
from scipy.interpolate import interp1d
import pandas as pd
from pathlib import Path


class TrajectoryPreprocessor:
    """Class to preprocess detected vehicle trajectories to real ones."""
    def __init__(self, net_file_path):
        self.net = sumolib.net.readNet(net_file_path)
        
    def interpolate_trajectory(self, vehicle_df, target_fps=30, method='linear'):
        """
        Interpolate trajectory to achieve consistent time sampling
        
        Args:
            vehicle_df: DataFrame for single vehicle with columns [time, x_sumo, y_sumo]
            target_fps: Target frames per second for interpolation
            method: 'linear', 'cubic', or 'spline'
        """
        if len(vehicle_df) < 2:
            return vehicle_df
            
        # Sort by time
        vehicle_df = vehicle_df.sort('time')
        
        # Get time range and create uniform time grid
        time_min = vehicle_df['time'].min()
        time_max = vehicle_df['time'].max()
        target_dt = 1.0 / target_fps
        
        # Create new time points
        new_times = np.arange(time_min, time_max + target_dt, target_dt)
        
        # Convert to numpy for interpolation
        original_times = vehicle_df['time'].to_numpy()
        x_coords = vehicle_df['x_sumo'].to_numpy()
        y_coords = vehicle_df['y_sumo'].to_numpy()
        
        # Handle different interpolation methods
        if method == 'linear':
            f_x = interp1d(original_times, x_coords, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
            f_y = interp1d(original_times, y_coords, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            if len(original_times) >= 4:  # Cubic needs at least 4 points
                f_x = interp1d(original_times, x_coords, kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
                f_y = interp1d(original_times, y_coords, kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
            else:
                # Fall back to linear if not enough points
                f_x = interp1d(original_times, x_coords, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
                f_y = interp1d(original_times, y_coords, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Interpolate
        new_x = f_x(new_times)
        new_y = f_y(new_times)
        
        # Create new DataFrame
        interpolated_df = pl.DataFrame({
            'time': new_times,
            'x_sumo': new_x,
            'y_sumo': new_y,
            'id': [vehicle_df['id'][0]] * len(new_times), # Keep same vehicle ID
            'interpolated': [True] * len(new_times)
        })
        
        return interpolated_df
    
    def smooth_trajectory_with_physics(self, vehicle_df, max_speed=50.0, max_acceleration=5.0):
        """
        Apply physics-based constraints to smooth unrealistic movements
        
        Args:
            vehicle_df: DataFrame for single vehicle
            max_speed: Maximum realistic speed (m/s)
            max_acceleration: Maximum realistic acceleration (m/s²)
        """
        if len(vehicle_df) < 2:
            return vehicle_df
            
        vehicle_df = vehicle_df.sort('time')
        
        # Convert to numpy
        times = vehicle_df['time'].to_numpy()
        x_coords = vehicle_df['x_sumo'].to_numpy()
        y_coords = vehicle_df['y_sumo'].to_numpy()
        
        # Calculate velocities and accelerations
        dt = np.diff(times)
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        
        # Speed at each point
        speeds = np.sqrt(dx**2 + dy**2) / dt
        
        # Find unrealistic speed jumps
        valid_indices = [0] # Always keep first point
        
        for i in range(1, len(speeds) + 1):
            if i == len(speeds): # Last point
                valid_indices.append(i)
            elif speeds[i-1] <= max_speed:
                # Check acceleration if we have previous valid point
                if len(valid_indices) > 1:
                    prev_idx = valid_indices[-1]
                    time_diff = times[i] - times[prev_idx]
                    if time_diff > 0:
                        prev_speed = speeds[prev_idx-1] if prev_idx > 0 else 0
                        acceleration = abs(speeds[i-1] - prev_speed) / time_diff
                        if acceleration <= max_acceleration:
                            valid_indices.append(i)
                else:
                    valid_indices.append(i)
        
        # Filter to valid points
        filtered_df = vehicle_df[valid_indices]
        return filtered_df
    
    def validate_lane_consistency(self, vehicle_df, max_lane_jump_distance=20.0):
        """
        Remove trajectory segments where vehicle jumps to unrealistic lanes
        
        Args:
            vehicle_df: DataFrame for single vehicle with edge assignments
            max_lane_jump_distance: Maximum realistic distance for lane changes (meters)
        """
        if len(vehicle_df) < 2:
            return vehicle_df
            
        vehicle_df = vehicle_df.sort('time')
        valid_indices = [0] # Always keep first point
        
        for i in range(1, len(vehicle_df)):
            current_edge = vehicle_df['edgeID'][i]
            prev_edge = vehicle_df['edgeID'][valid_indices[-1]]
            
            if current_edge == prev_edge:
                # Same edge - always valid
                valid_indices.append(i)
            else:
                # Different edge - check if transition makes sense
                if self.is_valid_edge_transition(prev_edge, current_edge, vehicle_df, valid_indices[-1], i):
                    valid_indices.append(i)
        
        return vehicle_df[valid_indices]
    
    def is_valid_edge_transition(self, from_edge_id, to_edge_id, vehicle_df, from_idx, to_idx):
        """Check if transition between edges is realistic"""
        try:
            from_edge = self.net.getEdge(from_edge_id)
            to_edge = self.net.getEdge(to_edge_id)
            
            # Check if edges are directly connected
            outgoing_edges = [e.getID() for e in from_edge.getOutgoing()]
            if to_edge_id in outgoing_edges:
                return True
            
            # Check if edges are parallel (lane change scenario)
            from_pos = (vehicle_df['x_sumo'][from_idx], vehicle_df['y_sumo'][from_idx])
            to_pos = (vehicle_df['x_sumo'][to_idx], vehicle_df['y_sumo'][to_idx])
            
            distance = np.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)
            
            # If distance is small, might be parallel lanes
            if distance < 20.0: # 20 meters threshold
                return True
            
            # Check if there's a reasonable path between edges
            return self.has_reasonable_path(from_edge, to_edge, max_hops=3)
            
        except:
            return False
    
    def has_reasonable_path(self, from_edge, to_edge, max_hops=3):
        """Check if there's a path between edges within max_hops"""
        if max_hops <= 0:
            return False
            
        for outgoing in from_edge.getOutgoing():
            if outgoing == to_edge:
                return True
            if self.has_reasonable_path(outgoing, to_edge, max_hops - 1):
                return True
        
        return False
    
    def remove_outlier_trajectories(self, trajectory_df, min_points=5, max_speed_percentile=95):
        """
        Remove entire trajectories that seem unrealistic
        
        Args:
            trajectory_df: Full trajectory DataFrame
            min_points: Minimum points required for a valid trajectory
            max_speed_percentile: Remove trajectories with speeds above this percentile
        """
        # Group by vehicle ID
        vehicle_groups = trajectory_df.group_by('id')
        
        valid_trajectories = []
        
        for vehicle_id, vehicle_data in vehicle_groups:
            vehicle_df = vehicle_data.sort('time')
            
            # Check minimum points
            if len(vehicle_df) < min_points:
                print(f"Removing vehicle {vehicle_id}: too few points ({len(vehicle_df)})")
                continue
            
            # Check speed distribution
            if len(vehicle_df) > 1:
                times = vehicle_df['time'].to_numpy()
                x_coords = vehicle_df['x_sumo'].to_numpy()
                y_coords = vehicle_df['y_sumo'].to_numpy()
                
                dt = np.diff(times)
                dx = np.diff(x_coords)
                dy = np.diff(y_coords)
                
                speeds = np.sqrt(dx**2 + dy**2) / dt
                max_speed = np.percentile(speeds, max_speed_percentile)
                
                # Remove if consistently too fast (likely detection errors)
                if np.mean(speeds) > 30.0: # 30 m/s = 108 km/h average
                    print(f"Removing vehicle {vehicle_id}: unrealistic average speed ({np.mean(speeds):.2f} m/s)")
                    continue
            
            # Check trajectory length (too short might be noise)
            start_pos = (vehicle_df['x_sumo'][0], vehicle_df['y_sumo'][0])
            end_pos = (vehicle_df['x_sumo'][-1], vehicle_df['y_sumo'][-1])
            total_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            
            if total_distance < 10.0: # Less than 10 meters total movement
                print(f"Removing vehicle {vehicle_id}: too little movement ({total_distance:.2f}m)")
                continue
            
            valid_trajectories.append(vehicle_df)
        
        if valid_trajectories:
            return pl.concat(valid_trajectories)
        else:
            return pl.DataFrame() # Return empty DataFrame if no valid trajectories

class Det2SumoSync:
    def __init__(
            self,
            saving_file_path,
            camera_loc,
            dataset_path,
            csv_name="trajectory.csv"
        ):
        self.camera_loc = camera_loc
        # self.sumo_config = sumo_config
        self.saving_file_path = saving_file_path # Path to the directory where the SUMO files are saved
        self.sumo_filepath = Path(self.saving_file_path, camera_loc, "sumo")
        self.pre_filepath = Path(self.saving_file_path, camera_loc, "preprocess")
        self.net_file = Path(self.sumo_filepath, f"{camera_loc}.net.xml")
        self.sumo_netfile = "osm.net.xml"
        self.dataset_path = Path(dataset_path) # Path to the dataset directory
        # Load the trajectory CSV once at initialization
        # self.trajectory_df = pl.read_csv(Path(self.pre_filepath, csv_name))
        # trajectory_csv = f"../results/511video/{args.mode}/{args.camera_loc}/{args.mode}_trajectory_clustering.csv"
        trajectory_csv = f"../results/511video/federated/{camera_loc}/federated_trajectory_clustering.csv"
        self.trajectory_df = pl.read_csv(trajectory_csv, schema_overrides={"target_lane_id": pl.Utf8})
        
        # Get gps, pixel points from text file in each camera location
        self.net = sumolib.net.readNet(Path("../", "LaneDetection", "osm_extraction", self.camera_loc, self.sumo_netfile))

        # Get available edges
        self.available_edges = self.get_valid_edges()
        print(f"Network has {len(self.available_edges)} valid edges")

        self.vehicle_data = {}
    
    def pixel_to_global(self, camera_loc):
        # Load as (pixel_x, pixel_y, lat, lon)
        csv_path = Path(self.dataset_path, "511calibration", f"{camera_loc}.csv")
        df = pl.read_csv(csv_path)

        # Extract arrays
        pixel_points = df.select(["pixel_x", "pixel_y"]).to_numpy().astype(np.float64)
        gps_points = df.select(["latitude", "longitude"]).to_numpy().astype(np.float64)

        return gps_points, pixel_points

    def calibrate(self):
        # Compute the homography matrix
        traj = self.trajectory_df.select(["x", "y"]).to_numpy().astype(np.float32)
        world_points, pixel_points = self.pixel_to_global(self.camera_loc)
        H, _ = cv2.findHomography(pixel_points, world_points)

        # Convert a new pixel coordinate to SUMO world coordinates
        gps_points = cv2.perspectiveTransform(traj.reshape(-1, 1, 2), H).reshape(-1, 2)

        # Convert world to SUMO coordinates
        lane_points = np.array([
            self.net.convertLonLat2XY(lon, lat)
            for lat, lon in zip(gps_points[:, 0], gps_points[:, 1])
        ])

        # Convert back to Polars DataFrame and attach to the original DataFrame
        new_df = pl.DataFrame({
            "x_world": gps_points[:, 0],
            "y_world": gps_points[:, 1],
            "x_sumo": lane_points[:, 0],
            "y_sumo": lane_points[:, 1]
        })

        # Add the transformed coordinates as new columns
        self.trajectory_df = self.trajectory_df.with_columns(new_df)

    def get_valid_edges(self):
        """Get list of valid edges from network"""
        edges = []
        for edge in self.net.getEdges():
            # Skip internal edges (junctions)
            if edge.getFunction() != 'internal' and edge.getLength() > 5:
                edges.append(edge.getID())
        return edges

    def find_nearest_edge(self, x, y):
        radius = 50
        nearby_edges = self.net.getNeighboringEdges(x, y, radius)
        # pick the closest edge
        try:
            if len(nearby_edges) > 0:
                distancesAndEdges = sorted([(dist, edge) for edge, dist in nearby_edges], key=lambda x:x[0])
                # print(f"Found {len(distancesAndEdges)} nearby edges, {distancesAndEdges}")
                dist, closestEdge = distancesAndEdges[0]
                return closestEdge.getID()
        except:
            pass

        if self.available_edges:
            return self.available_edges[0]
        return None

    def prepare_data(self):
        """Prepare trajectory data and assign edges"""
        print("Preparing trajectory data...")

        # self.trajectory_df = self.trajectory_df.with_columns(
        #     pl.struct(["x_sumo", "y_sumo"]).map_elements(lambda row: self.find_nearest_edge(row["x_sumo"], row["y_sumo"])).alias("edgeID")
        # )
        # self.trajectory_df = self.trajectory_df.filter(pl.col("edgeID").is_not_null())

        fps = 30
        step_size = 0.05
        self.trajectory_df = self.trajectory_df.with_columns([
            ((pl.col("time") / step_size).round() * step_size).alias("vid_time")
        ])

        for vehicle_id in self.trajectory_df["id"].unique():
            vehicle_traj = self.trajectory_df.filter(pl.col("id") == vehicle_id)
            vehicle_traj = vehicle_traj.sort(["time"])

            # If trajectory has SUMO coordinates, use them to find edges
            if 'x_sumo' in vehicle_traj.columns and 'y_sumo' in vehicle_traj.columns:
                # Find edge for first position
                first_x = vehicle_traj['x_sumo'][0]
                first_y = vehicle_traj['y_sumo'][0]
                start_edge = self.find_nearest_edge(first_x, first_y)
            else:
                # Use a default edge
                start_edge = self.available_edges[0] if self.available_edges else "edge_0"
            
            vehicle_traj = vehicle_traj.with_columns(pl.lit(start_edge).alias("start_edge"))
            self.vehicle_data[str(vehicle_id)] = vehicle_traj

        print(f"Prepared data for {len(self.vehicle_data)} vehicles")

    def create_route(self, output_file="vehicles.rou.xml"):
        """Create routes file with actual network edges"""
        with open(output_file, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')

            used_routes = set()
            route_counter = 0

            # Sort vehicle data by depart_time
            sorted_vehicles = sorted(self.vehicle_data.items(), key=lambda x: x[1]['vid_time'][0])

            # for vehicle_id, vehicle_data in self.vehicle_data.items():
            for vehicle_id, vehicle_data in sorted_vehicles:
                start_edge = vehicle_data['start_edge'][0]
                
                # Create simple route
                if start_edge not in used_routes:
                    # Try to find connected edges for longer route
                    try:
                        edge_obj = self.net.getEdge(start_edge)
                        outgoing = [e.getID() for e in edge_obj.getOutgoing()]
                        if outgoing:
                            route_edges = [start_edge] + outgoing[:2] # Max 3 edges
                        else:
                            route_edges = [start_edge]
                    except:
                        route_edges = [start_edge]
                    
                    route_id = f"route_{route_counter}"
                    edges_str = ' '.join(route_edges)
                    f.write(f'    <route id="{route_id}" edges="{edges_str}"/>\n')
                    used_routes.add(start_edge)
                    route_counter += 1
                else:
                    route_id = f"route_{list(used_routes).index(start_edge)}"
                
                # Add vehicle
                depart_time = vehicle_data['vid_time'][0]
                if vehicle_data['lane_id'][0] != "-1":
                    depart_lane = vehicle_data['lane_id'][0].split('_')[1] if 'lane_id' in vehicle_data.columns else 0
                else:
                    depart_lane = 0
                f.write(f'    <vehicle id="{vehicle_id}" depart="{depart_time}" route="{route_id}" departLane="{depart_lane}"/>\n')

            f.write('</routes>\n')

        print(f"Created {output_file}")

    def get_position_at_time(self, vehicle_id, sim_time):
        """Get vehicle position at specific time"""
        if vehicle_id not in self.vehicle_data:
            return None, None
        
        df = self.vehicle_data[vehicle_id]
        
        # Use SUMO coordinates if available, otherwise use pixel coordinates
        if 'x_sumo' in df.columns and 'y_sumo' in df.columns:
            x_col, y_col = 'x_sumo', 'y_sumo'
        else:
            x_col, y_col = 'x', 'y' # Fallback to original coordinates
        
        # Simple interpolation
        time_diffs = np.abs(df['time'] - sim_time)
        closest_idx = time_diffs.idxmin()
        
        if len(df) == 1 or time_diffs.iloc[closest_idx] < 0.01:
            row = df.iloc[closest_idx]
            return float(row[x_col]), float(row[y_col])
        
        # Linear interpolation between closest points
        if closest_idx == 0:
            row = df.iloc[0]
        elif closest_idx == len(df) - 1:
            row = df.iloc[-1]
        else:
            if df.iloc[closest_idx]['time'] > sim_time:
                idx2, idx1 = closest_idx, closest_idx - 1
            else:
                idx1, idx2 = closest_idx, closest_idx + 1
            
            t1, t2 = df.iloc[idx1]['time'], df.iloc[idx2]['time']
            if t2 != t1:
                alpha = (sim_time - t1) / (t2 - t1)
                x = df.iloc[idx1][x_col] + alpha * (df.iloc[idx2][x_col] - df.iloc[idx1][x_col])
                y = df.iloc[idx1][y_col] + alpha * (df.iloc[idx2][y_col] - df.iloc[idx1][y_col])
                return float(x), float(y)
            else:
                return float(df.iloc[idx1][x_col]), float(df.iloc[idx1][y_col])
        
        return float(row[x_col]), float(row[y_col])

    def add_position_jitter(self, x, y, vehicle_id):
        """Add small random offset to prevent exact overlaps"""
        np.random.seed(hash(str(vehicle_id)) % 2**32)  # Consistent seed per vehicle
        jitter_range = 2.0  # meters
        x_offset = np.random.uniform(-jitter_range, jitter_range)
        y_offset = np.random.uniform(-jitter_range, jitter_range)
        return x + x_offset, y + y_offset

    def run_simulation(self, max_steps=1000, save_replay=True, replay_filename="replay_results.csv"):
        """ Here we need some estimation of the vehicle position/time at the beginning of the lane """
        print("Starting SUMO simulation...")

        traci.start([
            "sumo-gui",
            "--net-file", str(self.net_file),
            "-r", "vehicles.rou.xml",
            "--step-length", "0.1",
            "--collision.action", "none" # Disable collision detection for teleporting vehicles
        ])
        
        step = 0
        successful_moves = 0
        failed_moves = 0
        replay_log = []

        try:
            while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
                # time in detection has details than SUMO time, so we can change step
                traci.simulationStep()
        
                active_vehicles = traci.vehicle.getIDList()
                # print(f"Step {step}: Active Vehicles → {active_vehicles}")
                sim_time = traci.simulation.getTime()
    
                for vehicle_id in active_vehicles:
                    try:
                        x_target, y_target = self.get_position_at_time(int(vehicle_id), sim_time)
                        # print(x_target, y_target)
                        if x_target is not None and y_target is not None:
                            # Add small jitter to prevent exact overlaps
                            # x_jittered, y_jittered = self.add_position_jitter(x_sumo, y_sumo, vehicle_id)
                            traci.vehicle.moveToXY(vehicle_id, "", 0, x_target, y_target, keepRoute=2)

                            # Get actual position after move
                            actual_pos = traci.vehicle.getPosition(vehicle_id)
                            actual_speed = traci.vehicle.getSpeed(vehicle_id)
                            current_edge = traci.vehicle.getRoadID(vehicle_id)
                            current_lane = traci.vehicle.getLaneIndex(vehicle_id)

                            # Log the replay data
                            replay_log.append({
                                'step': step,
                                'sim_time': sim_time,
                                'vehicle_id': vehicle_id,
                                'target_x': x_target,
                                'target_y': y_target,
                                'actual_x': actual_pos[0],
                                'actual_y': actual_pos[1],
                                'speed': actual_speed,
                                'edge_id': current_edge,
                                'lane_index': current_lane,
                                'move_successful': True
                            })

                            successful_moves += 1

                            if step % 100 == 0: # Print occasionally
                                print(f"Step {step}: Vehicle {vehicle_id} -> Target({x_target:.1f}, {y_target:.1f}) Actual({actual_pos[0]:.1f}, {actual_pos[1]:.1f})")
                    except Exception as e:
                        # Log failed moves too
                        replay_log.append({
                            'step': step,
                            'sim_time': sim_time,
                            'vehicle_id': vehicle_id,
                            'target_x': x_target if 'x_target' in locals() else None,
                            'target_y': y_target if 'y_target' in locals() else None,
                            'actual_x': None,
                            'actual_y': None,
                            'speed': None,
                            'edge_id': None,
                            'lane_index': None,
                            'move_successful': False,
                            'error': str(e)
                        })
                        
                        failed_moves += 1
                        if step % 100 == 0:
                            print(f"Failed to move vehicle {vehicle_id}: {e}")
                
                step += 1
        
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            traci.close()
            print(f"Simulation ended. Successful moves: {successful_moves}, Failed: {failed_moves}")

            # Save replay results
            if save_replay and replay_log:
                replay_df = pd.DataFrame(replay_log)
                replay_df.to_csv(replay_filename, index=False)
                print(f"Saved replay results to {replay_filename} ({len(replay_log)} records)")
                
                # Print summary statistics
                if len(replay_df) > 0:
                    successful_records = replay_df[replay_df['move_successful'] == True]
                    print(f"Replay Summary:")
                    print(f"  - Total records: {len(replay_df)}")
                    print(f"  - Successful moves: {len(successful_records)}")
                    print(f"  - Failed moves: {len(replay_df) - len(successful_records)}")
                    print(f"  - Unique vehicles: {replay_df['vehicle_id'].nunique()}")
                    print(f"  - Time range: {replay_df['sim_time'].min():.1f} - {replay_df['sim_time'].max():.1f}s")
            
            print(f"Simulation ended. Successful moves: {successful_moves}, Failed: {failed_moves}")


class ImprovedDet2SumoSync(Det2SumoSync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = None
    
    def preprocess_trajectories(self, interpolate=True, validate_lanes=True, remove_outliers=True):
        """
        Preprocess trajectories before SUMO simulation
        """
        print("Preprocessing trajectories...")
        
        # Initialize preprocessor
        net_path = Path("../", "LaneDetection", "osm_extraction", self.camera_loc, self.sumo_netfile)
        self.preprocessor = TrajectoryPreprocessor(net_path)
        
        # Remove outlier trajectories first
        if remove_outliers:
            print("Removing outlier trajectories...")
            self.trajectory_df = self.preprocessor.remove_outlier_trajectories(self.trajectory_df)
            print(f"Remaining vehicles: {self.trajectory_df['id'].n_unique()}")
        
        # Process each vehicle's trajectory
        processed_trajectories = []
        
        for vehicle_id in self.trajectory_df['id'].unique():
            vehicle_data = self.trajectory_df.filter(pl.col('id') == vehicle_id).sort('time')
            
            # Apply physics-based smoothing
            vehicle_data = self.preprocessor.smooth_trajectory_with_physics(vehicle_data)
            
            # Interpolate if requested
            if interpolate and len(vehicle_data) >= 2:
                vehicle_data = self.preprocessor.interpolate_trajectory(vehicle_data, target_fps=30)
            
            if len(vehicle_data) > 0:
                processed_trajectories.append(vehicle_data)
        
        if processed_trajectories:
            self.trajectory_df = pl.concat(processed_trajectories)
        
        print(f"Preprocessing complete. Final vehicle count: {self.trajectory_df['id'].n_unique()}")
    
    def create_route(self):
        # Add edge IDs
        self.trajectory_df = self.trajectory_df.with_columns(
            pl.struct(["x_sumo", "y_sumo"]).map_elements(
                lambda row: self.find_nearest_edge(row["x_sumo"], row["y_sumo"]), 
                return_dtype=pl.Utf8
            ).alias("edgeID")
        )

        self.trajectory_df = self.trajectory_df.filter(pl.col("edgeID").is_not_null())

        # Validate lane consistency for each vehicle
        if self.preprocessor:
            validated_trajectories = []
            for vehicle_id in self.trajectory_df['id'].unique():
                vehicle_data = self.trajectory_df.filter(pl.col('id') == vehicle_id)
                validated_data = self.preprocessor.validate_lane_consistency(vehicle_data)
                if len(validated_data) > 0:
                    validated_trajectories.append(validated_data)
            
            if validated_trajectories:
                self.trajectory_df = pl.concat(validated_trajectories)
        
        super().create_route()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Settings for Training")
    parser.add_argument('--dataset_path', default='../dataset/', help='dataset path')
    parser.add_argument('--video_path', default='../dataset/511video', help='camera ip or local video path')
    parser.add_argument('--saving_path', default='../results/', help='path to save results')
    parser.add_argument('--osm_path', type=str, default="../LaneDetection/osm_extraction/", help='The path of the OSM file to extract data from')
    parser.add_argument('--camera_loc', type=str, default="US12_Yahara", help='The camera location to use for synchronization')
    parser.add_argument('--mode', type=str, default="federated", help='Mode of lane detection, federated, meta, and baseline.')
    args = parser.parse_args()

    # trajectory_csv = f"../results/511video/{args.camera_loc}/preprocess/trajectory.csv"
    trajectory_csv = f"../results/511video/{args.mode}/{args.camera_loc}/{args.mode}_trajectory_clustering.csv"
    net_file = f"{args.camera_loc}.net.xml"

    file_name_ = Path(args.video_path).stem
    saving_file_path = Path(args.saving_path, file_name_)

    # Synchronize trajectories
    syncer = ImprovedDet2SumoSync(saving_file_path, args.camera_loc, args.dataset_path)
    syncer.calibrate()
    syncer.prepare_data()

    # syncer.preprocess_trajectories(
    #     interpolate=True,
    #     validate_lanes=True,
    #     remove_outliers=True
    # )

    syncer.create_route()
    syncer.run_simulation(max_steps=36000)

