import cv2
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
import polars as pl
from pathlib import Path
from PIL import Image

matplotlib.set_loglevel(level='warning')
matplotlib.use('Agg')

from LaneDetection.osm_extraction.utils import compute_lane_width_from_gps
from LaneDetection.osm_extraction.connect_to_osm import OSMConnection

logger = logging.getLogger(__name__)


def compute_loss_for_baseline(geo_learning, traj_df, lane_boundaries_for_contour, processed_data, client_id):
    """Wrapper function of compute_loss in GeometricLeanrning, executing geometric learning and compute loss."""
    try:
        
        # Convert to pandas if still in Polars
        traj_df_pd = traj_df.to_pandas() if hasattr(traj_df, 'to_pandas') else traj_df
        # Filter out unassigned or invalid lane clusters
        traj_df_pd = traj_df_pd[traj_df_pd["clustered_id"] != -1]

        # Create osm_connection if not provided
        osm_connection = OSMConnection(geo_learning.args, geo_learning.filepath)

        # Visualization setup
        if geo_learning.is_save:
            _visualize_lane_detection(
                traj_df_pd, 
                lane_boundaries_for_contour, 
                processed_data, 
                osm_connection, 
                client_id,
                geo_learning,
            )
        
        # Extract detected centers and compute lane widths
        detected_center_list = []
        lane_width_list = []
        
        for boundaries in lane_boundaries_for_contour:
            for lane_id, data in boundaries.items():
                detected_center_list.append(data["center"])
                widths, avg_width = compute_lane_width_from_gps(data["left"], data["right"])
                lane_width_list.append(widths)

        # Get SUMO data for comparison
        if len(detected_center_list) > 0:
            _, cluster_to_edge_map, lane_shape = osm_connection.get_sumo_data(
                np.mean(detected_center_list, axis=1), 
                client_id, 
                trial='0'
            )
            
            # Get sumo_node data from processed_data if available
            sumo_node, _ = processed_data.get('sumo_graph', ([], []))
            
            # Flatten SUMO nodes into tensor format
            sumo_center_tensor = []
            for group in sumo_node:
                for line in group:
                    line_tensor = torch.tensor(np.array(line), dtype=torch.float32)
                    sumo_center_tensor.append(line_tensor)
            
            # Convert detected centers to tensor
            detected_center_tensor = torch.tensor(np.array(detected_center_list), dtype=torch.float32)
            
            # Compute loss using geo_learning's compute_loss method
            l_total, l_lane_count, l_cons, l_trip, l_geo = geo_learning.compute_loss(
                detected_center_tensor, 
                sumo_center_tensor, 
                lane_width_list, 
                lane_shape, 
                cluster_to_edge_map
            )
            
            # Convert losses to float values
            total_loss = l_total.item() if isinstance(l_total, torch.Tensor) else float(l_total)
            
            metrics = {
                'lane_count': len(detected_center_list),
                'l_total': total_loss,
                'l_lane_count': l_lane_count.item() if isinstance(l_lane_count, torch.Tensor) else float(l_lane_count),
                'l_cons': l_cons.item() if isinstance(l_cons, torch.Tensor) else float(l_cons),
                'l_trip': l_trip.item() if isinstance(l_trip, torch.Tensor) else float(l_trip),
                'l_geo': l_geo.item() if isinstance(l_geo, torch.Tensor) else float(l_geo),
                'detected_lanes': len(detected_center_list),
                'sumo_lanes': len(sumo_center_tensor) if sumo_center_tensor and sumo_center_tensor[0].shape[0] > 0 else 0
            }

            logger.info(f"Client {client_id} - Total Loss: {total_loss:.4f}, "
                        f"Lane Count Loss: {metrics['l_lane_count']:.4f}, "
                        f"Consistency Loss: {metrics['l_cons']:.4f}, "
                        f"Triplet Loss: {metrics['l_trip']:.4f}, "
                        f"Geometry Loss: {metrics['l_geo']:.4f}")
            
            return total_loss, metrics
            
        else:
            # No lanes detected
            logger.warning(f"No lanes detected for client {client_id}")
            return float('inf'), {'lane_count': 0, 'error': 'No lanes detected'}
            
    except Exception as e:
        logger.error(f"Error in geo_learning for client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), {'error': str(e)}


def _visualize_lane_detection(traj_df_pd, lane_boundaries_for_contour, 
                              processed_data, osm_connection, client_id, geo_learning):
    """Visualize lane detection results on the camera image."""
    
    colors = [
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
    
    fig_filepath = Path(geo_learning.filepath, client_id, "figures")
    
    fig1, ax1 = plt.subplots(figsize=(12, 10)) # For trajectory plots
    fig2, ax2 = plt.subplots(figsize=(12, 10)) # For lane plots
    
    # Load background image if available
    c_epoch = processed_data.get('c_epoch', 0)
    g_epoch = processed_data.get('g_epoch', 0)
    img_path = Path(fig_filepath, f"{g_epoch}_contour_on_roadmap.png")
    if img_path.exists():
        try:
            img = np.array(Image.open(img_path))
            ax1.imshow(img)
            ax2.imshow(img)
        except:
            logger.error(f"Failed to load image from {img_path}. Using blank background.")

    # Plot lane detection results
    for boundaries in lane_boundaries_for_contour:
        for lane_count, (lane_id, data) in enumerate(boundaries.items()):
            color = colors[lane_count % len(colors)]
            
            # Get trajectory points for this lane
            lane_df = traj_df_pd[traj_df_pd["clustered_id"] == lane_id]
            
            if 'pixel_hom' in processed_data and processed_data['pixel_hom'] is not None:
                # Convert to pixel coordinates if homography is available
                pixel_center = osm_connection.global_to_pixel(data["center"], processed_data["pixel_hom"])
                pixel_left = osm_connection.global_to_pixel(data["left"], processed_data["pixel_hom"])
                pixel_right = osm_connection.global_to_pixel(data["right"], processed_data["pixel_hom"])
                
                # Plot trajectory points
                if len(lane_df) > 0:
                    try:
                        ax1.scatter(lane_df["x"], lane_df["y"], s=2, alpha=0.6, 
                                color=color, label=f"Traj Lane {lane_id}")
                    except:
                        logger.error(f"Error plotting lane_df for lane_id {lane_id}")
                
                # Plot lane boundaries
                ax2.plot(pixel_center[:, 0], pixel_center[:, 1], 
                        color=color, linewidth=2.5, label=f"Center Lane {lane_id}")
                ax2.plot(pixel_left[:, 0], pixel_left[:, 1], 
                        color=color, linewidth=2.0, linestyle='--', alpha=0.7)
                ax2.plot(pixel_right[:, 0], pixel_right[:, 1], 
                        color=color, linewidth=2.0, linestyle='--', alpha=0.7)
            else:
                # Plot in GPS coordinates if no homography
                if len(lane_df) > 0 and 'x_gps' in lane_df.columns:
                    ax1.scatter(lane_df["x_gps"], lane_df["y_gps"], s=2, alpha=0.6, 
                                color=color, label=f"Traj Lane {lane_id}")
                
                # Plot lane boundaries in GPS
                ax2.plot(data["center"][:, 0], data["center"][:, 1], 
                        color=color, linewidth=2.5, label=f"Center Lane {lane_id}")
                ax2.plot(data["left"][:, 0], data["left"][:, 1], 
                        color=color, linewidth=2.0, linestyle='--', alpha=0.7)
                ax2.plot(data["right"][:, 0], data["right"][:, 1], 
                        color=color, linewidth=2.0, linestyle='--', alpha=0.7)

    # Deduplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    ax1.set_axis_off()

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    ax2.set_axis_off()
    ax2.set_xlim(0, 1920)
    ax2.set_ylim(1080, 0) 

    # Save the visualization
    fig1_path = Path(fig_filepath, f"{g_epoch}_trajectory_clustering_{c_epoch}.png")
    fig2_path = Path(fig_filepath, f"{g_epoch}_lane_detection_{c_epoch}.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    logger.info(f"Lane detection visualization saved to {fig1_path} and {fig2_path}")


def save_training_results(system, training_history, save_path):
    """Save final training results and model."""
    results_dir = Path(save_path, "training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training history
    import json
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    
    # Plot training curve
    if training_history:
        plt.figure(figsize=(10, 6))
        epochs = [h.get('epoch', i) for i, h in enumerate(training_history)]
        avg_losses = [h.get('avg_loss', float('inf')) for h in training_history]
        std_losses = [h.get('std_loss', 0) for h in training_history]
        
        valid_indices = [i for i, loss in enumerate(avg_losses) if loss != float('inf')]
        
        if valid_indices:
            epochs = [epochs[i] for i in valid_indices]
            avg_losses = [avg_losses[i] for i in valid_indices]
            std_losses = [std_losses[i] for i in valid_indices]
            
            plt.plot(epochs, avg_losses, 'b-', label='Average Loss')
            plt.fill_between(epochs, 
                             np.array(avg_losses) - np.array(std_losses),
                             np.array(avg_losses) + np.array(std_losses),
                             alpha=0.3, color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Progress - {system.strategy.upper()}')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_dir / 'training_curve.png')
        plt.close()
    
    # Save final model
    system.save_checkpoint(results_dir)
    
    logger.info(f"Training results saved to {results_dir}")


def estimate_object_size_bytes(obj):
    """Serialize and estimate object size in bytes."""
    return len(pickle.dumps(obj))


# class LaneAssignmentPostProcessor:
#     """
#     Post-processes trajectory CSV to add lane assignments after geometric learning
#     """
#     def __init__(self, args, saving_file_path):
#         self.args = args
#         self.saving_file_path = saving_file_path
        
#     def assign_vehicles_to_detected_lanes(self, traj_df, lane_boundaries_for_contour, 
#                                         pixel_hom, camera_loc, epoch):
#         """
#         Assign vehicles in trajectory DataFrame to detected lanes
        
#         Args:
#             traj_df: Trajectory DataFrame from continuous process
#             lane_boundaries_for_contour: Detected lane boundaries from geometric learning
#             pixel_hom: Pixel homography transformation
#             camera_loc: Camera location identifier
#             epoch: Current epoch number
            
#         Returns:
#             Enhanced DataFrame with lane assignments
#         """
#         try:
#             # Convert to pandas for easier processing
#             if hasattr(traj_df, 'to_pandas'):
#                 traj_df_pd = traj_df.to_pandas()
#             else:
#                 traj_df_pd = traj_df
            
#             # Initialize new columns
#             traj_df_pd['assigned_lane'] = -1
#             traj_df_pd['lane_distance'] = -1.0
#             traj_df_pd['lane_confidence'] = 0.0
#             traj_df_pd['num_lanes_detected'] = 0
            
#             # Process each detected lane boundary
#             lane_centers = {}
#             total_lanes = 0
            
#             for contour_idx, boundaries in enumerate(lane_boundaries_for_contour):
#                 for lane_id, lane_data in boundaries.items():
#                     # Extract lane center coordinates
#                     lane_center = lane_data.get("center", np.array([]))
#                     if len(lane_center) > 0:
#                         lane_centers[total_lanes] = {
#                             'center_coords': lane_center,
#                             'original_lane_id': lane_id,
#                             'contour_idx': contour_idx,
#                             'width': lane_data.get("width", 3.5)
#                         }
#                         total_lanes += 1
            
#             # Update total lanes detected
#             traj_df_pd['num_lanes_detected'] = total_lanes
            
#             if total_lanes == 0:
#                 logger.warning(f"No lanes detected for {camera_loc} at epoch {epoch}")
#                 return pl.from_pandas(traj_df_pd)
            
#             # Just put each vehicle position to detected lane id
#             # Assign each vehicle position to closest lane
#             for idx, row in traj_df_pd.iterrows():
#                 vehicle_x, vehicle_y = row['x'], row['y']
                
#                 # Convert vehicle position if homography is available
#                 if pixel_hom is not None:
#                     try:
#                         # Transform vehicle pixel coordinates to GPS/world coordinates
#                         vehicle_pos = np.array([[vehicle_x, vehicle_y]], dtype=np.float32)
#                         vehicle_world = cv2.perspectiveTransform(
#                             vehicle_pos.reshape(-1, 1, 2), pixel_hom
#                         ).reshape(-1, 2)[0]
#                         vehicle_x_world, vehicle_y_world = vehicle_world
#                     except:
#                         vehicle_x_world, vehicle_y_world = vehicle_x, vehicle_y
#                 else:
#                     vehicle_x_world, vehicle_y_world = vehicle_x, vehicle_y
                
#                 # Find closest lane
#                 min_distance = float('inf')
#                 closest_lane = -1
#                 confidence = 0.0
                
#                 for lane_idx, lane_info in lane_centers.items():
#                     lane_center_coords = lane_info['center_coords']
                    
#                     # Calculate minimum distance to lane centerline
#                     distances = []
#                     for center_point in lane_center_coords:
#                         if len(center_point) >= 2:
#                             dist = np.sqrt(
#                                 (vehicle_x_world - center_point[0])**2 + 
#                                 (vehicle_y_world - center_point[1])**2
#                             )
#                             distances.append(dist)
                    
#                     if distances:
#                         avg_distance = np.mean(distances)
#                         min_dist_to_lane = min(distances)
                        
#                         if min_dist_to_lane < min_distance:
#                             min_distance = min_dist_to_lane
#                             closest_lane = lane_idx
                            
#                             # Calculate confidence based on distance and lane width
#                             lane_width = lane_info.get('width', 3.5)
#                             # Confidence decreases with distance, max at lane center
#                             confidence = max(0, 1.0 - (min_dist_to_lane / (lane_width * 2)))
                
#                 # Assign vehicle to lane if within reasonable distance
#                 distance_threshold = 100 # pixels, or adjust based on your coordinate system
#                 if min_distance < distance_threshold and confidence > 0.1:
#                     # THIS IS WRONG
#                     traj_df_pd.at[idx, 'assigned_lane'] = closest_lane
#                     traj_df_pd.at[idx, 'lane_distance'] = float(min_distance)
#                     traj_df_pd.at[idx, 'lane_confidence'] = float(confidence)
            
#             # Add additional lane statistics
#             enhanced_df = self._add_lane_statistics(pl.from_pandas(traj_df_pd))
            
#             # Save enhanced trajectory CSV
#             self._save_enhanced_trajectory(enhanced_df, camera_loc, epoch)
            
#             logger.info(f"Successfully assigned {total_lanes} lanes to vehicles for {camera_loc}")
#             return enhanced_df
            
#         except Exception as e:
#             logger.error(f"Error in lane assignment for {camera_loc}: {e}")
#             return traj_df
    
#     def _add_lane_statistics(self, df):
#         """Add lane-related statistics to the DataFrame"""
#         try:
#             # Sort by vehicle ID and time for proper sequence analysis
#             df = df.sort(["id", "time"])
            
#             # Add lane change detection
#             df = df.with_columns([
#                 # Detect lane changes (when assigned_lane changes for same vehicle)
#                 (pl.col("assigned_lane").diff().over("id") != 0).cast(pl.Int8).alias("lane_changed"),
                
#                 # Count consecutive frames in same lane
#                 pl.col("assigned_lane").rle_id().over("id").alias("lane_segment_id"),
                
#                 # Calculate vehicle's primary (most frequent) lane
#                 pl.col("assigned_lane").mode().over("id").alias("primary_lane")
#             ])
            
#             # Add lane stability metrics
#             df = df.with_columns([
#                 # Time spent in current lane segment
#                 pl.col("lane_segment_id").count().over(["id", "lane_segment_id"]).alias("time_in_current_lane"),
                
#                 # Average confidence in lane assignment for this vehicle
#                 pl.col("lane_confidence").mean().over("id").alias("avg_lane_confidence")
#             ])
            
#             return df
            
#         except Exception as e:
#             logger.warning(f"Could not add lane statistics: {e}")
#             return df
    
#     def _save_enhanced_trajectory(self, enhanced_df, camera_loc, epoch):
#         """Save the enhanced trajectory CSV with lane assignments"""
#         try:
#             pre_filepath = Path(self.saving_file_path, camera_loc, "preprocess")
            
#             # Save enhanced trajectory
#             enhanced_csv_path = Path(pre_filepath, f"trajectory_with_lanes_epoch_{epoch}.csv")
#             enhanced_df.write_csv(enhanced_csv_path)
            
#             # Also update the main trajectory.csv
#             main_csv_path = Path(pre_filepath, "trajectory.csv")
#             enhanced_df.write_csv(main_csv_path)
            
#             # Save lane assignment summary
#             lane_summary = self._create_lane_summary(enhanced_df)
#             summary_path = Path(pre_filepath, f"lane_summary_epoch_{epoch}.json")
            
#             with open(summary_path, 'w') as f:
#                 json.dump(lane_summary, f, indent=2)
            
#             logger.info(f"Saved enhanced trajectory to {enhanced_csv_path}")
            
#         except Exception as e:
#             logger.error(f"Error saving enhanced trajectory: {e}")
    
#     def _create_lane_summary(self, df):
#         """Create a summary of lane assignments"""
#         try:
#             summary = {
#                 'total_vehicles': df['id'].n_unique(),
#                 'total_frames': len(df),
#                 'lanes_detected': int(df['num_lanes_detected'].max()),
#                 'vehicles_assigned': len(df.filter(pl.col('assigned_lane') >= 0)),
#                 'assignment_rate': float(len(df.filter(pl.col('assigned_lane') >= 0)) / len(df)),
#                 'avg_confidence': float(df['lane_confidence'].mean()),
#                 # 'lane_changes_detected': int(df['lane_changed'].sum())
#             }
            
#             # Per-lane statistics
#             lane_stats = {}
#             for lane_id in range(int(df['num_lanes_detected'].max())):
#                 lane_vehicles = df.filter(pl.col('assigned_lane') == lane_id)
#                 if len(lane_vehicles) > 0:
#                     lane_stats[f'lane_{lane_id}'] = {
#                         'vehicle_count': lane_vehicles['id'].n_unique(),
#                         'total_detections': len(lane_vehicles),
#                         'avg_confidence': float(lane_vehicles['lane_confidence'].mean()),
#                         'avg_distance': float(lane_vehicles['lane_distance'].mean())
#                     }
            
#             summary['per_lane_stats'] = lane_stats
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error creating lane summary: {e}")
#             return {}
