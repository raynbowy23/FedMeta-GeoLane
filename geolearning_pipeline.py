import json
import time
import cv2
import polars as pl
import numpy as np
import torch
import mlflow
import logging
import traceback
from tqdm import tqdm
from pathlib import Path

# Initialize other components
from LaneDetection.lane_detection.lane_continuous_process import LaneContinuousProcess
from LaneDetection.lane_detection.geo_learning import GeometricLearning
from LaneDetection.osm_extraction.connect_to_osm import OSMConnection
from data_pipeline import OrbitDataPipeline
from geolearning_system import GeoLearningSystem
# from utils import LaneAssignmentPostProcessor

logger = logging.getLogger(__name__)


class GeometricLearningPipeline:
    """Pipeline for geometric learning in lane detection."""
    
    def __init__(self, args, saving_file_path, camera_loc_list, strategy_type='baseline'):
        self.args = args
        self.saving_file_path = saving_file_path
        self.camera_loc_list = camera_loc_list
        
        # Initialize learning system
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_system = GeoLearningSystem(args, device, strategy_type)
        self.learning_system.setup_clients(camera_loc_list)
        
        self.lane_detection = LaneContinuousProcess(args, saving_file_path, camera_loc_list)
        self.data_pipeline = OrbitDataPipeline(args, saving_file_path)
        self.osm_connection = OSMConnection(args, saving_file_path)
        
        # Create geo_learning instances for each camera
        self.geo_learning_instances = {}
        for camera_loc in camera_loc_list:
            self.geo_learning_instances[camera_loc] = GeometricLearning(args, saving_file_path)

    def run_learning_round(self, c_epoch, g_epoch, preprocessed_by_camera):
        """Run one round of learning"""
        self.learning_system.round_counter = g_epoch

        # Select clients
        available_clients = list(preprocessed_by_camera.keys())
        selected_clients = self.learning_system.select_clients(available_clients)

        logger.info(f"Epoch {g_epoch} - Strategy: {self.learning_system.strategy} - "
                   f"Selected clients: {selected_clients}")
        
        # Process each client
        client_results = {}
        for client_id in selected_clients:
            try:
                processed_data = preprocessed_by_camera[client_id]
                processed_data.update({'c_epoch': c_epoch, 'g_epoch': g_epoch})

                geo_learning = self.geo_learning_instances[client_id]
                
                loss, theta, metrics = self.learning_system.client_update(
                    client_id, processed_data, geo_learning
                )
                
                client_results[client_id] = (loss, theta, metrics)
                
                logger.info(f"Client {client_id}: Loss = {loss:.4f}, "
                           f"BPS = {metrics.get('bps', 0):.2f}")
                
            except Exception as e:
                logger.error(f"Error processing client {client_id}: {e}")
                continue
        
        # Aggregate results
        aggregated = self.learning_system.aggregate_client_updates(client_results)
        
        # Train models
        self.learning_system.train_models(g_epoch)
        
        logger.info(f"Epoch {g_epoch} - Avg Loss: {aggregated['avg_loss']:.4f} "
                   f"(±{aggregated.get('std_loss', 0):.4f}), "
                   f"Strategy: {aggregated.get('strategy', 'unknown')}")
        
        return aggregated
    
    def switch_to_deployment(self):
        """Switch to deployment mode"""
        self.learning_system.switch_to_deployment()
    
    def save_results(self, training_history):
        """Save training results"""
        results_dir = Path(self.saving_file_path, "training_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save training history
        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2, default=str)
        
        # Save model checkpoints
        self.learning_system.save_checkpoint(results_dir)
        
        logger.info(f"Results saved to {results_dir}")

def process_camera_data(data_pipeline, osm_connection, camera_loc, c_epoch, frame, collect_cars, collect_dots, traj_df):
    """
    Process camera data 
    """
    frame_copy = frame.copy()

    try:
        binary_heatmap_area, detected_vehs = data_pipeline.create_binary_image(camera_loc, c_epoch, frame_copy, collect_cars)
        detected_cnts = data_pipeline.generate_contour(camera_loc, frame_copy, binary_heatmap_area)

        gps_df, _, pixel_hom, sumo_graph = osm_connection.visualize_gps_network(camera_loc, traj_df, detected_cnts)

        contour_ids = []

        # Find index of x and y columns
        x_idx = gps_df.columns.index("x")
        y_idx = gps_df.columns.index("y")

        # Iterate over each row of the dataframe
        for row in gps_df.iter_rows():
            x, y = row[x_idx], row[y_idx]
            
            found = -1
            for cid, contour in enumerate(detected_cnts):
                if cv2.pointPolygonTest(contour, (x, y), False) > 0:
                    found = cid
                    break # stop at the first matching contour

            contour_ids.append(found)

        # Add result as new column
        gps_df = gps_df.with_columns(pl.Series("contour_id", contour_ids))

        # Apply this logic using Polars expression
        out_df = gps_df.sort(["id", "frame_num"])

        # Step 2: Remove rows with null dx/dy before computing heading
        out_df = out_df.drop_nulls(["x_gps", "y_gps"])

        # Compute dx and dy per object
        out_df = out_df.with_columns([
            pl.col("x_gps").diff().over("id").alias("dx"),
            pl.col("y_gps").diff().over("id").alias("dy"),
        ])

        # Filter rows with non-null dx/dy only
        out_df = out_df.filter(
            (pl.col("dx").is_not_null()) & (pl.col("dy").is_not_null())
        )

        # Compute heading angle
        out_df = out_df.with_columns([
            pl.struct(["dx", "dy"])
            .map_elements(lambda row: float(np.arctan2(row["dy"], row["dx"])), return_dtype=pl.Float64)
            .alias("theta_rad")
        ])

        return {
            'frame': frame_copy,
            'collect_cars': collect_cars,
            'collect_dots': collect_dots,
            'detected_cnts': detected_cnts,
            'gps_df': out_df,
            'pixel_hom': pixel_hom,
            'sumo_graph': sumo_graph
        }
    except Exception as e:
        logger.error(f"Error processing camera data for {camera_loc} at epoch {c_epoch}: {e}")
        raise

def continuous_process(args, c_epoch, data_queue, stop_event, saving_file_path, camera_loc_list, barrier):
    """
    Continuous data collection process
    """
    logger.info("[Continuous Process] Started")

    try:
        lane_detection = LaneContinuousProcess(args, saving_file_path, camera_loc_list)
        data_pipeline = OrbitDataPipeline(args, saving_file_path)
        osm_connection = OSMConnection(args, saving_file_path)
        
        # Multi-threading for continuous learning
        for epoch in tqdm(range(c_epoch), desc="Continuous Process", position=0):
            if stop_event.is_set():
                logger.info("Stopping continuous process...")
                return
                
            logger.info(f"Continuous Process Epoch: {epoch}")
            
            # Detection
            data_by_camera = lane_detection.continuous_process(args, epoch)
            preprocessed_by_camera = {}

            for camera_loc, data in data_by_camera.items():
                # Process data for each camera
                result = process_camera_data(
                    data_pipeline,
                    osm_connection,
                    camera_loc,
                    c_epoch=epoch,
                    frame=data.last_frame,
                    collect_cars=data.collect_cars,
                    collect_dots=data.collect_dots,
                    traj_df=data.out_df,
                )
                
                preprocessed_by_camera[camera_loc] = result
            
            # Put processed data in the queue for geometric learning thread
            data_queue.put((epoch, preprocessed_by_camera))
            logger.info("[Continuous Process] Epoch finished")

            time.sleep(0.1)
            barrier.wait()
            
    except Exception as e:
        logger.error(f"Error in continuous process: {e}")
        traceback.print_exc()
        stop_event.set()


def geometric_learning(pipeline, g_epoch, data_queue, stop_event, barrier):
    """
    Geomtric Learning thread.

    Geometric learning with federated meta-learning.
    Pipeline:
    (scene features) --> MetaMLModel --> theta_pred --> geo_learning.run_pipeline()
                                                        │
                                        geo_learning.compute_loss(...)
                                                        │
                                        loss.backward() + optimizer.step()
    """
    logger.info(f"[Geometric Learning] Start - Strategy: {pipeline.learning_system.strategy}")

    training_history = []
    # lane_processor = LaneAssignmentPostProcessor(pipeline.args, pipeline.saving_file_path)
    
    try:
        for epoch in tqdm(range(g_epoch), desc="Federated Geometric Learning", position=1):
            barrier.wait()
            
            if stop_event.is_set():
                logger.info("Stopping geometric learning...")
                return
            
            # Get preprocessed data
            c_epoch, preprocessed_by_camera = data_queue.get()
            
            # Run learning round
            aggregated = pipeline.run_learning_round(c_epoch, epoch, preprocessed_by_camera)
            training_history.append(aggregated)
            
            # MLflow logging
            try:
                mlflow.log_metric("Loss/Avg", aggregated['avg_loss'], step=epoch)
                if 'avg_bps' in aggregated:
                    mlflow.log_metric("Communication/Avg_BPS", aggregated['avg_bps'], step=epoch)
                if 'std_loss' in aggregated:
                    mlflow.log_metric("Loss/Std", aggregated['std_loss'], step=epoch)
                
                # Log detailed loss components if available
                for key, value in aggregated.items():
                    if key.startswith('avg_l_'):
                        component_name = key.replace('avg_l_', '')
                        mlflow.log_metric(f"Loss_Components/Avg_{component_name}", value, step=epoch)
                    elif key.startswith('std_l_'):
                        component_name = key.replace('std_l_', '')
                        mlflow.log_metric(f"Loss_Components/Std_{component_name}", value, step=epoch)
                
                # Log strategy-specific metrics
                strategy = aggregated.get('strategy', 'unknown')
                mlflow.log_metric(f"{strategy.title()}/Avg_Loss", aggregated['avg_loss'], step=epoch)
                if 'total_bps' in aggregated:
                    mlflow.log_metric(f"{strategy.title()}/Total_BPS", aggregated['total_bps'], step=epoch)
                    
            except ImportError:
                logger.warning("MLflow not available, skipping metric logging")
            
            # Switch to deployment mode after sufficient training
            if epoch >= 10:
                pipeline.switch_to_deployment()

            # if epoch == 10:
            #     for camera_loc, processed_data in preprocessed_by_camera.items():
            #         try:
            #             geo_learning = pipeline.geo_learning_instances[camera_loc]
            #             traj_df, lane_boundaries = geo_learning.run(
            #                 c_epoch=c_epoch,
            #                 g_epoch=epoch,
            #                 traj_df=processed_data.get('gps_df', None),
            #                 camera_loc=camera_loc,
            #                 trial='0',
            #                 is_save=False
            #             )

            #             # Assign vehicles to detected lanes with the best model
            #             _ = lane_processor.assign_vehicles_to_detected_lanes(
            #                 traj_df=processed_data.get('gps_df', None),
            #                 lane_boundaries_for_contour=lane_boundaries,
            #                 pixel_hom=processed_data.get('pixel_hom', None),
            #                 camera_loc=camera_loc,
            #                 epoch=epoch
            #             )
            #             logger.info(f"Update trajectory CSV with lane assignments for {camera_loc}")
            #         except Exception as e:
            #             logger.error(f"Error in lane assignment for {camera_loc}: {e}")
            #             traceback.print_exc()
            
            time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Error in geometric learning: {e}")
        traceback.print_exc()
        stop_event.set()
    
    finally:
        pipeline.save_results(training_history)
        logger.info("[Geometric Learning] Finish")