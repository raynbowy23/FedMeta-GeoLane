import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import json
import time
import mlflow

from LaneDetection.osm_extraction.utils import compute_lane_width_from_gps
from LaneDetection.osm_extraction.connect_to_osm import OSMConnection

from .utils import FederatedConfig, SceneFeatureExtractor

from PIL import Image
import pickle

logger = logging.getLogger(__name__)


def estimate_object_size_bytes(obj):
    """Serialize and estimate object size in bytes."""
    return len(pickle.dumps(obj))

class MetaMLModel(nn.Module):
    """
    Black-box meta-learner that maps scene features to optimal theta parameters.
    No gradient-based adaptation - directly predicts parameters from features.
    """
    def __init__(self, feature_dim=5, hidden_dim=128, num_theta_params=5, config_path=None):
        super(MetaMLModel, self).__init__()
        
        self.config = FederatedConfig(config_path)
        self.feature_dim = feature_dim
        self.num_theta_params = num_theta_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Parameter-specific heads
        self.theta_heads = nn.ModuleDict({
            'width_scale': nn.Linear(hidden_dim, 1),
            'consistency_weight': nn.Linear(hidden_dim, 1),
            'triplet_margin': nn.Linear(hidden_dim, 1),
            'smoothing_factor': nn.Linear(hidden_dim, 1),
        })

        # Learnable loss weights
        self.loss_weights = nn.ParameterDict({
            'lane_count': nn.Parameter(torch.tensor(1.0)),
            'consistency': nn.Parameter(torch.tensor(1.0)),
            'triplet': nn.Parameter(torch.tensor(1.0)),
            'geometry': nn.Parameter(torch.tensor(1.0)),
        })
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, scene_features):
        """
        Args:
            scene_features: Tensor of shape (batch_size, feature_dim)
        
        Returns:
            theta_dict: Dictionary of predicted parameters
        """
        # Handle both single samples and batches
        if scene_features.dim() == 1:
            scene_features = scene_features.unsqueeze(0)
        
        # Extract shared features
        features = self.feature_extractor(scene_features.to(self.device))

        print("Extracted features shape:", features.shape)
        
        # Predict each theta parameter
        theta_dict = {}
        for param_name, head in self.theta_heads.items():
            # Use appropriate activation for each parameter
            if param_name == 'triplet_margin':
                # Margin should be positive and potentially > 1
                theta_dict[param_name] = torch.relu(head(features)).squeeze() + 0.1
            elif param_name == 'width_scale':
                # Width scale should be positive, typically 0.5-2.0
                theta_dict[param_name] = torch.sigmoid(head(features)).squeeze() * 1.5 + 0.5
            elif param_name == 'smoothing_factor':
                # Smoothing factor typically 1-20
                theta_dict[param_name] = torch.sigmoid(head(features)).squeeze() * 19 + 1
            else:
                # Others typically 0-1
                theta_dict[param_name] = torch.sigmoid(head(features)).squeeze()

        print("Predicted theta parameters:", theta_dict)
        
        loss_weights = {}
        weight_sum = sum(torch.abs(w) for w in self.loss_weights.values())
        for name, weight in self.loss_weights.items():
            loss_weights[f'weight_{name}'] = torch.abs(weight) / weight_sum

        print("Loss weights:", loss_weights)

        
        output_dict = {**theta_dict, **loss_weights}

        return output_dict


class FederatedMetaLearner:
    """
    Orchestrates federated learning across multiple camera clients with meta-learning.
    """
    def __init__(self, meta_model, device='cpu', visualize_lanes=True):
        self.meta_model = meta_model.to(device)
        self.device = device
        self.client_data_buffer = defaultdict(list)
        self.global_data_buffer = []
        self.training_history = []
        self.visualize_lanes = visualize_lanes

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
    
    def client_update(self, client_id, processed_data, geo_learning, trial_mode=True):
        """
        Perform local update for a single client (camera).
        
        Args:
            client_id: Camera location identifier
            processed_data: Preprocessed data from the camera
            geo_learning: GeometricalLearning instance
            trial_mode: Whether to run multiple trials or use predicted theta directly
        
        Returns:
            best_loss, best_theta, client_metrics
        """
        download_start = time.time()
        # scene_features = self.extract_scene_features(processed_data)
        scene_features = SceneFeatureExtractor.extract_features(processed_data)
        
        if trial_mode:
            # Training mode: try multiple theta configurations
            best_loss = float('inf')
            best_theta = None
            trial_results = []

            # First trial: Use meta-model prediction
            with torch.no_grad():
                predicted_theta = self.meta_model(scene_features)
                # Convert to simple dict with float values
                predicted_theta_values = {}
                for k, v in predicted_theta.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 0: # scalar
                            predicted_theta_values[k] = v.item()
                        else: # has dimensions
                            predicted_theta_values[k] = v.squeeze().item()
                    else:
                        predicted_theta_values[k] = float(v)
            download_end = time.time()

            download_time = download_end - download_start
            download_size_bytes = estimate_object_size_bytes(predicted_theta_values)
            
            # Update geo_learning theta - but geo_learning.theta is empty dict
            # So we need to initialize it with the predicted values
            for k, v in predicted_theta_values.items():
                geo_learning.theta[k] = torch.tensor(v)
            
            # Run geometric learning and measure upload
            upload_start = time.time()
            try:
                loss, metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
                
                trial_results.append({
                    'theta': predicted_theta_values,
                    'loss': loss,
                    'metrics': metrics
                })
                
                if loss < best_loss:
                    best_loss = loss
                    best_theta = predicted_theta_values
            except Exception as e:
                logger.error(f"Error in first trial for client {client_id}: {e}")
                # Return default values if geo_learning fails
                return 1.0, predicted_theta_values, {}
            upload_end = time.time()
            upload_time = upload_end - upload_start
            upload_size_bytes = estimate_object_size_bytes({
                "theta": predicted_theta_values,
                "metrics": metrics,
                "loss": loss
            })

            # Additional trials with perturbations
            for trial in range(2):
                perturbed_theta = {}
                for k, v in predicted_theta_values.items():
                    noise = torch.randn(1).item() * 0.1
                    if k == 'width_scale':
                        perturbed_theta[k] = max(0.5, min(2.0, v + noise))
                    elif k == 'smoothing_factor':
                        perturbed_theta[k] = max(1, min(20, v + noise * 10))
                    elif k == 'triplet_margin':
                        perturbed_theta[k] = max(0.1, min(2.0, v + noise))
                    else:
                        perturbed_theta[k] = max(0.1, min(1.0, v + noise))
                
                # Update geo_learning with perturbed theta
                for k, v in perturbed_theta.items():
                    geo_learning.theta[k] = torch.tensor(v)
                
                try:
                    loss, metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
                    
                    trial_results.append({
                        'theta': perturbed_theta,
                        'loss': loss,
                        'metrics': metrics
                    })
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_theta = perturbed_theta
                except Exception as e:
                    logger.error(f"Error in trial {trial} for client {client_id}: {e}")
                    continue

            # Compute BPS
            bps_upload = (upload_size_bytes * 8) / upload_time if upload_time > 0 else 0
            bps_download = (download_size_bytes * 8) / download_time if download_time > 0 else 0
            total_data_mb = (upload_size_bytes + download_size_bytes) / (1024 ** 2)
            metrics.update({
                'bps_upload': bps_upload,
                'bps_download': bps_download,
                'bps': (bps_upload + bps_download) / 2,
                'latency': upload_time + download_time,
                'data_size_mb': total_data_mb
            })

            logger.info(f"Client {client_id}: Upload = {upload_size_bytes} bytes, Download = {download_size_bytes} bytes, Latency = {metrics['latency']:.2f}s, BPS = {metrics['bps']:.2f} bps")
            
            # Store data for meta-model training
            self.client_data_buffer[client_id].append({
                'scene_features': scene_features.cpu(),
                'best_theta': best_theta,
                'best_loss': best_loss,
                'trial_results': trial_results
            })
            
        else:
            # Deployment mode: use predicted theta directly
            with torch.no_grad():
                predicted_theta = self.meta_model(scene_features)
                predicted_theta_values = {}
                for k, v in predicted_theta.items():
                    if isinstance(v, torch.Tensor):
                        predicted_theta_values[k] = v.item() if v.dim() == 0 else v.squeeze().item()
                    else:
                        predicted_theta_values[k] = float(v)
            
            for k, v in predicted_theta_values.items():
                geo_learning.theta[k] = torch.tensor(v)
                
            try:
                best_loss, metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
                best_theta = predicted_theta_values
            except Exception as e:
                logger.error(f"Error in deployment mode for client {client_id}: {e}")
                return 1.0, predicted_theta_values, {}
        
        return best_loss, best_theta, metrics
    
    def _run_geo_learning(self, geo_learning, processed_data, client_id):
        """Execute geometric learning and compute loss."""

        try:
            # Create osm_connection for this client if not exists
            if not hasattr(self, 'osm_connections'):
                self.osm_connections = {}
            
            if client_id not in self.osm_connections:
                self.osm_connections[client_id] = OSMConnection(geo_learning.args, geo_learning.filepath)
            
            osm_connection = self.osm_connections[client_id]
            
            # Run geometric learning
            traj_df, lane_boundaries_for_contour = geo_learning.run(
                c_epoch=processed_data.get('c_epoch', 0),
                g_epoch=processed_data.get('g_epoch', 0),
                traj_df=processed_data['gps_df'],
                camera_loc=client_id,
                trial='0',
                is_save=geo_learning.is_save
            )

            # Convert to pandas if still in Polars
            traj_df_pd = traj_df.to_pandas() if hasattr(traj_df, 'to_pandas') else traj_df
            # Filter out unassigned or invalid lane clusters
            traj_df_pd = traj_df_pd[traj_df_pd["clustered_id"] != -1]

            # Save trajectory data with lane correspondence
            # self._save_lane_detection_csv(traj_df_pd, lane_boundaries_for_contour, processed_data, client_id, geo_learning)

            # Extract detected centers and compute lane widths
            detected_center_list = []
            lane_width_list = []
            
            for cnts, boundaries in lane_boundaries_for_contour.items():
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

                # print(lane_shape)
                # print(f"Cluster to edge map: {cluster_to_edge_map}")

                # Visualization setup
                if geo_learning.is_save and hasattr(self, 'visualize_lanes') and self.visualize_lanes:
                    self._visualize_lane_detection(
                        traj_df_pd, 
                        lane_boundaries_for_contour, 
                        processed_data, 
                        osm_connection, 
                        client_id,
                        geo_learning,
                        lane_shape,
                        cluster_to_edge_map
                    )
                
                # Get sumo_node data from processed_data if available
                sumo_node, _ = processed_data.get('sumo_graph', ([], []))
                
                sumo_center_tensor = []
                for group in sumo_node:
                    for line in group:
                        line_tensor = torch.tensor(np.array(line), dtype=torch.float32)
                        # Convert each line (list of arrays) to a 2D list
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
                    'sumo_lanes': len(sumo_center_tensor) if sumo_center_tensor[0].shape[0] > 0 else 0
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

    def _visualize_lane_detection(
            self, traj_df_pd, lane_boundaries_for_contour, 
            processed_data, osm_connection, client_id, geo_learning,
            sumo_lane_shape, cluster_to_edge_map
        ):
        """Visualize lane detection results on the camera image."""
        
        fig_filepath = Path(geo_learning.filepath, client_id, "figures")
        csv_filepath = Path(geo_learning.filepath, client_id)

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
                # TODO: Handle image loading errors. This happened because of the threading timing
                logger.error(f"Failed to load image from {img_path}. Using blank background.")

        traj_df_pd["lane_id"] = traj_df_pd["lane_id"].astype("object")

        # print(client_id)
        # Plot lane detection results
        for cnt_id, boundaries in lane_boundaries_for_contour.items():
            for lane_count, (lane_id, data) in enumerate(boundaries.items()):
                color = self.colors[lane_count]
                
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
                            logger.error(f"Error at plotting lane_df for lane_id {lane_id} with color {color}")
                    
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
                                  c=color, label=f"Traj Lane {lane_id}")
                    
                    # Plot lane boundaries in GPS
                    ax2.plot(data["center"][:, 0], data["center"][:, 1], 
                            color=color, linewidth=2.5, label=f"Center Lane {lane_id}")
                    ax2.plot(data["left"][:, 0], data["left"][:, 1], 
                            color=color, linewidth=2.0, linestyle='--', alpha=0.7)
                    ax2.plot(data["right"][:, 0], data["right"][:, 1], 
                            color=color, linewidth=2.0, linestyle='--', alpha=0.7)

                sumo_lane_id = list(sumo_lane_shape.keys())[cluster_to_edge_map[lane_id][1]]
                # This cnt should be valid contour id (some of them start from 2 and 3)
                # print(cnt_id, lane_id, sumo_lane_id)
                traj_df_pd.loc[
                    (traj_df_pd["contour_id"] == cnt_id) & (traj_df_pd["clustered_id"] == lane_id),
                    "lane_id"
                ] = sumo_lane_id

        if g_epoch == 11:
            traj_df_pd.to_csv(Path(csv_filepath, f"federated_trajectory_clustering.csv"))

        # ax1.set_title(f"Lane Detection Results - Client {client_id} - Epoch {c_epoch}")
        # Deduplicate legend entries
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # overwrites duplicates
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        ax1.set_axis_off()

        # ax2.set_title(f"Lane Boundaries - Client {client_id} - Epoch {c_epoch}")
        # Deduplicate legend entries
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # overwrites duplicates
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        ax2.set_axis_off()
        ax2.set_xlim(0, 1920)
        ax2.set_ylim(1080, 0) 

        # Save the visualization
        fig1_path = Path(fig_filepath, f"{g_epoch}_federated_trajectory_clustering_{c_epoch}.png")
        fig2_path = Path(fig_filepath, f"{g_epoch}_federated_lane_detection_{c_epoch}.png")
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        logger.info(f"Lane detection visualization saved to {fig1_path} and {fig2_path}")

    def aggregate_client_updates(self, client_results):
        """
        Aggregate results from multiple clients for federated learning.
        We take FedAvg approach.
        """
        aggregated_metrics = defaultdict(list)
        loss_component_keys = ['l_lane_count', 'l_cons', 'l_trip', 'l_geo']
        client_thetas = []
        
        for client_id, (loss, theta, metrics) in client_results.items():
            aggregated_metrics['losses'].append(loss)
            client_thetas.append(theta)
            for k, v in metrics.items():
                aggregated_metrics[k].append(v)
        
        # Compute statistics
        avg_loss = np.mean(aggregated_metrics['losses'])
        std_loss = np.std(aggregated_metrics['losses'])

        # Compute average of each metric starting with 'l_'
        avg_metrics = {
            k: float(np.mean(v_list))
            for k, v_list in aggregated_metrics.items()
            if k.startswith("l_")
        }

        # Average theta parameters
        avg_theta = {}
        for key in client_thetas[0]:
            stacked = torch.stack([torch.tensor(theta[key], dtype=torch.float32) for theta in client_thetas])
            avg_theta[key] = stacked.mean(dim=0)
        
        # Log individual client metrics to MLflow
        try:
            for client_id, (loss, theta, metrics) in client_results.items():
                mlflow.log_metric(f"Federated/Loss_{client_id}", loss, step=self.round_counter)
                
                # Log detailed loss components per client
                for key in loss_component_keys:
                    if key in metrics:
                        mlflow.log_metric(f"Federated/{key}_{client_id}", metrics[key], step=self.round_counter)
                        
        except ImportError:
            logger.warning("MLflow not available for detailed logging")

        return {
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'client_count': len(client_results),
            'aggregated_metrics': dict(aggregated_metrics),
            'avg_aggregated_metrics': dict(avg_metrics),
            'avg_theta': avg_theta,
            'strategy': 'federated',
        }
    
    def train_meta_model(self, num_epochs=10, lr=1e-3):
        """
        Train the meta-model using collected client data.
        """
        # Prepare training data
        training_data = []
        for client_id, buffer in self.client_data_buffer.items():
            for entry in buffer:
                training_data.append({
                    'features': entry['scene_features'],
                    'target_theta': entry['best_theta'],
                    'target_loss': entry['best_loss'],
                    'trial_results': entry['trial_results']
                })
        
        if len(training_data) < 10:
            logger.warning("Insufficient data for meta-model training")
            return
        
        # Create optimizer
        optimizer = optim.Adam(self.meta_model.parameters(), lr=lr)
        
        # Training loop
        self.meta_model.train()
        epoch_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_losses = []

            np.random.shuffle(training_data)
            for data in training_data:
                optimizer.zero_grad()
                
                # Predict theta
                predicted_theta = self.meta_model(data['features'].to(self.device))
                
                # Compute loss (MSE between predicted and best theta)
                loss = 0
                theta_losses = {}
                for param_name, pred_value in predicted_theta.items():
                    if param_name.startswith('weight_'):
                        continue # Skip weight parameters in theta loss

                    if param_name in data['target_theta']:
                        target_value = torch.tensor(data['target_theta'][param_name], dtype=torch.float32).to(self.device)
                        param_loss = nn.MSELoss()(pred_value, target_value)
                        loss += param_loss
                        theta_losses[param_name] = param_loss.item()

                # Add loss weight optimization based on trial results
                if data['trial_results']:
                    # Get the best trial (lowest loss)
                    best_trial = min(data['trial_results'], key=lambda x: x['loss'])
                    
                    if 'metrics' in best_trial:
                        metrics = best_trial['metrics']
                        # Normalize loss components
                        total_component_loss = 0
                        component_losses = {}
                        
                        for comp in ['lane_count', 'cons', 'trip', 'geo']:
                            if f'l_{comp}' in metrics:
                                component_losses[comp] = metrics[f'l_{comp}']
                                total_component_loss += metrics[f'l_{comp}']
                        
                        # Encourage weights to be proportional to inverse of component losses
                        if total_component_loss > 0:
                            for comp, comp_loss in component_losses.items():
                                weight_key = f'weight_{comp.replace("cons", "consistency").replace("trip", "triplet")}'
                                if weight_key in predicted_theta:
                                    # Target weight inversely proportional to loss contribution
                                    target_weight = 1.0 / (comp_loss + 0.1) # Add small constant to avoid division by zero
                                    weight_penalty = nn.MSELoss()(predicted_theta[weight_key].to(self.device), 
                                                                torch.tensor(target_weight).to(self.device))
                                    loss += 0.1 * weight_penalty # Small coefficient

                # Add loss prediction objective
                target_detection_loss = torch.tensor(data['target_loss']).to(self.device)
                loss_weight = 0.1
                
                # Simple loss predictor from theta values
                theta_vector = torch.stack([predicted_theta[k].to(self.device) for k in sorted(predicted_theta.keys()) 
                                        if not k.startswith('weight_')])
                predicted_loss = torch.sum(theta_vector) * 0.5
                loss_prediction_error = nn.MSELoss()(predicted_loss, target_detection_loss)
                loss += loss_weight * loss_prediction_error
                
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(training_data)
            epoch_losses.append(avg_epoch_loss)

            # Log detailed information every 5 epochs
            if epoch % 5 == 0:
                logger.info(f"Meta-model training epoch {epoch}/{num_epochs}: "
                        f"Loss = {avg_epoch_loss:.4f}, "
                        f"Min batch loss = {min(batch_losses):.4f}, "
                        f"Max batch loss = {max(batch_losses):.4f}")
            
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_epoch_loss,
                'theta_losses': theta_losses,
                'num_samples': len(training_data)
            })
        # Log final statistics
        logger.info(f"Meta-model training completed. "
                f"Initial loss: {epoch_losses[0]:.4f}, "
                f"Final loss: {epoch_losses[-1]:.4f}, "
                f"Improvement: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.1f}%")
    
    def save_model(self, path):
        """Save meta-model and training history."""
        save_dict = {
            'model_state_dict': self.meta_model.state_dict(),
            'training_history': self.training_history,
            'client_data_buffer': dict(self.client_data_buffer)
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load meta-model and training history."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.client_data_buffer = defaultdict(list, checkpoint.get('client_data_buffer', {}))
        logger.info(f"Model loaded from {path}")


class FederatedLaneDetectionSystem:
    """
    Main system orchestrating federated meta-learning for lane detection.
    """
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        
        # Initialize meta-model
        self.meta_model = MetaMLModel(
            feature_dim=5,
            hidden_dim=128,
            num_theta_params=5
        ).to(self.device)
        
        # Initialize federated learner
        self.fed_learner = FederatedMetaLearner(self.meta_model, device)
        
        # Training settings
        self.training_mode = True
        self.round_counter = 0
        self.client_selection_ratio = 0.8 # Select 80% of clients per round
        
    def select_clients(self, available_clients, ratio=0.8):
        """Randomly select a subset of clients for each round."""
        num_selected = max(1, int(len(available_clients) * ratio))
        selected = np.random.choice(available_clients, num_selected, replace=False)
        return selected.tolist()
    
    def switch_to_deployment(self):
        """Switch from training to deployment mode."""
        self.training_mode = False
        self.meta_model.eval()
        logger.info("Switched to deployment mode")
    
    def save_checkpoint(self, path):
        """Save complete system state."""
        checkpoint_path = Path(path) / f"federated_checkpoint_round_{self.round_counter}.pth"
        self.fed_learner.save_model(checkpoint_path)
        
        # Save additional metadata
        metadata = {
            'round_counter': self.round_counter,
            'training_mode': self.training_mode,
            'client_selection_ratio': self.client_selection_ratio
        }
        
        with open(Path(path) / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def load_checkpoint(self, checkpoint_path, metadata_path):
        """Load complete system state."""
        self.fed_learner.load_model(checkpoint_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.round_counter = metadata['round_counter']
        self.training_mode = metadata['training_mode']
        self.client_selection_ratio = metadata['client_selection_ratio']
