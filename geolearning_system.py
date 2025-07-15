import time
import json
import sys
import torch
import torch.nn as nn
import numpy as np
import mlflow
import logging
from collections import defaultdict
from pathlib import Path

from utils import compute_loss_for_baseline

from LaneDetection.lane_detection.utils import SceneFeatureExtractor
from LaneDetection.lane_detection.meta_federated_lane_detection import (
    MetaMLModel, FederatedMetaLearner
)

logger = logging.getLogger(__name__)


class GeoLearningSystem:
    """
    GeoLearning system that handles all learning strategies through a single interface.
    """
    def __init__(self, args, device='cpu', strategy='baseline'):
        self.args = args
        self.device = device
        self.strategy = strategy
        self.training_mode = True
        self.round_counter = 0
        
        # Initialize components based on strategy
        self._initialize_strategy_components()
        
        # Shared components for all strategies
        self.client_data_buffer = defaultdict(list)
        self.training_history = []
        
    def _initialize_strategy_components(self):
        """Initialize components specific to each strategy"""
        if self.strategy == 'baseline':
            self._init_baseline()
        elif self.strategy == 'meta':
            self._init_meta_learning()
        elif self.strategy == 'federated':
            self._init_federated()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _init_baseline(self):
        """Initialize baseline strategy with fixed parameters"""
        self.fixed_theta = {
            'angle_penalty': torch.tensor(0.5),
            'width_scale': torch.tensor(1.0),
            'consistency_weight': torch.tensor(0.5),
            'triplet_margin': torch.tensor(0.8),
            'smoothing_factor': torch.tensor(10.0),
            'edge_trim_ratio': torch.tensor(0.1),
            'weight_lane_count': torch.tensor(1.0),
            'weight_consistency': torch.tensor(1.0),
            'weight_triplet': torch.tensor(1.0),
            'weight_geometry': torch.tensor(1.0),
        }
        logger.info("Initialized baseline strategy with fixed theta")
    
    def _init_meta_learning(self):
        """Initialize meta-learning strategy"""
        self.meta_models = {}
        self.optimizers = {}
        logger.info("Initialized meta-learning strategy")
    
    def _init_federated(self):
        """Initialize federated meta-learning strategy"""
        self.meta_model = MetaMLModel(
            feature_dim=5,
            hidden_dim=128,
            num_theta_params=5
        ).to(self.device)
        
        self.fed_learner = FederatedMetaLearner(self.meta_model, self.device)
        self.client_selection_ratio = 0.8
        
        logger.info("Initialized federated strategy")
    
    def setup_clients(self, camera_loc_list):
        """Setup client-specific components after knowing camera locations"""
        self.camera_loc_list = camera_loc_list
        
        if self.strategy == 'meta':
            self._setup_meta_clients()
    
    def _setup_meta_clients(self):
        """Setup individual meta-models for meta-learning strategy"""
        for camera_loc in self.camera_loc_list:
            self.meta_models[camera_loc] = MetaMLModel(
                feature_dim=5, hidden_dim=128, num_theta_params=5
            ).to(self.device)
            self.optimizers[camera_loc] = torch.optim.Adam(
                self.meta_models[camera_loc].parameters(), lr=1e-3
            )
        
        logger.info(f"Setup meta-models for {len(self.camera_loc_list)} clients")
    
    def client_update(self, client_id, processed_data, geo_learning):
        """
        Client update method that delegates to strategy-specific implementation
        """
        if self.strategy == 'baseline':
            return self._baseline_client_update(client_id, processed_data, geo_learning)
        elif self.strategy == 'meta':
            return self._meta_client_update(client_id, processed_data, geo_learning)
        elif self.strategy == 'federated':
            self.fed_learner.round_counter = processed_data.get('g_epoch', 0)
            return self._federated_client_update(client_id, processed_data, geo_learning)
    
    def _baseline_client_update(self, client_id, processed_data, geo_learning):
        """Baseline client update with fixed theta"""
        start_time = time.time()
        
        # Set fixed theta parameters
        for k, v in self.fixed_theta.items():
            geo_learning.theta[k] = v
        
        # Run geometric learning
        loss, metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
        
        # Calculate communication metrics
        duration = time.time() - start_time
        # TODO: Implement size estimation for raw video data. Ignore the current BPS and simply find it in the folder.
        upload_size_bytes = self._estimate_data_size(processed_data['gps_df'])
        download_size_bytes = 0 # No model download for baseline
        
        metrics.update(self._calculate_communication_metrics(
            upload_size_bytes, download_size_bytes, duration
        ))
        
        return loss, self.fixed_theta, metrics
    
    def _meta_client_update(self, client_id, processed_data, geo_learning):
        """Meta-learning client update with individual models"""
        start_time = time.time()

        # Extract scene features and predict theta
        scene_features = SceneFeatureExtractor.extract_features(processed_data).to(self.device)
        
        self.meta_models[client_id].eval()
        with torch.no_grad():
            predicted_theta = self.meta_models[client_id](scene_features)
            predicted_theta_values = {
                k: v.item() if v.dim() == 0 else v.squeeze().item() 
                for k, v in predicted_theta.items()
            }
        
        # Update geo_learning with predicted theta
        for k, v in predicted_theta_values.items():
            geo_learning.theta[k] = torch.tensor(v)

        # Run geometric learning
        loss, metrics = self._run_geo_learning(geo_learning, processed_data, client_id)

        # Store data for meta-model training
        self.client_data_buffer[client_id].append({
            'scene_features': scene_features.cpu(),
            'best_theta': predicted_theta_values,
            'best_loss': loss,
            'metrics': metrics
        })
        
        # Calculate communication metrics
        duration = time.time() - start_time
        # TODO: Implement size estimation for raw video data. Ignore the current BPS and simply find it in the folder.
        upload_size_bytes = self._estimate_data_size(processed_data['gps_df'])
        download_size_bytes = 0 # No global model download in pure meta-learning
        
        metrics.update(self._calculate_communication_metrics(
            upload_size_bytes, download_size_bytes, duration
        ))
        
        return loss, predicted_theta_values, metrics
    
    def _federated_client_update(self, client_id, processed_data, geo_learning):
        """Federated client update using federated learner"""
        return self.fed_learner.client_update(
            client_id=client_id,
            processed_data=processed_data,
            geo_learning=geo_learning,
            trial_mode=self.training_mode
        )
    
    def aggregate_client_updates(self, client_results):
        """
        Aggregation method that delegates to strategy-specific implementation
        """
        if self.strategy == 'baseline':
            return self._baseline_aggregate(client_results)
        elif self.strategy == 'meta':
            return self._meta_aggregate(client_results)
        elif self.strategy == 'federated':
            return self.fed_learner.aggregate_client_updates(client_results)
    
    def _baseline_aggregate(self, client_results):
        """Simple averaging for baseline"""
        losses = [result[0] for result in client_results.values()]
        metrics_list = [result[2] for result in client_results.values()]
        
        # Aggregate detailed loss components
        aggregated_loss_components = {}
        loss_component_keys = ['l_lane_count', 'l_cons', 'l_trip', 'l_geo']
        
        for key in loss_component_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated_loss_components[f'avg_{key}'] = np.mean(values)
                aggregated_loss_components[f'std_{key}'] = np.std(values)

        # Log individual client metrics to MLflow
        try:
            for client_id, (loss, theta, metrics) in client_results.items():
                mlflow.log_metric(f"Baseline/Loss_{client_id}", loss, step=self.round_counter)
                
                # Log detailed loss components per client
                for key in loss_component_keys:
                    if key in metrics:
                        mlflow.log_metric(f"Baseline/{key}_{client_id}", metrics[key], step=self.round_counter)
                        
        except ImportError:
            logger.warning("MLflow not available for detailed logging")
        
        return {
            'avg_loss': np.mean([l for l in losses if l != float('inf')]),
            'std_loss': np.std([l for l in losses if l != float('inf')]),
            'client_count': len(client_results),
            'avg_bps': 0,
            'total_bps': 0,
            'strategy': 'baseline',
            **aggregated_loss_components
        }
    
    def _meta_aggregate(self, client_results):
        """Aggregation for meta-learning (similar to baseline but with different metrics)"""
        losses = [result[0] for result in client_results.values()]
        metrics_list = [result[2] for result in client_results.values()]

        # Aggregate detailed loss components
        aggregated_loss_components = {}
        loss_component_keys = ['l_lane_count', 'l_cons', 'l_trip', 'l_geo']
        
        for key in loss_component_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated_loss_components[f'avg_{key}'] = np.mean(values)
                aggregated_loss_components[f'std_{key}'] = np.std(values)
        
        # Log individual client metrics to MLflow
        try:
            for client_id, (loss, theta, metrics) in client_results.items():
                mlflow.log_metric(f"Meta/Loss_{client_id}", loss, step=self.round_counter)
                mlflow.log_metric(f"Meta/BPS_{client_id}", metrics.get('bps', 0), step=self.round_counter)
                
                # Log detailed loss components per client
                for key in loss_component_keys:
                    if key in metrics:
                        mlflow.log_metric(f"Meta/{key}_{client_id}", metrics[key], step=self.round_counter)
                
                # Log predicted theta parameters for meta-learning
                if isinstance(theta, dict):
                    for param_name, param_value in theta.items():
                        if not param_name.startswith('weight_'):  # Skip weight parameters
                            mlflow.log_metric(f"Meta/Theta_{param_name}_{client_id}", 
                                            float(param_value), step=self.round_counter)
                        
        except ImportError:
            logger.warning("MLflow not available for detailed logging")
        
        return {
            'avg_loss': np.mean([l for l in losses if l != float('inf')]),
            'std_loss': np.std([l for l in losses if l != float('inf')]),
            'client_count': len(client_results),
            'avg_bps': 0,
            'total_bps': 0,
            'strategy': 'meta_learning',
            **aggregated_loss_components
        }
    
    def train_models(self, epoch):
        """
        Train models based on strategy
        """
        if self.strategy == 'meta' and self.training_mode:
            self._train_meta_models()
        elif self.strategy == 'federated' and self.training_mode:
            logger.info("Training federated meta-model...")
            self.fed_learner.train_meta_model(num_epochs=20)
    
    def _train_meta_models(self):
        """Train individual meta-models for meta-learning strategy"""
        selected_clients = ['US12_Todd', 'US12_Monona', 'US12_Yahara']
        
        for client_id in selected_clients:
            if len(self.client_data_buffer[client_id]) >= 10:
                self._train_individual_meta_model(client_id)
    
    def _train_individual_meta_model(self, client_id, num_epochs=10):
        """Train meta-model for a single client"""
        logger.info(f"Training meta-model for client {client_id}")
        
        model = self.meta_models[client_id]
        optimizer = self.optimizers[client_id]
        data_buffer = self.client_data_buffer[client_id]
        
        model.train()
        epoch_losses = []
        batch_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            np.random.shuffle(data_buffer)
            
            for data in data_buffer:
                optimizer.zero_grad()
                
                predicted_theta = model(data['scene_features'].to(self.device))
                
                # Compute loss
                loss = 0
                for param_name, pred_value in predicted_theta.items():
                    if param_name.startswith('weight_'):
                        continue
                    
                    if param_name in data['best_theta']:
                        target_value = torch.tensor(
                            data['best_theta'][param_name], dtype=torch.float32
                        ).to(self.device)
                        loss += nn.MSELoss()(pred_value, target_value)
                
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(data_buffer)
            epoch_losses.append(avg_epoch_loss)

            # Log detailed information every 5 epochs
            if epoch % 5 == 0:
                logger.info(f"Meta-model training epoch {epoch}/{num_epochs}: "
                        f"Loss = {avg_epoch_loss:.4f}, "
                        f"Min batch loss = {min(batch_losses):.4f}, "
                        f"Max batch loss = {max(batch_losses):.4f}")
            
        logger.info(f"Meta-model training completed. "
                f"Initial loss: {epoch_losses[0]:.4f}, "
                f"Final loss: {epoch_losses[-1]:.4f}, "
                f"Improvement: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.1f}%")
    
    def select_clients(self, available_clients):
        """Select clients based on strategy and training mode"""
        if self.strategy == 'federated':
            if self.training_mode:
                return ['US12_Todd', 'US12_Monona', 'US12_Yahara']
                # return ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_Stoughton', 
                #        'US12_Whitney', 'US12_JohnNolen', 'US12_Park']
            else:
                return ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_Park']
        elif self.strategy == 'meta':
            if self.training_mode:
                return ['US12_Todd', 'US12_Monona', 'US12_Yahara']
            else:
                return ['US12_Todd', 'US12_Monona', 'US12_Yahara', 'US12_Park']
        else: # Baseline
            return available_clients
    
    def switch_to_deployment(self):
        """Switch to deployment mode"""
        self.training_mode = False
        if self.strategy == 'federated':
            self.fed_learner.meta_model.eval()
        elif self.strategy == 'meta':
            for model in self.meta_models.values():
                model.eval()
        
        logger.info(f"Switched {self.strategy} strategy to deployment mode")
    
    def save_checkpoint(self, path):
        """Save strategy-specific checkpoints"""
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(exist_ok=True)
        
        if self.strategy == 'federated':
            self.fed_learner.save_model(checkpoint_dir / 'federated_model.pth')
        elif self.strategy == 'meta':
            for client_id, model in self.meta_models.items():
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizers[client_id].state_dict(),
                    'client_buffer': self.client_data_buffer[client_id]
                }, checkpoint_dir / f'meta_model_{client_id}.pth')
        
        # Save common metadata
        metadata = {
            'strategy': self.strategy,
            'round_counter': self.round_counter,
            'training_mode': self.training_mode
        }
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved {self.strategy} checkpoint to {checkpoint_dir}")
    
    def _run_geo_learning(self, geo_learning, processed_data, client_id):
        """Shared method to run geometric learning and compute loss"""

        try:
            # Run geometric learning
            traj_df, lane_boundaries_for_contour = geo_learning.run(
                c_epoch=processed_data.get('c_epoch', 0),
                g_epoch=processed_data.get('g_epoch', 0),
                traj_df=processed_data['gps_df'],
                camera_loc=client_id,
                trial='0',
                is_save=geo_learning.is_save
            )

            # Extract detected centers and compute lane widths
            detected_center_list = []
            
            for cnts, boundaries in lane_boundaries_for_contour.items():
                for lane_id, data in boundaries.items():

                    detected_center_list.append(data["center"])

            if len(detected_center_list) > 0:
                loss, metrics = compute_loss_for_baseline(
                    geo_learning, traj_df, lane_boundaries_for_contour, 
                    processed_data, client_id
                )
                
                return loss, metrics
            else:
                # No lanes detected
                logger.warning(f"No lanes detected for client {client_id}")
                return float('inf'), {'lane_count': 0, 'error': 'No lanes detected'}
            
        except Exception as e:
            logger.error(f"Error in geo_learning for client {client_id}: {e}")
            return float('inf'), {'error': str(e)}
    
    def _estimate_data_size(self, data):
        """Estimate data size in bytes (for dataframes for now)"""
        # TODO: Implement size estimation for raw video data for baseline and meta-learning
        return sys.getsizeof(data)
    
    def _calculate_communication_metrics(self, upload_bytes, download_bytes, duration):
        """Calculate BPS and other communication metrics"""
        bps_upload = upload_bytes * 8 / duration if duration > 0 else 0
        bps_download = download_bytes * 8 / duration if duration > 0 else 0
        
        return {
            'bps_upload': bps_upload,
            'bps_download': bps_download,
            'bps': bps_upload + bps_download,
            'total_bits': (upload_bytes + download_bytes) * 8,
            'latency': duration,
            'data_size_mb': (upload_bytes + download_bytes) / 1e6
        }