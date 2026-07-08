import math
import cv2
import yaml
import torch
import numpy as np
import polars as pl
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import logging

# Scene feature vector length. The last three are relative peak-salience
# statistics added so the meta-model can see the structure it thresholds.
SCENE_FEATURE_DIM = 8

logger = logging.getLogger(__name__)


def create_binary_image_utils(learning_cycle, vehicles, frame, filepath, is_save):
    """ create a binary image based on the positions of detected vehicles in the whole frame
    Args:
        learning_cycle: the index of lane learning time, ex: if learning_cycle=2 means it is the second time learning
        vehicles: accumulated detected vehicles from the whole frame, the format of each item inside it is [x,y,w,h,fid,c]
        frame: current input video frame
        filepath: saving folder path
    Returns: 
        binary_image, remain_vehicles
            binary_image is the generated 2D binary image and remain_vehicles are the detected vehicle
            data inside the heatmap_area whose value is 1.
    """
    binary_image = np.zeros_like(frame[:, :, 0]).astype(np.float32)
    conf = [veh[-1] for veh in vehicles]

    p1 = np.percentile(conf, 50) if conf else 0.5

    for veh in vehicles:
        # if vehile confidence is higher than the 25th percentile, then it is a detected vehicle
        if veh[-1] > p1:
            x, y, w, h, fid, confidence = veh
            center = (int(x), int(y))
            axes = (int(w // 3.5), int(h // 3.5))
            # axes = (int(w // 8), int(h // 8))
            cv2.ellipse(binary_image, center, axes, angle=0, startAngle=0, endAngle=360, color=(1), thickness=-1)
        
    binary_image = (binary_image * 255).astype(np.uint8)
    if is_save:
        cv2.imwrite(Path(filepath, f"{learning_cycle}_binary_heatmap_area.png"), binary_image)
    return binary_image, vehicles

def gps_point_distance_m(lat1, lon1, lat2, lon2):
    avg_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111_320 * math.cos(avg_lat)
    dy = (lat2 - lat1) * 111_320
    return math.sqrt(dx**2 + dy**2)

def compute_lane_length_from_gps(left, right):
    """
    left, right: np.ndarray of shape (N, 2), where columns are [lon, lat]
    """
    assert left.shape == right.shape
    lengths = []
    for (lat1, lon1), (lat2, lon2) in zip(left, right):
        length = gps_point_distance_m(lat1, lon1, lat2, lon2)
        lengths.append(length)
    lengths = np.array(lengths)
    return lengths, lengths.mean()

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


class FederatedConfig:
    """Configuration for federated learning system."""
    """
    Utility functions and configuration for federated meta-learning lane detection.
    """
    
    def __init__(self, config_path=None):
        # Default configuration
        self.config = {
            'meta_model': {
                'feature_dim': 5,
                'hidden_dim': 128,
                'num_theta_params': 5,
                'dropout_rate': 0.2
            },
            'federated': {
                'num_rounds': 100,
                'clients_per_round_ratio': 0.8,
                'min_clients_per_round': 2,
                'aggregation_strategy': 'average',  # 'average', 'weighted', 'median'
                'client_sampling': 'random',  # 'random', 'importance', 'round_robin'
            },
            'training': {
                'meta_epochs_per_round': 10,
                'meta_learning_rate': 1e-3,
                'meta_batch_size': 32,
                'trial_perturbation_std': 0.1,
                'num_trials_per_client': 3,
                'early_stopping_patience': 20,
                'convergence_threshold': 0.01
            },
            'theta_params': {
                'angle_penalty': {'min': 0.1, 'max': 1.0, 'default': 0.5},
                'width_scale': {'min': 0.5, 'max': 2.0, 'default': 1.0},
                'consistency_weight': {'min': 0.1, 'max': 1.0, 'default': 0.5},
                'triplet_margin': {'min': 0.1, 'max': 2.0, 'default': 0.8},
                'smoothing_factor': {'min': 1, 'max': 20, 'default': 10}
            },
            'privacy': {
                'differential_privacy': False,
                'noise_multiplier': 1.0,
                'max_grad_norm': 1.0,
                'secure_aggregation': False
            },
            'communication': {
                'compression': True,
                'compression_ratio': 0.1,
                'async_updates': False,
                'max_staleness': 5
            }
        }
        
        # Load from file if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Update default config with loaded values
        self._deep_update(self.config, loaded_config)
    
    def save_config(self, save_path):
        """Save configuration to YAML file."""
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'meta_model.feature_dim')."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class ClientSelector:
    """Strategies for selecting clients in federated rounds."""
    
    def __init__(self, strategy='random'):
        self.strategy = strategy
        self.client_importance = {}
        self.round_robin_index = 0
        
    def select_clients(self, available_clients, num_to_select, client_metrics=None):
        """Select clients based on strategy."""
        if self.strategy == 'random':
            return self._random_selection(available_clients, num_to_select)
        elif self.strategy == 'importance':
            return self._importance_selection(available_clients, num_to_select, client_metrics)
        elif self.strategy == 'round_robin':
            return self._round_robin_selection(available_clients, num_to_select)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _random_selection(self, clients, n):
        """Random client selection."""
        n = min(n, len(clients))
        return np.random.choice(clients, n, replace=False).tolist()
    
    def _importance_selection(self, clients, n, metrics):
        """Select clients based on importance scores."""
        if not metrics:
            return self._random_selection(clients, n)
        
        # Calculate importance scores based on metrics
        scores = []
        for client in clients:
            if client in metrics:
                # Higher score for clients with higher loss variance (more to learn)
                loss_variance = metrics[client].get('loss_variance', 0)
                data_size = metrics[client].get('data_size', 1)
                score = loss_variance * np.log(data_size + 1)
                scores.append(score)
            else:
                scores.append(0)
        
        # Normalize scores to probabilities
        scores = np.array(scores)
        if scores.sum() > 0:
            probs = scores / scores.sum()
        else:
            probs = np.ones(len(clients)) / len(clients)
        
        n = min(n, len(clients))
        selected = np.random.choice(clients, n, replace=False, p=probs)
        return selected.tolist()
    
    def _round_robin_selection(self, clients, n):
        """Round-robin client selection."""
        n = min(n, len(clients))
        selected = []
        
        for _ in range(n):
            selected.append(clients[self.round_robin_index % len(clients)])
            self.round_robin_index += 1
        
        return selected


class ModelCompressor:
    """Compress model updates for efficient communication."""
    
    @staticmethod
    def compress_gradients(gradients, compression_ratio=0.1):
        """Compress gradients using top-k sparsification."""
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                compressed[name] = None
                continue
            
            # Flatten gradient
            flat_grad = grad.flatten()
            num_elements = flat_grad.numel()
            
            # Keep only top-k elements
            k = max(1, int(num_elements * compression_ratio))
            topk_values, topk_indices = torch.topk(flat_grad.abs(), k)
            
            # Create sparse representation
            compressed[name] = {
                'shape': grad.shape,
                'indices': topk_indices,
                'values': flat_grad[topk_indices],
                'sign': torch.sign(flat_grad[topk_indices])
            }
        
        return compressed
    
    @staticmethod
    def decompress_gradients(compressed_gradients):
        """Decompress gradients from sparse representation."""
        decompressed = {}
        for name, compressed in compressed_gradients.items():
            if compressed is None:
                decompressed[name] = None
                continue
            
            # Reconstruct full gradient
            grad = torch.zeros(compressed['shape']).flatten()
            grad[compressed['indices']] = compressed['values'].abs() * compressed['sign']
            decompressed[name] = grad.reshape(compressed['shape'])
        
        return decompressed


class PrivacyManager:
    """Manage differential privacy for federated learning."""
    
    def __init__(self, noise_multiplier=1.0, max_grad_norm=1.0):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def add_noise_to_gradients(self, gradients, num_samples):
        """Add Gaussian noise for differential privacy."""
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            if grad is None:
                noisy_gradients[name] = None
                continue
            
            # Clip gradient norm
            grad_norm = grad.norm()
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)
            
            # Add noise
            noise_std = self.noise_multiplier * self.max_grad_norm / num_samples
            noise = torch.randn_like(grad) * noise_std
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients


class FederatedMetrics:
    """Track and analyze federated learning metrics."""
    
    def __init__(self):
        self.round_metrics = []
        self.client_metrics = {}
        
    def add_round_metrics(self, round_num, metrics):
        """Add metrics for a federated round."""
        metrics['round'] = round_num
        self.round_metrics.append(metrics)
    
    def add_client_metrics(self, client_id, round_num, metrics):
        """Add metrics for a specific client."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        
        metrics['round'] = round_num
        self.client_metrics[client_id].append(metrics)
    
    def get_client_statistics(self, client_id):
        """Get statistics for a specific client."""
        if client_id not in self.client_metrics:
            return None
        
        client_data = self.client_metrics[client_id]
        losses = [m['loss'] for m in client_data if 'loss' in m]
        
        return {
            'num_rounds': len(client_data),
            'avg_loss': np.mean(losses) if losses else None,
            'loss_variance': np.var(losses) if losses else None,
            'last_round': client_data[-1]['round'] if client_data else None
        }
    
    def plot_training_progress(self, save_path):
        """Plot federated training progress."""
        if not self.round_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Average loss over rounds
        rounds = [m['round'] for m in self.round_metrics]
        avg_losses = [m['avg_loss'] for m in self.round_metrics]
        std_losses = [m.get('std_loss', 0) for m in self.round_metrics]
        
        ax = axes[0, 0]
        ax.plot(rounds, avg_losses, 'b-', label='Average Loss')
        ax.fill_between(rounds,
                       np.array(avg_losses) - np.array(std_losses),
                       np.array(avg_losses) + np.array(std_losses),
                       alpha=0.3)
        ax.set_xlabel('Round')
        ax.set_ylabel('Loss')
        ax.set_title('Average Loss Across Clients')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Number of participating clients
        num_clients = [m.get('client_count', 0) for m in self.round_metrics]
        ax = axes[0, 1]
        ax.plot(rounds, num_clients, 'g-')
        ax.set_xlabel('Round')
        ax.set_ylabel('Number of Clients')
        ax.set_title('Client Participation')
        ax.grid(True)
        
        # Plot 3: Client loss distribution
        ax = axes[1, 0]
        client_losses = {}
        for client_id, metrics_list in self.client_metrics.items():
            losses = [m['loss'] for m in metrics_list if 'loss' in m]
            if losses:
                client_losses[client_id] = losses
        
        if client_losses:
            ax.boxplot(client_losses.values(), labels=client_losses.keys())
            ax.set_xlabel('Client ID')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Distribution by Client')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Convergence rate
        ax = axes[1, 1]
        if len(avg_losses) > 1:
            convergence_rates = np.diff(avg_losses)
            ax.plot(rounds[1:], convergence_rates, 'r-')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Round')
            ax.set_ylabel('Loss Change')
            ax.set_title('Convergence Rate')
            ax.grid(True)
        
        plt.savefig(save_path)
        plt.close()


def trial_score(loss, metrics):
    """Model-independent score for ranking black-box trials.

    Ranks by the raw geometric total in meters when available. The weighted
    l_total is gameable by the predicted loss weights (compute_loss does not
    renormalize them), so ranking trials by it lets weight-lowering noise win
    over genuinely better detections. Falls back to the loss when no raw metric
    exists (e.g. a no-detection trial).
    """
    v = metrics.get('geo_total_m') if isinstance(metrics, dict) else None
    if v is not None and not (isinstance(v, float) and np.isnan(v)):
        return float(v)
    return float(loss) if np.isfinite(loss) else float('inf')


# Weakly-supervised trial budget for test-time calibration at sites with no
# local training history. Both FedMeta (global-model init) and Meta
# (nearest-scene donor init) receive the SAME budget so the comparison measures
# initialization quality, not adaptation opportunity.
DEPLOYMENT_CALIBRATION_TRIALS = 2

# The SUMO lane matching threshold: every reference lane in a no-detection scene
# is at least one full matching threshold away from any detected lane, so this is
# a conservative floor for the coverage charge, not an estimate.
NO_DETECTION_COVERAGE_PENALTY_M = 10.0


def no_detection_result(processed_data):
    """Loss and metrics charged when a theta detects zero lanes.

    Keeps the trial search finite (inf poisons loss averages and best-theta
    selection) and keeps failed epochs visible in the reported lane-count and
    coverage columns instead of silently dropping out of the averages. Geometry
    columns that were never measured stay NaN rather than being fabricated.
    """
    sumo_node, _ = processed_data.get('sumo_graph', ([], []))
    ref_count = sum(len(group) for group in sumo_node)
    loss = 10.0 * max(ref_count, 1)
    metrics = {
        'lane_count': 0,
        'error': 'No lanes detected',
        'l_lane_count': float(ref_count),
        'lane_count_err': float(ref_count),
        'lane_count_exact': 0.0,
        'geo_coverage_m': NO_DETECTION_COVERAGE_PENALTY_M,
        'geo_consistency_m': float('nan'),
        'geo_centerline_m': float('nan'),
        'geo_width_m': float('nan'),
        'geo_total_m': float('nan'),
    }
    return loss, metrics


def perturb_theta(theta_values):
    """One black-box exploration step around a predicted theta.

    Shared by the federated and meta strategies so both run the identical search rule (Eq. 9 trial selection).
    """
    perturbed = {}
    for k, v in theta_values.items():
        # Loss weights are not search dimensions. Perturbing them (unnormalized)
        # lets a trial score a lower weighted loss without any detection change,
        # so keep the model's softmax-normalized values fixed across trials.
        if k.startswith('weight_'):
            perturbed[k] = v
            continue
        noise = torch.randn(1).item() * 0.1
        if k == 'width_scale':
            perturbed[k] = max(0.5, min(2.0, v + noise))
        elif k == 'smoothing_factor':
            perturbed[k] = max(1, min(20, v + noise * 10))
        elif k == 'triplet_margin':
            perturbed[k] = max(0.1, min(2.0, v + noise))
        elif k == 'peak_prominence':
            # Wide exploration across the full RELATIVE range (fraction of the
            # scene's max smoothed peak): the useful value can sit far from the
            # initial prediction, so sample the range instead of a local step
            # the black-box search cannot escape.
            perturbed[k] = float(np.random.uniform(0.05, 0.95))
        else:
            perturbed[k] = max(0.1, min(1.0, v + noise))
    return perturbed


class SceneFeatureExtractor:
    """Advanced scene feature extraction for meta-learning."""
    
    @staticmethod
    def peak_statistics(processed_data):
        """Relative peak-salience statistics of the per-contour lane histograms.

        Mirrors the histogram path in geo_learning.run (per-id mean x per
        contour, smoothed, normalized by the max peak) and summarizes the
        candidate-peak structure. Without these the correct peak_prominence
        must be inferred indirectly from coarse count features, which is the
        measured cause of the global model's compressed scene response.
        Returns [n_candidate_peaks/10, median relative prominence, spread of
        relative prominences]; zeros when trajectories are unavailable.
        """
        gps_df = processed_data.get('gps_df') if isinstance(processed_data, dict) else None
        try:
            if gps_df is None or len(gps_df) == 0:
                return [0.0, 0.0, 0.0]
            df = gps_df if isinstance(gps_df, pl.DataFrame) else pl.DataFrame(gps_df)
            agg = df.group_by(['contour_id', 'id']).agg(pl.col('x_gps').mean().alias('xm')).drop_nulls()
        except Exception:
            return [0.0, 0.0, 0.0]

        proms = []
        for cnt in agg['contour_id'].unique().to_list():
            means = agg.filter(pl.col('contour_id') == cnt)['xm'].to_numpy()
            # NaN GPS points survive drop_nulls (NaN is not null in polars) and
            # make np.histogram's range autodetection blow up.
            means = means[np.isfinite(means)]
            if len(means) < 3:
                continue
            hist, _ = np.histogram(means, bins=50)
            smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
            m = float(smoothed.max())
            if m <= 0:
                continue
            _, props = find_peaks(smoothed / m, prominence=0.02)
            proms.extend(props['prominences'].tolist())
        if not proms:
            return [0.0, 0.0, 0.0]
        return [len(proms) / 10.0, float(np.median(proms)), float(np.std(proms))]

    @staticmethod
    def extract_features(processed_data, include_advanced=False):
        """Extract comprehensive scene features."""
        # Basic features
        contour_count = len(processed_data['detected_cnts'])
        roi_height = processed_data['frame'].shape[0]
        roi_width = processed_data['frame'].shape[1]
        vehicle_count = len(processed_data['collect_cars'])
        point_density = len(processed_data['collect_dots']) / (roi_height * roi_width)
        aspect_ratio = roi_width / roi_height
        
        # Count features are log-scaled: raw detection counts accumulate over the
        # whole video history (tens of thousands), and linear scaling saturates the
        # meta-model's sigmoid heads, pinning predictions to their range bounds for
        # any busy scene.
        features = [
            contour_count / 10.0, # Normalize
            float(np.log1p(vehicle_count)) / 10.0,
            float(np.log1p(len(processed_data['collect_dots']))) / 10.0,
            aspect_ratio,
            roi_height / 1000.0
        ] + SceneFeatureExtractor.peak_statistics(processed_data)
        
        # TODO: not included in the journal
        if include_advanced:
            # Advanced features
            # Traffic flow direction
            if 'gps_df' in processed_data and len(processed_data['gps_df']) > 0:
                df = processed_data['gps_df']
                if 'theta_rad' in df.columns:
                    angles = df['theta_rad'].to_numpy()
                    # Circular mean of angles
                    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
                    angle_variance = 1 - np.abs(np.mean(np.exp(1j * angles)))
                    features.extend([mean_angle / np.pi, angle_variance])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # Contour complexity
            if len(processed_data['detected_cnts']) > 0:
                contour_areas = [cv2.contourArea(cnt) for cnt in processed_data['detected_cnts']]
                avg_area = np.mean(contour_areas) / (roi_height * roi_width)
                area_variance = np.var(contour_areas) / (roi_height * roi_width)**2
                features.extend([avg_area, area_variance])
            else:
                features.extend([0.0, 0.0])
            
            # Time of day encoding (if available)
            if 'timestamp' in processed_data:
                hour = processed_data['timestamp'].hour
                # Cyclic encoding
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                features.extend([hour_sin, hour_cos])
            else:
                features.extend([0.0, 1.0]) # Default to midnight
        
        return torch.tensor(features, dtype=torch.float32)