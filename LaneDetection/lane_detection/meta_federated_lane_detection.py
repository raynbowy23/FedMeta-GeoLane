import copy
import random
import logging
import json
import time
import mlflow
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from LaneDetection.osm_extraction.utils import compute_lane_width_from_gps
from LaneDetection.osm_extraction.connect_to_osm import OSMConnection

from .utils import (FederatedConfig, SceneFeatureExtractor, perturb_theta, no_detection_result,
                    trial_score, SCENE_FEATURE_DIM, DEPLOYMENT_CALIBRATION_TRIALS)

logger = logging.getLogger(__name__)

# ---- Federated aggregation algorithm ----
# 'fedavg'    : plain sample-weighted parameter averaging of locally MSE-fit
#               regressors. Provably regresses the global model toward the
#               population mean of best_theta (amplitude compression) -> the old
#               behaviour, kept as an ablation.
# 'perfedavg' : first-order Per-FedAvg (Fallah et al., NeurIPS 2020). The global
#               model is trained to be a good INITIALIZATION that reaches each
#               site's scene->theta mapping in one local adaptation step, which
#               is the actual federated-meta-learning objective and what the
#               deployment / adaptation-curve evaluation adapts from.
# 'central'   : pooled supervised regression. All clients' (scene_features ->
#               best_theta) pairs are trained into the ONE shared model directly.
#               Unlike fedavg -- where each client fits only its own near-constant
#               features and learns a constant, so averaging collapses amplitude --
#               pooled training sees the cross-camera variation and can learn the
#               scene->theta slope the features carry. Centralized upper bound for
#               the federated method; used to test whether the FedAvg/Per-FedAvg
#               amplitude collapse is the aggregation (fixable) or a feature limit.
PERFEDAVG_INNER_LR = 1e-2     # alpha: local adaptation step size
PERFEDAVG_OUTER_LR = 1e-2     # beta: server meta-gradient step size
PERFEDAVG_INNER_STEPS = 1     # gradient steps taken from the global init per client
DEPLOYMENT_ADAPT_STEPS = 1    # gradient steps taken at an unseen site before predicting
CENTRAL_EPOCHS = 10           # supervised passes over the pooled buffer per round ('central')


def _scalarize(t):
    # robustly get a scalar tensor from model outputs that may have shape [B] or [B,1] or []
    if isinstance(t, torch.Tensor):
        if t.dim() == 0: 
            return t
        if t.dim() == 1 and t.size(-1) == 1:
            return t.squeeze(-1)
        return t.squeeze()
    return torch.tensor(float(t))

def estimate_object_size_bytes(obj):
    """Serialize and estimate object size in bytes."""
    return len(pickle.dumps(obj))

class MetaMLModel(nn.Module):
    """
    Black-box meta-learner that maps scene features to optimal theta parameters.
    No gradient-based adaptation - directly predicts parameters from features.
    """
    def __init__(self, feature_dim=SCENE_FEATURE_DIM, hidden_dim=128, num_theta_params=6, config_path=None):
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
            'peak_prominence': nn.Linear(hidden_dim, 1),
            # Absolute recall lever, added LAST to keep saved-checkpoint layer
            # order stable for the Rust export. Unlike peak_prominence (a
            # normalized fraction that collapses to a scene-independent
            # constant), the minimum vehicles-per-lane gate is in absolute
            # counts, so its optimum stays scene-dependent and gives the
            # meta-learner something to actually adapt.
            'min_lane_evidence': nn.Linear(hidden_dim, 1),
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

        logger.debug("Extracted features shape: %s", features.shape)
        
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
            elif param_name == 'peak_prominence':
                # RELATIVE peak salience in (0.05, 0.95): a fraction of the
                # scene's max smoothed histogram peak (the histogram is
                # normalized before peak finding), so the value is scale-free
                # across sparse and busy scenes. Lower recovers weak lanes,
                # higher rejects spurious peaks.
                theta_dict[param_name] = torch.sigmoid(head(features)).squeeze() * 0.9 + 0.05
            elif param_name == 'min_lane_evidence':
                # Absolute minimum vehicles per lane, 1-20. A cluster with fewer
                # supporting vehicles is dropped, so a low peak_prominence can
                # recover weak lanes without admitting spurious low-support peaks.
                theta_dict[param_name] = torch.sigmoid(head(features)).squeeze() * 19 + 1
            else:
                # Others typically 0-1
                theta_dict[param_name] = torch.sigmoid(head(features)).squeeze()

        logger.debug("Predicted theta parameters: %s", theta_dict)
        
        loss_weights = {}
        weight_sum = sum(torch.abs(w) for w in self.loss_weights.values())
        for name, weight in self.loss_weights.items():
            loss_weights[f'weight_{name}'] = torch.abs(weight) / weight_sum

        logger.debug("Loss weights: %s", loss_weights)

        
        output_dict = {**theta_dict, **loss_weights}

        return output_dict


class FederatedMetaLearner:
    """
    Orchestrates federated learning across multiple camera clients with meta-learning.
    """
    def __init__(self, meta_model, device='cpu', visualize_lanes=True, fed_algo='perfedavg',
                 seen_deploy='buffer'):
        self.meta_model = meta_model.to(device)
        self.device = device
        # 'perfedavg' (meta-init, default), 'fedavg' (mean-regressor ablation), or
        # 'central' (pooled supervised, centralized upper bound).
        self.fed_algo = fed_algo
        # Seen-site deployment: 'buffer' reuses the best black-box theta from the
        # site's training history (model bypassed); 'model' deploys the trained
        # model's prediction + calibration, so a recovered model can reach seen.
        self.seen_deploy = seen_deploy
        self.client_data_buffer = defaultdict(list)
        self.global_data_buffer = []
        self.training_history = []
        self.visualize_lanes = visualize_lanes
        # Cached test-time calibrated thetas for clients with no local training
        # history (unseen sites), filled on first deployment contact.
        self.deployment_theta = {}

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

                # Trial 0 is the reference unconditionally: its loss can be inf
                # (no lanes detected), and inf < inf is false, which would leave
                # best_theta as None and crash theta aggregation downstream.
                best_loss = loss
                best_theta = predicted_theta_values
                best_metrics = metrics
                best_score = trial_score(loss, metrics)
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
                perturbed_theta = perturb_theta(predicted_theta_values)

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
                    
                    if trial_score(loss, metrics) < best_score:
                        best_score = trial_score(loss, metrics)
                        best_loss = loss
                        best_theta = perturbed_theta
                        best_metrics = metrics
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
            
            # Store data for meta-model training. 'metrics' (the BEST trial's raw
            # metrics) is what trial_score ranks the deployed theta by — the same
            # rule the meta strategy uses, so the two pickers cannot diverge.
            self.client_data_buffer[client_id].append({
                'scene_features': scene_features.cpu(),
                'best_theta': best_theta,
                'best_loss': best_loss,
                'metrics': best_metrics,
                'trial_results': trial_results
            })
            
        else:
            # Deployment: every site runs its best-known configuration, one
            # uniform rule for Meta and FedMeta. Seen sites reuse the best theta
            # from their own federated training history (local evidence beats
            # re-predicting through the amplitude-compressed global model).
            # Unseen sites get the global prediction plus a small
            # weakly-supervised calibration budget on arrival (the OSM reference
            # exists there too); the result is cached. No data leaves any site.
            calibrate = False
            if client_id in self.deployment_theta:
                predicted_theta_values = self.deployment_theta[client_id]
            elif self.seen_deploy == 'buffer' and self.client_data_buffer.get(client_id):
                predicted_theta_values = self._best_theta_from_buffer(client_id)
                self.deployment_theta[client_id] = predicted_theta_values
            else:
                # seen_deploy=='model' routes seen sites through the trained model
                # too (predict + on-arrival calibration), so the recovered global
                # model can actually reach the seen table instead of being bypassed
                # by the buffer. Unseen always lands here.
                with torch.no_grad():
                    predicted_theta = self.meta_model(scene_features)
                    predicted_theta_values = {}
                    for k, v in predicted_theta.items():
                        if isinstance(v, torch.Tensor):
                            predicted_theta_values[k] = v.item() if v.dim() == 0 else v.squeeze().item()
                        else:
                            predicted_theta_values[k] = float(v)
                calibrate = True

            for k, v in predicted_theta_values.items():
                geo_learning.theta[k] = torch.tensor(v)

            try:
                best_loss, metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
                best_theta = predicted_theta_values
                if calibrate:
                    best_score = trial_score(best_loss, metrics)
                    for _ in range(DEPLOYMENT_CALIBRATION_TRIALS):
                        cand = perturb_theta(predicted_theta_values)
                        for k, v in cand.items():
                            geo_learning.theta[k] = torch.tensor(v)
                        try:
                            c_loss, c_metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
                        except Exception as e:
                            logger.error(f"Calibration trial failed for {client_id}: {e}")
                            continue
                        if trial_score(c_loss, c_metrics) < best_score:
                            best_score = trial_score(c_loss, c_metrics)
                            best_loss, best_theta, metrics = c_loss, cand, c_metrics

                    # Meta-init cash-in: adapt the global model one step on the
                    # site's own calibration evidence and try its prediction. The
                    # black-box budget above is identical to Meta's; the only
                    # difference is that Per-FedAvg's init was trained to be
                    # adaptable, so this candidate only wins when the init is good.
                    if self.fed_algo == 'perfedavg':
                        support = [{'x': scene_features.to(self.device), 'y': best_theta}]
                        try:
                            adapted = self._adapt_from_global(support)
                            with torch.no_grad():
                                a_pred = adapted(scene_features)
                                a_theta = {k: (v.item() if v.dim() == 0 else v.squeeze().item())
                                           for k, v in a_pred.items()}
                            for k, v in a_theta.items():
                                geo_learning.theta[k] = torch.tensor(v)
                            a_loss, a_metrics = self._run_geo_learning(geo_learning, processed_data, client_id)
                            if trial_score(a_loss, a_metrics) < best_score:
                                best_score = trial_score(a_loss, a_metrics)
                                best_loss, best_theta, metrics = a_loss, a_theta, a_metrics
                                logger.info(f"[Deploy] {client_id}: meta-init adaptation improved theta")
                        except Exception as e:
                            logger.error(f"Meta-init adaptation failed for {client_id}: {e}")

                    self.deployment_theta[client_id] = best_theta
                    logger.info(f"[Deploy] {client_id}: test-time calibration done (prominence {best_theta.get('peak_prominence', float('nan')):.3f})")
            except Exception as e:
                logger.error(f"Error in deployment mode for client {client_id}: {e}")
                return 1.0, predicted_theta_values, {}
        
        return best_loss, best_theta, metrics
    
    def _best_theta_from_buffer(self, client_id):
        """The client's best recorded theta from its training-time trial history,
        ranked by trial_score — the SAME rule the meta strategy's picker uses.
        Ranking by weighted best_loss here while meta ranked by trial_score gave
        the two strategies different deployment-selection rules (the Mineral
        lane_err 8.3-vs-1.1 artifact). Entries from old checkpoints without a
        'metrics' key fall back to the loss inside trial_score."""
        buf = self.client_data_buffer.get(client_id, [])
        finite = [b for b in buf if np.isfinite(b.get('best_loss', float('inf'))) and b.get('best_theta')]
        pick = (min(finite, key=lambda b: trial_score(b['best_loss'], b.get('metrics', {}) or {}))
                if finite else buf[-1])
        return pick['best_theta']

    def _fedavg_state_dicts(self, global_state, client_states, client_sizes):
        assert len(client_states) == len(client_sizes) > 0
        new_state = {k: torch.zeros_like(v) for k, v in global_state.items()}
        N = sum(client_sizes)
        for state, n in zip(client_states, client_sizes):
            w = n / float(N)
            for k in new_state:
                new_state[k] += w * state[k]
        return new_state

    def client_meta_step(self, client_id, global_state, epochs=1, lr=1e-3):
        """
        Trains a local copy of meta_model on the client's buffer:
        ({scene_features} -> {best_theta}) pairs generated by black-box trials.
        Returns: (updated_state_dict, num_samples). If no data, returns (None, 0).
        """
        local_buffer = self.client_data_buffer.get(client_id, [])
        if not local_buffer:
            return None, 0

        # local copy & init from global
        local_meta = copy.deepcopy(self.meta_model).to(self.device)
        local_meta.load_state_dict(global_state)
        local_meta.train()

        opt = torch.optim.Adam(local_meta.parameters(), lr=lr)
        mse = nn.MSELoss()

        # flatten buffer into simple per-sample items
        samples = []
        for entry in local_buffer:
            # entry: {'scene_features', 'best_theta', 'best_loss', 'trial_results'}
            samples.append({
                'x': entry['scene_features'].to(self.device),
                'y': entry['best_theta'],  # dict of target scalars
            })

        for _ in range(epochs):
            random.shuffle(samples)
            for s in samples:
                pred = local_meta(s['x'])
                loss = 0.0
                # supervise ONLY real theta params (skip weight_* if present)
                for k, p in pred.items():
                    if k.startswith('weight_'):
                        continue
                    tgt = torch.tensor(float(s['y'][k]), dtype=torch.float32, device=self.device)
                    loss = loss + mse(_scalarize(p), tgt)

                opt.zero_grad()
                loss.backward()
                opt.step()

        return local_meta.state_dict(), len(samples)

    # ---------------- Per-FedAvg (meta-init) ----------------

    def _theta_mse(self, model, samples):
        """Mean MSE over the real theta params (skipping weight_*) for a list of
        {'x': scene_features, 'y': best_theta dict} samples."""
        mse = nn.MSELoss()
        total = 0.0
        for s in samples:
            pred = model(s['x'])
            for k, p in pred.items():
                if k.startswith('weight_'):
                    continue
                tgt = torch.tensor(float(s['y'][k]), dtype=torch.float32, device=self.device)
                total = total + mse(_scalarize(p), tgt)
        return total / max(len(samples), 1)

    def _client_samples(self, client_id):
        """Flatten a client's buffer into [{'x': scene_features, 'y': best_theta}]."""
        out = []
        for entry in self.client_data_buffer.get(client_id, []):
            if entry.get('best_theta') is None:
                continue
            out.append({'x': entry['scene_features'].to(self.device),
                        'y': entry['best_theta']})
        return out

    def _client_meta_gradient(self, client_id, global_state,
                              inner_lr=PERFEDAVG_INNER_LR, inner_steps=PERFEDAVG_INNER_STEPS):
        """First-order Per-FedAvg meta-gradient for one client.

        Adapt a local copy from the global init on a support split, then return
        the gradient of the query loss evaluated AT the adapted parameters (the
        first-order approximation drops the Hessian term). With fewer than two
        buffered samples the support/query split is impossible, so fall back to a
        Reptile pseudo-gradient (init - adapted) that points the same way and lets
        early rounds still contribute.

        Returns (grad_dict keyed by named_parameters, num_samples) or (None, 0).
        """
        samples = self._client_samples(client_id)
        if not samples:
            return None, 0

        local = copy.deepcopy(self.meta_model).to(self.device)
        local.load_state_dict(global_state)
        local.train()

        if len(samples) >= 2:
            random.shuffle(samples)
            split = max(1, len(samples) // 2)
            support, query = samples[:split], samples[split:]
            inner_opt = torch.optim.SGD(local.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                inner_opt.zero_grad()
                self._theta_mse(local, support).backward()
                inner_opt.step()
            local.zero_grad()
            self._theta_mse(local, query).backward()
            grad = {n: (p.grad.detach().clone() if p.grad is not None
                        else torch.zeros_like(p))
                    for n, p in local.named_parameters()}
        else:
            inner_opt = torch.optim.SGD(local.parameters(), lr=inner_lr)
            for _ in range(max(inner_steps, 2)):
                inner_opt.zero_grad()
                self._theta_mse(local, samples).backward()
                inner_opt.step()
            adapted = dict(local.named_parameters())
            grad = {n: (global_state[n] - adapted[n].detach()).clone()
                    for n in adapted}
        return grad, len(samples)

    def _perfedavg_update(self, selected_clients,
                          inner_lr=PERFEDAVG_INNER_LR, inner_steps=PERFEDAVG_INNER_STEPS,
                          outer_lr=PERFEDAVG_OUTER_LR):
        """Server Per-FedAvg step: aggregate client meta-gradients, step the global
        model so one local adaptation step reaches each site's scene->theta map."""
        global_state = copy.deepcopy(self.meta_model.state_dict())
        grads, sizes = [], []
        for cid in selected_clients:
            g_i, n_i = self._client_meta_gradient(cid, global_state, inner_lr, inner_steps)
            if g_i is not None and n_i > 0:
                grads.append(g_i)
                sizes.append(n_i)
        if not grads:
            logger.warning("No client meta-gradients available for Per-FedAvg step.")
            return
        N = float(sum(sizes))
        new_state = copy.deepcopy(global_state)
        for name in grads[0]:
            agg = sum((n / N) * g[name] for g, n in zip(grads, sizes))
            new_state[name] = new_state[name] - outer_lr * agg
        self.meta_model.load_state_dict(new_state)
        logger.info(f"Per-FedAvg meta update: aggregated meta-gradients from {len(grads)} clients")

    def _central_supervised_update(self, selected_clients, epochs=CENTRAL_EPOCHS, lr=1e-3):
        """Pool every client's (scene_features -> best_theta) pairs and train the
        one shared model directly (centralized supervised meta-learning). FedAvg
        of per-client fits never sees cross-camera input variation and collapses
        to a constant; pooled training exposes the scene->theta slope."""
        samples = []
        for cid in selected_clients:
            samples.extend(self._client_samples(cid))
        if not samples:
            logger.warning("No pooled samples for central supervised update.")
            return
        self.meta_model.train()
        opt = torch.optim.Adam(self.meta_model.parameters(), lr=lr)
        for _ in range(epochs):
            random.shuffle(samples)
            for s in samples:
                loss = self._theta_mse(self.meta_model, [s])
                opt.zero_grad()
                loss.backward()
                opt.step()
        logger.info(f"Central supervised update: trained shared model on {len(samples)} pooled samples "
                    f"from {len(selected_clients)} clients")

    def _adapt_from_global(self, support_samples,
                           inner_lr=PERFEDAVG_INNER_LR, inner_steps=DEPLOYMENT_ADAPT_STEPS):
        """Return a copy of the global meta-model adapted a few gradient steps on a
        site's locally generated (scene -> best_theta) samples. This is how an
        unseen site cashes in the meta-init and what the adaptation curve measures."""
        local = copy.deepcopy(self.meta_model).to(self.device)
        local.train()
        opt = torch.optim.SGD(local.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            opt.zero_grad()
            self._theta_mse(local, support_samples).backward()
            opt.step()
        local.eval()
        return local

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
                l_total, l_lane_count, l_cons, l_trip, l_geo, raw_metrics = geo_learning.compute_loss(
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

                metrics.update(raw_metrics)

                logger.info(f"Client {client_id} - Total Loss: {total_loss:.4f}, "
                          f"Lane Count Loss: {metrics['l_lane_count']:.4f}, "
                          f"Consistency Loss: {metrics['l_cons']:.4f}, "
                          f"Triplet Loss: {metrics['l_trip']:.4f}, "
                          f"Geometry Loss: {metrics['l_geo']:.4f}")
                logger.info(f"Client {client_id} - [raw m] consistency={raw_metrics['geo_consistency_m']:.3f}, "
                          f"centerline={raw_metrics['geo_centerline_m']:.3f}, width={raw_metrics['geo_width_m']:.3f}, "
                          f"geo_total={raw_metrics['geo_total_m']:.3f}, lane_count_err={raw_metrics['lane_count_err']:.0f}")

                return total_loss, metrics
                
            else:
                logger.warning(f"No lanes detected for client {client_id}")
                return no_detection_result(processed_data)
                
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

                # Map the detected cluster back to a SUMO lane id for the saved CSV.
                # A cluster without an edge mapping (e.g. an extra detected lane
                # beyond the reference count) or an out-of-range index has no SUMO
                # id; skip the labeling for it rather than crash the whole client.
                sumo_keys = list(sumo_lane_shape.keys())
                edge = cluster_to_edge_map.get(lane_id)
                if edge is None or not (0 <= edge[1] < len(sumo_keys)):
                    continue
                sumo_lane_id = sumo_keys[edge[1]]
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
        loss_component_keys = ['l_lane_count', 'l_cons', 'l_trip', 'l_geo',
                               'geo_consistency_m', 'geo_coverage_m', 'geo_centerline_m', 'geo_width_m',
                               'geo_total_m', 'lane_count_err', 'lane_count_exact']
        client_thetas = []
        
        for client_id, (loss, theta, metrics) in client_results.items():
            aggregated_metrics['losses'].append(loss)
            client_thetas.append(theta)
            for k, v in metrics.items():
                aggregated_metrics[k].append(v)
        
        # Compute statistics over finite losses only (a client can still report
        # inf on an unexpected exception; it must not poison the round average)
        finite_losses = [l for l in aggregated_metrics['losses'] if np.isfinite(l)]
        avg_loss = np.mean(finite_losses) if finite_losses else float('inf')
        std_loss = np.std(finite_losses) if finite_losses else 0.0

        # Average the reported components (incl. model-independent raw metrics).
        # nanmean so a client with no matched lanes (nan) doesn't void the average.
        avg_metrics = {
            k: float(np.nanmean(np.asarray(aggregated_metrics[k], dtype=float)))
            for k in loss_component_keys
            if k in aggregated_metrics
            and not np.all(np.isnan(np.asarray(aggregated_metrics[k], dtype=float)))
        }

        # Average theta parameters (defensive: a failed client can report no theta)
        client_thetas = [t for t in client_thetas if t is not None]
        avg_theta = {}
        if client_thetas:
            for key in client_thetas[0]:
                stacked = torch.stack([torch.tensor(theta[key], dtype=torch.float32) for theta in client_thetas])
                avg_theta[key] = stacked.mean(dim=0)
        
        # Log individual client metrics to MLflow
        try:
            for client_id, (loss, theta, metrics) in client_results.items():
                mlflow.log_metric(f"Federated/Loss_{client_id}", loss, step=self.round_counter)
                
                # Log detailed loss components per client (incl. raw model-independent metrics)
                for key in loss_component_keys:
                    val = metrics.get(key)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mlflow.log_metric(f"Federated/{key}_{client_id}", val, step=self.round_counter)
                        
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
    
    def train_models(self, selected_clients, local_epochs=1, lr=1e-3):
        """
        Federated update of the meta_model across selected clients.
        Uses each client's local buffers produced in client_update().
        """
        if self.fed_algo == 'central':
            # Pooled supervised regression: does the shared model recover the
            # per-scene signal that FedAvg/Per-FedAvg collapse?
            self._central_supervised_update(selected_clients, epochs=CENTRAL_EPOCHS, lr=lr)
            return

        if self.fed_algo == 'perfedavg':
            # Meta-init objective: train the global model to be adapted in one
            # local step, rather than to predict the population-mean theta.
            self._perfedavg_update(selected_clients,
                                   inner_lr=PERFEDAVG_INNER_LR,
                                   inner_steps=PERFEDAVG_INNER_STEPS,
                                   outer_lr=PERFEDAVG_OUTER_LR)
            return

        # ---- plain FedAvg (mean-regressor ablation) ----
        # 1) broadcast current global meta-model
        global_state = copy.deepcopy(self.meta_model.state_dict())

        # 2) local client updates (black-box supervision stays local)
        client_states, client_sizes = [], []
        for cid in selected_clients:
            state_i, n_i = self.client_meta_step(cid, global_state, epochs=local_epochs, lr=lr)
            if state_i is not None and n_i > 0:
                client_states.append(state_i)
                client_sizes.append(n_i)

        if not client_states:
            logger.warning("No client updates available for federated meta step.")
            return

        # 3) aggregate (FedAvg) and load
        new_state = self._fedavg_state_dicts(global_state, client_states, client_sizes)
        self.meta_model.load_state_dict(new_state)
        logger.info(f"Federated meta update: aggregated from {len(client_states)} clients")
    
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
            feature_dim=SCENE_FEATURE_DIM,
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
