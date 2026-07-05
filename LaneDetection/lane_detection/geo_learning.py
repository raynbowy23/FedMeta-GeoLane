import time
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
matplotlib.set_loglevel(level = 'warning')
import logging
logger = logging.getLogger(__name__)

from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import polars as pl

from .utils import *
from .loss import *
from LaneDetection.osm_extraction.utils import interpolate_edge


class GeometricLearning:
    """
      Implement geometric learning for lane detection.
    """
    def __init__(self, args, saving_file_path):
        self.args = args
        self.filepath = saving_file_path
        self.rootpath = Path(args.dataset_path, '511calibration')
        self.sumo_filepath = args.osm_path
        self.sumo_netfile = 'osm.net.xml'
        self.sumo_savefile = 'out.net.xml'

        self.lambda_thres = args.lambda_thres
        self.cnts_threshold = args.cnts_threshold
        self.is_save = args.is_save

        # Define learnable rule parameters (initialized manually or from input)
        self.theta = {
            'width_scale': torch.tensor(1.0),
            'consistency_weight': torch.tensor(0.5),
            'triplet_margin': torch.tensor(0.8),
            'smoothing_factor': torch.tensor(10.0),
            'sigma': torch.tensor(2.0),
            'peak_prominence': torch.tensor(1.0),
        }

        self.colors = [
            (0.98, 0.80, 0.18),   # light yellow
            (0.68, 0.85, 0.90),   # light blue
            (0.60, 0.80, 0.60),   # light green
            (0.99, 0.70, 0.40),   # orange-peach
            (0.80, 0.80, 0.60),   # khaki
            (0.95, 0.75, 0.85),   # soft pink
            (0.67, 0.87, 0.64),   # mint green
            (0.85, 0.85, 0.95),   # lavender gray
            (0.90, 0.95, 0.75),   # pale lime
            (0.60, 0.90, 0.90),   # cyan mist
            (0.95, 0.85, 0.65),   # sand
            (0.75, 0.95, 0.85),   # soft aqua
            (0.88, 0.88, 0.88),   # light gray
            (0.93, 0.93, 0.73),   # pastel yellow
            (0.85, 0.70, 0.55),   # clay
            (0.73, 0.93, 0.93),   # ice blue
            (0.98, 0.85, 0.73),   # light apricot
            (0.85, 0.95, 0.80),   # soft green-yellow
            (0.78, 0.85, 0.95),   # light periwinkle
            (0.98, 0.98, 0.75),   # cream
        ]


    # Gaussian filter using a weighted moving average
    def gaussian_filter(
            self,
            data,
            window_size=5,
            sigma=2
        ):
        kernel = np.exp(-np.linspace(-2, 2, window_size) ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return np.convolve(data, kernel, mode='same')

    def estimate_lane_width(self, lane_df):
        """
        Estimate width from trajectory spread (x spread for vertical lanes).
        """
        x = lane_df["x_gps"].values
        width_est = 2 * np.std(x) # ~95% coverage if Gaussian
        return width_est

    def compute_lane_geometry(self, df_plot, smoothing=10, num_points=30):
        """
        For each clustered_id in df_plot, compute centerline, lane width, and left/right boundaries.
        
        Returns:
            lane_boundaries_by_id: {
                lane_id: {
                    "center": (N, 2),
                    "left": (N, 2),
                    "right": (N, 2),
                    "width": float
                }
            }
        """
        if 'smoothing_factor' in self.theta:
            smoothing = self.theta['smoothing_factor'].item() if isinstance(self.theta['smoothing_factor'], torch.Tensor) else self.theta['smoothing_factor']
            clustered_ids = sorted(df_plot["clustered_id"].dropna().unique())

        lane_boundaries_by_id = {}

        for lane_id in clustered_ids:
            lane_df = df_plot[df_plot["clustered_id"] == lane_id]
            if len(lane_df) < 5:
                continue

            # Sort by y_gps for vertical alignment
            lane_df_sorted = lane_df.sort_values(by="y_gps")
            x = lane_df_sorted["x_gps"].values
            y = lane_df_sorted["y_gps"].values

            try:
                # Fit the spline in local meter coordinates. s is a squared-residual
                # budget, so in raw GPS degrees (residuals ~1e-5) any s >= ~1e-6
                # collapses to one maximally smoothed curve and the meta-learned
                # smoothing_factor has no effect at all. In meters, s in [1, 20]
                # spans tight-to-smooth fits as intended. x_gps is latitude and
                # y_gps is longitude in this pipeline.
                lat0, lon0 = float(np.mean(x)), float(np.mean(y))
                m_lat = 111_320.0
                m_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
                y_m = (y - lon0) * m_lon
                x_m = (x - lat0) * m_lat
                spline = UnivariateSpline(y_m, x_m, s=smoothing)
                y_fit_m = np.linspace(y_m.min(), y_m.max(), num=num_points)
                x_fit_m = spline(y_fit_m)
                y_fit = y_fit_m / m_lon + lon0
                x_fit = x_fit_m / m_lat + lat0

                # Estimate lane width
                lane_width = self.estimate_lane_width(lane_df)
                if 'width_scale' in self.theta:
                    width_scale = self.theta['width_scale'].item() if isinstance(self.theta['width_scale'], torch.Tensor) else self.theta['width_scale']
                    lane_width *= width_scale

                # Compute direction and normals
                dx = np.gradient(x_fit)
                dy = np.gradient(y_fit)
                norm = np.sqrt(dx**2 + dy**2)
                dx /= norm
                dy /= norm
                nx = -dy
                ny = dx

                # Compute boundaries
                offset = lane_width / 2
                x_left = x_fit + nx * offset
                y_left = y_fit + ny * offset
                x_right = x_fit - nx * offset
                y_right = y_fit - ny * offset

                lane_boundaries_by_id[int(lane_id)] = {
                    "center": np.stack([x_fit, y_fit], axis=1),
                    "left": np.stack([x_left, y_left], axis=1),
                    "right": np.stack([x_right, y_right], axis=1),
                    "width": lane_width
                }

            except Exception as e:
                logger.error(f"[Warning] Lane {lane_id} spline fit failed: {e}")
                continue

        return lane_boundaries_by_id


    def run(self, c_epoch, g_epoch, traj_df,
            camera_loc, trial, is_save):
        """
        Process Geometrical Learning

        Args:
            g_epoch (_type_): _description_
            frame (_type_): _description_
            collect_cars (_type_): _description_
            collect_det_dots_including_truck (_type_): _description_
            init_detected_centers (_type_): _description_
            adjusted_points (_type_): _description_
            cluster_to_edge_map (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.start = time.time()
        self.c_epoch = c_epoch
        self.g_epoch = g_epoch
        self.trial = trial

        self.fig_filepath = Path(self.filepath, camera_loc, "figures")
        traj_df = traj_df.filter((~traj_df["x_gps"].is_nan()) & (~traj_df["y_gps"].is_nan()))
        # lane_width = 3.5
        lane_width = 0.00004311
        lane_boundaries_for_contour = {}

        # cmap = cm.get_cmap("tab20")
        # colors = [cmap(i) for i in range(10)]
        # colors = [tuple(c / 255 for c in rgb) + (1.0,) for rgb in self.colors]
        # cmap = ListedColormap(colors)
        # colors = [cmap(i) for i in range(20)]
        lane_num_list = []

        # Initialize frame for lane visualization
        fig1, ax1 = plt.subplots(figsize=(12, 10)) # For histogram
        fig2, ax2 = plt.subplots(figsize=(12, 10)) # For clustered point
        fig3, ax3 = plt.subplots(figsize=(12, 10)) # For lane visualization

        cluster_assignments = []

        for c, cnts in enumerate(traj_df["contour_id"].unique().to_list()):
            df_contour = traj_df.filter(pl.col("contour_id") == cnts)

            trajectory_summary = (
                df_contour.group_by("id")
                .agg([
                    pl.mean("x_gps").alias("x_mean"),
                    pl.mean("y_gps").alias("y_mean"),
                    pl.mean("theta_rad").alias("theta_mean")
                ])
            )

            # Histogram
            X = trajectory_summary["x_mean"].to_numpy().reshape(-1, 1)

            hist_vals, bin_edges = np.histogram(X, bins=50)
            # hist_vals, bin_edges = np.histogram(X, bins=50)
            if 'sigma' in self.theta:
                smoothed_hist = self.gaussian_filter(hist_vals, window_size=5, sigma=self.theta['sigma'].item() if isinstance(self.theta['sigma'], torch.Tensor) else self.theta['sigma'])
            else:
                smoothed_hist = self.gaussian_filter(hist_vals, window_size=5, sigma=2)

            # Adaptive peak detection: meta-learned peak salience (theta_prominence).
            # Lower values recover small peaks from sparse / low-traffic lanes;
            # higher values suppress spurious peaks in busy or noisy scenes.
            prominence = self.theta.get('peak_prominence', torch.tensor(1.0))
            if isinstance(prominence, torch.Tensor):
                prominence = prominence.item()
            # Use the same salience floor for height so that genuine peaks
            # attenuated by Gaussian smoothing are not pre-filtered by the
            # height gate before the prominence test is applied.
            peaks, _ = find_peaks(smoothed_hist, height=prominence, distance=3, prominence=prominence)
            n_lanes = len(peaks)

            logger.info(f"Estimated number of lanes: {n_lanes}")
            lane_num_list.append(n_lanes)

            if is_save:
                ax1.hist(X, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if is_save:
                ax1.plot(bin_centers, hist_vals, label="Original", alpha=0.5)
                ax1.plot(bin_centers, smoothed_hist, label="Smoothed (Gaussian)", linewidth=2)
                ax1.set_title("Distribution of Mean X (Lane Width Separation)")
                ax1.set_xlabel("x_mean")
                ax1.grid(True)
                fig1.savefig(Path(self.fig_filepath, f"x_mean_distribution_contour_{cnts}.png"))
                plt.close('fig1')

            logger.info(f"Detected {lane_num_list[c]} lanes to cluster")

            if lane_num_list[c] != 0:
                kmeans = KMeans(n_clusters=lane_num_list[c], random_state=0).fit(X)
                trajectory_summary = trajectory_summary.with_columns([
                    pl.lit(kmeans.labels_).cast(pl.Int64).alias("clustered_id")
                ])
                cluster_assignments.append(trajectory_summary.select(["id", "clustered_id"]))

                df_labeled = df_contour.join(
                    trajectory_summary,
                    on="id",
                    how="left"
                )

                df_plot = df_labeled.to_pandas()

                lane_boundaries_by_id = self.compute_lane_geometry(df_plot)
                lane_boundaries_for_contour[cnts] = lane_boundaries_by_id

                # Plot points by lane
                logger.info(f"Detected {len(df_plot['clustered_id'].unique())} lanes")

                for lane_id, data in lane_boundaries_by_id.items():
                    lane_df = df_plot[df_plot["clustered_id"] == lane_id]

                    if is_save:
                        # Temporary lat and lon is opposite
                        ax2.scatter(lane_df["y_gps"], lane_df["x_gps"], s=1, alpha=0.5, label=f"Lane {lane_id}", color=self.colors[int(lane_id)]) 
                        ax3.scatter(lane_df["y_gps"], lane_df["x_gps"], s=1, alpha=0.5, label=f"Lane {lane_id}", color=self.colors[int(lane_id)])
                        ax3.plot(data["center"][:, 1], data["center"][:, 0], color=self.colors[int(lane_id)], linewidth=2.5, label=f"Lane Centerline {lane_id}")
                        ax3.plot(data["left"][:, 1], data["left"][:, 0], color=self.colors[int(lane_id)], linewidth=1.0)
                        ax3.plot(data["right"][:, 1], data["right"][:, 0], color=self.colors[int(lane_id)], linewidth=1.0)

        if not cluster_assignments:
            # No contour produced any lane cluster (e.g. an aggressive theta such as a
            # high peak_prominence suppressing every histogram peak). Return the designed
            # "no lanes detected" outcome so the trial search scores this theta as bad
            # instead of crashing the whole client round.
            logger.warning(f"{camera_loc}: no lane clusters found for any contour (theta={self.theta})")
            traj_df = traj_df.with_columns(pl.lit(-1, dtype=pl.Int64).alias("clustered_id"))
            return traj_df, lane_boundaries_for_contour

        all_assignments = pl.concat(cluster_assignments)

        # Make sure to drop the placeholder column before joining
        traj_df = traj_df.join(all_assignments, on=["id"], how="left")

        if is_save:
            ax2.set_xlabel("X (Longitude)")
            ax2.set_ylabel("Y (Latitude)")
            ax2.set_title("Trajectory Clusters")
            ax2.grid(True)
            # Deduplicate legend entries
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # overwrites duplicates
            ax2.legend(by_label.values(), by_label.keys())
            fig2.savefig(Path(self.fig_filepath, f"trajectory_clusters.png"))
            plt.close('fig2')

            ax3.set_xlabel("X (Longitude)")
            ax3.set_ylabel("Y (Latitude)")
            ax3.set_title("Trajectory Clusters with Lane Center Lines")
            ax3.grid(True)
            handles, labels = ax3.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # overwrites duplicates
            ax3.legend(by_label.values(), by_label.keys())
            fig3.savefig(Path(self.fig_filepath, f"lane_center_lines.png"))
            plt.close('fig3')
            plt.close('all')

        return traj_df, lane_boundaries_for_contour

    def update_theta(self, theta_dict):
        """Update theta parameters from dictionary, handling both tensor and float inputs."""
        for k, v in theta_dict.items():
            if isinstance(v, torch.Tensor):
                self.theta[k] = v.clone()
            else:
                self.theta[k] = torch.tensor(float(v))
            
            logger.info(f"[update_theta] Updated {k}: {self.theta[k].item():.4f}")

    def compute_bps(
            self,
            data_size,
            unit='MB',
            duration_seconds=1
        ):
        """
        Calculate Bits Per Second (BPS).

        Parameters:
        - data_size (float): Size of the data transferred.
        - unit (str): Unit of the data ('B', 'KB', 'MB', 'GB'). Default is 'MB'.
        - duration_seconds (float): Duration in seconds over which the data was transferred.

        Returns:
        - bps (float): Bits per second.
        """
        unit_multipliers = {
            'B': 8,
            'KB': 8 * 1024,
            'MB': 8 * 1024 ** 2,
            'GB': 8 * 1024 ** 3,
        }

        if unit not in unit_multipliers:
            raise ValueError("Unit must be one of: 'B', 'KB', 'MB', 'GB'")

        total_bits = data_size * unit_multipliers[unit]
        bps = total_bits / duration_seconds
        return bps

    def compute_loss(
            self,
            detected_center_list,
            sumo_center_list,
            detected_width,
            sumo_lane_shape=None,
            cluster_to_edge_map=None
        ):
        lane_num = len(detected_center_list)

        # Lane Count Loss
        l_lane_count = abs(len(detected_center_list) - len(sumo_center_list))

        if len(detected_center_list) != len(sumo_center_list):
            logger.warning(f"Lane count mismatch: Detected {len(detected_center_list)}, SUMO {len(sumo_center_list)}")
            # print(detected_center_list, sumo_center_list)

        # Lane consistency loss: Calculate for whole lane -> soft label cross-entropy
        # Ensure device consistency and gradient requirements
        if lane_num > 0:
            device = detected_center_list[0].device if hasattr(detected_center_list[0], 'device') else torch.device('cpu')
        else:
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device('cpu')

        l_cons_list = []
        l_trip = 0.0
        num_triplets = 0

        triplet_margin = self.theta.get('triplet_margin', torch.tensor(0.8))
        if isinstance(triplet_margin, torch.Tensor):
            triplet_margin = triplet_margin.item()
        
        triplet_loss_fn = LaneTripletLoss(margin=triplet_margin)

        # SUMO polylines can lose points to the graph dedup in create_sumo_graph
        # (coordinates shared across lanes in a group are dropped), so resample any
        # lane that doesn't match the detected 30-point resolution before stacking.
        target_points = detected_center_list.shape[1] if detected_center_list.dim() == 3 else 30
        sumo_center_list = [s for s in sumo_center_list if s.shape[0] >= 2]
        sumo_center_list = [
            s if s.shape[0] == target_points else torch.tensor(
                np.asarray(interpolate_edge(s.numpy(), num_points=target_points)), dtype=torch.float32)
            for s in sumo_center_list
        ]

        # Associate each detected lane with its reference lane by mean nearest-point
        # distance in meters. Polyline parameterization direction must not affect the
        # match (SUMO lane shapes often run opposite to travel direction), so no
        # index-aligned point comparison.
        matched_sumo_lanes = []
        matched_idx = []

        for detected in detected_center_list:
            min_dist = float('inf')
            closest, closest_j = None, -1

            for j, sumo in enumerate(sumo_center_list):
                dist = gps_pairwise_distance(detected, sumo).min(dim=1).values.mean()

                if dist < min_dist:
                    min_dist = dist
                    closest, closest_j = sumo, j

            matched_sumo_lanes.append(closest.unsqueeze(0)) # shape (1, 30, 2)
            matched_idx.append(closest_j)

        sumo_lanes = torch.cat(matched_sumo_lanes, dim=0) # shape (4, 30, 2)

        # Orient each matched reference to the detected lane's direction before the
        # order-sensitive comparisons below (Frechet walks both curves in sequence,
        # the triplet compares points index-aligned). SUMO lane shapes often run
        # opposite to the detected travel direction; without this the consistency
        # term measures lane length instead of deviation for reversed matches.
        oriented = []
        for lane in range(lane_num):
            P, Q = detected_center_list[lane], sumo_lanes[lane]
            ends = gps_pairwise_distance(P[[0, -1]], Q[[0, -1]])
            if ends[0, 1] + ends[1, 0] < ends[0, 0] + ends[1, 1]:
                Q = torch.flip(Q, dims=[0])
            oriented.append(Q.unsqueeze(0))
        sumo_lanes = torch.cat(oriented, dim=0)


        for lane in range(lane_num):
            ## DEBUG plot
            # plt.figure(figsize=(8, 6))

            # Plot all SUMO lanes in light gray
            # for j, sumo_lane in enumerate(sumo_center_list):
            #     x_sumo, y_sumo = sumo_lane[:, 0], sumo_lane[:, 1]
            #     plt.plot(x_sumo, y_sumo, 'o--', label=f'SUMO Lane {j}', alpha=0.4, color='gray')

            # Plot the detected lane
            # x_sumo, y_sumo = sumo_lanes[lane, :, 0], sumo_lanes[lane, :, 1]
            # plt.plot(x_sumo, y_sumo, 'o--', label=f'SUMO Lane {lane}', alpha=0.4, color='gray')
            # x_det, y_det = detected_center_list[lane, :, 0], detected_center_list[lane, :, 1]
            # plt.plot(x_det, y_det, 'ro-', linewidth=2, label=f'Detected Lane {lane}')

            # plt.title(f'Detected Lane {lane} vs All SUMO Lanes')
            # plt.xlabel("SUMO X")
            # plt.ylabel("SUMO Y")
            # plt.axis('equal')
            # plt.grid(True)
            # plt.legend()
            # plt.savefig(Path(self.fig_filepath, f"detected_vs_sumo_lane_{lane}.png"))

            if detected_center_list[lane].shape[0] > 1 and sumo_lanes[lane].shape[1] > 1:
                frechet_dist = frechet_distance(detected_center_list[lane], sumo_lanes[lane])
                l_cons_list.append(frechet_dist)

            # Eq. 7 contrastive term: anchor = detected lane, positive = its matched
            # reference, negative = the nearest reference lane that is NOT the
            # positive (hard negative). With a single reference lane in view there is
            # no valid negative and the term is skipped for that lane.
            anchor = detected_center_list[lane].to(device)
            positive = sumo_lanes[lane].to(device)
            neg_candidates = [s for j, s in enumerate(sumo_center_list) if j != matched_idx[lane]]
            if neg_candidates:
                neg_dists = torch.stack([
                    gps_pairwise_distance(anchor, s.to(device)).min(dim=1).values.mean()
                    for s in neg_candidates
                ])
                negative = neg_candidates[int(neg_dists.argmin())].to(device)
                ends = gps_pairwise_distance(anchor[[0, -1]], negative[[0, -1]])
                if ends[0, 1] + ends[1, 0] < ends[0, 0] + ends[1, 1]:
                    negative = torch.flip(negative, dims=[0])
                l_trip += triplet_loss_fn(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                num_triplets += 1

        # Combine consistency losses
        if l_cons_list:
            l_cons = torch.stack(l_cons_list).mean()
            if 'consistency_weight' in self.theta:
                cons_weight = self.theta['consistency_weight']
                if isinstance(cons_weight, torch.Tensor):
                    cons_weight = cons_weight.item()
                l_cons = l_cons * cons_weight
        else:
            l_cons = torch.tensor(0.0, device=device)
        l_trip = l_trip / max(1, num_triplets)

        # Geometry loss: width + length comparison
        raw_width_m = float('nan')
        if sumo_lane_shape is not None and cluster_to_edge_map is not None:
            detected_length_list = []
            # detected_length = torch.zeros(lane_num, device=device, requires_grad=True)
            sumo_width = torch.zeros(lane_num, device=device)

            for i in range(lane_num):
                detected_center_gps = gps_to_cartesian(detected_center_list[i])
                # x, y = detected_center_list[i][:, 0], detected_center_list[i][:, 1]
                x, y = detected_center_gps[:, 0], detected_center_gps[:, 1]
                max_x, min_x = torch.max(x), torch.min(x)
                max_y, min_y = torch.max(y), torch.min(y)

                # detected_length[i] = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
                length_calc = torch.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + 1e-8)
                detected_length_list.append(length_calc)

            # Fill OSM length and width arrays
            sumo_length_list = []
            sumo_width_list = []
            
            for i in range(lane_num):
                sumo_length_list.append(torch.tensor(0.0, device=device))
                sumo_width_list.append(torch.tensor(0.0, device=device))

            for i, (cluster_id, (_, lane_id)) in enumerate(cluster_to_edge_map.items()):
                lane_keys = list(sumo_lane_shape.keys())
                lane_shape = sumo_lane_shape[lane_keys[int(lane_id)]]
                sumo_length_list[i] = torch.tensor(float(lane_shape[0]), dtype=torch.float32, device=device)
                sumo_width_list[i] = torch.tensor(float(lane_shape[1]), dtype=torch.float32, device=device)

            detected_width = [
                dw.to(device) if isinstance(dw, torch.Tensor) else torch.tensor(dw, dtype=torch.float32, device=device)
                for dw in detected_width
            ]
            detected_width = torch.stack(detected_width) if detected_width else torch.zeros(lane_num, device=device)

            # Per-lane reference widths from the SUMO net (Eq. 6's map-derived c_mw).
            # Unmatched lanes (sumo_width_list left at 0.0) fall back to the 3.2 m default.
            ref_widths = torch.stack([
                w if float(w) > 0 else torch.tensor(3.2, device=device)
                for w in sumo_width_list
            ])
            sumo_width = ref_widths.unsqueeze(1).expand(lane_num, detected_width.shape[1])

            width_errors = torch.stack([
                torch.mean((sumo_width[i] - detected_width[i]) ** 2)
                for i in range(lane_num)
            ])
            width_term = torch.sqrt(torch.sum(width_errors))

            # Not sure if we need to use length_term here as SUMO length is articulated by finding the closest lanes
            # l_geo = (width_term + length_term) / lane_num
            l_geo = width_term

            # Raw, unweighted mean absolute width error in meters against the real
            # per-lane SUMO reference. The learnable width_scale is divided back out
            # so the metric measures the trajectory-derived width estimate itself and
            # stays comparable across models with different learned scales.
            ws = self.theta.get('width_scale', 1.0)
            ws = float(ws.item()) if isinstance(ws, torch.Tensor) else float(ws)
            unscaled_width = detected_width / ws if ws > 0 else detected_width
            raw_width_m = float(torch.mean(torch.abs(unscaled_width - sumo_width)).item())


        weight_lane = self.theta.get('weight_lane_count', torch.tensor(1.0))
        weight_cons = self.theta.get('weight_consistency', torch.tensor(1.0))
        weight_trip = self.theta.get('weight_triplet', torch.tensor(1.0))
        weight_geo = self.theta.get('weight_geometry', torch.tensor(1.0))
        
        # Convert to float if tensor
        if isinstance(weight_lane, torch.Tensor):
            weight_lane = weight_lane.item()
        if isinstance(weight_cons, torch.Tensor):
            weight_cons = weight_cons.item()
        if isinstance(weight_trip, torch.Tensor):
            weight_trip = weight_trip.item()
        if isinstance(weight_geo, torch.Tensor):
            weight_geo = weight_geo.item()
        
        # Weighted total loss
        l_total = (weight_lane * l_lane_count * 10 + weight_cons * l_cons + \
                    weight_trip * l_trip + weight_geo * l_geo)

        # ---- Model-independent geometric metrics (meters), free of learned weights ----
        # Reported for fair cross-model comparison (Table 1). Unlike l_cons (scaled by
        # the learned consistency_weight), l_trip (a margin-offset triplet) and l_total
        # (baseline weights sum to 4 vs. meta's softmax weights sum to 1), these raw
        # quantities are computed identically for every model.
        if l_cons_list:
            raw_consistency_m = float(torch.stack(l_cons_list).mean().item())
        else:
            raw_consistency_m = float('nan')

        dev_list = []
        for lane in range(lane_num):
            P = detected_center_list[lane]
            Q = sumo_lanes[lane]
            if P.shape[0] > 0 and Q.shape[0] > 0:
                D = gps_pairwise_distance(P, Q)  # (m, n) in meters
                dev_list.append(D.min(dim=1).values.mean())
        raw_centerline_m = float(torch.stack(dev_list).mean().item()) if dev_list else float('nan')

        # Recall direction: how far is each reference SUMO lane from its nearest
        # DETECTED lane. Centerline/consistency above only score the lanes a model
        # chose to detect, so a model that skips hard lanes is never charged for
        # them; coverage is where a missed lane shows up.
        cov_list = []
        for Q in sumo_center_list:
            if Q.shape[0] == 0:
                continue
            per_detected = []
            for lane in range(lane_num):
                P = detected_center_list[lane]
                if P.shape[0] > 0:
                    D = gps_pairwise_distance(Q, P)  # (points_Q, points_P) in meters
                    per_detected.append(D.min(dim=1).values.mean())
            if per_detected:
                cov_list.append(torch.stack(per_detected).min())
        raw_coverage_m = float(torch.stack(cov_list).mean().item()) if cov_list else float('nan')

        comps = [raw_consistency_m, raw_centerline_m, raw_width_m, raw_coverage_m]
        geo_total_m = float(np.nansum(comps)) if any(not np.isnan(c) for c in comps) else float('nan')

        raw_metrics = {
            'geo_consistency_m': raw_consistency_m,  # mean Frechet distance to reference centerline
            'geo_coverage_m': raw_coverage_m,        # mean reference-lane distance to nearest detected lane (recall)
            'geo_centerline_m': raw_centerline_m,    # mean nearest-point centerline deviation
            'geo_width_m': raw_width_m,              # mean |detected - reference| lane width
            'geo_total_m': geo_total_m,              # equal-weight sum of the three (all meters)
            'lane_count_err': float(l_lane_count),   # |N_det - N_ref|
            'lane_count_exact': 1.0 if l_lane_count == 0 else 0.0,  # exact match -> accuracy %
        }

        return l_total, l_lane_count, l_cons, l_trip, l_geo, raw_metrics