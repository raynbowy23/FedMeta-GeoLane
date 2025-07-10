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
                # Fit spline
                spline = UnivariateSpline(y, x, s=smoothing)
                y_fit = np.linspace(y.min(), y.max(), num=num_points)
                x_fit = spline(y_fit)

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


    def run(
            self, c_epoch, g_epoch, traj_df,
            camera_loc, trial, is_save
        ):
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

            # Adaptive peak detection based on theta
            # if 'angle_penalty' in self.theta:
            #     angle_val = self.theta['angle_penalty'].item() if isinstance(self.theta['angle_penalty'], torch.Tensor) else self.theta['angle_penalty']
            #     # Adjust peak detection sensitivity based on angle penalty
            #     prominence = 1 + angle_val  # Higher angle penalty = more strict peak detection
            #     peaks, _ = find_peaks(smoothed_hist, height=1, distance=3, prominence=prominence)
            # else:
            #     peaks, _ = find_peaks(smoothed_hist, height=1, distance=3, prominence=1)
            peaks, _ = find_peaks(smoothed_hist, height=1, distance=3, prominence=1)
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
                        # ax2.scatter(lane_df["x_gps"], lane_df["y_gps"], s=1, alpha=0.5, label=f"Lane {lane_id}", color=self.colors[int(lane_id)]) 
                        # ax3.scatter(lane_df["x_gps"], lane_df["y_gps"], s=1, alpha=0.5, label=f"Lane {lane_id}", color=self.colors[int(lane_id)])
                        # ax3.plot(data["center"][:, 0], data["center"][:, 1], color=self.colors[int(lane_id)], linewidth=2.5, label=f"Lane Centerline {lane_id}")
                        # ax3.plot(data["left"][:, 0], data["left"][:, 1], color=self.colors[int(lane_id)], linewidth=1.0)
                        # ax3.plot(data["right"][:, 0], data["right"][:, 1], color=self.colors[int(lane_id)], linewidth=1.0)

                        # Temporary lat and lon is opposite
                        ax2.scatter(lane_df["y_gps"], lane_df["x_gps"], s=1, alpha=0.5, label=f"Lane {lane_id}", color=self.colors[int(lane_id)]) 
                        ax3.scatter(lane_df["y_gps"], lane_df["x_gps"], s=1, alpha=0.5, label=f"Lane {lane_id}", color=self.colors[int(lane_id)])
                        ax3.plot(data["center"][:, 1], data["center"][:, 0], color=self.colors[int(lane_id)], linewidth=2.5, label=f"Lane Centerline {lane_id}")
                        ax3.plot(data["left"][:, 1], data["left"][:, 0], color=self.colors[int(lane_id)], linewidth=1.0)
                        ax3.plot(data["right"][:, 1], data["right"][:, 0], color=self.colors[int(lane_id)], linewidth=1.0)

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

        # Adjust sumo lane number for detected results.
        matched_sumo_lanes = []

        for detected in detected_center_list:
            min_dist = float('inf')
            closest = None

            for sumo in sumo_center_list:
                # dist = torch.cdist(detected, sumo, p=2).mean()
                try:
                    dist = torch.sqrt(torch.sum((detected[15] - sumo[15])**2))
                except:
                    dist = torch.sqrt(torch.sum((torch.mean(detected) - torch.mean(sumo))**2))

                if dist < min_dist:
                    min_dist = dist
                    closest = sumo

            matched_sumo_lanes.append(closest.unsqueeze(0)) # shape (1, 30, 2)

        sumo_lanes = torch.cat(matched_sumo_lanes, dim=0) # shape (4, 30, 2)


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

            # print(f"SUMO length: {len(sumo_lanes)}, Detected length: {len(detected_center_list)}")
            anchor = detected_center_list[lane].to(device)
            positive = sumo_lanes[lane].to(device)
            for j in range(lane_num):
                if lane == j:
                    negative = detected_center_list[j].to(device)
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
            sumo_width = torch.full((len(detected_width),len(detected_width[0])), 3.2)
            detected_width = torch.stack(detected_width) if detected_width else torch.zeros(lane_num, device=device)

            width_errors = torch.stack([
                torch.mean((sumo_width[i] - detected_width[i]) ** 2)
                for i in range(lane_num)
            ])
            width_term = torch.sqrt(torch.sum(width_errors))

            # Not sure if we need to use length_term here as SUMO length is articulated by finding the closest lanes
            # l_geo = (width_term + length_term) / lane_num
            l_geo = width_term


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

        return l_total, l_lane_count, l_cons, l_trip, l_geo