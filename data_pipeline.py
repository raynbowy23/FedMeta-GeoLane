import os
import cv2
import numpy as np
from pathlib import Path
from collections import deque

import matplotlib
from concurrent.futures import ThreadPoolExecutor
matplotlib.set_loglevel(level = 'warning')
import logging
logger = logging.getLogger(__name__)

import argparse
from tqdm import tqdm

from LaneDetection.lane_detection.lane_continuous_process import LaneContinuousProcess
from LaneDetection.lane_detection.utils import *
from LaneDetection.osm_extraction.connect_to_osm import OSMConnection
from LaneDetection.osm_extraction.utils import *


class OrbitDataPipeline:
    def __init__(self, args, saving_file_path):
        self.args = args
        self.filepath = saving_file_path
        self.lambda_thres = args.lambda_thres
        self.cnts_threshold = args.cnts_threshold
        self.is_save = args.is_save

        self.num_grids = args.num_grids
        self.lane_class = 5

        self.graph_buffer = deque(maxlen=500) # Store recent 500 frames
        self.osm_connection = OSMConnection(args, self.filepath)

    def create_binary_image(self, camera_loc, c_epoch, frame, collect_cars):
        """
          Generate a binary Image with detected Vehicles
        """
        self.c_epoch = c_epoch
        fig_filepath = Path(self.filepath, camera_loc, "figures")  
        binary_heatmap_area, detected_vehs = create_binary_image_utils(c_epoch, collect_cars, frame, fig_filepath, self.is_save)
        return binary_heatmap_area, detected_vehs

    def generate_contour(self, camera_loc, frame, binary_heatmap_area):
        """ generate valid road contour based on generated binary image in the last step

        Args:
            binrary_headmap_area: binary image created in file
            frame: current input video frame

        Returns: remain_cnts: valid road contours' set
        """
        min_area = 12000
        max_area = 6000000

        fig_filepath = Path(self.filepath, camera_loc, "figures")  

        def process_contour(c):
            x, y, w, h = cv2.boundingRect(c)
            approx_area = w * h
            if not (min_area < approx_area < max_area):
                return None

            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                return None

            [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.atan(-vy / vx) * 180 / np.pi

            if abs(angle) > 5:
                return c
            return None

        # Detect contours
        cnts = cv2.findContours(binary_heatmap_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # If binary lane detection detects separate contours which is supposed to be one contour, merge them
        min_merge_area = 10000
        if self.cnts_threshold > 0:
            # target_x = binary_heatmap_area.shape[1] // self.cnts_threshold
            # target_contours = [cnt for cnt in cnts if np.max(cnt[:, :, 0]) > target_x]
            target_contours = [cnt for cnt in cnts if cv2.contourArea(cnt) < min_merge_area]

            connected_lanes = np.zeros_like(binary_heatmap_area)
            cv2.drawContours(connected_lanes, target_contours, -1, 255, thickness=cv2.FILLED)

            # Apply dilation to connect the lanes. 
            # Change the kernel size to adjust the connection (larger kernel size = more connection)
            kernel = np.ones((15, 15), np.uint8)
            connected_lanes = cv2.dilate(connected_lanes, kernel, iterations=1)

            merged_lanes = cv2.bitwise_or(binary_heatmap_area, connected_lanes)
            cnts = cv2.morphologyEx(merged_lanes, cv2.MORPH_CLOSE, kernel)


        # Process contours in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_contour, cnts))

        valid_contours = [c for c in results if c is not None]

        # Save the output image
        if self.is_save:
            # Draw all valid contours at once
            cv2.drawContours(frame, valid_contours, -1, (255, 0, 0), 2)
            cv2.imwrite(Path(fig_filepath, f"{self.c_epoch}_contour_on_roadmap.png"), frame)
        return valid_contours
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='./dataset/511video/test2.mp4', help='camera ip or local video path')
    parser.add_argument('--saving_path', default='./results/', help='path to save results')
    parser.add_argument('--T', type=int, default=60, help='Time interval of each cycle, the unit is second')
    parser.add_argument('--is_save', action='store_true', help='Save the results or not')
    parser.add_argument('--conf_thre', type=float, default='0.25', help='Detection confidence score threshold when creating '
                                                                        'the road segment')
    parser.add_argument('--osm_save_date', type=str, default="./LaneDetection/osm_extraction/2024-12-04-23-05-50", help='The date of the OSM file to extract data from')
    parser.add_argument('--use_historical_data', action='store_true', help='Use historical data or not')
    parser.add_argument('--skip_continuous_learning', action='store_true', help='Skip continuous learning or not')
    parser.add_argument('--lambda_thres', type=int, default='120', help='Criteria of stopping the cycle learning')
    parser.add_argument('--cnts_threshold', type=int, default='0', help='Contours threshold')
    args = parser.parse_args()
    print(args)

    file_name_ = Path(args.video_path).stem
    saving_file_path = Path(args.saving_path, file_name_)
    os.makedirs(saving_file_path, exist_ok=True)
    logger.info(f"Filepath: {saving_file_path}")

    # Lane detection algorithm
    lane_detection = LaneContinuousProcess(args, saving_file_path)
    data_pipeline = OrbitDataPipeline(args, saving_file_path)

    # Initializations
    new_detected_centers = None
    adjusted_points = None

    CONT_EPOCHS = 1 # Epochs for continuous loop for data addition
    GEO_EPOCHS = 1 # Epochs for each feedback loop for learning process
    for c_epoch in tqdm(range(CONT_EPOCHS), desc="Continuous Process"):
        logger.info(f"Continuous Process Epoch: {c_epoch}")
        last_frame, collect_cars, collect_det_dots_including_truck, traj_df = lane_detection.continuous_process(args, c_epoch)

        logger.info(f"Data Processing...")
        frame = last_frame.copy()
        node_matrix, edge_matrix = data_pipeline.generate_adjacency_matrix(c_epoch, frame, collect_cars, traj_df)
        data_pipeline.update_graph_buffer(node_matrix, edge_matrix)
        logger.info(f"Graph Buffer Updated")