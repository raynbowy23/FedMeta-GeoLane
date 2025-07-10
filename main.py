import os
from pathlib import Path
import matplotlib
matplotlib.set_loglevel(level = 'warning')
matplotlib.use('Agg')
import random
import mlflow


import numpy as np
import cv2
import time
import argparse
import shutil
import queue
import threading
import traceback
import torch

from geolearning_pipeline import GeometricLearningPipeline, continuous_process, geometric_learning
from OpenDriveConversion.det2sumo_sync import Det2SumoSync

import logging

parser = argparse.ArgumentParser(description="Parameter Settings for Training")
parser.add_argument('--num_grids', type=int, default=50, help='Number of grids in the graph')
parser.add_argument('--seed', default=42, help="seed number", type=int)
parser.add_argument('--dataset_path', default='./dataset/', help='dataset path')
parser.add_argument('--video_path', default='./dataset/511video', help='camera ip or local video path')
parser.add_argument('--saving_path', default='./results/', help='path to save results')
parser.add_argument('--T', type=int, default=60, help='Time interval of each cycle, the unit is second')
parser.add_argument('--is_save', action='store_true', help='Save the results or not')
parser.add_argument('--conf_thre', type=float, default='0.25', help='Detection confidence score threshold when creating '
                                                                    'the road segment')
parser.add_argument('--osm_path', type=str, default="./LaneDetection/osm_extraction/", help='The path of the OSM file to extract data from')
parser.add_argument('--model', type=str, default='federated', help='Model type: federated, baseline, meta')
parser.add_argument('--use_historical_data', action='store_true', help='Use historical data or not')
parser.add_argument('--skip_continuous_learning', action='store_true', help='Skip continuous learning or not')
parser.add_argument('--lambda_thres', type=int, default='120', help='Criteria of stopping the cycle learning')
parser.add_argument('--cnts_threshold', type=int, default='0', help='Contours threshold')
parser.add_argument('--centralized', action='store_true', help='Centralized learning or not')

# Federated learning specific arguments
parser.add_argument('--federated', action='store_true', help='Use federated meta-learning')
parser.add_argument('--fed_rounds', type=int, default=100, help='Number of federated learning rounds')
parser.add_argument('--client_selection_ratio', type=float, default=0.8, help='Ratio of clients to select per round')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate for meta-model')

args = parser.parse_args()
print(args)

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'logs/{args.model}_test.log', filemode='w', encoding='utf-8', level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

if not logger.hasHandlers():
    handler = logging.FileHandler("test_log_output.log")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
logger.info("Logging is working!")

file_name_ = Path(args.video_path).stem
saving_file_path = Path(args.saving_path, file_name_, args.model)
os.makedirs(saving_file_path, exist_ok=True)
logger.info(f"Filepath: {saving_file_path}")
fig_filepath = Path(saving_file_path, "figures")

# meta_model = MetaMLModel()
# optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup camera locations
camera_loc_list = []
with open(Path(args.dataset_path, "camera_location_list.txt"), 'r') as f:
    for camera_loc in f:
        camera_loc_name = camera_loc.strip()
        print(camera_loc_name)

        # Set up file path for each camera_loc_name
        Path(saving_file_path, camera_loc_name).mkdir(parents=True, exist_ok=True)
        Path(saving_file_path, camera_loc_name, "figures").mkdir(parents=True, exist_ok=True) # Figure path
        Path(saving_file_path, camera_loc_name, "preprocess").mkdir(parents=True, exist_ok=True) # Preprocess
        Path(saving_file_path, camera_loc_name, "preprocess", "graph").mkdir(parents=True, exist_ok=True) # Graph path
        Path(saving_file_path, camera_loc_name, "pixel").mkdir(parents=True, exist_ok=True) # Pixel path
        Path(saving_file_path, camera_loc_name, "sumo").mkdir(parents=True, exist_ok=True) # Sumo path

        # Copy SUMO files
        sumo_input_file = Path(args.osm_path, camera_loc_name, "osm.net.xml")
        sumo_output_file = Path(saving_file_path, camera_loc_name, "sumo", str(camera_loc_name) + ".net.xml")
        if sumo_input_file.exists():
            shutil.copy(sumo_input_file, sumo_output_file)

        camera_loc_list.append(camera_loc_name)


barrier = threading.Barrier(2) # 2 threads need to sync

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # If using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    mlflow.set_experiment(f"Federated Meta Lane Detection Experiment - {args.model}")

    with mlflow.start_run():
        mlflow.log_param("strategy", args.model)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("device", device)
        mlflow.log_param("num_cameras", len(camera_loc_list))

        # Create pipeline
        pipeline = GeometricLearningPipeline(
            args, saving_file_path, camera_loc_list, args.model
        )

        # Set epochs based on strategy
        if args.model == 'baseline':
            CONT_EPOCHS = 1 # Epochs for continuous data collection
            GEO_EPOCHS = 1  # Epochs for federated learning
        else:
            CONT_EPOCHS = 20
            GEO_EPOCHS = 20

        # Setup threading
        data_queue = queue.Queue()
        stop_event = threading.Event()
        barrier = threading.Barrier(2)

        # Start continuous data collection thread
        continuous_thread = threading.Thread(
            target=continuous_process,
            args=(args, CONT_EPOCHS, data_queue, stop_event, saving_file_path, camera_loc_list, barrier)
        )
    
        # Start geometric learning thread
        geometric_thread = threading.Thread(
            target=geometric_learning, 
            args=(pipeline, GEO_EPOCHS, data_queue, stop_event, barrier)
        )
    
    
        logger.info(f"Starting geometric learning pipeline with strategy: {args.model}")
        logger.info(f"Continuous epochs: {CONT_EPOCHS}, Geometric epochs: {GEO_EPOCHS}")

        continuous_thread.start()
        geometric_thread.start()
        
        continuous_thread.join()
        geometric_thread.join()
        
        cv2.destroyAllWindows()
        logger.info("Geometric learning finished successfully!")


def simulator_sync():
    """Synchronize with simulator"""
    logger.info("[Simulator Sync] Start")

    try:
        Det2SumoSync(args, saving_file_path).run()
        time.sleep(1.5)
        logger.info("[Simulator Sync] Finish")
    except Exception as e:
        logger.error(f"Error in simulator sync: {e}")
    
if __name__ == '__main__':
    logger.info(f"Starting lane detection with arguments: {args}")

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        traceback.print_exc()
    finally:
        logger.info("Program terminated")