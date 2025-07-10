import cv2
import argparse
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
import polars as pl
from collections import defaultdict
from dataclasses import dataclass

from ultralytics import YOLO
from .detector import YOLO_darknet as YOLOV11


@dataclass
class CameraData:
    collect_cars: list
    collect_dots: list
    out_df: pl.DataFrame
    last_frame: np.ndarray


class LaneContinuousProcess:
    """Continuous Data Collection
    """
    def __init__(self, args, saving_file_path, camera_loc_list):
        self.left_lane_nums = 0
        self.right_lane_nums = 0
        self.lambda_thres = args.lambda_thres
        self.filepath = saving_file_path
        self.saving_file_path = Path(saving_file_path)
        self.camera_loc_list = camera_loc_list

        self.data_by_camera = {}

        self.conf_settings(args)

    def conf_settings(self, args):
        self.video_path = Path(args.video_path)
        self.is_save = args.is_save
        # self.current_frame = 0

        '''Define global variables for cycle learning'''
        # self.collect_cars = []
        # self.collect_det_dots_including_truck = []
        self.detection_period = args.T # 60 seocnds
        if not args.skip_continuous_learning:
            self.detector = YOLOV11()

    def continuous_process(
            self,
            args,
            c_epoch):
        """Continuous Data Collection Process for Lane Learning

        Args:
            args (_type_): _description_
            epoch (_type_): _description_

        Returns:
            global_lane_centers (list): The global lane centers
        """

        self.vehicle_collected = False
        frame_id = 0 

        # Multiple video data in the self.video_path directory
        if self.video_path.is_dir():
            files = [f for f in self.video_path.iterdir() if f.is_file()]
            
            for v_path in files:
                camera_loc = v_path.stem

                if camera_loc in self.camera_loc_list:
                    logger.info(f"Reading: {camera_loc}")
                    current_frame = 0
                    collect_cars = []
                    collect_det_dots_including_truck = []

                    video = cv2.VideoCapture(v_path)
                    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Video FPS: {fps}")
                    if fps != 0: # It works even if you don't have the video
                        duration = frame_count / fps
                        logger.info(f"Video Duration: {duration}")
                    logger.info(f"Total frames: {frame_count}")

                    self.track_model = YOLO("yolo11n.pt")

                    fig_filepath = Path(self.saving_file_path, camera_loc, "figures")
                    pre_filepath = Path(self.saving_file_path, camera_loc, "preprocess")

                    # Accumulate historical data to current streams
                    if (args.use_historical_data or args.skip_continuous_learning): #and c_epoch != 0:
                        last_frame = np.load(Path(pre_filepath, f"last_frame.npy"))
                        collect_cars_list = np.load(Path(pre_filepath, f"collect_cars.npy")).tolist()
                        collect_det_dots_including_truck_list = np.load(Path(pre_filepath, f"collect_det_dots_including_truck.npy")).tolist()
                        # Convert back to tuples
                        collect_cars = [tuple(item) if isinstance(item, list) else item for item in collect_cars_list]
                        collect_det_dots_including_truck = [tuple(item) if isinstance(item, list) else item for item in collect_det_dots_including_truck_list]
                        out_df = pl.read_csv(
                            Path(pre_filepath, f"trajectory.csv"),
                            schema_overrides={"target_lane_id": pl.Utf8}
                        )

                    if not args.skip_continuous_learning: #or c_epoch == 0:

                        track_history = defaultdict(lambda: [])
                        frame_time = 0
                        is_track_show = True
                        frame_data_list = []

                        while video.isOpened():
                            ret, frame = video.read()
                            if frame_time == 0:
                                first_frame = frame

                            if ret:
                                # current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
                                video_time = current_frame / fps

                                if video_time >= self.detection_period:
                                    break
                                else:
                                    results = self.track_model.track(frame, persist=True, verbose=False)

                                    # Get the boxes and track IDs
                                    if results[0].boxes.id is None:
                                        continue
                                    boxes = results[0].boxes.xywh.cpu()
                                    track_ids = results[0].boxes.id.int().cpu().tolist()
                                    obj_cls = results[0].boxes.cls.cpu().tolist()
                                    confs = results[0].boxes.conf.cpu().tolist()

                                    # Visualize the results on the frame
                                    annotated_frame = results[0].plot()

                                    # Plot the tracks
                                    for box, track_id, obj_cls, conf in zip(boxes, track_ids, obj_cls, confs):
                                        x, y, w, h = box
                                        if is_track_show:
                                            # This is the trajectory for each object
                                            track = track_history[track_id]
                                            track.append((float(x), float(y))) # x, y center point
                                            if len(track) > 30: # retain 90 tracks for 90 frames
                                                track.pop(0)

                                            # Draw the tracking lines
                                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
                                            cv2.circle(first_frame, (int(x), int(y)), 2, (0, 255, 0), 2)

                                        # Save to csv
                                        cap_data = {
                                            "id": track_id,
                                            "time": frame_time / fps,
                                            "frame_num": frame_time,
                                            "class": obj_cls,
                                            "conf": conf,
                                            "x": int(x),
                                            "y": int(y),
                                            "w": int(w),
                                            "h": int(h),
                                        }

                                        if 10 < box[2] < 550 and 10 < box[3] < 550:
                                            if obj_cls == 2:
                                                collect_cars.append((x, y, w, h, frame_id, conf))
                                            if obj_cls == 2 or obj_cls == 7:
                                                collect_det_dots_including_truck.append((x, y, int(w / 4 * 3), h, frame_id, conf))

                                        frame_data_list.append(cap_data)

                                    frame_time += 1
                                    # if is_track_show:
                                        # Display the annotated frame
                                        # cv2.imshow("YOLO11 Tracking", annotated_frame)

                                        # Break the loop if 'q' is pressed
                                        # if cv2.waitKey(1) & 0xFF == ord("q"):
                                        #     break
                                last_frame = frame
                            else:
                                break

                            current_frame += 1

                        out_df = pl.DataFrame(frame_data_list, orient="row")
                        out_df.write_csv(Path(pre_filepath, f"trajectory.csv"))
                        cv2.imwrite(Path(fig_filepath, f"{c_epoch}_trajectory_plotted.png"), first_frame)

                        # Release the video capture object and close the display window
                        video.release()
                        cv2.destroyAllWindows()

                        # Save collected info
                        if self.is_save:
                            np.save(Path(pre_filepath, f"last_frame.npy"), last_frame)
                            np.save(Path(pre_filepath, f"collect_cars.npy"), collect_cars)
                            np.save(Path(pre_filepath, f"collect_det_dots_including_truck.npy"), collect_det_dots_including_truck)

                    # One time save for the first epoch
                    if self.is_save and c_epoch == 0:
                        filename = Path(fig_filepath, f"input_image_{c_epoch}.png")
                        cv2.imwrite(filename, last_frame)

                    self.data_by_camera[camera_loc] = CameraData(
                        collect_cars=collect_cars,
                        collect_dots=collect_det_dots_including_truck,
                        out_df=out_df,
                        last_frame=last_frame
                    )

            return self.data_by_camera


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='../../../dataset/511video/test2.mp4', help='camera ip or local video path')
    parser.add_argument('--saving_path', default='./results/',
                        help='path to save results')
    parser.add_argument('--T', type=int, default=60, help='Time interval of each cycle, the unit is second')
    parser.add_argument('--is_save', action='store_true', help='Save the results or not')
    parser.add_argument('--conf_thre', type=float, default='0.25', help='Detection confidence score threshold when creating '
                                                                        'the road segment')
    parser.add_argument('--use_historical_data', action='store_true', help='Use historical data or not')
    parser.add_argument('--skip_continuous_learning', action='store_true', help='Skip continuous learning or not')
    parser.add_argument('--lambda_thre', type=int, default='120', help='Criteria of stopping the cycle learning')
    args = parser.parse_args()
    print(args)

    lane_detection = LaneContinuousProcess(args)
    lane_detection.continuous_learning(args)
    cv2.destroyAllWindows()
    print("Finished!")