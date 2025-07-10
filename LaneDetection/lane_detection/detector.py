from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

import torch
import time
from collections import defaultdict

class YOLO_darknet:
    def __init__(self, is_classify=False, is_trace=False, half=False):

        self.model = YOLO("yolo11n.pt")

        self.stride = int(self.model.stride.max())
        self.img_size = 640
        self.imgsz = check_imgsz(self.img_size, stride=self.stride)
        self.is_classify = is_classify
        self.is_trace = is_trace
        self.half= half
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = "cuda:0"
        self.auto = True
        self.max_det = 1000
        self.agnostic_nms = False
        self.classes = None

        # Store the track history
        self.track_history = defaultdict(lambda: [])

        # Run inference
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters()))) # run once

    def detect(self, frame):
        # Padded resize
        prev_time = time.time()

        preds = self.model.predict(
            source=frame,
            data="bytetrack.yaml",
            epochs=100,
            imgsz=self.imgsz,
            vid_stride=self.stride,
            iou=self.iou_thres,
            conf=self.conf_thres,
            device="0",
            classes=self.classes,
            agnostic_nms=self.agnostic_nms,
            verbose=False
        )

        fps = int(1 / (time.time() - prev_time))
        # print("FPS: {}".format(fps))
        detection_list = []
        boxes_list = []
        confidence = []
        classes = []
        ratios = []
        # print("lenpred",len(pred))
        for pred in preds:
            boxes = pred.boxes.xywh
            confs = pred.boxes.conf
            track_ids = pred.boxes.id
            clss = pred.boxes.cls

            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                class_id = int(clss[i])
                conf = float(confs[i])
                x_center, y_center, width, height = box.tolist()

                track = self.track_history[track_id]
                track.append((float(x_center), float(y_center)))

                if class_id in [2, 7]:
                    label = "car" if class_id == 2 else "truck"

                    detection_list.append([label, conf, int(x_center - width / 2), int(y_center - height / 2), int(width), int(height), track_id])
                    boxes_list.append([int(x_center - width / 2), int(y_center - height / 2), int(width), int(height)])
                    confidence.append(conf)
                    classes.append(label)

            return detection_list, boxes_list, confidence, classes, self.track_history
