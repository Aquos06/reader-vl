from pathlib import Path
from typing import Optional, Dict
from ultralytics import YOLO as YOLOUltra

YOLO_CONF = 0.05
YOLO_IOU = 0.3
YOLO_IMGZ = 640
YOLO_AUGMENT = True
YOLO_NMS = True

class YOLO:
    def __init__(self,
                 weight_path: Path,
                 YOLO_CONF: Optional[float] = YOLO_CONF,
                 YOLO_IOU: Optional[float] = YOLO_IOU,
                 YOLO_IMGZ: Optional[float] = YOLO_IMGZ,
                 YOLO_AUGMENT: Optional[bool] = YOLO_AUGMENT,
                 YOLO_NMS: Optional[bool] = YOLO_NMS
                ) -> None:
        self.model = YOLOUltra(weight_path)
        self.yolo_conf = YOLO_CONF
        self.yolo_iou = YOLO_IOU
        self.yolo_imgz = YOLO_IMGZ
        self.yolo_augment = YOLO_AUGMENT
        self.yolo_nms = YOLO_NMS
        
    def __call__(self,image):
        results = self.model(
            image, 
            iou=self.yolo_iou, 
            conf=self.yolo_conf, 
            imgsz=self.yolo_imgz, 
            augment=self.yolo_augment,
            agnostic_nms=self.yolo_nms)
        return results,
