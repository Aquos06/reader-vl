from pathlib import Path
from typing import Optional

from ultralytics import YOLO as YOLOUltra

YOLO_CONF = 0.05
YOLO_IOU = 0.3
YOLO_IMGZ = 640
YOLO_AUGMENT = True
YOLO_NMS = True


class YOLO:
    """
    Wrapper class for the Ultralytics YOLO model.
    Provides a simplified interface for object detection.
    """

    def __init__(
        self,
        weight_path: Path,
        YOLO_CONF: Optional[float] = YOLO_CONF,
        YOLO_IOU: Optional[float] = YOLO_IOU,
        YOLO_IMGZ: Optional[float] = YOLO_IMGZ,
        YOLO_AUGMENT: Optional[bool] = YOLO_AUGMENT,
        YOLO_NMS: Optional[bool] = YOLO_NMS,
    ) -> None:
        """
        Initializes the YOLO object.

        Args:
            weight_path: Path to the YOLO model weights file.
            YOLO_CONF: Confidence threshold for object detection.
            YOLO_IOU: IOU threshold for non-maximum suppression.
            YOLO_IMGZ: Image size for inference.
            YOLO_AUGMENT: Whether to use augmentation during inference.
            YOLO_NMS: Whether to apply non-maximum suppression.
        """
        self.model = YOLOUltra(weight_path)
        self.yolo_conf = YOLO_CONF
        self.yolo_iou = YOLO_IOU
        self.yolo_imgz = YOLO_IMGZ
        self.yolo_augment = YOLO_AUGMENT
        self.yolo_nms = YOLO_NMS

    def __call__(self, image):
        """
        Performs object detection on an image.

        Args:
            image: The input image.

        Returns:
            The YOLO model's prediction results.
        """
        results = self.model(
            image,
            iou=self.yolo_iou,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgz,
            augment=self.yolo_augment,
            agnostic_nms=self.yolo_nms,
        )
        return results
