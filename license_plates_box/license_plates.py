import torch
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class Detection:
    x1: int  # Top-left x-coordinate
    y1: int  # Top-left y-coordinate
    x2: int  # Bottom-right x-coordinate
    y2: int  # Bottom-right y-coordinate
    name: str  # Detected object's name
    confidence: float  # Confidence score

class LicensePlateDetector():
    def __init__(self, cfg):
        """
        Initializes the LicensePlateDetector with a YOLOv11 model.
        
        :param model_path: Path to the YOLOv11 model file.
        """
        # Load the YOLOv11 model
        self.model = YOLO(cfg.basic.yolo_model)

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess the input image for detection.
        
        :param image_path: Path to the input image file.
        :return: Preprocessed image in tensor format.
        """
        # Resize and normalize the image
        image_resized = cv2.resize(image, (640, 640))
        image_resized = image_resized / 255.0  # Normalize to [0, 1]
        image_resized = np.transpose(image_resized, (2, 0, 1))  # Change from HWC to CHW format
        image_resized = torch.tensor(image_resized, dtype=torch.float32)
        image_resized = image_resized.unsqueeze(0)  # Add batch dimension
        return image_resized

    def detect_license_plate(self, image: np.ndarray):
        """
        Detect license plates in the given image.
        
        :param image_path: Path to the input image file.
        :return: List of detected license plates with bounding boxes.
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        detections = self.model(image)

        results = []

        # Post-process the predictions to extract bounding boxes and labels
        for detection in detections[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Get the bounding box coordinates
            confidence = detection.conf[0]  # Get confidence score
            class_id = detection.cls[0]  # Get class id

            result = Detection(x1=x1, y1=y1, x2=x2, y2=y2, name = "",confidence=confidence)
            results.append(result)

        return results