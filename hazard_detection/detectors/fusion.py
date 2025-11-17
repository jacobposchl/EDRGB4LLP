import cv2
import numpy as np
from .base import BaseDetector


class FusionDetector(BaseDetector):
    """Event-RGB fusion detector"""
    def __init__(self):
        super().__init__("Fusion")

    def process_frame(self, timestamp: float, rgb_frame: np.ndarray, 
                     voxel_grid: np.ndarray = None) -> tuple:
        detections = []

        if voxel_grid is None or not np.any(voxel_grid):
            return detections, 0.0

        # Calculate motion energy from events
        event_motion = np.sum(np.abs(voxel_grid))

        # Create spatial motion map
        spatial_motion = np.sum(np.abs(voxel_grid), axis=0)

        # Normalize and threshold
        if np.max(spatial_motion) > 0:
            spatial_motion_normalized = (spatial_motion / np.max(spatial_motion) * 255).astype(np.uint8)
        else:
            return detections, float(event_motion)

        # Use fixed threshold instead of percentile
        _, motion_mask = cv2.threshold(spatial_motion_normalized, 30, 255, cv2.THRESH_BINARY)

        # Find motion regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Lower threshold - events are more sensitive
                x, y, w, h = cv2.boundingRect(contour)
                region_motion = np.sum(np.abs(voxel_grid[:, y:y+h, x:x+w]))

                detections.append({
                    'timestamp': timestamp,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'motion_magnitude': float(region_motion),
                    'center': (x + w//2, y + h//2)
                })

        self.detection_history.append({
            'timestamp': timestamp,
            'motion_energy': event_motion,
            'num_detections': len(detections)
        })

        return detections, float(event_motion)
