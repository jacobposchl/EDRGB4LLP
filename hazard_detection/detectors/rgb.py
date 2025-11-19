import cv2
import numpy as np
from .base import BaseDetector


class RGBDetector(BaseDetector):
    """RGB-only detector using frame differencing"""
    def __init__(self):
        super().__init__("RGB_Only")

    def process_frame(self, timestamp: float, rgb_frame: np.ndarray, 
                     voxel_grid=None) -> tuple:
        detections = []
        motion_energy = 0.0

        if self.previous_frame is None:
            # Initialize previous frame and return nothing this first call
            self.previous_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            return detections, motion_energy

        current_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        frame_diff = cv2.absdiff(current_gray, self.previous_frame)
        # Lowered pixel threshold from 25 -> 15 for increased sensitivity during testing
        _, motion_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

        motion_energy = float(np.sum(motion_mask)) / 255.0

        # Find regions with significant motion
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Lowered area threshold from 400 -> 200 to detect smaller motions
            if area > 200:
                x, y, w, h = cv2.boundingRect(contour)
                region_motion = float(np.sum(motion_mask[y:y+h, x:x+w])) / 255.0

                det = {
                    'timestamp': timestamp,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'motion_magnitude': region_motion,
                    'center': (x + w//2, y + h//2)
                }
                detections.append(det)

                # Debug: print first few detection summaries so we can tune thresholds
                if len(self.detection_history) == 0 or len(self.recent_detections) < 5:
                    try:
                        print(f"[RGB DEBUG] det area={area:.1f} motion={region_motion:.1f} center={det['center']}")
                    except Exception:
                        pass

        self.previous_frame = current_gray
        self.detection_history.append({
            'timestamp': timestamp,
            'motion_energy': motion_energy,
            'num_detections': len(detections)
        })

        # Store individual detections for recent-query APIs
        for d in detections:
            # ensure timestamp present and append
            self.recent_detections.append(d)

        return detections, motion_energy
