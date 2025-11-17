class BaseDetector:
    """Base class for detectors"""
    def __init__(self, name: str):
        self.name = name
        self.previous_frame = None
        self.detection_history = []

    def process_frame(self, timestamp: float, rgb_frame, 
                     voxel_grid=None) -> tuple:
        """Process a frame and return (detections, motion_energy)"""
        raise NotImplementedError
