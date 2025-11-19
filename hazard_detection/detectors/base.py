from collections import deque


class BaseDetector:
    """Base class for detectors"""
    def __init__(self, name: str):
        self.name = name
        self.previous_frame = None
        # Summary history entries: {'timestamp','motion_energy','num_detections'}
        self.detection_history = []
        # Store recent individual detections (dictionaries including 'timestamp' and bbox)
        self.recent_detections = deque(maxlen=1024)

    def process_frame(self, timestamp: float, rgb_frame, 
                     voxel_grid=None) -> tuple:
        """Process a frame and return (detections, motion_energy)"""
        raise NotImplementedError

    def get_recent_detections(self, window: float = 0.1, latest_time: float = None):
        """Return list of recent detection dicts within `window` seconds.

        - `window`: time span in seconds to look back from `latest_time`.
        - `latest_time`: optional reference time; if None uses the most recent
          detection timestamp available.
        """
        if len(self.recent_detections) == 0:
            return []

        if latest_time is None:
            latest_time = float(self.recent_detections[-1]['timestamp'])

        cutoff = latest_time - float(window)
        results = [d for d in self.recent_detections if float(d.get('timestamp', 0.0)) >= cutoff]
        return results
