import carla
import numpy as np
import cv2
from typing import List, Dict, Optional

from .detectors.rgb import RGBDetector
from .detectors.fusion import FusionDetector
from .events import HazardEvent


class DualSensorSystem:
    """System that captures both RGB and DVS data simultaneously"""
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = []

        # Data buffers
        self.rgb_frame = None
        self.rgb_timestamp = None
        self.event_buffer = []

        # Voxel grid parameters
        self.voxel_shape = (4, 600, 800)
        self.temporal_window_ms = 10.0

        # Detectors
        self.rgb_detector = RGBDetector()
        self.fusion_detector = FusionDetector()

        # Detection zones (for hazard checking)
        self.critical_zone = {
            'x_min': 300, 'x_max': 500,  # Center region horizontally
            'y_min': 300, 'y_max': 600   # Lower half vertically (road ahead)
        }

        self.setup_sensors()

    def setup_sensors(self):
        """Setup RGB and DVS cameras"""
        bp_lib = self.world.get_blueprint_library()
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # RGB Camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20 Hz

        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda data: self.on_rgb_frame(data))
        self.sensors.append(camera)

        # DVS Event Camera
        dvs_bp = bp_lib.find('sensor.camera.dvs')
        dvs_bp.set_attribute('image_size_x', '800')
        dvs_bp.set_attribute('image_size_y', '600')
        dvs_bp.set_attribute('fov', '90')
        dvs_bp.set_attribute('positive_threshold', '0.3')
        dvs_bp.set_attribute('negative_threshold', '0.3')
        dvs_bp.set_attribute('sigma_positive_threshold', '0')
        dvs_bp.set_attribute('sigma_negative_threshold', '0')
        dvs_bp.set_attribute('use_log', 'true')
        dvs_bp.set_attribute('sensor_tick', '0.0')

        dvs = self.world.spawn_actor(dvs_bp, camera_transform, attach_to=self.vehicle)
        dvs.listen(lambda data: self.on_dvs_events(data))
        self.sensors.append(dvs)

    def on_rgb_frame(self, image):
        """Store RGB frame"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.rgb_frame = array[:, :, :3]
        self.rgb_timestamp = image.timestamp
        # Debug: count frames and show a few samples
        if not hasattr(self, '_rgb_frame_count'):
            self._rgb_frame_count = 0
        self._rgb_frame_count += 1
        if self._rgb_frame_count <= 5:
            try:
                print(f"[DEBUG] on_rgb_frame #{self._rgb_frame_count}: ts={self.rgb_timestamp} shape={self.rgb_frame.shape}")
            except Exception:
                print(f"[DEBUG] on_rgb_frame #{self._rgb_frame_count}: ts={self.rgb_timestamp}")

    def on_dvs_events(self, data):
        """Accumulate DVS events"""
        events = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.uint16),
            ('y', np.uint16),
            ('t', np.int64),
            ('pol', np.bool_)
        ]))

        for event in events:
            self.event_buffer.append({
                'x': int(event['x']),
                'y': int(event['y']),
                't': float(event['t']) / 1e6,  # Convert to milliseconds
                'pol': bool(event['pol'])
            })
        # Debug: report event buffer length occasionally
        if not hasattr(self, '_dvs_event_reports'):
            self._dvs_event_reports = 0
        self._dvs_event_reports += 1
        if self._dvs_event_reports <= 5:
            print(f"[DEBUG] on_dvs_events: added {len(events)} events; buffer_len={len(self.event_buffer)}")

    def create_voxel_grid(self, current_timestamp_ms: float) -> np.ndarray:
        """Create voxel grid from recent events"""
        voxel_grid = np.zeros(self.voxel_shape, dtype=np.float32)

        if len(self.event_buffer) == 0:
            return voxel_grid

        t_start = current_timestamp_ms - self.temporal_window_ms
        t_end = current_timestamp_ms

        window_events = [e for e in self.event_buffer if t_start <= e['t'] <= t_end]

        if len(window_events) == 0:
            return voxel_grid

        num_bins = self.voxel_shape[0]
        bin_duration = self.temporal_window_ms / num_bins

        for event in window_events:
            x, y, t, pol = event['x'], event['y'], event['t'], event['pol']

            if not (0 <= x < self.voxel_shape[2] and 0 <= y < self.voxel_shape[1]):
                continue

            relative_time = t - t_start
            bin_idx = int(relative_time / bin_duration)
            bin_idx = min(bin_idx, num_bins - 1)

            voxel_grid[bin_idx, y, x] += 1.0 if pol else -1.0

        return voxel_grid

    def cleanup_old_events(self, current_timestamp_ms: float):
        """Remove old events"""
        cutoff = current_timestamp_ms - (self.temporal_window_ms * 2)
        self.event_buffer = [e for e in self.event_buffer if e['t'] >= cutoff]

    def process_detections(self, hazard_events: List[HazardEvent]):
        """Run both detectors and check for hazard detections"""
        if self.rgb_frame is None or self.rgb_timestamp is None:
            if hasattr(self, '_rgb_frame_count') and self._rgb_frame_count > 0:
                # got frames earlier but currently missing
                print(f"[DEBUG] process_detections called but rgb_frame missing; rgb_count={getattr(self,'_rgb_frame_count',0)} event_buffer_len={len(self.event_buffer)}")
            return

        timestamp = self.rgb_timestamp
        timestamp_ms = timestamp * 1000.0
        if hasattr(self, '_debug_process_count'):
            self._debug_process_count += 1
        else:
            self._debug_process_count = 1
        if self._debug_process_count <= 5:
            print(f"[DEBUG] process_detections #{self._debug_process_count}: ts={timestamp} rgb_shape={self.rgb_frame.shape} event_buffer_len={len(self.event_buffer)}")

        # Create voxel grid
        voxel_grid = self.create_voxel_grid(timestamp_ms)

        # Run both detectors
        rgb_detections, rgb_energy = self.rgb_detector.process_frame(
            timestamp, self.rgb_frame, None
        )

        fusion_detections, fusion_energy = self.fusion_detector.process_frame(
            timestamp, self.rgb_frame, voxel_grid
        )

        # Check if any detections are in critical zone (potential hazard detection)
        rgb_critical = self.check_critical_detections(rgb_detections)
        fusion_critical = self.check_critical_detections(fusion_detections)

        # Update hazard events with detection times
        for hazard in hazard_events:
            # Only update if not already detected and hazard has been triggered
            if hazard.trigger_time <= timestamp:
                if rgb_critical and hazard.detected_rgb is None:
                    hazard.detected_rgb = timestamp
                if fusion_critical and hazard.detected_fusion is None:
                    hazard.detected_fusion = timestamp

        # Cleanup
        self.cleanup_old_events(timestamp_ms)

    def check_critical_detections(self, detections: List[Dict]) -> bool:
        """Check if any detection is in critical zone with significant motion"""
        for det in detections:
            x, y, w, h = det['bbox']
            center_x, center_y = det['center']

            # Check if detection center is in critical zone
            if (self.critical_zone['x_min'] <= center_x <= self.critical_zone['x_max'] and
                self.critical_zone['y_min'] <= center_y <= self.critical_zone['y_max']):
                # Check if motion is significant
                if det['motion_magnitude'] > 1000:  # Threshold for "dangerous" motion
                    return True
        return False

    def cleanup(self):
        """Destroy sensors"""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
