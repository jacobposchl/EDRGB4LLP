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
        self.rgb_timestamp = None  # Simulation time in seconds
        self.event_buffer = []

        # Event timestamp management - CRITICAL FIX
        # DVS events have timestamps in nanoseconds since sensor start
        # We need to align them with RGB simulation timestamps
        self.dvs_reference_time_ns = None  # First DVS timestamp we see
        self.dvs_reference_sim_time = None  # Corresponding simulation time

        # Voxel grid parameters
        self.voxel_shape = (4, 600, 800)
        self.temporal_window_ms = 50.0  # Increased from 10ms - more forgiving for initial testing

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
        dvs_bp.set_attribute('sensor_tick', '0.0')  # Stream continuously

        dvs = self.world.spawn_actor(dvs_bp, camera_transform, attach_to=self.vehicle)
        dvs.listen(lambda data: self.on_dvs_events(data))
        self.sensors.append(dvs)

    def on_rgb_frame(self, image):
        """Store RGB frame with simulation timestamp"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.rgb_frame = array[:, :, :3]
        self.rgb_timestamp = image.timestamp  # Simulation time in seconds
        
        # Debug output for first few frames
        if not hasattr(self, '_rgb_frame_count'):
            self._rgb_frame_count = 0
        self._rgb_frame_count += 1
        if self._rgb_frame_count <= 3:
            print(f"[DEBUG] RGB frame #{self._rgb_frame_count}: sim_time={self.rgb_timestamp:.6f}s shape={self.rgb_frame.shape}")

    def on_dvs_events(self, data):
        """Accumulate DVS events with proper timestamp alignment"""
        # Parse event data
        events = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.uint16),
            ('y', np.uint16),
            ('t', np.int64),  # Nanoseconds since DVS sensor start
            ('pol', np.bool_)
        ]))

        if len(events) == 0:
            return

        # CRITICAL FIX: Establish time reference on first event batch
        # DVS data.timestamp is the CARLA simulation time when this event batch was captured
        if self.dvs_reference_time_ns is None:
            # Use the first event's timestamp as our reference point
            self.dvs_reference_time_ns = events[0]['t']
            # The simulation time for this reference
            self.dvs_reference_sim_time = data.timestamp
            print(f"[DEBUG] DVS reference established: sim_time={data.timestamp:.6f}s, dvs_ref_ns={self.dvs_reference_time_ns}")

        # Convert each event's timestamp to simulation time
        for event in events:
            # Calculate time elapsed since reference in nanoseconds
            time_offset_ns = event['t'] - self.dvs_reference_time_ns
            # Convert to seconds and add to reference simulation time
            sim_time = self.dvs_reference_sim_time + (time_offset_ns / 1e9)
            
            self.event_buffer.append({
                'x': int(event['x']),
                'y': int(event['y']),
                't': sim_time,  # Now in simulation seconds - aligned with RGB!
                'pol': bool(event['pol'])
            })
        
        # Debug output for first few event batches
        if not hasattr(self, '_dvs_event_reports'):
            self._dvs_event_reports = 0
        self._dvs_event_reports += 1
        if self._dvs_event_reports <= 3:
            if len(self.event_buffer) > 0:
                times = [e['t'] for e in self.event_buffer[-min(5, len(events)):]]
                print(f"[DEBUG] DVS batch #{self._dvs_event_reports}: added {len(events)} events, "
                      f"buffer={len(self.event_buffer)}, recent_times={[f'{t:.6f}' for t in times]}")

    def create_voxel_grid(self, current_timestamp: float) -> np.ndarray:
        """Create voxel grid from recent events using simulation time"""
        voxel_grid = np.zeros(self.voxel_shape, dtype=np.float32)

        if len(self.event_buffer) == 0:
            return voxel_grid

        # Convert temporal window to seconds to match our event timestamps
        temporal_window_s = self.temporal_window_ms / 1000.0
        t_start = current_timestamp - temporal_window_s
        t_end = current_timestamp

        # Filter events within temporal window
        window_events = [e for e in self.event_buffer if t_start <= e['t'] <= t_end]

        if len(window_events) == 0:
            return voxel_grid

        # Create voxel grid
        num_bins = self.voxel_shape[0]
        
        for event in window_events:
            x, y, t, pol = event['x'], event['y'], event['t'], event['pol']

            # Check bounds
            if not (0 <= x < self.voxel_shape[2] and 0 <= y < self.voxel_shape[1]):
                continue

            # Calculate temporal bin
            relative_time = t - t_start
            bin_idx = int((relative_time / temporal_window_s) * num_bins)
            bin_idx = min(bin_idx, num_bins - 1)

            # Accumulate event
            voxel_grid[bin_idx, y, x] += 1.0 if pol else -1.0

        return voxel_grid

    def cleanup_old_events(self, current_timestamp: float):
        """Remove old events based on simulation time"""
        # Keep events from the last 2x temporal window
        cutoff = current_timestamp - (2 * self.temporal_window_ms / 1000.0)
        self.event_buffer = [e for e in self.event_buffer if e['t'] >= cutoff]

    def process_detections(self, hazard_events: List[HazardEvent]):
        """Run both detectors and check for hazard detections"""
        if self.rgb_frame is None or self.rgb_timestamp is None:
            return

        timestamp = self.rgb_timestamp  # Already in seconds
        
        # Debug output
        if not hasattr(self, '_debug_process_count'):
            self._debug_process_count = 0
        self._debug_process_count += 1
        
        # Create voxel grid using simulation time
        voxel_grid = self.create_voxel_grid(timestamp)
        
        # Debug voxel grid for first few frames
        if self._debug_process_count <= 5:
            voxel_sum = np.sum(np.abs(voxel_grid))
            has_events = len(self.event_buffer) > 0
            print(f"[DEBUG] Process #{self._debug_process_count}: "
                  f"sim_time={timestamp:.6f}s, buffer={len(self.event_buffer)} events, "
                  f"voxel_sum={voxel_sum:.1f}, voxel_has_data={voxel_sum > 0}")

        # Run both detectors
        rgb_detections, rgb_energy = self.rgb_detector.process_frame(
            timestamp, self.rgb_frame, None
        )

        fusion_detections, fusion_energy = self.fusion_detector.process_frame(
            timestamp, self.rgb_frame, voxel_grid
        )

        # Debug detection counts
        if self._debug_process_count <= 5:
            print(f"[DEBUG] Detections: RGB={len(rgb_detections)}, Fusion={len(fusion_detections)}")

        # Check if any detections are in critical zone
        rgb_critical = self.check_critical_detections(rgb_detections)
        fusion_critical = self.check_critical_detections(fusion_detections)

        # Update hazard events with detection times
        for hazard in hazard_events:
            if hazard.trigger_time <= timestamp:
                if rgb_critical and hazard.detected_rgb is None:
                    hazard.detected_rgb = timestamp
                    print(f"[DETECTION] RGB detected hazard at t={timestamp:.3f}s (lag={((timestamp - hazard.trigger_time)*1000):.1f}ms)")
                if fusion_critical and hazard.detected_fusion is None:
                    hazard.detected_fusion = timestamp
                    print(f"[DETECTION] Fusion detected hazard at t={timestamp:.3f}s (lag={((timestamp - hazard.trigger_time)*1000):.1f}ms)")

        # Cleanup old events
        self.cleanup_old_events(timestamp)

    def check_critical_detections(self, detections: List[Dict]) -> bool:
        """Check if any detection is in critical zone with significant motion"""
        for det in detections:
            x, y, w, h = det['bbox']
            center_x, center_y = det['center']

            # Check if detection center is in critical zone
            if (self.critical_zone['x_min'] <= center_x <= self.critical_zone['x_max'] and
                self.critical_zone['y_min'] <= center_y <= self.critical_zone['y_max']):
                # Check if motion is significant (lowered threshold for testing)
                if det['motion_magnitude'] > 500:  # Lowered from 1000
                    return True
        return False

    def cleanup(self):
        """Destroy sensors"""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()