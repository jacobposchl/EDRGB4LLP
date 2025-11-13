import carla  
import random  
import time  
import numpy as np  
import pandas as pd  
import cv2  
from collections import deque  
import psutil  
import os  
  
class AVConfiguration:  
    """Base class for AV sensor configurations"""  
    def __init__(self, world, vehicle, config_name):  
        self.world = world  
        self.vehicle = vehicle  
        self.config_name = config_name  
        self.sensors = []  
        self.metrics = {  
            'frame': [],  
            'timestamp': [],  
            'latency_ms': [],  
            'cpu_percent': [],  
            'memory_mb': [],  
            'detections': [],
            'motion_energy': []
        }  
        self.previous_frame = None
          
    def cleanup(self):  
        for sensor in self.sensors:  
            if sensor.is_alive:  
                sensor.destroy()  
  
class DefaultAVConfig(AVConfiguration):  
    """Default AV with only RGB camera"""  
    def __init__(self, world, vehicle):  
        super().__init__(world, vehicle, "Default_RGB_Only")  
        self.setup_sensors()  
          
    def setup_sensors(self):  
        # RGB Camera setup  
        bp_lib = self.world.get_blueprint_library()  
        camera_bp = bp_lib.find('sensor.camera.rgb')  
        camera_bp.set_attribute('image_size_x', '800')  
        camera_bp.set_attribute('image_size_y', '600')  
        camera_bp.set_attribute('fov', '90')  
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20 Hz  
          
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)  
        camera.listen(lambda data: self.process_rgb(data))  
        self.sensors.append(camera)  
          
    def process_rgb(self, image):  
        start_time = time.time()  
          
        # Convert to numpy array  
        array = np.frombuffer(image.raw_data, dtype=np.uint8)  
        array = array.reshape((image.height, image.width, 4))  
        array = array[:, :, :3]  # Remove alpha channel  
          
        # Motion-based hazard detection using frame differencing
        detections, motion_energy = self.detect_hazards_motion(array)  
          
        latency = (time.time() - start_time) * 1000  # Convert to ms  
          
        self.metrics['frame'].append(image.frame)  
        self.metrics['timestamp'].append(image.timestamp)  
        self.metrics['latency_ms'].append(latency)  
        self.metrics['cpu_percent'].append(psutil.cpu_percent())  
        self.metrics['memory_mb'].append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  
        self.metrics['detections'].append(len(detections))
        self.metrics['motion_energy'].append(motion_energy)
        
        # Store current frame for next iteration
        self.previous_frame = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
          
    def detect_hazards_motion(self, image):  
        """Detect hazards based on motion between frames"""
        detections = []  
        motion_energy = 0.0
        
        if self.previous_frame is None:
            return detections, motion_energy
            
        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute frame difference
        frame_diff = cv2.absdiff(current_gray, self.previous_frame)
        
        # Apply threshold to get significant motion
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate total motion energy
        motion_energy = float(np.sum(motion_mask)) / 255.0
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        
        for contour in contours:  
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold  
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'type': 'moving_object', 
                    'area': area,
                    'bbox': (x, y, w, h),
                    'motion_magnitude': float(np.sum(motion_mask[y:y+h, x:x+w])) / 255.0
                })  
                  
        return detections, motion_energy
  
class FusionAVConfig(AVConfiguration):  
    """Fusion AV with RGB + DVS Event Camera"""  
    def __init__(self, world, vehicle):  
        super().__init__(world, vehicle, "Fusion_RGB_DVS")  
        self.event_buffer = []  # Buffer to accumulate events
        self.last_rgb_timestamp = None
        self.voxel_grid_shape = (4, 600, 800)  # (temporal_bins, height, width)
        self.temporal_window_ms = 10.0  # 10ms window for event accumulation
        self.setup_sensors()  
          
    def setup_sensors(self):  
        bp_lib = self.world.get_blueprint_library()  
          
        # RGB Camera  
        camera_bp = bp_lib.find('sensor.camera.rgb')  
        camera_bp.set_attribute('image_size_x', '800')  
        camera_bp.set_attribute('image_size_y', '600')  
        camera_bp.set_attribute('fov', '90')  
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20 Hz
          
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)  
        camera.listen(lambda data: self.process_rgb(data))  
        self.sensors.append(camera)  
          
        # DVS Event Camera - runs continuously to accumulate events
        dvs_bp = bp_lib.find('sensor.camera.dvs')  
        dvs_bp.set_attribute('image_size_x', '800')  
        dvs_bp.set_attribute('image_size_y', '600')  
        dvs_bp.set_attribute('fov', '90')  
        dvs_bp.set_attribute('positive_threshold', '0.3')  
        dvs_bp.set_attribute('negative_threshold', '0.3')  
        dvs_bp.set_attribute('sigma_positive_threshold', '0')  
        dvs_bp.set_attribute('sigma_negative_threshold', '0')  
        dvs_bp.set_attribute('use_log', 'true')  
        dvs_bp.set_attribute('sensor_tick', '0.0')  # Continuous event capture
          
        dvs = self.world.spawn_actor(dvs_bp, camera_transform, attach_to=self.vehicle)  
        dvs.listen(lambda data: self.accumulate_events(data))  
        self.sensors.append(dvs)  
          
    def accumulate_events(self, data):  
        """Continuously accumulate events into buffer"""
        # Parse events from CARLA DVS data
        events = np.frombuffer(data.raw_data, dtype=np.dtype([  
            ('x', np.uint16),  
            ('y', np.uint16),  
            ('t', np.int64),  
            ('pol', np.bool_)  
        ]))  
        
        # Store events with their timestamps
        for event in events:
            self.event_buffer.append({
                'x': int(event['x']),
                'y': int(event['y']),
                't': float(event['t']) / 1e6,  # Convert to milliseconds
                'pol': bool(event['pol'])
            })
          
    def process_rgb(self, image):  
        """Process RGB frame and fuse with recent events"""
        start_time = time.time()  
        
        # Convert RGB image to numpy array  
        array = np.frombuffer(image.raw_data, dtype=np.uint8)  
        array = array.reshape((image.height, image.width, 4))  
        rgb_frame = array[:, :, :3]
        
        # Get current timestamp in milliseconds
        current_timestamp_ms = image.timestamp * 1000.0
        
        # Create voxel grid from events in the temporal window
        voxel_grid = self.create_voxel_grid(current_timestamp_ms)
        
        # Fusion detection using both RGB and events
        detections, motion_energy = self.detect_hazards_fusion(rgb_frame, voxel_grid)  
          
        latency = (time.time() - start_time) * 1000  # Convert to ms
          
        self.metrics['frame'].append(image.frame)  
        self.metrics['timestamp'].append(image.timestamp)  
        self.metrics['latency_ms'].append(latency)  
        self.metrics['cpu_percent'].append(psutil.cpu_percent())  
        self.metrics['memory_mb'].append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  
        self.metrics['detections'].append(len(detections))
        self.metrics['motion_energy'].append(motion_energy)
        
        # Clean up old events to prevent memory overflow
        self.cleanup_old_events(current_timestamp_ms)
        self.last_rgb_timestamp = current_timestamp_ms
    
    def create_voxel_grid(self, current_timestamp_ms):
        """Create voxel grid from events in the temporal window"""
        voxel_grid = np.zeros(self.voxel_grid_shape, dtype=np.float32)
        
        if len(self.event_buffer) == 0:
            return voxel_grid
        
        # Define temporal window: [current_time - window_size, current_time]
        t_start = current_timestamp_ms - self.temporal_window_ms
        t_end = current_timestamp_ms
        
        # Filter events within the temporal window
        window_events = [e for e in self.event_buffer if t_start <= e['t'] <= t_end]
        
        if len(window_events) == 0:
            return voxel_grid
        
        # Assign events to temporal bins
        num_bins = self.voxel_grid_shape[0]
        bin_duration = self.temporal_window_ms / num_bins
        
        for event in window_events:
            x, y, t, pol = event['x'], event['y'], event['t'], event['pol']
            
            # Ensure coordinates are within bounds
            if not (0 <= x < self.voxel_grid_shape[2] and 0 <= y < self.voxel_grid_shape[1]):
                continue
            
            # Calculate which temporal bin this event belongs to
            relative_time = t - t_start
            bin_idx = int(relative_time / bin_duration)
            bin_idx = min(bin_idx, num_bins - 1)  # Clamp to valid range
            
            # Accumulate event in voxel grid with polarity
            # Positive events add +1, negative events add -1
            voxel_grid[bin_idx, y, x] += 1.0 if pol else -1.0
        
        return voxel_grid
    
    def cleanup_old_events(self, current_timestamp_ms):
        """Remove events older than needed to prevent memory overflow"""
        cutoff_time = current_timestamp_ms - (self.temporal_window_ms * 2)  # Keep 2x window for safety
        self.event_buffer = [e for e in self.event_buffer if e['t'] >= cutoff_time]
    
    def detect_hazards_fusion(self, rgb_frame, voxel_grid):  
        """Detect hazards using event-RGB fusion"""
        detections = []  
        
        # Calculate motion energy from event voxel grid
        # Sum absolute values across all temporal bins
        event_motion = np.sum(np.abs(voxel_grid))
        
        # Create a spatial motion map by summing across temporal dimension
        spatial_motion = np.sum(np.abs(voxel_grid), axis=0)
        
        # Threshold to find regions with significant event activity
        motion_threshold = np.percentile(spatial_motion[spatial_motion > 0], 75) if np.any(spatial_motion > 0) else 0
        
        if motion_threshold > 0:
            _, motion_mask = cv2.threshold(spatial_motion.astype(np.uint8), 
                                          int(motion_threshold), 255, cv2.THRESH_BINARY)
            
            # Find contours of moving regions from events
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            for contour in contours:  
                area = cv2.contourArea(contour)
                if area > 300:  # Lower threshold since events are more sensitive
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract motion magnitude in this region across all temporal bins
                    region_motion = np.sum(np.abs(voxel_grid[:, y:y+h, x:x+w]))
                    
                    detections.append({
                        'type': 'moving_object', 
                        'area': area,
                        'bbox': (x, y, w, h),
                        'motion_magnitude': float(region_motion)
                    })  
        
        return detections, float(event_motion)
  
def setup_urban_environment(client, town_name='Town03'):  
    """Setup realistic urban environment"""  
    world = client.load_world(town_name)  
      
    # Configure weather for realistic conditions  
    weather = carla.WeatherParameters(  
        cloudiness=30.0,  
        precipitation=0.0,  
        sun_altitude_angle=70.0,  
        fog_density=10.0  
    )  
    world.set_weather(weather)  
      
    # Set synchronous mode for deterministic testing  
    settings = world.get_settings()  
    settings.synchronous_mode = True  
    settings.fixed_delta_seconds = 0.05  
    world.apply_settings(settings)  
      
    # Spawn traffic vehicles  
    spawn_points = world.get_map().get_spawn_points()  
    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')  
      
    traffic_vehicles = []  
    for i in range(30):  # Spawn 30 traffic vehicles  
        vehicle_bp = random.choice(vehicle_bps)  
        spawn_point = random.choice(spawn_points)  
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)  
        if vehicle:  
            vehicle.set_autopilot(True)  
            traffic_vehicles.append(vehicle)  
      
    # Spawn pedestrians  
    walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')  
    walker_spawn_points = []  
    for i in range(20):  # Spawn 20 pedestrians  
        spawn_point = carla.Transform()  
        loc = world.get_random_location_from_navigation()  
        if loc:  
            spawn_point.location = loc  
            walker_bp = random.choice(walker_bps)  
            walker = world.try_spawn_actor(walker_bp, spawn_point)  
            if walker:  
                walker_spawn_points.append(walker)  
      
    return world, traffic_vehicles, walker_spawn_points  
  
def run_experiment(client, config_class, duration_seconds=60):  
    """Run experiment for a given configuration"""  
    print(f"\n{'='*60}")  
    print(f"Running experiment: {config_class.__name__}")  
    print(f"{'='*60}\n")  
      
    # Setup environment  
    world, traffic_vehicles, pedestrians = setup_urban_environment(client)  
      
    # Spawn ego vehicle  
    spawn_points = world.get_map().get_spawn_points()  
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')  
    ego_vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))  
    ego_vehicle.set_autopilot(True)  
      
    # Create configuration  
    config = config_class(world, ego_vehicle)  
      
    # Run simulation  
    start_time = time.time()  
    frame_count = 0  
      
    try:  
        while time.time() - start_time < duration_seconds:  
            world.tick()  
            frame_count += 1  
              
            if frame_count % 100 == 0:  
                elapsed = time.time() - start_time  
                print(f"Progress: {elapsed:.1f}s / {duration_seconds}s ({frame_count} frames)")  
                  
    finally:  
        # Cleanup  
        config.cleanup()  
        ego_vehicle.destroy()  
        for vehicle in traffic_vehicles:  
            if vehicle.is_alive:  
                vehicle.destroy()  
        for walker in pedestrians:  
            if walker.is_alive:  
                walker.destroy()  
      
    return config.metrics  
  
def save_metrics_to_csv(metrics_dict, output_dir='results'):  
    """Save metrics to labeled CSV files"""  
    os.makedirs(output_dir, exist_ok=True)  
      
    for config_name, metrics in metrics_dict.items():  
        df = pd.DataFrame(metrics)  
          
        # Calculate summary statistics  
        summary = {  
            'Configuration': config_name,  
            'Avg_Latency_ms': df['latency_ms'].mean(),  
            'Std_Latency_ms': df['latency_ms'].std(),  
            'Max_Latency_ms': df['latency_ms'].max(),
            'Min_Latency_ms': df['latency_ms'].min(),
            'Avg_CPU_percent': df['cpu_percent'].mean(),  
            'Avg_Memory_MB': df['memory_mb'].mean(),  
            'Total_Detections': df['detections'].sum(),  
            'Avg_Detections_per_frame': df['detections'].mean(),
            'Avg_Motion_Energy': df['motion_energy'].mean(),
            'Total_Frames': len(df)
        }  
          
        # Save detailed metrics  
        filename = f"{output_dir}/{config_name}_detailed.csv"  
        df.to_csv(filename, index=False)  
        print(f"Saved detailed metrics to: {filename}")  
          
        # Save summary  
        summary_df = pd.DataFrame([summary])  
        summary_filename = f"{output_dir}/{config_name}_summary.csv"  
        summary_df.to_csv(summary_filename, index=False)  
        print(f"Saved summary to: {summary_filename}")  
      
    # Create comparison table  
    comparison_data = []  
    for config_name, metrics in metrics_dict.items():  
        df = pd.DataFrame(metrics)  
        comparison_data.append({  
            'Configuration': config_name,  
            'Frames_Processed': len(df),
            'Avg_Latency_ms': df['latency_ms'].mean(),  
            'Avg_CPU_%': df['cpu_percent'].mean(),  
            'Avg_Memory_MB': df['memory_mb'].mean(),  
            'Total_Detections': df['detections'].sum(),
            'Avg_Motion_Energy': df['motion_energy'].mean()
        })  
      
    comparison_df = pd.DataFrame(comparison_data)  
    comparison_filename = f"{output_dir}/comparison.csv"  
    comparison_df.to_csv(comparison_filename, index=False)  
    print(f"\nSaved comparison table to: {comparison_filename}")  
    print("\nComparison Results:")  
    print(comparison_df.to_string(index=False))  
  
def main():  
    # Connect to CARLA  
    client = carla.Client('localhost', 2000)  
    client.set_timeout(10.0)  
      
    # Run experiments  
    metrics_dict = {}  
      
    # Test Default Configuration  
    metrics_dict['Default_RGB_Only'] = run_experiment(client, DefaultAVConfig, duration_seconds=60)  
      
    # Test Fusion Configuration  
    metrics_dict['Fusion_RGB_DVS'] = run_experiment(client, FusionAVConfig, duration_seconds=60)  
      
    # Save metrics  
    save_metrics_to_csv(metrics_dict)  
  
if __name__ == '__main__':  
    try:  
        main()  
    except KeyboardInterrupt:  
        print('\nCancelled by user. Bye!')  
    except Exception as e:  
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()