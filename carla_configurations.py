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
        self.data_queue = deque(maxlen=100)  
        self.metrics = {  
            'frame': [],  
            'timestamp': [],  
            'latency_ms': [],  
            'cpu_percent': [],  
            'memory_mb': [],  
            'detections': []  
        }  
          
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
          
        # Simple hazard detection (semantic segmentation simulation)  
        detections = self.detect_hazards_rgb(array)  
          
        latency = (time.time() - start_time) * 1000  # Convert to ms  
          
        self.metrics['frame'].append(image.frame)  
        self.metrics['timestamp'].append(image.timestamp)  
        self.metrics['latency_ms'].append(latency)  
        self.metrics['cpu_percent'].append(psutil.cpu_percent())  
        self.metrics['memory_mb'].append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  
        self.metrics['detections'].append(len(detections))  
          
    def detect_hazards_rgb(self, image):  
        # Simplified detection: use color thresholding as proxy for object detection  
        # In real implementation, you'd use a neural network  
        detections = []  
          
        # Detect pedestrians (approximate with skin tones)  
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)  
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)  
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)  
          
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            if cv2.contourArea(contour) > 500:  # Minimum area threshold  
                detections.append({'type': 'pedestrian', 'area': cv2.contourArea(contour)})  
                  
        return detections  
  
class FusionAVConfig(AVConfiguration):  
    """Fusion AV with RGB + DVS Event Camera"""  
    def __init__(self, world, vehicle):  
        super().__init__(world, vehicle, "Fusion_RGB_DVS")  
        self.rgb_data = None  
        self.dvs_data = None  
        self.setup_sensors()  
          
    def setup_sensors(self):  
        bp_lib = self.world.get_blueprint_library()  
          
        # RGB Camera  
        camera_bp = bp_lib.find('sensor.camera.rgb')  
        camera_bp.set_attribute('image_size_x', '800')  
        camera_bp.set_attribute('image_size_y', '600')  
        camera_bp.set_attribute('fov', '90')  
        camera_bp.set_attribute('sensor_tick', '0.05')  
          
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)  
        camera.listen(lambda data: self.process_rgb(data))  
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
        dvs_bp.set_attribute('sensor_tick', '0.0')  # High frequency  
          
        dvs = self.world.spawn_actor(dvs_bp, camera_transform, attach_to=self.vehicle)  
        dvs.listen(lambda data: self.process_dvs(data))  
        self.sensors.append(dvs)  
          
    def process_rgb(self, image):  
        array = np.frombuffer(image.raw_data, dtype=np.uint8)  
        array = array.reshape((image.height, image.width, 4))  
        self.rgb_data = array[:, :, :3]  
          
    def process_dvs(self, data):  
        start_time = time.time()  
          
        # Process DVS events  
        events = np.frombuffer(data.raw_data, dtype=np.dtype([  
            ('x', np.uint16),  
            ('y', np.uint16),  
            ('t', np.int64),  
            ('pol', np.bool_)  
        ]))  
          
        # Fusion detection  
        detections = self.detect_hazards_fusion(events)  
          
        latency = (time.time() - start_time) * 1000  
          
        self.metrics['frame'].append(data.frame)  
        self.metrics['timestamp'].append(data.timestamp)  
        self.metrics['latency_ms'].append(latency)  
        self.metrics['cpu_percent'].append(psutil.cpu_percent())  
        self.metrics['memory_mb'].append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  
        self.metrics['detections'].append(len(detections))  
          
    def detect_hazards_fusion(self, events):  
        # Event-based detection: cluster events to find moving objects  
        detections = []  
          
        if len(events) > 0:  
            # Create event frame  
            event_frame = np.zeros((600, 800), dtype=np.uint8)  
            for event in events:  
                if 0 <= event['y'] < 600 and 0 <= event['x'] < 800:  
                    event_frame[event['y'], event['x']] = 255 if event['pol'] else 128  
              
            # Find clusters of events (moving objects)  
            _, thresh = cv2.threshold(event_frame, 100, 255, cv2.THRESH_BINARY)  
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
              
            for contour in contours:  
                if cv2.contourArea(contour) > 300:  
                    detections.append({'type': 'moving_object', 'area': cv2.contourArea(contour)})  
                      
        return detections  
  
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
            'Avg_CPU_percent': df['cpu_percent'].mean(),  
            'Avg_Memory_MB': df['memory_mb'].mean(),  
            'Total_Detections': df['detections'].sum(),  
            'Avg_Detections_per_frame': df['detections'].mean()  
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
            'Avg_Latency_ms': df['latency_ms'].mean(),  
            'Avg_CPU_%': df['cpu_percent'].mean(),  
            'Avg_Memory_MB': df['memory_mb'].mean(),  
            'Total_Detections': df['detections'].sum()  
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