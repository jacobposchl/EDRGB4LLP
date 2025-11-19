"""Record a single hazard from the ego RGB camera for visual verification.

Saves a video and CSV with timestamps into `results/rgb_single_hazard/`.

Usage (from project root):
  python scripts\rgb_single_hazard.py --hazard vehicle --debug --target-speed 50
"""
import os
import sys
import time
import threading
import argparse
import traceback
from collections import deque
import cv2
import numpy as np
import carla
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hazard_detection.system import DualSensorSystem
from hazard_detection.hazards import create_overtake_hazard, create_pedestrian_crossing_hazard


def setup_straight_road_scene(client, seed=None):
    """Setup on Town10HD which has long straight highways"""
    import random
    # Use a fixed default seed when none provided so runs are reproducible
    if seed is None:
        seed = 2
    random.seed(seed)
    
    world = client.load_world('Town10HD')
    
    weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=0.0,
        sun_altitude_angle=70.0,
        fog_density=5.0
    )
    world.set_weather(weather)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    spawn_points = world.get_map().get_spawn_points()
    
    # Find straight highway sections
    straight_spawns = []
    for sp in spawn_points:
        if sp.location.z < 5.0:  # Ground level highway
            straight_spawns.append(sp)
    
    if not straight_spawns:
        straight_spawns = spawn_points
    
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_vehicle = None
    
    random.shuffle(straight_spawns)
    lane_width = 3.5  # approximate lane width in meters
    for sp in straight_spawns[:10]:
        # compute forward/right vectors from spawn yaw so we can offset into left lane
        yaw_rad = math.radians(sp.rotation.yaw)
        forward = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
        right = carla.Vector3D(-forward.y, forward.x, 0.0)
        # place ego in left lane (one lane left of center)
        left = carla.Vector3D(-right.x, -right.y, 0.0)
        spawn_loc = carla.Location(
            x=sp.location.x + left.x * lane_width,
            y=sp.location.y + left.y * lane_width,
            z=sp.location.z
        )
        spawn_tf = carla.Transform(spawn_loc, sp.rotation)
        ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if ego_vehicle:
            world.tick()
            break
    
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle")
    
    return world, ego_vehicle



def record_single_hazard(client, pre_seconds=2.0, post_seconds=6.0,
                        min_video_seconds=6.0, output_dir='results/rgb_single_hazard', 
                        debug=False, seed=None, hazard='vehicle', target_speed: float = None):
    os.makedirs(output_dir, exist_ok=True)
    
    world, ego_vehicle = setup_straight_road_scene(client, seed=seed)
    sensor_system = DualSensorSystem(world, ego_vehicle)
    
    # NO AUTOPILOT - just drive straight forward
    ego_vehicle.set_autopilot(False)
    
    # Warm up
    for _ in range(20):
        # Apply constant throttle to move straight
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        world.tick()
        try:
            sensor_system.process_detections([])
        except:
            pass
    
    fps = 20  # 0.05s per frame
    pre_frames = max(1, int(pre_seconds * fps))
    post_frames = max(1, int(post_seconds * fps))
    
    buffer = deque(maxlen=pre_frames)
    
    print(f"Recording pedestrian hazard: pre={pre_seconds}s post={post_seconds}s fps={fps}")
    print("Car driving straight forward (no autopilot, no turning)")
    print("Collecting pre-spawn frames...")
    
    # Collect pre-frames while driving straight
    for _ in range(pre_frames * 2):
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        world.tick()
        
        if sensor_system.rgb_frame is not None:
            frame_copy = sensor_system.rgb_frame.copy()
            ts = float(sensor_system.rgb_timestamp)
            buffer.append((ts, frame_copy))
    
    # Spawn hazard NOW - crossing directly in front
    walker = None
    vehicle = None
    if hazard.lower().startswith('ped'):
        print("Spawning pedestrian to cross IN FRONT of car...")
        ev = create_pedestrian_crossing_hazard(world, ego_vehicle, debug=debug)
        walker = ev.actor if ev is not None else None
        if walker is None:
            print("Failed to spawn pedestrian hazard")
            return
        # Keep reference to hazard event so detectors can be notified
        hazard_event = ev
        spawn_time = world.get_snapshot().timestamp.elapsed_seconds
        print(f"Pedestrian spawned at t={spawn_time:.2f}s - running across car's path!")
    else:
        print("Spawning vehicle OVERTAKE hazard (behind, faster)...")
        # Spawn hazard slightly behind ego and make it faster so it overtakes
        # Use explicit `target_speed` if provided; otherwise let the hazard
        # factory use its own default (keeps behavior simple and predictable).
        if target_speed is not None:
            ev = create_overtake_hazard(world, ego_vehicle, back_distance=8.0, lateral_offset=3.5, target_speed=target_speed, duration=post_seconds + 2.0, debug=debug)
        else:
            ev = create_overtake_hazard(world, ego_vehicle, back_distance=8.0, lateral_offset=3.5, duration=post_seconds + 2.0, debug=debug)
        vehicle = ev.actor if ev is not None else None
        if vehicle is None:
            print("Failed to spawn vehicle hazard")
            return
        # Keep reference to hazard event so detectors can be notified
        hazard_event = ev
        spawn_time = world.get_snapshot().timestamp.elapsed_seconds
        print(f"Vehicle spawned at t={spawn_time:.2f}s - crossing car's path!")
        print(f"Hazard Vehicle Speed: {target_speed}")
    
    # Start with pre-buffer
    captured_frames = list(buffer)

    # Create a run-specific folder early so we can save annotated frames during recording
    run_folder = os.path.join(output_dir, f"hazard_{hazard}_{int(time.time())}")
    os.makedirs(run_folder, exist_ok=True)
    rgb_annotated_saved = False
    fusion_annotated_saved = False
    both_annotated_saved = False
    
    # Collect post-spawn frames - keep driving straight
    print("Recording post-spawn frames...")
    for i in range(post_frames):
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        world.tick()
        # Let detectors process this tick and check for hazard detection
        try:
            sensor_system.process_detections([hazard_event])
        except Exception:
            pass

        # After processing, check recent detections from each detector and save annotated frames
        try:
            # Get recent detections within a short window
            rgb_recent = sensor_system.rgb_detector.get_recent_detections(window=0.5)
            fusion_recent = sensor_system.fusion_detector.get_recent_detections(window=0.5)

            # If RGB detections exist and not saved yet, annotate and save
            if rgb_recent and sensor_system.rgb_frame is not None and not rgb_annotated_saved:
                img = sensor_system.rgb_frame.copy()
                for d in rgb_recent:
                    x, y, w, h = d['bbox']
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cx, cy = d['center']
                    cv2.putText(img, f"RGB area={int(d['area'])}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                out_path = os.path.join(run_folder, 'annotated_rgb.png')
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved annotated RGB frame: {out_path}")
                rgb_annotated_saved = True

            # If Fusion detections exist and not saved yet, annotate and save
            if fusion_recent and sensor_system.rgb_frame is not None and not fusion_annotated_saved:
                img = sensor_system.rgb_frame.copy()
                for d in fusion_recent:
                    x, y, w, h = d['bbox']
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img, f"FUS motion={int(d['motion_magnitude'])}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                out_path = os.path.join(run_folder, 'annotated_fusion.png')
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved annotated Fusion frame: {out_path}")
                fusion_annotated_saved = True

            # Save combined overlay if both detected
            if rgb_recent and fusion_recent and sensor_system.rgb_frame is not None and not both_annotated_saved:
                img = sensor_system.rgb_frame.copy()
                for d in rgb_recent:
                    x, y, w, h = d['bbox']
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                for d in fusion_recent:
                    x, y, w, h = d['bbox']
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                out_path = os.path.join(run_folder, 'annotated_both.png')
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved annotated combined frame: {out_path}")
                both_annotated_saved = True
        except Exception:
            pass
        
        if sensor_system.rgb_frame is not None:
            frame_copy = sensor_system.rgb_frame.copy()
            ts = float(sensor_system.rgb_timestamp)
            captured_frames.append((ts, frame_copy))
        
        if (i + 1) % 20 == 0:
            print(f"  Captured {i+1}/{post_frames} frames...")
    
    # Ensure minimum duration
    total_frames = len(captured_frames)
    desired_frames = max(total_frames, int(math.ceil(min_video_seconds * fps)))
    extra_needed = desired_frames - total_frames
    
    if extra_needed > 0:
        print(f"Collecting {extra_needed} extra frames for minimum duration...")
        for _ in range(extra_needed):
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
            world.tick()
            # Process extra frames for detections as well
            try:
                sensor_system.process_detections([hazard_event])
            except Exception:
                pass
            if sensor_system.rgb_frame is not None:
                frame_copy = sensor_system.rgb_frame.copy()
                ts = float(sensor_system.rgb_timestamp)
                captured_frames.append((ts, frame_copy))
    
    print(f"Captured {len(captured_frames)} total frames")
    
    # Save video into the run folder created earlier
    if captured_frames:
        video_path = os.path.join(run_folder, 'video.mp4')
        csv_path = os.path.join(run_folder, 'timestamps.csv')

        h, w = captured_frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        with open(csv_path, 'w') as f:
            f.write('frame_idx,timestamp,seconds_since_spawn\n')
            for i, (ts, frame) in enumerate(captured_frames):
                out.write(frame)
                f.write(f'{i},{ts:.6f},{ts - spawn_time:.6f}\n')

        out.release()
        print(f"\nSaved video: {video_path}")
        print(f"Saved timestamps: {csv_path}")
        # Write a simple detection summary for this hazard
        try:
            det_csv = os.path.join(run_folder, 'detections.csv')
            with open(det_csv, 'w') as df:
                df.write('event_type,trigger_time,detected_rgb,detected_fusion,rgb_lag_ms,fusion_lag_ms,advantage_ms\n')
                if 'hazard_event' in locals():
                    he = hazard_event
                    drgb = f"{he.detected_rgb:.6f}" if he.detected_rgb is not None else ''
                    dfus = f"{he.detected_fusion:.6f}" if he.detected_fusion is not None else ''
                    rgb_lag = f"{he.detection_lag_rgb():.1f}" if he.detection_lag_rgb() is not None else ''
                    fusion_lag = f"{he.detection_lag_fusion():.1f}" if he.detection_lag_fusion() is not None else ''
                    adv = f"{he.latency_advantage():.1f}" if he.latency_advantage() is not None else ''
                    df.write(f"{he.event_type},{he.trigger_time:.6f},{drgb},{dfus},{rgb_lag},{fusion_lag},{adv}\n")
                else:
                    # No hazard recorded (shouldn't happen), write empty row
                    df.write('\n')
            print(f"Saved detections summary: {det_csv}")
        except Exception as e:
            print(f"Warning: failed to write detections.csv: {e}")
    
    # Cleanup
    try:
        # Prefer cleanup() if provided by the system, otherwise try destroy()
        if hasattr(sensor_system, 'cleanup'):
            sensor_system.cleanup()
        elif hasattr(sensor_system, 'destroy'):
            sensor_system.destroy()
        if walker is not None:
            walker.destroy()
        if vehicle is not None:
            vehicle.destroy()
        ego_vehicle.destroy()
    except:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--pre', type=float, default=2.0, help='Seconds before spawn')
    parser.add_argument('--post', type=float, default=6.0, help='Seconds after spawn')
    parser.add_argument('--min-video-seconds', type=float, default=6.0)
    parser.add_argument('--output', default='results/rgb_single_hazard')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hazard', choices=['pedestrian', 'vehicle'], default='vehicle')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--target-speed', type=float, default=None, help='Absolute target speed (m/s) for overtake hazard; if omitted a default delta over ego speed is used')
    
    args = parser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    try:
        record_single_hazard(client, pre_seconds=args.pre, post_seconds=args.post,
               min_video_seconds=args.min_video_seconds, 
               output_dir=args.output, debug=args.debug, seed=args.seed, hazard=args.hazard, target_speed=args.target_speed)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()