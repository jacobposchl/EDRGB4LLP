"""Record a single hazard from the ego RGB camera for visual verification.

Saves a video and CSV with timestamps into `results/rgb_single_hazard/`.

Usage (from project root):
  python scripts\rgb_single_hazard.py --hazard pedestrian --pre 2 --post 6

Defaults record 2s before spawn and 6s after (total >= 8s). Change `--post` to
set total post-spawn recording; `--min-video-seconds` ensures final video is
at least that long (defaults to 6s).
"""
import os
import sys
import time
import argparse
import traceback
from collections import deque

import cv2
import numpy as np
import carla
import math

# Make project importable when running script directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hazard_detection.environment import setup_controlled_environment, _spawn_ego_vehicle
from hazard_detection.system import DualSensorSystem
from hazard_detection.hazards import create_pedestrian_crossing_hazard, create_sudden_brake_hazard


def record_single_hazard(client, hazard_type='pedestrian', pre_seconds=2.0, post_seconds=6.0,
                         min_video_seconds=6.0, output_dir='results/rgb_single_hazard', debug=False, seed: int = None):
    os.makedirs(output_dir, exist_ok=True)

    world, traffic_vehicles = setup_controlled_environment(client, seed=seed)

    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()

    ego_vehicle, _ = _spawn_ego_vehicle(world, vehicle_bp, spawn_points, debug=debug)
    sensor_system = DualSensorSystem(world, ego_vehicle)

    # Warm up
    for _ in range(20):
        try:
            world.tick()
        except Exception:
            pass
        try:
            sensor_system.process_detections([])
        except Exception:
            pass

    # Rolling buffer for pre-spawn frames
    # Determine fps from world fixed delta if possible; otherwise default 20
    try:
        fixed_dt = world.get_settings().fixed_delta_seconds
        fps = int(round(1.0 / fixed_dt)) if fixed_dt and fixed_dt > 0 else 20
    except Exception:
        fps = 20

    pre_frames = max(1, int(round(pre_seconds * fps)))
    post_frames = max(1, int(round(post_seconds * fps)))
    min_total_seconds = float(min_video_seconds)

    buffer = deque(maxlen=pre_frames)
    captured_frames = []

    print(f"Recording single hazard: type={hazard_type} pre={pre_seconds}s post={post_seconds}s fps={fps}")

    hazard_spawned = False
    spawn_info = None

    start_time = time.time()

    try:
        # Run simulation until we complete post-capture
        while True:
            try:
                world.tick()
            except Exception:
                pass

            # collect latest frame into buffer (if present)
            if sensor_system.rgb_frame is not None and sensor_system.rgb_timestamp is not None:
                # Copy to avoid mutation
                frame_copy = sensor_system.rgb_frame.copy()
                ts = float(sensor_system.rgb_timestamp)
                buffer.append((ts, frame_copy))

            # If hazard not spawned yet, spawn it and then start collecting post frames
            if not hazard_spawned:
                # Spawn hazard immediately (we want to see it play out)
                if hazard_type.lower().startswith('ped'):
                    hazard = create_pedestrian_crossing_hazard(world, ego_vehicle, debug=debug)
                else:
                    hazard = create_sudden_brake_hazard(world, ego_vehicle, debug=debug)

                if hazard is None:
                    print('Failed to spawn hazard; aborting.')
                    return

                hazard_spawned = True
                spawn_info = {'type': hazard.event_type, 'trigger_time': hazard.trigger_time, 'actor_id': getattr(hazard.actor, 'id', None)}
                print(f"Spawned hazard: {spawn_info}")

                # Initialize captured_frames with pre-buffer
                captured_frames = list(buffer)
                post_collected = 0
                # Now collect post_frames frames
                while post_collected < post_frames:
                    try:
                        world.tick()
                    except Exception:
                        pass

                    if sensor_system.rgb_frame is None or sensor_system.rgb_timestamp is None:
                        time.sleep(0.01)
                        continue

                    ts = float(sensor_system.rgb_timestamp)
                    frame_copy = sensor_system.rgb_frame.copy()
                    captured_frames.append((ts, frame_copy))
                    post_collected += 1

                # After collecting requested post frames, ensure minimum video seconds
                total_frames = len(captured_frames)
                desired_frames = max(total_frames, int(math.ceil(min_total_seconds * fps)))
                # If need to collect more frames to meet min duration, collect additional frames
                extra_needed = desired_frames - total_frames
                extra_collected = 0
                while extra_collected < extra_needed:
                    try:
                        world.tick()
                    except Exception:
                        pass

                    if sensor_system.rgb_frame is None or sensor_system.rgb_timestamp is None:
                        time.sleep(0.01)
                        continue

                    ts = float(sensor_system.rgb_timestamp)
                    frame_copy = sensor_system.rgb_frame.copy()
                    captured_frames.append((ts, frame_copy))
                    extra_collected += 1

                # Done collecting; break out
                break

            # safety: timeout
            if time.time() - start_time > 60.0:
                print('Timeout waiting for hazard to spawn/frames; aborting.')
                return

    except Exception:
        print('Error during recording:', traceback.format_exc())

    # Save video and CSV
    try:
        folder = os.path.join(output_dir, f"hazard_single_{hazard_type}_{int(time.time())}")
        os.makedirs(folder, exist_ok=True)

        # Save frames as PNG and write CSV rows
        csv_rows = []
        for i, (ts, frame) in enumerate(captured_frames):
            img_path = os.path.join(folder, f"frame_{i:04d}.png")
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, bgr)
            except Exception:
                cv2.imwrite(img_path, frame)
            csv_rows.append({'idx': i, 'timestamp': ts, 'image': img_path})

        # Encode video at fps; pad if needed by repeating last frame
        if len(captured_frames) == 0:
            print('No frames captured; aborting')
            return

        # Determine video length and padding
        first_img = cv2.imread(csv_rows[0]['image'])
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        total_frames = len(csv_rows)
        desired_frames = max(total_frames, int(math.ceil(min_total_seconds * fps)))
        pad_count = desired_frames - total_frames

        out_path = os.path.join(folder, 'video.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        last_im = None
        for r in csv_rows:
            im = cv2.imread(r['image'])
            if im is None:
                continue
            writer.write(im)
            last_im = im

        for _ in range(pad_count):
            if last_im is not None:
                writer.write(last_im)

        writer.release()

        # Save CSV
        import csv as _csv
        with open(os.path.join(folder, 'frames.csv'), 'w', newline='') as cf:
            w = _csv.DictWriter(cf, fieldnames=['idx', 'timestamp', 'image'])
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)

        print('Saved single-hazard video to', folder)

    except Exception:
        print('Failed to save video/CSV:', traceback.format_exc())

    # Cleanup
    try:
        sensor_system.cleanup()
    except Exception:
        pass
    try:
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()
    except Exception:
        pass
    for v in traffic_vehicles:
        try:
            if v and v.is_alive:
                v.destroy()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--hazard', choices=['pedestrian', 'sudden_brake'], default='pedestrian')
    parser.add_argument('--pre', type=float, default=2.0, help='Seconds to include before spawn')
    parser.add_argument('--post', type=float, default=6.0, help='Seconds to record after spawn')
    parser.add_argument('--min-video-seconds', type=float, default=6.0, help='Minimum encoded video length in seconds')
    parser.add_argument('--output', default='results/rgb_single_hazard')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for deterministic environment')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    record_single_hazard(client, hazard_type=args.hazard, pre_seconds=args.pre, post_seconds=args.post,
                         min_video_seconds=args.min_video_seconds, output_dir=args.output, debug=args.debug,
                         seed=args.seed)


if __name__ == '__main__':
    main()
