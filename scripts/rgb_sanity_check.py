"""Run a short controlled experiment and save RGB frames/videos per hazard.

Usage:
    python scripts/rgb_sanity_check.py --duration 60 --hazards 4 --frames 15

Outputs saved to `results/rgb_sanity/` as per-hazard folders containing PNG frames,
a `video.mp4` inside each folder, and a top-level `frames.csv` index.
"""
import os
import time
import csv
import argparse
import traceback
import math

import cv2

import carla
import sys

# Ensure project root is on sys.path so `hazard_detection` can be imported
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hazard_detection.environment import setup_controlled_environment, _spawn_ego_vehicle
from hazard_detection.system import DualSensorSystem
from hazard_detection.hazards import create_pedestrian_crossing_hazard, create_sudden_brake_hazard


def run_rgb_sanity(client, duration_seconds=60, num_hazards=4, frames_per_hazard=15, output_dir='results/rgb_sanity', debug=False, video_seconds=6.0, seed: int = None):
    os.makedirs(output_dir, exist_ok=True)

    world, traffic_vehicles = setup_controlled_environment(client, seed=seed)

    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()

    ego_vehicle, _ = _spawn_ego_vehicle(world, vehicle_bp, spawn_points, debug=debug)

    sensor_system = DualSensorSystem(world, ego_vehicle)

    # Warm up detectors and simulation
    for _ in range(20):
        try:
            world.tick()
        except Exception:
            pass
        try:
            sensor_system.process_detections([])
        except Exception:
            pass

    hazard_interval = duration_seconds / (num_hazards + 1)
    next_hazard_time = 2.0 if debug else 10.0
    hazards_created = 0

    start_time = time.time()
    csv_rows = []

    try:
        while time.time() - start_time < duration_seconds:
            try:
                world.tick()
            except Exception:
                pass

            try:
                current_sim_time = world.get_snapshot().timestamp.elapsed_seconds
            except Exception:
                current_sim_time = 0.0

            # Spawn hazard if due
            if hazards_created < num_hazards and current_sim_time >= next_hazard_time:
                if hazards_created % 2 == 0:
                    hazard = create_pedestrian_crossing_hazard(world, ego_vehicle, debug=debug)
                    hazard_type = 'pedestrian_crossing'
                else:
                    hazard = create_sudden_brake_hazard(world, ego_vehicle, debug=debug)
                    hazard_type = 'sudden_brake'

                if hazard is not None:
                    hazards_created += 1
                    print(f"[T={current_sim_time:.1f}s] Hazard #{hazards_created}: {hazard_type}")

                    # Create folder for this hazard
                    folder = os.path.join(output_dir, f"hazard_{hazards_created}_{hazard_type}_{int(current_sim_time)}")
                    os.makedirs(folder, exist_ok=True)

                    # Collect a number of RGB frames (one per tick) after the spawn
                    saved = 0
                    safety = 0
                    last_ts = None
                    # Wait for new frames and only count genuinely new timestamps so we
                    # reliably capture `frames_per_hazard` distinct frames.
                    while saved < frames_per_hazard and safety < frames_per_hazard * 50:
                        try:
                            world.tick()
                        except Exception:
                            pass

                        # Ensure we have a frame
                        if sensor_system.rgb_frame is None or sensor_system.rgb_timestamp is None:
                            safety += 1
                            time.sleep(0.01)
                            continue

                        ts = sensor_system.rgb_timestamp
                        # Only save if this frame timestamp has advanced (new frame)
                        if last_ts is not None and ts <= last_ts:
                            # no new frame yet
                            safety += 1
                            time.sleep(0.005)
                            continue

                        frame = sensor_system.rgb_frame.copy()

                        img_path = os.path.join(folder, f"frame_{saved:03d}.png")
                        try:
                            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(img_path, bgr)
                        except Exception:
                            # fallback: try write as-is
                            try:
                                cv2.imwrite(img_path, frame)
                            except Exception:
                                print("Failed to save frame to", img_path)

                        csv_rows.append({'hazard_id': hazards_created, 'hazard_type': hazard_type, 'frame_idx': saved, 'timestamp': ts, 'image': img_path})
                        saved += 1
                        last_ts = ts

                    # advance next hazard time
                    next_hazard_time += hazard_interval

            # Regular processing (keeps detectors running)
            try:
                sensor_system.process_detections([])
            except Exception:
                pass

    finally:
        print('\nEncoding videos and writing CSV...')

        # Encode per-hazard mp4s
        for item in os.listdir(output_dir):
            folder = os.path.join(output_dir, item)
            if not os.path.isdir(folder):
                continue
            frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
            if not frames:
                continue
            first_path = os.path.join(folder, frames[0])
            img = cv2.imread(first_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            out_path = os.path.join(folder, 'video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            try:
                # Determine fps so that video length is at least `video_seconds`.
                frames_count = len(frames)
                desired_seconds = float(video_seconds)

                if frames_count <= 0:
                    continue

                fps = int(frames_count // desired_seconds) if frames_count >= desired_seconds else 1
                if fps < 1:
                    fps = 1

                # If at current fps the length is still less than desired, we'll pad the last frame.
                length = frames_count / float(fps)
                pad_count = 0
                if length < desired_seconds:
                    required_total = int(math.ceil(desired_seconds * fps))
                    pad_count = max(0, required_total - frames_count)

                writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
                last_im = None
                for f in frames:
                    p = os.path.join(folder, f)
                    im = cv2.imread(p)
                    if im is None:
                        continue
                    writer.write(im)
                    last_im = im

                # Pad with last frame if needed to reach desired duration
                for _ in range(pad_count):
                    if last_im is not None:
                        writer.write(last_im)

                writer.release()
            except Exception:
                print(f"Failed to encode video for {folder}:\n", traceback.format_exc())

        # Write CSV index
        csv_file = os.path.join(output_dir, 'frames.csv')
        try:
            with open(csv_file, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=['hazard_id', 'hazard_type', 'frame_idx', 'timestamp', 'image'])
                writer.writeheader()
                for r in csv_rows:
                    writer.writerow(r)
        except Exception:
            print('Failed to write CSV index:', traceback.format_exc())

        # Cleanup sensors and actors
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

        print('Done. Results are in:', os.path.abspath(output_dir))


def main():
    parser = argparse.ArgumentParser(description='RGB sanity check runner')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--hazards', type=int, default=4)
    parser.add_argument('--frames', type=int, default=15)
    parser.add_argument('--output', default='results/rgb_sanity')
    parser.add_argument('--video-seconds', type=float, default=6.0, help='Minimum seconds for encoded hazard videos')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for deterministic environment')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    run_rgb_sanity(client, duration_seconds=args.duration, num_hazards=args.hazards,
                   frames_per_hazard=args.frames, output_dir=args.output, debug=args.debug,
                   video_seconds=args.video_seconds, seed=args.seed)


if __name__ == '__main__':
    main()
