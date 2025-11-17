import time
import os
import numpy as np
import pandas as pd
import carla
from typing import List

from .events import HazardEvent
from .system import DualSensorSystem
from .environment import setup_controlled_environment, _spawn_ego_vehicle


def _warmup_detectors(sensor_system: DualSensorSystem, world, cycles: int = 20):
    """Tick the world and feed empty detections to initialize detectors."""
    for _ in range(cycles):
        try:
            world.tick()
        except Exception:
            pass
        sensor_system.process_detections([])


def _spawn_hazard_if_due(world, ego_vehicle, hazards_created, num_hazards, next_hazard_time, current_sim_time, debug: bool = False):
    """Spawn an appropriate hazard if the schedule says so. Returns (hazard, hazard_type, hazards_created, next_hazard_time)."""
    if hazards_created >= num_hazards or current_sim_time < next_hazard_time:
        return None, None, hazards_created, next_hazard_time

    # Alternate hazard types
    from .hazards import create_pedestrian_crossing_hazard, create_sudden_brake_hazard

    if hazards_created % 2 == 0:
        hazard = create_pedestrian_crossing_hazard(world, ego_vehicle, debug=debug)
        hazard_type = "Pedestrian Crossing"
    else:
        hazard = create_sudden_brake_hazard(world, ego_vehicle, debug=debug)
        hazard_type = "Sudden Brake"

    if hazard is not None:
        hazards_created += 1
        next_hazard_time += 1.0  # caller will add interval; increment slightly to avoid double-spawn on same tick

    return hazard, hazard_type, hazards_created, next_hazard_time


def _cleanup_resources(sensor_system: DualSensorSystem, hazard_events: List[HazardEvent], ego_vehicle, traffic_vehicles: List):
    """Destroy sensors, hazards, ego and traffic vehicles safely."""
    try:
        sensor_system.cleanup()
    except Exception:
        pass

    for hazard in hazard_events:
        try:
            if hazard.actor and hazard.actor.is_alive:
                if hazard.event_type == 'sudden_brake':
                    try:
                        hazard.actor.apply_control(carla.VehicleControl(brake=0.0))
                    except Exception:
                        pass
                if 'controller' in hazard.metadata and hazard.metadata['controller']:
                    try:
                        hazard.metadata['controller'].stop()
                    except Exception:
                        pass
                    try:
                        hazard.metadata['controller'].destroy()
                    except Exception:
                        pass
                try:
                    hazard.actor.destroy()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()
    except Exception:
        pass

    for vehicle in traffic_vehicles:
        try:
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass


def run_controlled_experiment(client, duration_seconds=120, num_hazards=6, debug: bool = False):
    """Run experiment with scripted hazards (clean, small high-level flow)."""
    print(f"\n{'='*70}")
    print("CONTROLLED HAZARD DETECTION EXPERIMENT")
    print(f"Duration: {duration_seconds}s | Target hazards: {num_hazards}")
    print(f"{'='*70}\n")

    world, traffic_vehicles = setup_controlled_environment(client)

    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()

    ego_vehicle, chosen_spawn = _spawn_ego_vehicle(world, vehicle_bp, spawn_points, debug=debug)

    sensor_system = DualSensorSystem(world, ego_vehicle)

    print("Warming up detectors...")
    _warmup_detectors(sensor_system, world, cycles=20)

    hazard_events: List[HazardEvent] = []
    next_hazard_time = 2.0 if debug else 10.0
    if debug:
        print(f"[DEBUG] Debug mode enabled: first hazard at {next_hazard_time}s")

    hazard_interval = duration_seconds / (num_hazards + 1)
    hazards_created = 0

    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < duration_seconds:
            try:
                world.tick()
            except Exception:
                pass

            frame_count += 1
            try:
                current_sim_time = world.get_snapshot().timestamp.elapsed_seconds
            except Exception:
                current_sim_time = 0.0

            if debug and frame_count % 5 == 0:
                print(f"[DEBUG] sim_time={current_sim_time:.3f} next_hazard={next_hazard_time:.3f} hazards_created={hazards_created} frame={frame_count}")

            # Spawn hazard when due
            hazard, hazard_type, hazards_created, next_hazard_time = _spawn_hazard_if_due(
                world, ego_vehicle, hazards_created, num_hazards, next_hazard_time, current_sim_time, debug=debug
            )
            if hazard is not None:
                hazard_events.append(hazard)
                print(f"[T={current_sim_time:.1f}s] Hazard #{hazards_created}: {hazard_type}")
                # add main interval after a successful spawn
                next_hazard_time += hazard_interval

            # Process detections
            try:
                sensor_system.process_detections(hazard_events)
            except Exception as e:
                print(f"[ERROR] Detector processing error: {e}")

            if frame_count % 200 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {elapsed:.1f}s / {duration_seconds}s ({frame_count} frames, {hazards_created} hazards)")

    finally:
        print("\nCleaning up...")
        _cleanup_resources(sensor_system, hazard_events, ego_vehicle, traffic_vehicles)

    return hazard_events, sensor_system


def analyze_results(hazard_events: List[HazardEvent], sensor_system: DualSensorSystem, 
                    output_dir='results'):
    """Analyze and save experimental results"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("HAZARD DETECTION ANALYSIS")
    print(f"{'='*70}\n")

    # Per-hazard analysis
    results = []
    for i, hazard in enumerate(hazard_events, 1):
        rgb_lag = hazard.detection_lag_rgb()
        fusion_lag = hazard.detection_lag_fusion()
        advantage = hazard.latency_advantage()

        result = {
            'Hazard_ID': i,
            'Event_Type': hazard.event_type,
            'Trigger_Time_s': hazard.trigger_time,
            'RGB_Detected': rgb_lag is not None,
            'RGB_Lag_ms': rgb_lag if rgb_lag is not None else -1,
            'Fusion_Detected': fusion_lag is not None,
            'Fusion_Lag_ms': fusion_lag if fusion_lag is not None else -1,
            'Latency_Advantage_ms': advantage if advantage is not None else None
        }
        results.append(result)

        # Print summary
        print(f"Hazard #{i} ({hazard.event_type}):")
        print(f"  RGB: {'Detected' if rgb_lag is not None else 'MISSED'}", end='')
        if rgb_lag is not None:
            print(f" in {rgb_lag:.1f}ms")
        else:
            print()
        print(f"  Fusion: {'Detected' if fusion_lag is not None else 'MISSED'}", end='')
        if fusion_lag is not None:
            print(f" in {fusion_lag:.1f}ms")
        else:
            print()
        if advantage is not None:
            if advantage > 0:
                print(f"  ✓ Fusion was {advantage:.1f}ms FASTER")
            else:
                print(f"  ✗ RGB was {-advantage:.1f}ms faster")
        print()

    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/hazard_detection_results.csv", index=False)

    # Calculate summary statistics
    detected_by_both = [r for r in results if r['RGB_Detected'] and r['Fusion_Detected']]

    if detected_by_both:
        avg_advantage = np.mean([r['Latency_Advantage_ms'] for r in detected_by_both])
        fusion_wins = sum(1 for r in detected_by_both if r['Latency_Advantage_ms'] > 0)

        summary = {
            'Total_Hazards': len(hazard_events),
            'RGB_Detection_Rate': sum(1 for r in results if r['RGB_Detected']) / len(results),
            'Fusion_Detection_Rate': sum(1 for r in results if r['Fusion_Detected']) / len(results),
            'Both_Detected': len(detected_by_both),
            'Avg_RGB_Lag_ms': np.mean([r['RGB_Lag_ms'] for r in results if r['RGB_Detected']]) if any(r['RGB_Detected'] for r in results) else None,
            'Avg_Fusion_Lag_ms': np.mean([r['Fusion_Lag_ms'] for r in results if r['Fusion_Detected']]) if any(r['Fusion_Detected'] for r in results) else None,
            'Avg_Latency_Advantage_ms': avg_advantage if detected_by_both else None,
            'Fusion_Faster_Count': fusion_wins if detected_by_both else 0,
            'RGB_Faster_Count': len(detected_by_both) - fusion_wins if detected_by_both else 0
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)

        print(f"{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Total Hazards: {summary['Total_Hazards']}")
        print(f"RGB Detection Rate: {summary['RGB_Detection_Rate']*100:.1f}%")
        print(f"Fusion Detection Rate: {summary['Fusion_Detection_Rate']*100:.1f}%")
        print(f"Both Detected: {summary['Both_Detected']}")

        if summary['Avg_RGB_Lag_ms'] is not None:
            print(f"\nAverage Detection Lags:")
            print(f"  RGB-only: {summary['Avg_RGB_Lag_ms']:.1f}ms")
        if summary['Avg_Fusion_Lag_ms'] is not None:
            print(f"  Fusion: {summary['Avg_Fusion_Lag_ms']:.1f}ms")

        if summary['Avg_Latency_Advantage_ms'] is not None:
            print(f"\nLatency Advantage:")
            if summary['Avg_Latency_Advantage_ms'] > 0:
                print(f"  ✓ Fusion average: {summary['Avg_Latency_Advantage_ms']:.1f}ms FASTER")
            else:
                print(f"  ✗ RGB average: {-summary['Avg_Latency_Advantage_ms']:.1f}ms faster")
            print(f"  Fusion faster: {summary['Fusion_Faster_Count']}/{summary['Both_Detected']} cases")

    # Save detector history
    try:
        rgb_history_df = pd.DataFrame(sensor_system.rgb_detector.detection_history)
        rgb_history_df.to_csv(f"{output_dir}/rgb_detector_history.csv", index=False)

        fusion_history_df = pd.DataFrame(sensor_system.fusion_detector.detection_history)
        fusion_history_df.to_csv(f"{output_dir}/fusion_detector_history.csv", index=False)
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"Results saved to '{output_dir}/' directory")
    print(f"{'='*70}\n")
