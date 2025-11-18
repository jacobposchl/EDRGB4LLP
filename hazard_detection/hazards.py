import random
import math
import carla
from typing import Optional

from .events import HazardEvent


def create_pedestrian_crossing_hazard(world, ego_vehicle, spawn_distance=30.0, debug: bool = False, start_controller: bool = True) -> Optional[HazardEvent]:
    """Create a pedestrian that crosses in front of ego vehicle

    If `start_controller` is False the walker AI controller won't be created/started
    (useful for quick debug spawn tests).
    """
    # Get ego vehicle location and forward vector
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    forward_vector = ego_transform.get_forward_vector()

    # Calculate spawn point ahead of vehicle, offset laterally so the walker starts
    # outside the RGB camera peripheral and then crosses toward the center.
    # Compute a lateral vector perpendicular to forward_vector.
    # Use the lateral vector perpendicular to forward. We'll spawn the walker
    # slightly to one lateral side and set the target to the opposite side so
    # its motion is predominantly lateral (left/right) rather than forward.
    lateral = carla.Vector3D(forward_vector.y, -forward_vector.x, 0.0)
    lateral_offset = 8.0  # meters to the side; tuned to be outside peripheral

    # Spawn further forward and offset laterally (we spawn on the right side and
    # target to the left so the walker moves left across the camera view).
    spawn_location = carla.Location(
        x=ego_location.x + forward_vector.x * spawn_distance - lateral.x * lateral_offset,
        y=ego_location.y + forward_vector.y * spawn_distance - lateral.y * lateral_offset,
        z=ego_location.z + 1.0
    )

    # Target location is lateral across (predominantly lateral movement)
    target_location = carla.Location(
        x=spawn_location.x + lateral.x * (2.0 * lateral_offset),
        y=spawn_location.y + lateral.y * (2.0 * lateral_offset),
        z=spawn_location.z
    )

    # Spawn pedestrian
    walker_bp = random.choice(world.get_blueprint_library().filter('walker.pedestrian.*'))
    spawn_transform = carla.Transform(spawn_location)

    if debug:
        print(f"[DEBUG] Pedestrian spawn: trying walker blueprint {walker_bp.id} at {spawn_location}")

    # Compute rotation so the walker initially faces the crossing target (avoids facing backwards)
    dx = target_location.x - spawn_location.x
    dy = target_location.y - spawn_location.y
    yaw = math.degrees(math.atan2(dy, dx))
    spawn_rotation = carla.Rotation(yaw=yaw)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    walker = world.try_spawn_actor(walker_bp, spawn_transform)

    if walker is None:
        if debug:
            print(f"[DEBUG] Pedestrian spawn failed at {spawn_location} (collision/invalid).")
        return None

    # Advance simulation so actor transform is initialized (synchronous mode)
    try:
        world.tick()
    except Exception:
        pass

    if debug:
        try:
            print(f"[DEBUG] Pedestrian spawned id={walker.id} at transform={walker.get_transform().location}")
        except Exception:
            print(f"[DEBUG] Pedestrian spawned but unable to read transform")

    # Optionally setup walker controller to move the pedestrian
    walker_controller = None
    if start_controller:
        try:
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            # Start controller and let the world tick so controller initializes properly
            walker_controller.start()
            try:
                world.tick()
            except Exception:
                pass

            # Issue movement command after controller initialized using the
            # previously-computed `target_location` (opposite lateral side).
            # NOTE: do not recompute the spawn position here — the earlier
            # `target_location` is the intended crossing destination.
            walker_controller.go_to_location(target_location)
            walker_controller.set_max_speed(2.0)
        except Exception as e:
            if debug:
                print(f"[DEBUG] Warning: failed to create/start walker controller: {e}")
            walker_controller = None

    metadata = {
        'controller': walker_controller,
        'spawn_location': spawn_location
    }
    if start_controller and walker_controller is not None:
        metadata['target_location'] = target_location

    return HazardEvent(
        event_type='pedestrian_crossing',
        trigger_time=world.get_snapshot().timestamp.elapsed_seconds,
        actor=walker,
        metadata=metadata
    )


def create_sudden_brake_hazard(world, ego_vehicle, debug: bool = False) -> Optional[HazardEvent]:
    """Create a vehicle ahead that suddenly brakes"""
    # Find vehicle ahead of ego
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    forward_vector = ego_transform.get_forward_vector()

    # Look for vehicles in front (relaxed criteria)
    all_vehicles = world.get_actors().filter('vehicle.*')
    target_vehicle = None
    min_distance = float('inf')

    checked = 0
    if all_vehicles is None:
        if debug:
            print("[DEBUG] Warning: world.get_actors().filter('vehicle.*') returned None")
    else:
        for vehicle in all_vehicles:
            checked += 1
            try:
                if vehicle.id == ego_vehicle.id:
                    continue

                v_location = vehicle.get_transform().location
                to_vehicle = v_location - ego_location
                # Check if vehicle is ahead (dot product > 0)
                forward_projection = to_vehicle.x * forward_vector.x + to_vehicle.y * forward_vector.y

                if forward_projection > 0:  # Ahead
                    distance = ego_location.distance(v_location)
                    # Relaxed distance window to improve chances of finding a target
                    if 3.0 <= distance <= 35.0 and distance < min_distance:
                        target_vehicle = vehicle
                        min_distance = distance
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Exception checking vehicle {getattr(vehicle,'id', 'unknown')}: {e}")

    if debug:
        print(f"[DEBUG] Sudden-brake: checked {checked} vehicles, selected target={getattr(target_vehicle,'id', None)} distance={min_distance if target_vehicle else 'N/A'}")

    # If no suitable vehicle was found, optionally force-spawn a temporary vehicle ahead
    forced_spawned = False
    temp_vehicle = None
    if target_vehicle is None:
        # Compute a spawn location ahead of ego (~12 meters ahead)
        spawn_distance = 12.0
        spawn_location = carla.Location(
            x=ego_location.x + forward_vector.x * spawn_distance - forward_vector.y * 1.5,
            y=ego_location.y + forward_vector.y * spawn_distance + forward_vector.x * 1.5,
            z=ego_location.z + 1.0
        )

        spawn_transform = carla.Transform(spawn_location, ego_transform.rotation)
        vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
        bp = random.choice(vehicle_bps)
        if debug:
            print(f"[DEBUG] No target found — attempting forced spawn ahead at {spawn_location} using {bp.id}")

        try:
            temp_vehicle = world.try_spawn_actor(bp, spawn_transform)
            if temp_vehicle is not None:
                # Advance simulation to initialize transform
                try:
                    world.tick()
                except Exception:
                    pass
                # Ensure it's not on autopilot so we can control braking
                try:
                    temp_vehicle.set_autopilot(False)
                except Exception:
                    pass
                target_vehicle = temp_vehicle
                forced_spawned = True
                min_distance = ego_location.distance(target_vehicle.get_transform().location)
                if debug:
                    print(f"[DEBUG] Forced spawn succeeded: id={target_vehicle.id} distance={min_distance:.2f}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] Forced spawn attempt failed: {e}")

    if target_vehicle is None:
        # Still no target found or spawned
        if debug:
            print("[DEBUG] Sudden-brake: no target available after forced spawn attempt")
        return None

    # Apply sudden brake to the chosen target
    try:
        target_vehicle.apply_control(carla.VehicleControl(brake=1.0))
    except Exception as e:
        if debug:
            print(f"[DEBUG] Failed to apply brake to vehicle {getattr(target_vehicle,'id', 'unknown')}: {e}")

    metadata = {'distance': min_distance}
    if forced_spawned:
        metadata['forced_spawn'] = True
        metadata['forced_bp'] = bp.id if 'bp' in locals() else 'unknown'

    return HazardEvent(
        event_type='sudden_brake',
        trigger_time=world.get_snapshot().timestamp.elapsed_seconds,
        actor=target_vehicle,
        metadata=metadata
    )
