import random
import math
import carla
import threading
import time
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

    # Compute a lateral vector perpendicular to forward_vector and a crossing
    # point a fixed distance in front of the ego where the walker should cross.
    lateral = carla.Vector3D(forward_vector.y, -forward_vector.x, 0.0)
    lateral_offset = 8.0  # meters to the side; tuned to be outside peripheral

    # We'll pick a crossing point a fixed distance in front of ego so the
    # walker crosses the ego's path predictably (rather than spawning far
    # ahead and sometimes missing the front of the vehicle).
    crossing_distance = min(spawn_distance, 12.0)
    crossing_point = carla.Location(
        x=ego_location.x + forward_vector.x * crossing_distance,
        y=ego_location.y + forward_vector.y * crossing_distance,
        z=ego_location.z + 1.0
    )

    # Spawn to one lateral side of the crossing point and target the opposite
    # side so motion is lateral across the ego's trajectory.
    spawn_location = carla.Location(
        x=crossing_point.x - lateral.x * lateral_offset,
        y=crossing_point.y - lateral.y * lateral_offset,
        z=crossing_point.z
    )

    target_location = carla.Location(
        x=crossing_point.x + lateral.x * lateral_offset,
        y=crossing_point.y + lateral.y * lateral_offset,
        z=crossing_point.z
    )

    # Project both spawn and target to the road surface to avoid z jitter
    try:
        wp_spawn = world.get_map().get_waypoint(spawn_location)
        spawn_location.z = wp_spawn.transform.location.z
    except Exception:
        pass
    try:
        wp_target = world.get_map().get_waypoint(target_location)
        target_location.z = wp_target.transform.location.z
    except Exception:
        pass

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
            # Use a higher max speed so the pedestrian runs across the road
            # quickly (helps ensure it crosses in front of the ego vehicle).
            walker_controller.go_to_location(target_location)
            walker_controller.set_max_speed(6.0)
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


def create_overtake_hazard(world, ego_vehicle, back_distance: float = 8.0, lateral_offset: float = 0.5,
                          target_speed: float = 50.0, duration: float = 6.0, debug: bool = False) -> Optional[HazardEvent]:
    """Spawn a hazard vehicle slightly behind the ego in the same lane and drive it at `target_speed`.

    - `back_distance`: meters behind the ego to spawn the hazard
    - `lateral_offset`: small lateral offset to avoid initial collision
    - `target_speed`: absolute speed (m/s) the hazard will aim for
    - `duration`: how long the hazard will drive before braking

    This function uses `target_speed` directly (no reading of ego velocity),
    keeping behavior simple and deterministic.
    """

    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    # lateral vector (right) perpendicular to forward
    lateral = carla.Vector3D(-forward.y, forward.x, 0.0)

    # Spawn a bit behind the ego and slightly to the side
    spawn_loc = carla.Location(
        x=ego_loc.x - forward.x * back_distance + lateral.x * lateral_offset,
        y=ego_loc.y - forward.y * back_distance + lateral.y * lateral_offset,
        z=ego_loc.z + 1.0
    )

    # Align yaw with ego so vehicle points forward along the road
    spawn_rot = ego_tf.rotation
    spawn_tf = carla.Transform(spawn_loc, spawn_rot)

    # Project spawn to road surface
    try:
        wp = world.get_map().get_waypoint(spawn_loc)
        spawn_loc.z = wp.transform.location.z
        spawn_tf = carla.Transform(spawn_loc, spawn_rot)
    except Exception:
        pass

    # pick same vehicle model as ego if possible
    try:
        vehicle_bp = world.get_blueprint_library().find(ego_vehicle.type_id)
    except Exception:
        vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))

    if debug:
        print(f"[DEBUG] Overtake spawn attempts for {vehicle_bp.id} near {spawn_loc}")

    # Try a few spawn attempts similar to other function
    vehicle = None
    attempts = [spawn_tf]
    for dz in (0.5, 1.0, -0.3):
        attempts.append(carla.Transform(carla.Location(spawn_loc.x, spawn_loc.y, spawn_loc.z + dz), spawn_rot))
    for df in (-2.0, 2.0, -5.0):
        tloc = carla.Location(spawn_loc.x - forward.x * df, spawn_loc.y - forward.y * df, spawn_loc.z)
        attempts.append(carla.Transform(tloc, spawn_rot))

    for idx, tf_try in enumerate(attempts):
        try:
            vehicle = world.try_spawn_actor(vehicle_bp, tf_try)
            if vehicle:
                if debug:
                    print(f"[DEBUG] Overtake vehicle spawned on attempt {idx} at {tf_try.location}")
                break
        except Exception as e:
            if debug:
                print(f"[DEBUG] Overtake spawn attempt {idx} exception: {e}")

    if vehicle is None:
        if debug:
            print(f"[DEBUG] Overtake: failed to spawn vehicle near {spawn_loc} after {len(attempts)} attempts")
            print(f"[DEBUG] Will try fallback random jitter attempts to find free space")

        # Fallback: try a few random nearby offsets to avoid collisions (helpful in crowded maps)
        import random as _rnd
        fallback_attempts = 8
        for i in range(fallback_attempts):
            jitter_x = _rnd.uniform(-6.0, 6.0)
            jitter_y = _rnd.uniform(-6.0, 6.0)
            jitter_z = _rnd.choice((0.5, 1.0, -0.3))
            tf_try = carla.Transform(carla.Location(spawn_loc.x + jitter_x, spawn_loc.y + jitter_y, spawn_loc.z + jitter_z), spawn_rot)
            try:
                vehicle = world.try_spawn_actor(vehicle_bp, tf_try)
                if vehicle:
                    if debug:
                        print(f"[DEBUG] Overtake fallback spawned on attempt {i} at {tf_try.location} (jitter x={jitter_x:.2f}, y={jitter_y:.2f})")
                    break
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Overtake fallback attempt {i} exception: {e}")

        if vehicle is None:
            if debug:
                print(f"[DEBUG] Overtake: all fallback attempts failed. Last spawn_loc={spawn_loc}")
            return None

    try:
        vehicle.set_autopilot(False)
    except Exception:
        pass

    try:
        world.tick()
    except Exception:
        pass

    # Use the provided absolute `target_speed` directly
    target_speed = float(target_speed)

    # Simple constant-throttle driver: apply steady forward throttle with no steering
    def _driver(actor, tgt_speed, dur):
        try:
            dt = world.get_settings().fixed_delta_seconds or 0.05
        except Exception:
            dt = 0.05
        throttle = max(0.2, min(1.0, tgt_speed / 30.0))
        steps = max(1, int(dur / dt))
        for _ in range(steps):
            try:
                actor.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0, brake=0.0))
            except Exception:
                pass
            time.sleep(dt)
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass

    drv = threading.Thread(target=_driver, args=(vehicle, target_speed, duration))
    drv.daemon = True
    drv.start()

    metadata = {'spawn_location': spawn_loc, 'target_speed': target_speed, 'duration': duration}
    return HazardEvent(event_type='vehicle_overtake', trigger_time=world.get_snapshot().timestamp.elapsed_seconds, actor=vehicle, metadata=metadata)


def create_oncoming_vehicle_hazard(world, ego_vehicle, front_distance: float = 60.0, lateral_offset: float = 0.0,
                                  target_speed: float = 25.0, duration: float = 6.0, debug: bool = False) -> Optional[HazardEvent]:
    """Spawn a vehicle ahead of the ego that drives directly toward it (opposite lane traffic)."""

    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    lateral = carla.Vector3D(-forward.y, forward.x, 0.0)

    spawn_loc = carla.Location(
        x=ego_loc.x + forward.x * front_distance + lateral.x * lateral_offset,
        y=ego_loc.y + forward.y * front_distance + lateral.y * lateral_offset,
        z=ego_loc.z + 1.0
    )

    spawn_yaw = (ego_tf.rotation.yaw + 180.0) % 360.0
    spawn_rot = carla.Rotation(pitch=ego_tf.rotation.pitch, yaw=spawn_yaw, roll=ego_tf.rotation.roll)
    spawn_tf = carla.Transform(spawn_loc, spawn_rot)

    try:
        wp = world.get_map().get_waypoint(spawn_loc)
        spawn_loc.z = wp.transform.location.z
        spawn_tf = carla.Transform(spawn_loc, spawn_rot)
    except Exception:
        pass

    try:
        vehicle_bp = world.get_blueprint_library().find(ego_vehicle.type_id)
    except Exception:
        vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))

    attempts = [spawn_tf]
    for dz in (0.5, 1.0, -0.5):
        attempts.append(carla.Transform(carla.Location(spawn_loc.x, spawn_loc.y, spawn_loc.z + dz), spawn_rot))
    for ds in (-4.0, 4.0):
        shifted_loc = carla.Location(spawn_loc.x + forward.x * ds, spawn_loc.y + forward.y * ds, spawn_loc.z)
        attempts.append(carla.Transform(shifted_loc, spawn_rot))

    vehicle = None
    for idx, tf_try in enumerate(attempts):
        try:
            vehicle = world.try_spawn_actor(vehicle_bp, tf_try)
            if vehicle:
                if debug:
                    print(f"[DEBUG] Oncoming vehicle spawned on attempt {idx} at {tf_try.location}")
                break
        except Exception as e:
            if debug:
                print(f"[DEBUG] Oncoming spawn attempt {idx} exception: {e}")

    if vehicle is None:
        if debug:
            print(f"[DEBUG] Oncoming: failed to spawn vehicle near {spawn_loc}")
        return None

    try:
        vehicle.set_autopilot(False)
    except Exception:
        pass

    try:
        world.tick()
    except Exception:
        pass

    target_speed = float(target_speed)

    def _driver(actor, tgt_speed, dur):
        try:
            dt = world.get_settings().fixed_delta_seconds or 0.05
        except Exception:
            dt = 0.05
        throttle = max(0.2, min(1.0, tgt_speed / 35.0))
        steps = max(1, int(dur / dt))
        for _ in range(steps):
            if not actor.is_alive:
                break
            try:
                actor.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0, brake=0.0))
            except Exception:
                break
            time.sleep(dt)
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass

    drv = threading.Thread(target=_driver, args=(vehicle, target_speed, duration))
    drv.daemon = True
    drv.start()

    metadata = {'spawn_location': spawn_loc, 'target_speed': target_speed, 'duration': duration}
    return HazardEvent(event_type='oncoming_vehicle', trigger_time=world.get_snapshot().timestamp.elapsed_seconds, actor=vehicle, metadata=metadata)


def create_drunk_driver_hazard(world, ego_vehicle, front_distance: float = 25.0, lateral_offset: float = -3.5,
                               target_speed: float = 35.0, duration: float = 8.0, weave_amplitude: float = 0.35,
                               debug: bool = False) -> Optional[HazardEvent]:
    """Spawn a weaving vehicle ahead of the ego to emulate an impaired driver."""

    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    lateral = carla.Vector3D(-forward.y, forward.x, 0.0)

    spawn_loc = carla.Location(
        x=ego_loc.x + forward.x * front_distance + lateral.x * lateral_offset,
        y=ego_loc.y + forward.y * front_distance + lateral.y * lateral_offset,
        z=ego_loc.z + 1.0
    )

    spawn_rot = ego_tf.rotation
    spawn_tf = carla.Transform(spawn_loc, spawn_rot)

    try:
        wp = world.get_map().get_waypoint(spawn_loc)
        spawn_loc.z = wp.transform.location.z
        spawn_tf = carla.Transform(spawn_loc, spawn_rot)
    except Exception:
        pass

    try:
        vehicle_bp = world.get_blueprint_library().find(ego_vehicle.type_id)
    except Exception:
        vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)

    if vehicle is None:
        if debug:
            print(f"[DEBUG] Drunk driver spawn failed at {spawn_loc}")
        return None

    try:
        vehicle.set_autopilot(False)
    except Exception:
        pass

    try:
        world.tick()
    except Exception:
        pass

    target_speed = float(target_speed)

    def _drunk_driver(actor, tgt_speed, dur, weave):
        try:
            dt = world.get_settings().fixed_delta_seconds or 0.05
        except Exception:
            dt = 0.05
        throttle = max(0.2, min(1.0, tgt_speed / 30.0))
        steps = max(1, int(dur / dt))
        for _ in range(steps):
            if not actor.is_alive:
                break
            steer = random.uniform(-weave, weave)
            try:
                actor.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
            except Exception:
                break
            time.sleep(dt)
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass

    drv = threading.Thread(target=_drunk_driver, args=(vehicle, target_speed, duration, weave_amplitude))
    drv.daemon = True
    drv.start()

    metadata = {
        'spawn_location': spawn_loc,
        'target_speed': target_speed,
        'duration': duration,
        'weave_amplitude': weave_amplitude
    }
    return HazardEvent(event_type='drunk_driver', trigger_time=world.get_snapshot().timestamp.elapsed_seconds, actor=vehicle, metadata=metadata)
