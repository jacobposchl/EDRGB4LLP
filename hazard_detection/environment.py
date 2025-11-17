import random
import carla


def setup_controlled_environment(client, town_name='Town03'):
    """Setup environment with controlled traffic"""
    world = client.load_world(town_name)

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

    # Spawn moderate traffic
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')

    traffic_vehicles = []
    for i in range(15):  # Fewer vehicles for more controlled environment
        vehicle_bp = random.choice(vehicle_bps)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            traffic_vehicles.append(vehicle)

    return world, traffic_vehicles


def _spawn_ego_vehicle(world, vehicle_bp, spawn_points, debug: bool = False):
    """Attempt to spawn the ego vehicle robustly and return (ego_vehicle, chosen_spawn)."""
    ego_vehicle = None
    chosen_spawn = None

    random.shuffle(spawn_points)
    for sp in spawn_points:
        ego_vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if ego_vehicle:
            chosen_spawn = sp
            try:
                world.tick()
            except Exception:
                pass
            break

    if ego_vehicle is None:
        # Try a few direct spawn attempts (may raise)
        for _ in range(3):
            sp = random.choice(spawn_points)
            try:
                ego_vehicle = world.spawn_actor(vehicle_bp, sp)
                chosen_spawn = sp
                break
            except Exception:
                continue

    if ego_vehicle is None:
        # Final fallback: try a small z-offset across a subset of spawn points
        for sp_try in spawn_points[:20]:
            try:
                sp_loc = carla.Location(sp_try.location.x, sp_try.location.y, sp_try.location.z + 1.0)
                sp_transform = carla.Transform(sp_loc, sp_try.rotation)
                ego_vehicle = world.try_spawn_actor(vehicle_bp, sp_transform)
                if ego_vehicle:
                    chosen_spawn = sp_try
                    try:
                        world.tick()
                    except Exception:
                        pass
                    break
            except Exception:
                continue

    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle: no free spawn point available.")

    try:
        ego_vehicle.set_autopilot(True)
    except Exception:
        pass

    if debug:
        try:
            loc = ego_vehicle.get_transform().location
            print(f"[DEBUG] Ego spawned id={ego_vehicle.id} loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})")
        except Exception:
            print(f"[DEBUG] Ego spawned id={getattr(ego_vehicle, 'id', 'unknown')}")

    return ego_vehicle, chosen_spawn
