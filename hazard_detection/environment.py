import random
import carla


def setup_controlled_environment(client, town_name='Town03', seed: int = None):
    """Setup environment with controlled traffic. If `seed` is provided,
    random operations will be deterministic for reproducible runs."""
    # Use a local RNG seeded when provided to avoid polluting global random state
    rng = random.Random(seed) if seed is not None else random
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

    # Spawn moderate traffic deterministically when a seed is provided
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')

    traffic_vehicles = []
    # Shuffle spawn points deterministically if a seed is set
    try:
        spawn_points_list = list(spawn_points)
    except Exception:
        spawn_points_list = spawn_points

    # Prefer spawn points that are not on junctions (i.e., straight road segments)
    non_junction_spawns = []
    try:
        non_junction_spawns = [sp for sp in spawn_points_list if not world.get_map().get_waypoint(sp.location).is_junction]
    except Exception:
        non_junction_spawns = []

    usable_spawn_points = non_junction_spawns if non_junction_spawns else spawn_points_list

    if seed is not None:
        rng.shuffle(usable_spawn_points)

    for sp in usable_spawn_points[:15]:  # try first 15 distinct spawn points
        vehicle_bp = rng.choice(vehicle_bps)
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle:
            try:
                vehicle.set_autopilot(True)
            except Exception:
                pass
            traffic_vehicles.append(vehicle)

    return world, traffic_vehicles


def _spawn_ego_vehicle(world, vehicle_bp, spawn_points, debug: bool = False, rng=None):
    """Attempt to spawn the ego vehicle robustly and return (ego_vehicle, chosen_spawn).

    Parameters:
    - world: CARLA world
    - vehicle_bp: blueprint for the ego vehicle
    - spawn_points: iterable of spawn points
    - debug: enable debug prints
    - rng: optional randomness source (random module or random.Random instance)
    """
    ego_vehicle = None
    chosen_spawn = None

    # Use provided RNG or fall back to module-level random
    rng_local = rng if rng is not None else random

    # Prefer spawn points that are not on junctions (straight road segments)
    try:
        spawn_points_list = list(spawn_points)
    except Exception:
        spawn_points_list = spawn_points

    non_junction = []
    try:
        non_junction = [sp for sp in spawn_points_list if not world.get_map().get_waypoint(sp.location).is_junction]
    except Exception:
        non_junction = []

    candidate_list = non_junction if non_junction else spawn_points_list

    # Deterministic shuffle when using an RNG
    try:
        rng_local.shuffle(candidate_list)
    except Exception:
        # rng_local may be the module `random` without a shuffle method bound; fallback
        random.shuffle(candidate_list)

    # Try to spawn at each candidate in order
    for sp in candidate_list:
        ego_vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if ego_vehicle:
            chosen_spawn = sp
            try:
                world.tick()
            except Exception:
                pass
            break

    if ego_vehicle is None:
        # Try a few direct spawn attempts (may raise). Prefer non-junction points
        fallback_candidates = non_junction if non_junction else spawn_points_list
        for _ in range(3):
            sp = rng_local.choice(fallback_candidates)
            try:
                ego_vehicle = world.spawn_actor(vehicle_bp, sp)
                chosen_spawn = sp
                break
            except Exception:
                continue

    if ego_vehicle is None:
        # Final fallback: try a small z-offset across a subset of spawn points
        subset = candidate_list[:20]
        for sp_try in subset:
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
