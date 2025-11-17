import time
import carla

def connect_to_carla(host: str = 'localhost', port: int = 2000,
                      timeout: float = 10.0, retries: int = 6, wait_seconds: float = 5.0):
    """Attempt to connect to a CARLA server with retries.

    Tries up to `retries` times, waiting `wait_seconds` between attempts.
    Returns a connected `carla.Client` with timeout set on success.
    Raises RuntimeError on failure.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            # Sanity check: try a lightweight request
            try:
                _ = client.get_available_maps()
            except Exception:
                _ = client.get_world()
            print(f"Connected to CARLA at {host}:{port} (attempt {attempt})")
            return client
        except Exception as e:
            last_exc = e
            print(f"Connection attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Could not connect to CARLA at {host}:{port} after {retries} attempts") from last_exc
