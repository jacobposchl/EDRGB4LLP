"""Thin CLI to run the hazard detection experiment."""
import os
import sys

# Ensure project root is on sys.path so the `hazard_detection` package is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hazard_detection.connection import connect_to_carla
from hazard_detection.experiment import run_controlled_experiment, analyze_results


def main():
    client = connect_to_carla('localhost', 2000, timeout=10.0, retries=6, wait_seconds=5.0)

    hazard_events, sensor_system = run_controlled_experiment(
        client,
        duration_seconds=120,
        num_hazards=6
    )

    analyze_results(hazard_events, sensor_system)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
