"""Hazard detection package public API."""
# Keep imports minimal to avoid heavy module imports on package import.
from .events import HazardEvent

__all__ = [
    "HazardEvent",
]
