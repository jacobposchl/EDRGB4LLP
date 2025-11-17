import carla
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HazardEvent:
    """Represents a scripted hazard event"""
    event_type: str
    trigger_time: float  # Simulation time when hazard begins
    actor: Optional[carla.Actor] = None
    detected_rgb: Optional[float] = None  # Time when RGB-only detected it
    detected_fusion: Optional[float] = None  # Time when fusion detected it
    metadata: Dict = field(default_factory=dict)

    def detection_lag_rgb(self) -> Optional[float]:
        """Calculate RGB detection lag in milliseconds"""
        if self.detected_rgb is None:
            return None
        return (self.detected_rgb - self.trigger_time) * 1000.0

    def detection_lag_fusion(self) -> Optional[float]:
        """Calculate fusion detection lag in milliseconds"""
        if self.detected_fusion is None:
            return None
        return (self.detected_fusion - self.trigger_time) * 1000.0

    def latency_advantage(self) -> Optional[float]:
        """Calculate fusion advantage (positive = fusion faster)"""
        rgb_lag = self.detection_lag_rgb()
        fusion_lag = self.detection_lag_fusion()
        if rgb_lag is None or fusion_lag is None:
            return None
        return rgb_lag - fusion_lag  # Positive means fusion was faster
