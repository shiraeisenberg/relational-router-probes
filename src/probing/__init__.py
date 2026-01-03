"""Linear probe training and evaluation.

This module provides:
- Generic linear probe utilities
- Signal-specific probing functions (intent, emotion, power, tension)
- Result dataclasses for consistent reporting

All probes use scikit-learn LogisticRegression with:
- max_iter=1000
- multi_class="multinomial" for >2 classes
- solver="lbfgs"
"""

from .linear_probe import train_probe, ProbeResult
from .intent_probe import train_intent_probe
from .emotion_probe import train_emotion_probe
from .power_probe import train_power_probe
from .tension_probe import train_tension_probe
from .relationship_probe import train_relationship_probe

__all__ = [
    "train_probe",
    "ProbeResult",
    "train_intent_probe",
    "train_emotion_probe",
    "train_power_probe",
    "train_tension_probe",
    "train_relationship_probe",
]
