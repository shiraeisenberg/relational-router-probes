"""Dataset loaders for relational signal probing.

Datasets:
    DailyDialog: Intent (4-class) and emotion (7-class) labels
    Wikipedia Talk: Power differential (admin/non-admin)
    SOTOPIA: Multi-turn social interactions with outcome labels
    MELD: Multi-party emotion recognition (transfer test)
    Synthetic: Generated tension/repair pairs
"""

from .dailydialog import load_dailydialog, DialogueTurn
from .wikipedia_talk import load_wikipedia_talk, WikipediaTurn
from .sotopia import load_sotopia_episodes, SOTOPIAEpisode
from .meld import load_meld, MELDUtterance
from .synthetic_tension import load_tension_pairs, generate_tension_pairs, TensionPair

__all__ = [
    "load_dailydialog",
    "DialogueTurn",
    "load_wikipedia_talk", 
    "WikipediaTurn",
    "load_sotopia_episodes",
    "SOTOPIAEpisode",
    "load_meld",
    "MELDUtterance",
    "load_tension_pairs",
    "generate_tension_pairs",
    "TensionPair",
]
