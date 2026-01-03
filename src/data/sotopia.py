"""SOTOPIA dataset loader.

SOTOPIA provides multi-turn social interactions with outcome labels:
- 90 scenarios
- 40 characters with distinct profiles
- 7-dimensional social outcomes

Source: https://github.com/sotopia-lab/sotopia
Paper: Zhou et al. (2024) "SOTOPIA: Interactive Evaluation for Social Intelligence"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SOTOPIAEpisode:
    """Complete SOTOPIA interaction episode."""
    episode_id: str
    scenario: str
    agent1_profile: dict
    agent2_profile: dict
    turns: list[dict]         # [{speaker, text, turn_idx}, ...]
    outcomes: dict = field(default_factory=dict)  # {believability, relationship, goal, ...}


def load_sotopia_episodes(
    data_dir: Optional[Path] = None,
    scenario_filter: Optional[list[str]] = None
) -> list[SOTOPIAEpisode]:
    """Load SOTOPIA interaction episodes.
    
    Args:
        data_dir: Path to data directory
        scenario_filter: Optional list of scenario IDs to include
        
    Returns:
        List of SOTOPIAEpisode objects
    """
    if data_dir is None:
        data_dir = Path("data/sotopia")
    
    # TODO: Implement loading logic
    # Generate episodes via SOTOPIA pipeline
    
    raise NotImplementedError(
        "SOTOPIA loader not yet implemented. "
        "Generate episodes via sotopia-lab/sotopia pipeline."
    )


def generate_sotopia_episodes(
    n_episodes: int = 100,
    output_dir: Optional[Path] = None
) -> list[SOTOPIAEpisode]:
    """Generate SOTOPIA episodes via their pipeline.
    
    Args:
        n_episodes: Number of episodes to generate
        output_dir: Where to save generated episodes
        
    Returns:
        List of generated SOTOPIAEpisode objects
    """
    raise NotImplementedError("SOTOPIA generation not yet implemented.")
