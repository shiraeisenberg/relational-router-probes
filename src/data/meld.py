"""MELD dataset loader.

MELD (Multimodal EmotionLines Dataset) from Friends TV show:
- 1,433 dialogues
- 13,708 utterances  
- 7 emotion classes (same as DailyDialog)
- 3 sentiment classes

Used for cross-dataset transfer tests.

Source: https://github.com/declare-lab/MELD
Paper: Poria et al. (2019)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MELDUtterance:
    """Single utterance from MELD dataset."""
    utterance_id: str
    dialogue_id: str
    speaker: str              # Character name
    text: str
    emotion: str              # Same 7 classes as DailyDialog
    sentiment: str            # positive, negative, neutral
    utterance_index: int


def load_meld(
    split: str = "train",
    data_dir: Optional[Path] = None
) -> list[MELDUtterance]:
    """Load MELD dataset.
    
    Args:
        split: One of "train", "dev", "test"
        data_dir: Path to data directory
        
    Returns:
        List of MELDUtterance objects
    """
    if data_dir is None:
        data_dir = Path("data/meld")
    
    # TODO: Implement loading logic
    # Download from https://github.com/declare-lab/MELD
    
    raise NotImplementedError(
        "MELD loader not yet implemented. "
        "Download from https://github.com/declare-lab/MELD"
    )
