"""DailyDialog dataset loader.

DailyDialog is a multi-turn dialogue dataset with:
- 13,118 dialogues
- 103,607 utterances
- Dialogue act labels (4 classes): inform, question, directive, commissive
- Emotion labels (7 classes): neutral, anger, disgust, fear, happiness, sadness, surprise

Source: http://yanran.li/dailydialog.html
Paper: Li et al. (2017) "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset"

Usage:
    turns = load_dailydialog(split="train")
    for turn in turns:
        print(f"{turn.dialogue_act}: {turn.text[:50]}...")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

# Mappings from integer labels to strings
DIALOGUE_ACTS = {
    1: "inform",
    2: "question", 
    3: "directive",
    4: "commissive",
}

EMOTIONS = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


@dataclass
class DialogueTurn:
    """Single turn in a DailyDialog conversation."""
    turn_id: str
    dialogue_id: str
    speaker: int              # 0 or 1 (alternating speakers)
    text: str
    dialogue_act: str         # inform, question, directive, commissive
    emotion: str              # neutral, anger, disgust, fear, happiness, sadness, surprise
    turn_index: int           # Position in dialogue (0-indexed)


def load_dailydialog(
    split: str = "train",
    data_dir: Optional[Path] = None
) -> list[DialogueTurn]:
    """Load DailyDialog dataset.
    
    Args:
        split: One of "train", "validation", "test"
        data_dir: Path to data directory. Defaults to data/dailydialog/
        
    Returns:
        List of DialogueTurn objects
        
    Raises:
        FileNotFoundError: If data files not found
        ValueError: If invalid split specified
    """
    if data_dir is None:
        data_dir = Path("data/dailydialog")
    
    if split not in ["train", "validation", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be train, validation, or test.")
    
    # TODO: Implement loading logic
    # Expected file structure:
    # data/dailydialog/
    #   train/
    #     dialogues_text.txt      # __eou__ separated turns, __eod__ separated dialogues
    #     dialogues_act.txt       # Space-separated act labels per dialogue
    #     dialogues_emotion.txt   # Space-separated emotion labels per dialogue
    #   validation/
    #     ...
    #   test/
    #     ...
    
    raise NotImplementedError(
        "DailyDialog loader not yet implemented. "
        "Download data from http://yanran.li/dailydialog.html "
        f"and place in {data_dir}/"
    )


def get_dialogue_act_distribution(turns: list[DialogueTurn]) -> dict[str, int]:
    """Get distribution of dialogue acts."""
    counts = {}
    for turn in turns:
        counts[turn.dialogue_act] = counts.get(turn.dialogue_act, 0) + 1
    return counts


def get_emotion_distribution(turns: list[DialogueTurn]) -> dict[str, int]:
    """Get distribution of emotions."""
    counts = {}
    for turn in turns:
        counts[turn.emotion] = counts.get(turn.emotion, 0) + 1
    return counts
