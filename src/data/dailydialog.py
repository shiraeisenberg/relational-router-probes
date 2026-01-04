"""DailyDialog dataset loader.

DailyDialog is a multi-turn dialogue dataset with:
- 13,118 dialogues
- 103,607 utterances
- Dialogue act labels (4 classes): inform, question, directive, commissive
- Emotion labels (7 classes): neutral, anger, disgust, fear, happiness, sadness, surprise

Source: http://yanran.li/dailydialog.html (also on HuggingFace)
Paper: Li et al. (2017) "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset"

Usage:
    turns, stats = load_dailydialog(split="train")
    print(f"Loaded {stats.n_dialogues} dialogues, {stats.n_turns} turns")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Mappings from integer labels to strings
DIALOGUE_ACTS = {
    0: "__dummy__",  # HF uses 0-indexed but 0 is rare/dummy
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


@dataclass
class LoadStats:
    """Statistics from loading DailyDialog."""
    n_dialogues: int
    n_turns: int
    n_skipped: int
    skipped_reasons: list[tuple[int, str]]  # (line_number, reason)
    act_distribution: dict[str, int]
    emotion_distribution: dict[str, int]
    
    def print_summary(self):
        """Print a summary of the loaded data."""
        print(f"Loaded {self.n_dialogues:,} dialogues ({self.n_turns:,} turns)")
        print(f"Skipped: {self.n_skipped} malformed lines")
        
        # Act distribution
        total_acts = sum(self.act_distribution.values())
        if total_acts > 0:
            act_pcts = {k: f"{v/total_acts*100:.1f}%" for k, v in sorted(
                self.act_distribution.items(), key=lambda x: -x[1]
            )}
            print(f"Act distribution: {act_pcts}")
        
        # Emotion distribution
        total_emo = sum(self.emotion_distribution.values())
        if total_emo > 0:
            emo_pcts = {k: f"{v/total_emo*100:.1f}%" for k, v in sorted(
                self.emotion_distribution.items(), key=lambda x: -x[1]
            )}
            print(f"Emotion distribution: {emo_pcts}")


def load_dailydialog(
    split: str = "train",
    data_dir: Optional[Path] = None,
    verbose: bool = True,
    use_huggingface: bool = True,
) -> tuple[list[DialogueTurn], LoadStats]:
    """Load DailyDialog dataset.
    
    Args:
        split: One of "train", "validation", "test"
        data_dir: Path to local data directory (only used if use_huggingface=False)
        verbose: Whether to print loading stats
        use_huggingface: If True, load from HuggingFace Hub (recommended)
        
    Returns:
        Tuple of (list of DialogueTurn objects, LoadStats)
        
    Raises:
        FileNotFoundError: If data files not found (local mode only)
        ValueError: If invalid split specified
    """
    if split not in ["train", "validation", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be train, validation, or test.")
    
    # Use local files if data_dir is provided
    if data_dir is not None:
        return _load_from_local(split, data_dir, verbose)
    
    if use_huggingface:
        return _load_from_huggingface(split, verbose)
    else:
        return _load_from_local(split, Path("data/dailydialog"), verbose)


def _load_from_huggingface(
    split: str,
    verbose: bool = True,
) -> tuple[list[DialogueTurn], LoadStats]:
    """Load DailyDialog from HuggingFace Hub."""
    from datasets import load_dataset
    from collections import defaultdict
    
    if verbose:
        print(f"Loading DailyDialog from HuggingFace ({split} split)...")
    
    # Use benjaminbeilharz/better_daily_dialog which is in Parquet format
    # Schema: dialog_id, utterance, turn_type (1-4), emotion (0-6)
    # Each row is one utterance, need to group by dialog_id
    dataset = load_dataset("benjaminbeilharz/better_daily_dialog", split=split)
    
    # Group by dialog_id to reconstruct dialogues
    dialogues = defaultdict(list)
    for example in dataset:
        dialogues[example["dialog_id"]].append({
            "text": example["utterance"],
            "act": example["turn_type"],
            "emotion": example["emotion"],
        })
    
    turns: list[DialogueTurn] = []
    act_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    skipped_reasons: list[tuple[int, str]] = []
    
    for dialog_idx, dialog_id in enumerate(sorted(dialogues.keys())):
        dialogue_id = f"daily_{split}_{dialog_idx:05d}"
        utterances = dialogues[dialog_id]
        
        for turn_idx, utt in enumerate(utterances):
            text = utt["text"]
            act = utt["act"]
            emotion = utt["emotion"]
            
            # Handle act labels (1-4 in this dataset, same as DIALOGUE_ACTS)
            if act == 0:
                act = 1  # Default to "inform" if dummy
            
            # Validate labels
            if act not in DIALOGUE_ACTS:
                skipped_reasons.append((dialog_idx, f"Invalid act label: {act}"))
                continue
            if emotion not in EMOTIONS:
                skipped_reasons.append((dialog_idx, f"Invalid emotion label: {emotion}"))
                continue
            
            act_str = DIALOGUE_ACTS[act]
            emotion_str = EMOTIONS[emotion]
            
            turn = DialogueTurn(
                turn_id=f"{dialogue_id}_t{turn_idx:02d}",
                dialogue_id=dialogue_id,
                speaker=turn_idx % 2,  # Alternating 0, 1, 0, 1, ...
                text=text.strip(),
                dialogue_act=act_str,
                emotion=emotion_str,
                turn_index=turn_idx,
            )
            turns.append(turn)
            
            # Update counts
            act_counts[act_str] = act_counts.get(act_str, 0) + 1
            emotion_counts[emotion_str] = emotion_counts.get(emotion_str, 0) + 1
    
    n_dialogues = len(dialogues)
    
    stats = LoadStats(
        n_dialogues=n_dialogues,
        n_turns=len(turns),
        n_skipped=len(skipped_reasons),
        skipped_reasons=skipped_reasons,
        act_distribution=act_counts,
        emotion_distribution=emotion_counts,
    )
    
    if verbose:
        stats.print_summary()
    
    return turns, stats


def _load_from_local(
    split: str,
    data_dir: Path,
    verbose: bool = True,
) -> tuple[list[DialogueTurn], LoadStats]:
    """Load DailyDialog from local files."""
    import warnings
    
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    text_file = split_dir / "dialogues_text.txt"
    act_file = split_dir / "dialogues_act.txt"
    emotion_file = split_dir / "dialogues_emotion.txt"
    
    # Check if files exist
    missing_files = []
    for f in [text_file, act_file, emotion_file]:
        if not f.exists():
            missing_files.append(str(f))
    
    if missing_files:
        raise FileNotFoundError(
            f"DailyDialog data files not found:\n"
            f"  {chr(10).join(missing_files)}\n\n"
            f"Please download DailyDialog from http://yanran.li/dailydialog.html\n"
            f"and place files in {data_dir}/\n"
            f"See data/README.md for detailed instructions.\n\n"
            f"Or use use_huggingface=True (default) to load from HuggingFace Hub."
        )
    
    # Read all files
    with open(text_file, "r", encoding="utf-8", errors="replace") as f:
        text_lines = f.readlines()
    with open(act_file, "r", encoding="utf-8", errors="replace") as f:
        act_lines = f.readlines()
    with open(emotion_file, "r", encoding="utf-8", errors="replace") as f:
        emotion_lines = f.readlines()
    
    # Verify line counts match
    if not (len(text_lines) == len(act_lines) == len(emotion_lines)):
        raise ValueError(
            f"File line counts don't match: "
            f"text={len(text_lines)}, act={len(act_lines)}, emotion={len(emotion_lines)}"
        )
    
    turns: list[DialogueTurn] = []
    skipped_reasons: list[tuple[int, str]] = []
    act_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    n_dialogues = 0
    
    for line_idx, (text_line, act_line, emotion_line) in enumerate(
        zip(text_lines, act_lines, emotion_lines), start=1
    ):
        # Parse text line - split by __eou__
        text_line = text_line.strip()
        act_line = act_line.strip()
        emotion_line = emotion_line.strip()
        
        # Skip empty lines
        if not text_line:
            continue
        
        # Split utterances (remove trailing __eou__ and split)
        utterances = [u.strip() for u in text_line.split("__eou__") if u.strip()]
        
        # Parse act labels
        try:
            act_labels = [int(x) for x in act_line.split()]
        except ValueError as e:
            skipped_reasons.append((line_idx, f"Invalid act labels: {e}"))
            warnings.warn(f"Line {line_idx}: Invalid act labels, skipping")
            continue
        
        # Parse emotion labels
        try:
            emotion_labels = [int(x) for x in emotion_line.split()]
        except ValueError as e:
            skipped_reasons.append((line_idx, f"Invalid emotion labels: {e}"))
            warnings.warn(f"Line {line_idx}: Invalid emotion labels, skipping")
            continue
        
        # Verify counts match
        if not (len(utterances) == len(act_labels) == len(emotion_labels)):
            skipped_reasons.append((
                line_idx, 
                f"Count mismatch: {len(utterances)} utterances, "
                f"{len(act_labels)} acts, {len(emotion_labels)} emotions"
            ))
            warnings.warn(f"Line {line_idx}: Turn count mismatch, skipping")
            continue
        
        # Validate labels are in expected range
        valid = True
        for i, (act, emo) in enumerate(zip(act_labels, emotion_labels)):
            if act not in DIALOGUE_ACTS:
                skipped_reasons.append((line_idx, f"Unknown act label: {act}"))
                warnings.warn(f"Line {line_idx}: Unknown act label {act}, skipping")
                valid = False
                break
            if emo not in EMOTIONS:
                skipped_reasons.append((line_idx, f"Unknown emotion label: {emo}"))
                warnings.warn(f"Line {line_idx}: Unknown emotion label {emo}, skipping")
                valid = False
                break
        
        if not valid:
            continue
        
        # Create DialogueTurn objects
        dialogue_id = f"daily_{split}_{line_idx:05d}"
        n_dialogues += 1
        
        for turn_idx, (text, act, emo) in enumerate(zip(utterances, act_labels, emotion_labels)):
            turn_id = f"{dialogue_id}_t{turn_idx:02d}"
            act_str = DIALOGUE_ACTS[act]
            emo_str = EMOTIONS[emo]
            
            turn = DialogueTurn(
                turn_id=turn_id,
                dialogue_id=dialogue_id,
                speaker=turn_idx % 2,  # Alternating 0, 1, 0, 1, ...
                text=text,
                dialogue_act=act_str,
                emotion=emo_str,
                turn_index=turn_idx,
            )
            turns.append(turn)
            
            # Update counts
            act_counts[act_str] = act_counts.get(act_str, 0) + 1
            emotion_counts[emo_str] = emotion_counts.get(emo_str, 0) + 1
    
    stats = LoadStats(
        n_dialogues=n_dialogues,
        n_turns=len(turns),
        n_skipped=len(skipped_reasons),
        skipped_reasons=skipped_reasons,
        act_distribution=act_counts,
        emotion_distribution=emotion_counts,
    )
    
    if verbose:
        stats.print_summary()
    
    return turns, stats


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


def get_turns_by_dialogue(turns: list[DialogueTurn]) -> dict[str, list[DialogueTurn]]:
    """Group turns by dialogue_id."""
    dialogues = {}
    for turn in turns:
        if turn.dialogue_id not in dialogues:
            dialogues[turn.dialogue_id] = []
        dialogues[turn.dialogue_id].append(turn)
    return dialogues
