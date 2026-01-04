"""Wikipedia Talk Pages dataset loader.

Wikipedia Talk Pages provide power differential labels via editor status:
- 391,294 utterances from 125,292 conversations
- Admin vs non-admin status (direct power label)
- Edit count as secondary power signal

Source: Cornell Conversational Analysis Toolkit (ConvoKit)
Paper: Danescu-Niculescu-Mizil et al. "Echoes of Power"

Usage:
    turns, stats = load_wikipedia_talk(n_samples=5000, balanced=True, split="train")
    stats.print_summary()
"""

from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class WikipediaTurn:
    """Single turn from Wikipedia Talk Pages."""
    turn_id: str              # Unique identifier
    conversation_id: str      # For split leakage prevention
    speaker_id: str
    text: str
    is_admin: bool            # THE LABEL (power signal)
    edit_count: int           # Secondary signal (for ablations)
    turn_index: int           # Position in conversation


@dataclass
class LoadStats:
    """Statistics from loading Wikipedia Talk Pages."""
    n_conversations: int
    n_utterances: int
    admin_count: int
    non_admin_count: int
    mean_text_length: float
    median_text_length: float
    skipped_empty: int
    
    def print_summary(self) -> None:
        """Print formatted loading summary."""
        total = self.admin_count + self.non_admin_count
        admin_pct = self.admin_count / total * 100 if total > 0 else 0
        
        print(f"Wikipedia Talk Pages: Loaded {self.n_utterances:,} utterances from {self.n_conversations:,} conversations")
        print(f"  Admin: {self.admin_count:,} ({admin_pct:.1f}%)")
        print(f"  Non-admin: {self.non_admin_count:,} ({100 - admin_pct:.1f}%)")
        print(f"  Text length: mean={self.mean_text_length:.0f}, median={self.median_text_length:.0f} chars")
        if self.skipped_empty > 0:
            print(f"  Skipped {self.skipped_empty} empty utterances")


def load_wikipedia_talk(
    n_samples: int = 5000,
    balanced: bool = True,
    split: str = "train",
    seed: int = 42,
    min_text_length: int = 10,
    verbose: bool = True,
) -> tuple[list[WikipediaTurn], LoadStats]:
    """Load Wikipedia Talk Pages dataset via ConvoKit.
    
    Args:
        n_samples: Target number of utterances to return
        balanced: If True, sample 50/50 admin/non-admin
        split: "train" (80%) or "validation" (20%), split by conversation
        seed: Random seed for reproducibility
        min_text_length: Minimum text length to include
        verbose: Whether to print progress
        
    Returns:
        Tuple of (list of WikipediaTurn objects, LoadStats)
        
    Raises:
        ValueError: If invalid split specified
    """
    if split not in ["train", "validation", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be train, validation, or test.")
    
    # Treat "test" as "validation" for this dataset (we only do 80/20)
    if split == "test":
        split = "validation"
    
    from convokit import Corpus, download
    import statistics
    
    if verbose:
        print(f"Loading Wikipedia Talk corpus via ConvoKit...")
    
    corpus = Corpus(download("wiki-corpus"))
    
    # Group utterances by conversation
    conv_to_utts: dict[str, list[dict]] = {}
    skipped_empty = 0
    
    for utt in corpus.iter_utterances():
        # Skip empty or too-short utterances
        if not utt.text or len(utt.text.strip()) < min_text_length:
            skipped_empty += 1
            continue
        
        conv_id = utt.conversation_id
        if conv_id not in conv_to_utts:
            conv_to_utts[conv_id] = []
        
        # Parse edit count (stored as string in metadata)
        edit_count_str = utt.speaker.meta.get('edit-count', '0')
        try:
            edit_count = int(edit_count_str)
        except (ValueError, TypeError):
            edit_count = 0
        
        # Get admin status (check both utterance and speaker metadata)
        is_admin = bool(utt.meta.get('is-admin') or utt.speaker.meta.get('is-admin'))
        
        conv_to_utts[conv_id].append({
            'utt_id': utt.id,
            'speaker_id': utt.speaker.id,
            'text': utt.text.strip(),
            'is_admin': is_admin,
            'edit_count': edit_count,
        })
    
    if verbose:
        print(f"  Found {len(conv_to_utts):,} conversations with valid utterances")
    
    # Split conversations into train/validation (80/20)
    rng = random.Random(seed)
    conv_ids = list(conv_to_utts.keys())
    rng.shuffle(conv_ids)
    
    split_idx = int(len(conv_ids) * 0.8)
    if split == "train":
        selected_conv_ids = conv_ids[:split_idx]
    else:  # validation
        selected_conv_ids = conv_ids[split_idx:]
    
    if verbose:
        print(f"  Split '{split}': {len(selected_conv_ids):,} conversations")
    
    # Collect all utterances from selected conversations
    admin_utts = []
    non_admin_utts = []
    
    for conv_id in selected_conv_ids:
        for turn_idx, utt_data in enumerate(conv_to_utts[conv_id]):
            turn = WikipediaTurn(
                turn_id=f"wiki_{utt_data['utt_id']}",
                conversation_id=conv_id,
                speaker_id=utt_data['speaker_id'],
                text=utt_data['text'],
                is_admin=utt_data['is_admin'],
                edit_count=utt_data['edit_count'],
                turn_index=turn_idx,
            )
            if utt_data['is_admin']:
                admin_utts.append(turn)
            else:
                non_admin_utts.append(turn)
    
    if verbose:
        print(f"  Available: {len(admin_utts):,} admin, {len(non_admin_utts):,} non-admin")
    
    # Sample according to parameters
    if balanced:
        # 50/50 sampling
        n_per_class = n_samples // 2
        
        # Cap at available samples
        n_admin = min(n_per_class, len(admin_utts))
        n_non_admin = min(n_per_class, len(non_admin_utts))
        
        rng.shuffle(admin_utts)
        rng.shuffle(non_admin_utts)
        
        selected_utts = admin_utts[:n_admin] + non_admin_utts[:n_non_admin]
        rng.shuffle(selected_utts)
    else:
        # Natural distribution sampling
        all_utts = admin_utts + non_admin_utts
        rng.shuffle(all_utts)
        selected_utts = all_utts[:n_samples]
    
    # Compute stats
    text_lengths = [len(t.text) for t in selected_utts]
    admin_count = sum(1 for t in selected_utts if t.is_admin)
    non_admin_count = len(selected_utts) - admin_count
    
    # Count unique conversations in selected utterances
    unique_convs = set(t.conversation_id for t in selected_utts)
    
    stats = LoadStats(
        n_conversations=len(unique_convs),
        n_utterances=len(selected_utts),
        admin_count=admin_count,
        non_admin_count=non_admin_count,
        mean_text_length=statistics.mean(text_lengths) if text_lengths else 0,
        median_text_length=statistics.median(text_lengths) if text_lengths else 0,
        skipped_empty=skipped_empty,
    )
    
    if verbose:
        stats.print_summary()
    
    return selected_utts, stats


def get_admin_distribution(turns: list[WikipediaTurn]) -> dict[str, int]:
    """Get distribution of admin vs non-admin."""
    return {
        'admin': sum(1 for t in turns if t.is_admin),
        'non_admin': sum(1 for t in turns if not t.is_admin),
    }


def get_turns_by_conversation(turns: list[WikipediaTurn]) -> dict[str, list[WikipediaTurn]]:
    """Group turns by conversation_id."""
    convs: dict[str, list[WikipediaTurn]] = {}
    for turn in turns:
        if turn.conversation_id not in convs:
            convs[turn.conversation_id] = []
        convs[turn.conversation_id].append(turn)
    return convs
