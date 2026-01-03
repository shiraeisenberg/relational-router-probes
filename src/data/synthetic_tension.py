"""Synthetic tension/repair pair generator.

Generates conversation contexts with three response types:
- Escalation: Increases interpersonal tension
- Repair: De-escalates and restores warmth
- Neutral: Neither escalates nor repairs

Used to probe whether router logits encode tension dynamics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class TensionPair:
    """Synthetic tension/repair pair."""
    pair_id: str
    context: str              # Shared conversation context
    escalation: str           # Response that escalates tension
    repair: str               # Response that de-escalates
    neutral: str              # Neutral continuation


TENSION_GENERATION_PROMPT = '''Generate a conversation context and three possible responses.

The context should be a 2-3 turn conversation where interpersonal tension could go either way.

Provide:
1. ESCALATION: A response that increases conflict or tension
2. REPAIR: A response that de-escalates and restores warmth
3. NEUTRAL: A response that neither escalates nor repairs

Output as JSON:
{
    "context": "Speaker A: ... Speaker B: ...",
    "escalation": "Response that escalates",
    "repair": "Response that de-escalates",
    "neutral": "Response that continues neutrally"
}
'''


def load_tension_pairs(
    data_dir: Optional[Path] = None
) -> list[TensionPair]:
    """Load pre-generated tension pairs.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of TensionPair objects
    """
    if data_dir is None:
        data_dir = Path("data/synthetic")
    
    pairs_file = data_dir / "tension_pairs.json"
    
    if not pairs_file.exists():
        raise FileNotFoundError(
            f"Tension pairs not found at {pairs_file}. "
            "Run generate_tension_pairs() first."
        )
    
    with open(pairs_file) as f:
        data = json.load(f)
    
    return [TensionPair(**item) for item in data]


async def generate_tension_pairs(
    n_pairs: int = 500,
    output_dir: Optional[Path] = None,
    model: str = "claude-sonnet-4-20250514"
) -> list[TensionPair]:
    """Generate synthetic tension/repair pairs via Claude.
    
    Args:
        n_pairs: Number of pairs to generate
        output_dir: Where to save generated pairs
        model: Claude model to use
        
    Returns:
        List of generated TensionPair objects
    """
    import anthropic
    
    if output_dir is None:
        output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = anthropic.AsyncAnthropic()
    pairs = []
    
    # TODO: Implement generation loop
    # - Call Claude with TENSION_GENERATION_PROMPT
    # - Parse JSON response
    # - Create TensionPair objects
    # - Save to output_dir / "tension_pairs.json"
    
    raise NotImplementedError("Tension pair generation not yet implemented.")
