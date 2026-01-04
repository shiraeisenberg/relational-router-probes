"""Synthetic tension pair generator.

Generates two-turn dialogue exchanges with three tension classes:
- Escalation: Speaker B's response increases conflict/tension
- Repair: Speaker B's response de-escalates or resolves tension
- Neutral: Speaker B's response neither escalates nor repairs

Used to probe whether router logits encode tension dynamics.

Usage:
    pairs, stats = generate_tension_pairs(n_per_class=167)
    stats.print_summary()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import time
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# Tension class labels
TENSION_LABELS = ["escalation", "repair", "neutral"]


@dataclass
class TensionPair:
    """A two-turn dialogue exchange with tension label."""
    pair_id: str
    turn_a: str               # First speaker's message
    turn_b: str               # Second speaker's response
    label: str                # escalation, repair, neutral
    scenario: str             # Context (workplace, family, online, etc.)


@dataclass
class LoadStats:
    """Statistics from generating or loading tension pairs."""
    n_pairs: int
    escalation_count: int
    repair_count: int
    neutral_count: int
    n_generation_errors: int
    mean_turn_a_length: float
    mean_turn_b_length: float
    scenarios: dict[str, int]
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        print(f"Tension Pairs: {self.n_pairs} total")
        print(f"  Escalation: {self.escalation_count}")
        print(f"  Repair: {self.repair_count}")
        print(f"  Neutral: {self.neutral_count}")
        print(f"  Generation errors: {self.n_generation_errors}")
        print(f"  Mean turn_a length: {self.mean_turn_a_length:.0f} chars")
        print(f"  Mean turn_b length: {self.mean_turn_b_length:.0f} chars")
        if self.scenarios:
            top_scenarios = sorted(self.scenarios.items(), key=lambda x: -x[1])[:5]
            print(f"  Top scenarios: {dict(top_scenarios)}")


SYSTEM_PROMPT = """You are a dialogue generation assistant for machine learning research.
Generate realistic two-turn dialogue exchanges for tension classification research.
Output valid JSON only, no markdown formatting or code blocks."""

GENERATION_PROMPT = """Generate {batch_size} dialogue pairs for tension class: {label}

Definition of "{label}":
{definition}

Requirements:
- turn_a: The first speaker's message that sets up a situation
- turn_b: The second speaker's response that demonstrates the tension class
- scenario: Brief description (e.g., "workplace conflict", "family dinner", "online forum")
- Make dialogues natural, varied, and realistic
- Cover diverse scenarios: workplace, family, romantic, online, customer service, etc.
- Avoid stereotypes and make both speakers feel like real people

Output as a JSON array with exactly {batch_size} objects:
[
  {{"turn_a": "...", "turn_b": "...", "scenario": "..."}},
  ...
]

Only output the JSON array, nothing else."""

LABEL_DEFINITIONS = {
    "escalation": """Speaker B's response INCREASES conflict or tension from A's message.
Examples: dismissive replies, accusations, raising voice, bringing up past grievances, 
insults, refusing to engage constructively, one-upping complaints.""",
    
    "repair": """Speaker B's response DE-ESCALATES or RESOLVES tension from A's message.
Examples: apologizing, acknowledging feelings, offering compromise, validating concerns,
expressing empathy, taking responsibility, proposing solutions.""",
    
    "neutral": """Speaker B's response NEITHER escalates NOR repairs (informational exchange).
Examples: answering a question factually, providing requested information, 
topic changes, logistical responses, neutral acknowledgments.""",
}


def generate_tension_pairs(
    n_per_class: int = 167,
    model: str = "claude-sonnet-4-20250514",
    batch_size: int = 10,
    output_dir: Optional[Path] = None,
    save_intermediate: bool = True,
    verbose: bool = True,
) -> tuple[list[TensionPair], LoadStats]:
    """Generate synthetic tension pairs via Claude API.
    
    Args:
        n_per_class: Number of pairs to generate per class (~167 for 500 total)
        model: Claude model to use
        batch_size: Pairs to request per API call
        output_dir: Where to save generated pairs
        save_intermediate: Save after each batch (resume on failure)
        verbose: Print progress
        
    Returns:
        Tuple of (list of TensionPair objects, LoadStats)
    """
    import anthropic
    
    if output_dir is None:
        output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = anthropic.Anthropic()
    
    all_pairs: list[TensionPair] = []
    generation_errors = 0
    pair_counter = 0
    
    # Check for existing intermediate file
    intermediate_file = output_dir / "tension_pairs_intermediate.json"
    if intermediate_file.exists() and save_intermediate:
        if verbose:
            print(f"Loading existing intermediate file: {intermediate_file}")
        with open(intermediate_file) as f:
            existing_data = json.load(f)
        all_pairs = [TensionPair(**p) for p in existing_data["pairs"]]
        pair_counter = len(all_pairs)
        generation_errors = existing_data.get("errors", 0)
        if verbose:
            print(f"  Resuming from {len(all_pairs)} existing pairs")
    
    # Count existing pairs per class
    existing_counts = {label: 0 for label in TENSION_LABELS}
    for p in all_pairs:
        existing_counts[p.label] = existing_counts.get(p.label, 0) + 1
    
    for label in TENSION_LABELS:
        needed = n_per_class - existing_counts.get(label, 0)
        if needed <= 0:
            if verbose:
                print(f"[{label}] Already have {existing_counts[label]}/{n_per_class}, skipping")
            continue
        
        if verbose:
            print(f"[{label}] Generating {needed} pairs...")
        
        generated_for_label = 0
        
        while generated_for_label < needed:
            current_batch = min(batch_size, needed - generated_for_label)
            
            prompt = GENERATION_PROMPT.format(
                batch_size=current_batch,
                label=label,
                definition=LABEL_DEFINITIONS[label],
            )
            
            try:
                if verbose:
                    print(f"  Calling Claude API for {label} (batch {current_batch})...")
                
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    timeout=30.0,  # 30 second timeout
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                if verbose:
                    print(f"  Got response, parsing JSON...")
                
                # Parse JSON response
                content = response.content[0].text.strip()
                
                # Handle potential markdown wrapping
                if content.startswith("```"):
                    # Remove markdown code blocks
                    lines = content.split("\n")
                    content = "\n".join(
                        line for line in lines 
                        if not line.startswith("```")
                    )
                
                batch_data = json.loads(content)
                
                for item in batch_data:
                    pair_counter += 1
                    pair = TensionPair(
                        pair_id=f"tension_{pair_counter:04d}",
                        turn_a=item["turn_a"],
                        turn_b=item["turn_b"],
                        label=label,
                        scenario=item.get("scenario", "unknown"),
                    )
                    all_pairs.append(pair)
                    generated_for_label += 1
                
                if verbose:
                    print(f"  Generated batch: {len(batch_data)} pairs (total: {generated_for_label}/{needed})")
                
                # Save intermediate progress
                if save_intermediate:
                    _save_pairs(all_pairs, generation_errors, intermediate_file)
                
                # Rate limiting
                time.sleep(0.5)
                
            except json.JSONDecodeError as e:
                generation_errors += 1
                if verbose:
                    print(f"  JSON parse error: {e}")
                time.sleep(1)
                
            except Exception as e:
                generation_errors += 1
                if verbose:
                    print(f"  API error: {e}")
                time.sleep(2)
    
    # Compute stats
    stats = _compute_stats(all_pairs, generation_errors)
    
    # Save final output
    final_file = output_dir / "tension_pairs.json"
    _save_pairs(all_pairs, generation_errors, final_file)
    
    if verbose:
        print(f"\nSaved {len(all_pairs)} pairs to {final_file}")
        stats.print_summary()
    
    return all_pairs, stats


def _save_pairs(pairs: list[TensionPair], errors: int, path: Path) -> None:
    """Save pairs to JSON file."""
    data = {
        "pairs": [
            {
                "pair_id": p.pair_id,
                "turn_a": p.turn_a,
                "turn_b": p.turn_b,
                "label": p.label,
                "scenario": p.scenario,
            }
            for p in pairs
        ],
        "errors": errors,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _compute_stats(pairs: list[TensionPair], errors: int) -> LoadStats:
    """Compute statistics from generated pairs."""
    import statistics
    
    escalation_count = sum(1 for p in pairs if p.label == "escalation")
    repair_count = sum(1 for p in pairs if p.label == "repair")
    neutral_count = sum(1 for p in pairs if p.label == "neutral")
    
    turn_a_lengths = [len(p.turn_a) for p in pairs]
    turn_b_lengths = [len(p.turn_b) for p in pairs]
    
    scenarios: dict[str, int] = {}
    for p in pairs:
        scenarios[p.scenario] = scenarios.get(p.scenario, 0) + 1
    
    return LoadStats(
        n_pairs=len(pairs),
        escalation_count=escalation_count,
        repair_count=repair_count,
        neutral_count=neutral_count,
        n_generation_errors=errors,
        mean_turn_a_length=statistics.mean(turn_a_lengths) if turn_a_lengths else 0,
        mean_turn_b_length=statistics.mean(turn_b_lengths) if turn_b_lengths else 0,
        scenarios=scenarios,
    )


def load_tension_pairs(
    data_dir: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[list[TensionPair], LoadStats]:
    """Load pre-generated tension pairs.
    
    Args:
        data_dir: Path to data directory
        verbose: Whether to print summary
        
    Returns:
        Tuple of (list of TensionPair objects, LoadStats)
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
    
    pairs = [TensionPair(**item) for item in data["pairs"]]
    errors = data.get("errors", 0)
    
    stats = _compute_stats(pairs, errors)
    
    if verbose:
        stats.print_summary()
    
    return pairs, stats


def get_tension_distribution(pairs: list[TensionPair]) -> dict[str, int]:
    """Get distribution of tension labels."""
    return {
        "escalation": sum(1 for p in pairs if p.label == "escalation"),
        "repair": sum(1 for p in pairs if p.label == "repair"),
        "neutral": sum(1 for p in pairs if p.label == "neutral"),
    }
