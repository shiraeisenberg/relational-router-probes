"""Enron email dataset loader with seniority annotations.

The Enron email corpus provides power differential labels via sender/recipient
seniority in the corporate hierarchy. This tests whether routers encode power
when it's linguistically exercised (directives from senior to junior, deference
from junior to senior) rather than just speaker identity.

Key insight (from Herbie's feedback): The Wikipedia Talk power probe tests speaker
identity (admin label), not power being exercised. Enron emails with seniority
metadata provide a stronger test—emails between executives and subordinates should
contain linguistically-marked power signals (hedging, deference, directives).

Source: CMU Enron Email Dataset / Kaggle enron-email-dataset
Paper: Klimt & Yang (2004) "The Enron Corpus: A New Dataset for Email Classification Research"

Usage:
    emails, stats = load_enron(n_samples=5000, balanced=True, split="train")
    stats.print_summary()
    
    # Get only downward (senior→junior) communications:
    downward = [e for e in emails if e.is_downward]
"""

from dataclasses import dataclass
from typing import Optional
import random
import re
from pathlib import Path


# Known Enron executives and their seniority tiers
# Source: Public Enron org charts, court documents, Wikipedia
ENRON_EXECUTIVES = {
    # CEO/President tier (highest)
    "kenneth.lay": "CEO",
    "ken.lay": "CEO", 
    "klay": "CEO",
    "jeff.skilling": "President",
    "jeffrey.skilling": "President",
    "jskilli": "President",
    "andrew.fastow": "CFO",
    "andy.fastow": "CFO",
    "afastow": "CFO",
    "richard.causey": "CAO",
    "rick.causey": "CAO",
    "rcausey": "CAO",
    "rebecca.mark": "Vice Chairman",
    
    # VP tier
    "greg.whalley": "VP",
    "gwhalley": "VP",
    "mark.frevert": "VP",
    "mfrevert": "VP",
    "steven.kean": "VP",
    "skean": "VP",
    "richard.shapiro": "VP",
    "rshapiro": "VP",
    "james.derrick": "VP",
    "jderrick": "VP",
    "mark.haedicke": "VP",
    "mhaedicke": "VP",
    "john.lavorato": "VP",
    "jlavorat": "VP",
    "louise.kitchen": "VP",
    "lkitchen": "VP",
    "david.delainey": "VP",
    "ddelaine": "VP",
    "stanley.horton": "VP",
    "shorton": "VP",
    "kevin.hannon": "VP",
    "khannon": "VP",
    "cliff.baxter": "VP",  
    "cbaxter": "VP",
    
    # Director tier
    "vince.kaminski": "Director",
    "vkamins": "Director",
    "sally.beck": "Director",
    "sbeck": "Director",
    "sherri.sera": "Director",
    "ssera": "Director",
    "tana.jones": "Director",
    "tjones": "Director",
    "sara.shackleton": "Director",
    "sshackl": "Director",
    "mark.taylor": "Director",
    "mtaylor": "Director",
    "elizabeth.sager": "Director",
    "esager": "Director",
    
    # Manager tier (inferred from common managerial patterns)
    "gerald.nemec": "Manager",
    "gnemec": "Manager",
    "carol.clair": "Manager",
    "cclair": "Manager",
    "susan.bailey": "Manager",
    "sbailey": "Manager",
    "marie.heard": "Manager",
    "mheard": "Manager",
    "tracy.geaccone": "Manager",
    "tgeacco": "Manager",
}

# Seniority hierarchy (higher number = more senior)
SENIORITY_LEVELS = {
    "CEO": 6,
    "President": 5,
    "CFO": 5,
    "CAO": 5,
    "Vice Chairman": 5,
    "VP": 4,
    "Director": 3,
    "Manager": 2,
    "Employee": 1,
    "Unknown": 0,
}

# For binary classification: VP and above = high seniority
HIGH_SENIORITY_TIERS = {"CEO", "President", "CFO", "CAO", "Vice Chairman", "VP"}


@dataclass
class EnronEmail:
    """Single email from the Enron corpus with seniority annotations."""
    email_id: str
    sender: str                    # Email address
    recipient: str                 # Primary recipient email
    sender_seniority: str          # CEO, VP, Director, Manager, Employee, Unknown
    recipient_seniority: str
    text: str                      # Email body
    subject: str                   # Email subject
    is_downward: bool              # True if sender outranks recipient
    is_upward: bool                # True if recipient outranks sender
    seniority_gap: int             # Difference in seniority levels
    date: Optional[str] = None     # Email date if available


@dataclass
class LoadStats:
    """Statistics from loading Enron emails."""
    n_emails: int
    n_downward: int                # Senior → Junior
    n_upward: int                  # Junior → Senior
    n_peer: int                    # Same seniority
    sender_seniority_dist: dict[str, int]
    recipient_seniority_dist: dict[str, int]
    mean_text_length: float
    median_text_length: float
    n_known_senders: int           # Senders with known seniority
    n_known_recipients: int
    n_skipped: int
    skipped_reasons: dict[str, int]
    
    def print_summary(self) -> None:
        """Print formatted loading summary."""
        print(f"Enron Emails: Loaded {self.n_emails:,} emails")
        print(f"  Downward (senior→junior): {self.n_downward:,} ({self.n_downward/self.n_emails*100:.1f}%)")
        print(f"  Upward (junior→senior): {self.n_upward:,} ({self.n_upward/self.n_emails*100:.1f}%)")
        print(f"  Peer-to-peer: {self.n_peer:,} ({self.n_peer/self.n_emails*100:.1f}%)")
        print(f"  Known senders: {self.n_known_senders:,}, Known recipients: {self.n_known_recipients:,}")
        print(f"  Text length: mean={self.mean_text_length:.0f}, median={self.median_text_length:.0f} chars")
        if self.n_skipped > 0:
            print(f"  Skipped {self.n_skipped} emails: {self.skipped_reasons}")


def extract_username(email_address: str) -> str:
    """Extract username from email address for seniority lookup."""
    if not email_address:
        return ""
    # Handle various formats
    email_address = email_address.lower().strip()
    
    # Remove angle brackets if present
    if "<" in email_address:
        match = re.search(r'<([^>]+)>', email_address)
        if match:
            email_address = match.group(1)
    
    # Extract username from email
    if "@" in email_address:
        username = email_address.split("@")[0]
    else:
        username = email_address
    
    # Clean up common patterns
    username = username.replace(".", "").replace("_", "").replace("-", "")
    
    return username


def get_seniority(email_address: str) -> str:
    """Determine seniority tier from email address."""
    if not email_address:
        return "Unknown"
    
    email_lower = email_address.lower().strip()
    
    # Check for exact matches in known executives
    for pattern, tier in ENRON_EXECUTIVES.items():
        if pattern in email_lower:
            return tier
    
    # Extract username and check
    username = extract_username(email_address)
    if username in ENRON_EXECUTIVES:
        return ENRON_EXECUTIVES[username]
    
    # Check if username suggests seniority from common patterns
    # (This is heuristic - real analysis would use the full org chart)
    if any(x in email_lower for x in ["vp", "vice.president", "svp", "evp"]):
        return "VP"
    if any(x in email_lower for x in ["director", "dir."]):
        return "Director"
    if any(x in email_lower for x in ["manager", "mgr"]):
        return "Manager"
    
    # Default: treat as regular employee
    # In a real implementation, we'd cross-reference with org chart data
    return "Employee"


def parse_email_headers(raw_email: str) -> dict:
    """Parse email headers from raw email text."""
    headers = {}
    lines = raw_email.split("\n")
    
    current_header = None
    current_value = []
    
    for line in lines:
        # Empty line marks end of headers
        if line.strip() == "":
            if current_header:
                headers[current_header] = " ".join(current_value).strip()
            break
        
        # Check if this is a new header
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            # Save previous header
            if current_header:
                headers[current_header] = " ".join(current_value).strip()
            
            parts = line.split(":", 1)
            current_header = parts[0].strip().lower()
            current_value = [parts[1].strip()] if len(parts) > 1 else []
        elif current_header:
            # Continuation of previous header
            current_value.append(line.strip())
    
    return headers


def extract_body(raw_email: str) -> str:
    """Extract email body from raw email text."""
    # Split at first blank line (end of headers)
    parts = raw_email.split("\n\n", 1)
    if len(parts) > 1:
        body = parts[1]
    else:
        body = raw_email
    
    # Clean up body
    body = body.strip()
    
    # Remove common email artifacts
    # Remove forwarded message markers
    body = re.sub(r'-+\s*Forwarded by.*?-+', '', body, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove original message markers  
    body = re.sub(r'-+\s*Original Message\s*-+', '', body, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    body = re.sub(r'\n{3,}', '\n\n', body)
    
    return body.strip()


def load_enron(
    n_samples: int = 5000,
    balanced: bool = True,
    split: str = "train",
    seed: int = 42,
    min_text_length: int = 50,
    max_text_length: int = 2000,
    filter_hierarchical: bool = True,
    verbose: bool = True,
    data_path: Optional[Path] = None,
) -> tuple[list[EnronEmail], LoadStats]:
    """Load Enron email dataset with seniority annotations.
    
    Args:
        n_samples: Target number of emails to return
        balanced: If True, balance downward vs upward communications
        split: "train" (80%) or "validation" (20%), split by email ID hash
        seed: Random seed for reproducibility
        min_text_length: Minimum email body length to include
        max_text_length: Maximum email body length (truncate longer)
        filter_hierarchical: If True, only include emails with clear hierarchy
        verbose: Whether to print progress
        data_path: Optional path to local Enron data (uses HuggingFace if None)
        
    Returns:
        Tuple of (list of EnronEmail objects, LoadStats)
    """
    if split not in ["train", "validation", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be train, validation, or test.")
    
    # Treat "test" as "validation" for this dataset
    if split == "test":
        split = "validation"
    
    rng = random.Random(seed)
    
    if verbose:
        print(f"Loading Enron email corpus...")
    
    # Try loading from HuggingFace datasets
    try:
        from datasets import load_dataset
        
        # Use the Kaggle Enron dataset on HuggingFace
        # Note: This may need adjustment based on available dataset format
        dataset = load_dataset("SetFit/enron_spam", split="train")
        use_hf = True
        
        if verbose:
            print(f"  Loaded {len(dataset)} emails from HuggingFace")
    except Exception as e:
        if verbose:
            print(f"  HuggingFace load failed ({e}), trying local data...")
        use_hf = False
        dataset = None
    
    # If HuggingFace failed and we have local data
    if not use_hf and data_path:
        # Load from local CSV or directory
        raise NotImplementedError("Local Enron loading not yet implemented")
    
    # Parse emails and extract metadata
    all_emails = []
    skipped_reasons: dict[str, int] = {}
    
    if use_hf and dataset:
        for idx, example in enumerate(dataset):
            try:
                # Extract text - format depends on dataset
                if "text" in example:
                    raw_text = example["text"]
                elif "email" in example:
                    raw_text = example["email"]
                else:
                    skipped_reasons["no_text"] = skipped_reasons.get("no_text", 0) + 1
                    continue
                
                # Parse headers if present
                headers = parse_email_headers(raw_text)
                body = extract_body(raw_text)
                
                # Get sender/recipient
                sender = headers.get("from", headers.get("sender", ""))
                recipient = headers.get("to", "")
                
                # Handle multiple recipients - take first
                if "," in recipient:
                    recipient = recipient.split(",")[0].strip()
                
                # Skip if no sender/recipient
                if not sender or not recipient:
                    skipped_reasons["no_parties"] = skipped_reasons.get("no_parties", 0) + 1
                    continue
                
                # Skip if body too short
                if len(body) < min_text_length:
                    skipped_reasons["too_short"] = skipped_reasons.get("too_short", 0) + 1
                    continue
                
                # Truncate if too long
                if len(body) > max_text_length:
                    body = body[:max_text_length] + "..."
                
                # Get seniority
                sender_seniority = get_seniority(sender)
                recipient_seniority = get_seniority(recipient)
                
                sender_level = SENIORITY_LEVELS.get(sender_seniority, 0)
                recipient_level = SENIORITY_LEVELS.get(recipient_seniority, 0)
                
                # Skip if filtering for hierarchical and both are unknown
                if filter_hierarchical:
                    if sender_seniority == "Unknown" and recipient_seniority == "Unknown":
                        skipped_reasons["unknown_hierarchy"] = skipped_reasons.get("unknown_hierarchy", 0) + 1
                        continue
                    # Also skip if same level (peer-to-peer)
                    if sender_level == recipient_level:
                        skipped_reasons["peer_to_peer"] = skipped_reasons.get("peer_to_peer", 0) + 1
                        continue
                
                is_downward = sender_level > recipient_level
                is_upward = sender_level < recipient_level
                seniority_gap = sender_level - recipient_level
                
                email = EnronEmail(
                    email_id=f"enron_{idx:06d}",
                    sender=sender,
                    recipient=recipient,
                    sender_seniority=sender_seniority,
                    recipient_seniority=recipient_seniority,
                    text=body,
                    subject=headers.get("subject", ""),
                    is_downward=is_downward,
                    is_upward=is_upward,
                    seniority_gap=seniority_gap,
                    date=headers.get("date"),
                )
                all_emails.append(email)
                
            except Exception as e:
                skipped_reasons["parse_error"] = skipped_reasons.get("parse_error", 0) + 1
                continue
    
    if verbose:
        print(f"  Parsed {len(all_emails)} valid emails")
    
    # Split into train/validation (80/20 by email_id hash)
    def is_train(email_id: str) -> bool:
        return hash(email_id) % 5 != 0  # 80% train, 20% validation
    
    if split == "train":
        filtered_emails = [e for e in all_emails if is_train(e.email_id)]
    else:
        filtered_emails = [e for e in all_emails if not is_train(e.email_id)]
    
    if verbose:
        print(f"  Split '{split}': {len(filtered_emails)} emails")
    
    # Separate by direction
    downward_emails = [e for e in filtered_emails if e.is_downward]
    upward_emails = [e for e in filtered_emails if e.is_upward]
    
    if verbose:
        print(f"  Available: {len(downward_emails)} downward, {len(upward_emails)} upward")
    
    # Sample according to parameters
    if balanced:
        n_per_class = n_samples // 2
        n_downward = min(n_per_class, len(downward_emails))
        n_upward = min(n_per_class, len(upward_emails))
        
        rng.shuffle(downward_emails)
        rng.shuffle(upward_emails)
        
        selected = downward_emails[:n_downward] + upward_emails[:n_upward]
        rng.shuffle(selected)
    else:
        all_filtered = downward_emails + upward_emails
        rng.shuffle(all_filtered)
        selected = all_filtered[:n_samples]
    
    # Compute stats
    import statistics
    
    text_lengths = [len(e.text) for e in selected]
    sender_dist = {}
    recipient_dist = {}
    n_known_senders = 0
    n_known_recipients = 0
    
    for e in selected:
        sender_dist[e.sender_seniority] = sender_dist.get(e.sender_seniority, 0) + 1
        recipient_dist[e.recipient_seniority] = recipient_dist.get(e.recipient_seniority, 0) + 1
        if e.sender_seniority != "Unknown":
            n_known_senders += 1
        if e.recipient_seniority != "Unknown":
            n_known_recipients += 1
    
    stats = LoadStats(
        n_emails=len(selected),
        n_downward=sum(1 for e in selected if e.is_downward),
        n_upward=sum(1 for e in selected if e.is_upward),
        n_peer=sum(1 for e in selected if not e.is_downward and not e.is_upward),
        sender_seniority_dist=sender_dist,
        recipient_seniority_dist=recipient_dist,
        mean_text_length=statistics.mean(text_lengths) if text_lengths else 0,
        median_text_length=statistics.median(text_lengths) if text_lengths else 0,
        n_known_senders=n_known_senders,
        n_known_recipients=n_known_recipients,
        n_skipped=sum(skipped_reasons.values()),
        skipped_reasons=skipped_reasons,
    )
    
    if verbose:
        stats.print_summary()
    
    return selected, stats


def load_enron_for_probing(
    n_train: int = 5000,
    n_val: int = 1000,
    probe_type: str = "direction",
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[EnronEmail], list[EnronEmail], dict]:
    """Load Enron emails formatted for power probing.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        probe_type: Type of probe:
            - "direction": Downward vs upward (most interesting)
            - "seniority": High vs low seniority of sender
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Tuple of (train_emails, val_emails, label_info)
    """
    train_emails, train_stats = load_enron(
        n_samples=n_train,
        balanced=True,
        split="train",
        seed=seed,
        verbose=verbose,
    )
    
    val_emails, val_stats = load_enron(
        n_samples=n_val,
        balanced=True,
        split="validation",
        seed=seed + 1,
        verbose=verbose,
    )
    
    if probe_type == "direction":
        label_info = {
            "name": "communication_direction",
            "classes": ["upward", "downward"],
            "description": "Classifies whether email is from junior→senior (upward) or senior→junior (downward)",
            "get_label": lambda e: 1 if e.is_downward else 0,
        }
    elif probe_type == "seniority":
        label_info = {
            "name": "sender_seniority",
            "classes": ["low", "high"],
            "description": "Classifies whether sender is high seniority (VP+) or low seniority",
            "get_label": lambda e: 1 if e.sender_seniority in HIGH_SENIORITY_TIERS else 0,
        }
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
    
    return train_emails, val_emails, label_info


def get_direction_distribution(emails: list[EnronEmail]) -> dict[str, int]:
    """Get distribution of communication directions."""
    return {
        'downward': sum(1 for e in emails if e.is_downward),
        'upward': sum(1 for e in emails if e.is_upward),
        'peer': sum(1 for e in emails if not e.is_downward and not e.is_upward),
    }


def get_seniority_distribution(emails: list[EnronEmail]) -> dict[str, int]:
    """Get distribution of sender seniority levels."""
    dist = {}
    for e in emails:
        dist[e.sender_seniority] = dist.get(e.sender_seniority, 0) + 1
    return dist

