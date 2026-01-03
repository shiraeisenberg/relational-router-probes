"""Tests for data loaders."""

import pytest
from pathlib import Path


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDailyDialogConstants:
    """Test DailyDialog label mappings."""
    
    def test_dialogue_acts_count(self):
        """Test that we have 4 dialogue acts."""
        from src.data.dailydialog import DIALOGUE_ACTS
        assert len(DIALOGUE_ACTS) == 4
    
    def test_dialogue_acts_values(self):
        """Test dialogue act values."""
        from src.data.dailydialog import DIALOGUE_ACTS
        expected = {"inform", "question", "directive", "commissive"}
        assert set(DIALOGUE_ACTS.values()) == expected
    
    def test_emotions_count(self):
        """Test that we have 7 emotions."""
        from src.data.dailydialog import EMOTIONS
        assert len(EMOTIONS) == 7
    
    def test_emotions_values(self):
        """Test emotion values."""
        from src.data.dailydialog import EMOTIONS
        expected = {"neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"}
        assert set(EMOTIONS.values()) == expected


class TestDailyDialogLoader:
    """Test DailyDialog loader functionality."""
    
    def test_load_train_split(self):
        """Test loading train split from fixtures."""
        from src.data.dailydialog import load_dailydialog
        
        turns, stats = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        # Check we got turns
        assert len(turns) > 0
        assert stats.n_dialogues == 4
        # Dialogue 1: 3 turns, Dialogue 2: 4 turns, Dialogue 3: 2 turns, Dialogue 4: 4 turns = 13
        assert stats.n_turns == 13
    
    def test_load_validation_split(self):
        """Test loading validation split from fixtures."""
        from src.data.dailydialog import load_dailydialog
        
        turns, stats = load_dailydialog(
            split="validation",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        assert stats.n_dialogues == 1
        assert stats.n_turns == 2
    
    def test_load_test_split(self):
        """Test loading test split from fixtures."""
        from src.data.dailydialog import load_dailydialog
        
        turns, stats = load_dailydialog(
            split="test",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        assert stats.n_dialogues == 1
        assert stats.n_turns == 3
    
    def test_invalid_split_raises(self):
        """Test that invalid split raises ValueError."""
        from src.data.dailydialog import load_dailydialog
        
        with pytest.raises(ValueError, match="Invalid split"):
            load_dailydialog(split="invalid")
    
    def test_missing_files_raises(self):
        """Test that missing files raises FileNotFoundError."""
        from src.data.dailydialog import load_dailydialog
        
        with pytest.raises(FileNotFoundError, match="DailyDialog data files not found"):
            load_dailydialog(
                split="train",
                data_dir=Path("/nonexistent/path")
            )
    
    def test_turn_structure(self):
        """Test that DialogueTurn objects have correct structure."""
        from src.data.dailydialog import load_dailydialog, DialogueTurn
        
        turns, _ = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        turn = turns[0]
        assert isinstance(turn, DialogueTurn)
        assert turn.turn_id is not None
        assert turn.dialogue_id is not None
        assert turn.speaker in [0, 1]
        assert len(turn.text) > 0
        assert turn.dialogue_act in ["inform", "question", "directive", "commissive"]
        assert turn.emotion in ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        assert turn.turn_index >= 0
    
    def test_alternating_speakers(self):
        """Test that speakers alternate within a dialogue."""
        from src.data.dailydialog import load_dailydialog, get_turns_by_dialogue
        
        turns, _ = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        dialogues = get_turns_by_dialogue(turns)
        
        for dialogue_id, dialogue_turns in dialogues.items():
            for i, turn in enumerate(dialogue_turns):
                expected_speaker = i % 2
                assert turn.speaker == expected_speaker, (
                    f"Turn {i} in {dialogue_id} has speaker {turn.speaker}, expected {expected_speaker}"
                )
    
    def test_stats_distribution(self):
        """Test that stats include proper distributions."""
        from src.data.dailydialog import load_dailydialog
        
        turns, stats = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        # Check distributions sum to total turns
        assert sum(stats.act_distribution.values()) == stats.n_turns
        assert sum(stats.emotion_distribution.values()) == stats.n_turns
    
    def test_dialogue_act_labels_correct(self):
        """Test that dialogue act labels are correctly mapped."""
        from src.data.dailydialog import load_dailydialog
        
        turns, _ = load_dailydialog(
            split="test",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        # Test fixture: "2 1 4" -> question, inform, commissive
        assert turns[0].dialogue_act == "question"
        assert turns[1].dialogue_act == "inform"
        assert turns[2].dialogue_act == "commissive"
    
    def test_emotion_labels_correct(self):
        """Test that emotion labels are correctly mapped."""
        from src.data.dailydialog import load_dailydialog
        
        turns, _ = load_dailydialog(
            split="validation",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        # Test fixture: "0 4" -> neutral, happiness
        assert turns[0].emotion == "neutral"
        assert turns[1].emotion == "happiness"


class TestDailyDialogHelpers:
    """Test DailyDialog helper functions."""
    
    def test_get_dialogue_act_distribution(self):
        """Test dialogue act distribution function."""
        from src.data.dailydialog import load_dailydialog, get_dialogue_act_distribution
        
        turns, _ = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        dist = get_dialogue_act_distribution(turns)
        
        assert isinstance(dist, dict)
        assert sum(dist.values()) == len(turns)
    
    def test_get_emotion_distribution(self):
        """Test emotion distribution function."""
        from src.data.dailydialog import load_dailydialog, get_emotion_distribution
        
        turns, _ = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        dist = get_emotion_distribution(turns)
        
        assert isinstance(dist, dict)
        assert sum(dist.values()) == len(turns)
    
    def test_get_turns_by_dialogue(self):
        """Test grouping turns by dialogue."""
        from src.data.dailydialog import load_dailydialog, get_turns_by_dialogue
        
        turns, stats = load_dailydialog(
            split="train",
            data_dir=FIXTURES_DIR / "dailydialog",
            verbose=False
        )
        
        dialogues = get_turns_by_dialogue(turns)
        
        assert len(dialogues) == stats.n_dialogues
        total_turns = sum(len(d) for d in dialogues.values())
        assert total_turns == stats.n_turns


class TestTensionPair:
    """Test tension pair data structure."""
    
    def test_tension_labels(self):
        """Test tension pair labels."""
        from src.data.synthetic_tension import TensionPair
        
        pair = TensionPair(
            pair_id="test",
            context="A: Hello. B: Hi.",
            escalation="That's ridiculous!",
            repair="I understand your point.",
            neutral="Okay.",
        )
        
        assert pair.pair_id == "test"
        assert "ridiculous" in pair.escalation
