"""Tests for data loaders."""

import pytest


def test_dailydialog_constants():
    """Test DailyDialog label mappings."""
    from src.data.dailydialog import DIALOGUE_ACTS, EMOTIONS
    
    assert len(DIALOGUE_ACTS) == 4
    assert "inform" in DIALOGUE_ACTS.values()
    assert "question" in DIALOGUE_ACTS.values()
    
    assert len(EMOTIONS) == 7
    assert "neutral" in EMOTIONS.values()


def test_tension_labels():
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
