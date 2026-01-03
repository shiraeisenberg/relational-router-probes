"""Cross-dataset transfer tests."""

from dataclasses import dataclass


@dataclass
class TransferResult:
    source_dataset: str
    target_dataset: str
    signal_type: str
    source_auc: float
    target_auc: float
    transfer_gap: float


def test_cross_dataset_transfer(
    source_X, source_y,
    target_X, target_y,
    signal_type: str,
    source_name: str,
    target_name: str
) -> TransferResult:
    """Test how well probe transfers across datasets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(source_X, source_y)
    
    source_auc = roc_auc_score(source_y, probe.predict_proba(source_X), multi_class="ovr")
    target_auc = roc_auc_score(target_y, probe.predict_proba(target_X), multi_class="ovr")
    
    return TransferResult(
        source_dataset=source_name,
        target_dataset=target_name,
        signal_type=signal_type,
        source_auc=source_auc,
        target_auc=target_auc,
        transfer_gap=source_auc - target_auc,
    )
