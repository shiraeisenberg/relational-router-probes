"""Dumb baselines for comparison."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def run_bow_baseline(texts: list[str], labels: np.ndarray, test_size: float = 0.2):
    """Bag-of-words baseline."""
    from sklearn.model_selection import train_test_split
    
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, stratify=labels
    )
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    
    y_prob = probe.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    
    return {"baseline": "bow", "auc": auc}


def run_punctuation_baseline(texts: list[str], labels: np.ndarray, test_size: float = 0.2):
    """Punctuation-based baseline."""
    from sklearn.model_selection import train_test_split
    
    # Simple punctuation features
    def extract_punct(text):
        return [
            text.count("?"),
            text.count("!"),
            text.count("."),
            text.count(","),
            len(text),
            len(text.split()),
        ]
    
    X = np.array([extract_punct(t) for t in texts])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, stratify=labels
    )
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    
    y_prob = probe.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    
    return {"baseline": "punctuation", "auc": auc}
