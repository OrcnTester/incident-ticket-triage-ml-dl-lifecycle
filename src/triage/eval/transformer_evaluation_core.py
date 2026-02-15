import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, log_loss

def compute_ece(probs, y_true, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(mask):
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * mask.mean()
    return float(ece)

def compute_severe_error_rate(y_true, y_pred, severe_pairs):
    count = 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        if (t, p) in severe_pairs:
            count += 1
    return count / total

def evaluate_transformer(y_true, y_pred, probs, labels, high_risk_label=None):
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))
    metrics["log_loss"] = float(log_loss(y_true, probs))
    metrics["ece"] = compute_ece(probs, y_true)

    per_class_recall = recall_score(y_true, y_pred, average=None, labels=labels)
    metrics["per_class_recall"] = {
        label: float(r) for label, r in zip(labels, per_class_recall)
    }

    if high_risk_label is not None:
        idx = labels.index(high_risk_label)
        metrics["high_risk_recall"] = float(per_class_recall[idx])

    return metrics

def check_acceptance(metrics, thresholds):
    failed = []
    for key, threshold in thresholds.items():
        if key not in metrics:
            continue
        if metrics[key] < threshold:
            failed.append(key)

    return {
        "passed": len(failed) == 0,
        "failed_checks": failed
    }