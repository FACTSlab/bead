"""Standard model evaluation metrics.

This module provides common machine learning evaluation metrics including
accuracy, precision, recall, F1 score, and confusion matrices.
"""

from __future__ import annotations

import numpy as np

# Type alias for classification labels (categorical, ordinal, or numeric)
type Label = int | str | float


class ModelMetrics:
    """Standard model evaluation metrics.

    Provides static methods for computing common classification metrics:
    - Accuracy
    - Precision, Recall, F1 score (with micro/macro/weighted averaging)
    - Confusion matrix
    - Classification report

    Examples
    --------
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> ModelMetrics.accuracy(y_true, y_pred)
    0.8
    >>> metrics = ModelMetrics.precision_recall_f1(y_true, y_pred)
    >>> 'precision' in metrics and 'recall' in metrics
    True
    """

    @staticmethod
    def accuracy(y_true: list[Label], y_pred: list[Label]) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        y_true : list[Label]
            True labels.
        y_pred : list[Label]
            Predicted labels.

        Returns
        -------
        float
            Accuracy (proportion of correct predictions).

        Raises
        ------
        ValueError
            If y_true and y_pred have different lengths or are empty.

        Examples
        --------
        >>> y_true = [1, 2, 3, 1, 2]
        >>> y_pred = [1, 2, 2, 1, 2]
        >>> ModelMetrics.accuracy(y_true, y_pred)
        0.8
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have same length: "
                f"{len(y_true)} != {len(y_pred)}"
            )

        if not y_true:
            raise ValueError("y_true and y_pred cannot be empty")

        correct = sum(yt == yp for yt, yp in zip(y_true, y_pred, strict=True))
        return correct / len(y_true)

    @staticmethod
    def confusion_matrix(  # type: ignore[return]
        y_true: list[Label],
        y_pred: list[Label],
        labels: list[Label] | None = None,
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Compute confusion matrix.

        Parameters
        ----------
        y_true : list[Label]
            True labels.
        y_pred : list[Label]
            Predicted labels.
        labels : list[Label] | None
            List of labels to include in matrix. If None, uses all labels
            found in y_true or y_pred (sorted).

        Returns
        -------
        np.ndarray
            Confusion matrix of shape (n_classes, n_classes).
            Element [i, j] is count of items with true label i and
            predicted label j.

        Raises
        ------
        ValueError
            If y_true and y_pred have different lengths or are empty.

        Examples
        --------
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> cm = ModelMetrics.confusion_matrix(y_true, y_pred)
        >>> cm.shape
        (2, 2)
        >>> cm[0, 0]  # True negatives
        2
        >>> cm[1, 1]  # True positives
        2
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have same length: "
                f"{len(y_true)} != {len(y_pred)}"
            )

        if not y_true:
            raise ValueError("y_true and y_pred cannot be empty")

        # Determine labels
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        n_labels = len(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        # Build confusion matrix
        cm = np.zeros((n_labels, n_labels), dtype=int)

        for yt, yp in zip(y_true, y_pred, strict=True):
            i = label_to_idx[yt]
            j = label_to_idx[yp]
            cm[i, j] += 1

        return cm

    @staticmethod
    def precision_recall_f1(
        y_true: list[Label],
        y_pred: list[Label],
        average: str = "macro",
        labels: list[Label] | None = None,
    ) -> dict[str, float]:
        """Compute precision, recall, and F1 score.

        Parameters
        ----------
        y_true : list[Label]
            True labels.
        y_pred : list[Label]
            Predicted labels.
        average : str, default="macro"
            Averaging strategy:
            - "macro": Unweighted mean per class
            - "micro": Global average (counts all instances)
            - "weighted": Weighted by support (number of true instances)
        labels : list[Label] | None
            Labels to include. If None, uses all labels in y_true or y_pred.

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'precision', 'recall', 'f1', 'support'.

        Raises
        ------
        ValueError
            If y_true and y_pred have different lengths or are empty.

        Examples
        --------
        >>> y_true = [0, 1, 1, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 1, 1]
        >>> metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average='macro')
        >>> 'precision' in metrics
        True
        >>> 0.0 <= metrics['f1'] <= 1.0
        True
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have same length: "
                f"{len(y_true)} != {len(y_pred)}"
            )

        if not y_true:
            raise ValueError("y_true and y_pred cannot be empty")

        # Determine labels
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        # Compute confusion matrix
        cm = ModelMetrics.confusion_matrix(y_true, y_pred, labels=labels)  # type: ignore[assignment]

        # Per-class metrics
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        supports: list[float] = []

        for i, _ in enumerate(labels):
            # True positives, false positives, false negatives
            tp = cm[i, i]  # type: ignore[index]
            fp = cm[:, i].sum() - tp  # type: ignore[operator, index]
            fn = cm[i, :].sum() - tp  # type: ignore[operator, index]

            # Support (number of true instances)
            support = cm[i, :].sum()  # type: ignore[index]
            supports.append(support)

            # Precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            precisions.append(precision)

            # Recall
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            recalls.append(recall)

            # F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1s.append(f1)

        # Aggregate based on averaging strategy
        if average == "macro":
            # Unweighted mean
            avg_precision = float(np.mean(precisions))  # type: ignore[arg-type]
            avg_recall = float(np.mean(recalls))  # type: ignore[arg-type]
            avg_f1 = float(np.mean(f1s))  # type: ignore[arg-type]
        elif average == "micro":
            # Global counts
            tp_total = np.trace(cm)  # type: ignore[arg-type]
            fp_total = cm.sum() - tp_total  # type: ignore[operator]
            fn_total = cm.sum() - tp_total  # type: ignore[operator]

            if (tp_total + fp_total) > 0:
                avg_precision = tp_total / (tp_total + fp_total)
            else:
                avg_precision = 0.0

            if (tp_total + fn_total) > 0:
                avg_recall = tp_total / (tp_total + fn_total)
            else:
                avg_recall = 0.0

            if avg_precision + avg_recall > 0:
                avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            else:
                avg_f1 = 0.0
        elif average == "weighted":
            # Weighted by support
            total_support: int = sum(supports)  # type: ignore[arg-type]
            if total_support > 0:
                avg_precision = (
                    sum(  # type: ignore[arg-type]
                        p * s for p, s in zip(precisions, supports, strict=True)
                    )
                    / total_support
                )
                avg_recall = (
                    sum(  # type: ignore[arg-type]
                        r * s for r, s in zip(recalls, supports, strict=True)
                    )
                    / total_support
                )
                avg_f1 = (
                    sum(  # type: ignore[arg-type]
                        f * s for f, s in zip(f1s, supports, strict=True)
                    )
                    / total_support
                )
            else:
                avg_precision = 0.0
                avg_recall = 0.0
                avg_f1 = 0.0
        else:
            raise ValueError(
                f"Unknown average strategy: {average}. "
                "Must be one of: 'macro', 'micro', 'weighted'"
            )

        return {
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
            "support": float(sum(supports)),
        }

    @staticmethod
    def classification_report(
        y_true: list[Label],
        y_pred: list[Label],
        labels: list[Label] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Generate comprehensive classification report.

        Computes precision, recall, F1, and support for each class,
        plus macro and weighted averages.

        Parameters
        ----------
        y_true : list[Label]
            True labels.
        y_pred : list[Label]
            Predicted labels.
        labels : list[Label] | None
            Labels to include. If None, uses all labels in data.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary with structure:
            {
                'class_0': {
                    'precision': 0.85, 'recall': 0.80,
                    'f1': 0.82, 'support': 100
                },
                'class_1': {...},
                'macro_avg': {
                    'precision': 0.83, 'recall': 0.81,
                    'f1': 0.82, 'support': 200
                },
                'weighted_avg': {...},
                'accuracy': 0.82
            }

        Examples
        --------
        >>> y_true = [0, 1, 1, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 1, 1]
        >>> report = ModelMetrics.classification_report(y_true, y_pred)
        >>> 'accuracy' in report
        True
        >>> '0' in report and '1' in report
        True
        >>> 'macro_avg' in report and 'weighted_avg' in report
        True
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have same length: "
                f"{len(y_true)} != {len(y_pred)}"
            )

        if not y_true:
            raise ValueError("y_true and y_pred cannot be empty")

        # Determine labels
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        # Compute confusion matrix
        cm = ModelMetrics.confusion_matrix(y_true, y_pred, labels=labels)  # type: ignore[assignment]

        # Per-class metrics
        report: dict[str, dict[str, float]] = {}

        for i, label in enumerate(labels):
            tp = cm[i, i]  # type: ignore[index]
            fp = cm[:, i].sum() - tp  # type: ignore[operator, index]
            fn = cm[i, :].sum() - tp  # type: ignore[operator, index]
            support = cm[i, :].sum()  # type: ignore[index]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            report[str(label)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": float(support),
            }

        # Macro average
        macro_metrics = ModelMetrics.precision_recall_f1(
            y_true, y_pred, average="macro", labels=labels
        )
        report["macro_avg"] = macro_metrics

        # Weighted average
        weighted_metrics = ModelMetrics.precision_recall_f1(
            y_true, y_pred, average="weighted", labels=labels
        )
        report["weighted_avg"] = weighted_metrics

        # Overall accuracy
        report["accuracy"] = {"value": ModelMetrics.accuracy(y_true, y_pred)}

        return report
