import time
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC, SVC


def build_classifier(cfg):
    if cfg.clf == "linear_svm":
        return LinearSVC(
            C=cfg.C,
            dual="auto",
            max_iter=cfg.max_iter,
            verbose=int(cfg.verbose),
        )

    if cfg.clf == "rbf_svm":
        return SVC(
            kernel="rbf",
            C=cfg.C,
            gamma="scale",
            verbose=cfg.verbose,
        )

    if cfg.clf == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbose=int(cfg.verbose),
        )

    raise ValueError(
        f"Unknown classifier: {cfg.clf}"
    )


def get_score_matrix(model, X):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        raise TypeError(
            "Classifier has neither decision_function "
            "nor predict_proba."
        )

    if scores.ndim == 1:
        scores = np.column_stack(
            [-scores, scores]
        )

    return scores


def topk_accuracy(
    model,
    y_true,
    scores,
    k,
):
    k = min(k, scores.shape[1])

    topk_columns = np.argsort(
        -scores,
        axis=1,
    )[:, :k]

    topk_labels = model.classes_[
        topk_columns
    ]

    hits = (
        topk_labels
        == y_true[:, None]
    ).any(axis=1)

    return float(hits.mean())


def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
    class_names,
):
    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        train_time = (
            time.time()
            - start_time
        )

    convergence_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    converged = (
        len(convergence_warnings) == 0
    )

    max_iterations_used = None

    if hasattr(model, "n_iter_"):
        max_iterations_used = int(
            np.max(model.n_iter_)
        )

    start_time = time.time()

    scores = get_score_matrix(
        model,
        X_eval,
    )

    prediction_columns = np.argmax(
        scores,
        axis=1,
    )

    y_pred = model.classes_[
        prediction_columns
    ]

    evaluation_time = (
        time.time()
        - start_time
    )

    top1 = accuracy_score(
        y_eval,
        y_pred,
    )

    top5 = topk_accuracy(
        model,
        y_eval,
        scores,
        k=5,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_eval,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    labels = np.arange(
        len(class_names)
    )

    cm = confusion_matrix(
        y_eval,
        y_pred,
        labels=labels,
    )

    return {
        "top1_accuracy": float(top1),
        "overall_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),

        "train_time_sec": float(train_time),
        "evaluation_time_sec": float(
            evaluation_time
        ),

        "converged": converged,
        "max_iterations_used": (
            max_iterations_used
        ),

        "confusion_matrix": cm,
        "scores": scores,
        "y_pred": y_pred,
    }


def most_confused_pairs(
    cm,
    class_names,
    top_n=10,
):
    cm_copy = cm.copy()

    np.fill_diagonal(
        cm_copy,
        0,
    )

    flat_indices = np.argsort(
        -cm_copy,
        axis=None,
    )[:top_n]

    pairs = []
    n_classes = cm.shape[0]

    for flat_index in flat_indices:
        true_index = (
            flat_index // n_classes
        )

        predicted_index = (
            flat_index % n_classes
        )

        count = int(
            cm_copy[
                true_index,
                predicted_index,
            ]
        )

        if count > 0:
            pairs.append(
                (
                    class_names[true_index],
                    class_names[predicted_index],
                    count,
                )
            )

    return pairs


def _short_label(full_name):
    parts = full_name.split("_")

    if len(parts) >= 2:
        return "_".join(
            parts[-2:]
        )

    return full_name


def plot_confusion_matrix(
    cm,
    class_names,
    out_path=None,
    subset_idx=None,
):
    row_sums = cm.sum(
        axis=1,
        keepdims=True,
    )

    row_sums[
        row_sums == 0
    ] = 1

    cm_normalized = (
        cm / row_sums
    )

    normalization = mcolors.PowerNorm(
        gamma=0.4,
        vmin=0,
        vmax=max(
            cm_normalized.max(),
            1e-12,
        ),
    )

    figure, axis = plt.subplots(
        figsize=(9, 9)
    )

    image = axis.imshow(
        cm_normalized,
        cmap="viridis",
        norm=normalization,
    )

    axis.set_title(
        "Confusion Matrix "
        "(All Classes, Row-Normalized)"
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Fraction of true class",
    )

    figure.tight_layout()

    if out_path is not None:
        figure.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(figure)

    if subset_idx is None:
        return

    subset_cm = cm_normalized[
        np.ix_(
            subset_idx,
            subset_idx,
        )
    ]

    subset_names = [
        _short_label(
            class_names[index]
        )
        for index in subset_idx
    ]

    n_names = len(
        subset_names
    )

    size = max(
        8,
        0.45 * n_names,
    )

    figure, axis = plt.subplots(
        figsize=(size, size)
    )

    image = axis.imshow(
        subset_cm,
        cmap="viridis",
        norm=mcolors.PowerNorm(
            gamma=0.4,
            vmin=0,
            vmax=1,
        ),
    )

    axis.set_xticks(
        range(n_names)
    )

    axis.set_yticks(
        range(n_names)
    )

    axis.set_xticklabels(
        subset_names,
        rotation=90,
        fontsize=8,
    )

    axis.set_yticklabels(
        subset_names,
        fontsize=8,
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_title(
        "Most-Confused Species Subset"
    )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Fraction of true class",
    )

    figure.tight_layout()

    if out_path is not None:
        subset_path = str(
            out_path
        ).replace(
            ".png",
            "_subset.png",
        )

        figure.savefig(
            subset_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(figure)