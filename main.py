"""
Run examples:
  python main.py --data "wifi_db/clean_dataset.txt" --prune
  python main.py --data "wifi_db/clean_dataset.txt" --visualize
  python main.py --data "wifi_db/noisy_dataset.txt" 

Outputs a concise text report to stdout.
"""
from __future__ import annotations
import argparse
import math
import sys
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import matplotlib.pyplot as plt
from viz_tree import visualize_tree

# -----------------------------
# Utilities
# -----------------------------

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset (2000x8). Last column is integer room label.
    Returns X (N, D) float, y (N,) int.
    """
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError("Expected 2D array from dataset.")
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)
    return X, y


def entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())


@dataclass
class Leaf:
    prediction: int               # majority class
    class_counts: Dict[int, int]  # for analysis/visualization


@dataclass
class Node:
    attribute: int
    threshold: float
    left: Any   # Node | Leaf
    right: Any  # Node | Leaf


Tree = Any  # Node | Leaf


# -----------------------------
# Split search (continuous features)
# -----------------------------

def best_split(X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float, float]]:
    """Return (best_attr, best_thresh, best_gain) or None if no improving split.
    We scan midpoints between sorted unique values for each feature.
    """
    n, d = X.shape
    base_H = entropy(y)
    best_gain = 0.0
    best_attr = -1
    best_thresh = 0.0

    for j in range(d):
        # Sort examples by feature j
        order = np.argsort(X[:, j], kind='mergesort')  # stable
        xj = X[order, j]
        yj = y[order]
        # Candidate thresholds between distinct consecutive values
        # Maintain running class counts to compute remainder quickly
        vals, counts = np.unique(yj, return_counts=True)
        total_counts = {int(k): int(c) for k, c in zip(vals, counts)}
        left_counts: Dict[int, int] = {int(k): 0 for k in total_counts}
        right_counts: Dict[int, int] = total_counts.copy()

        # Consider split between indices i and i+1 where xj[i] != xj[i+1]
        for i in range(n - 1):
            cls = int(yj[i])
            left_counts[cls] += 1
            right_counts[cls] -= 1
            if xj[i] == xj[i + 1]:
                continue
            left_n = i + 1
            right_n = n - left_n
            # Entropy for sides
            if left_n == 0 or right_n == 0:
                continue
            pL = np.array(list(left_counts.values()), dtype=float)
            pR = np.array(list(right_counts.values()), dtype=float)
            HL = 0.0
            HR = 0.0
            # compute entropy from counts safely
            if left_n:
                p = pL / left_n
                HL = float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())
            if right_n:
                p = pR / right_n
                HR = float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())
            remainder = (left_n / n) * HL + (right_n / n) * HR
            gain = base_H - remainder
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_attr = j
                best_thresh = 0.5 * (xj[i] + xj[i + 1])

    if best_attr == -1:
        return None
    return best_attr, best_thresh, best_gain


# -----------------------------
# Tree building
# -----------------------------

def majority_class(y: np.ndarray) -> Tuple[int, Dict[int, int]]:
    vals, counts = np.unique(y, return_counts=True)
    idx = np.argmax(counts)
    maj = int(vals[idx])
    cc = {int(k): int(c) for k, c in zip(vals, counts)}
    return maj, cc


def build_tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: Optional[int] = None,
               min_samples_split: int = 2) -> Tree:
    # Stopping conditions
    if y.size == 0:
        return Leaf(prediction=0, class_counts={})
    if np.all(y == y[0]):
        maj, cc = majority_class(y)
        return Leaf(prediction=maj, class_counts=cc)
    if max_depth is not None and depth >= max_depth:
        maj, cc = majority_class(y)
        return Leaf(prediction=maj, class_counts=cc)
    if y.size < min_samples_split:
        maj, cc = majority_class(y)
        return Leaf(prediction=maj, class_counts=cc)

    split = best_split(X, y)
    if split is None:
        maj, cc = majority_class(y)
        return Leaf(prediction=maj, class_counts=cc)

    attr, thresh, _ = split
    left_mask = X[:, attr] <= thresh
    right_mask = ~left_mask
    if not left_mask.any() or not right_mask.any():
        maj, cc = majority_class(y)
        return Leaf(prediction=maj, class_counts=cc)

    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split)
    return Node(attribute=attr, threshold=thresh, left=left, right=right)


# -----------------------------
# Prediction & evaluation
# -----------------------------

def predict_one(x: np.ndarray, tree: Tree) -> int:
    node = tree
    while isinstance(node, Node):
        if x[node.attribute] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.prediction


def predict(X: np.ndarray, tree: Tree) -> np.ndarray:
    return np.array([predict_one(x, tree) for x in X], dtype=int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[int]] = None) -> np.ndarray:
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    L = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[int(t)], idx[int(p)]] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    # cm[i, j] i=true, j=pred
    per_class = {}
    L = cm.shape[0]
    total = cm.sum()
    acc = np.trace(cm) / total if total else 0.0
    for k in range(L):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[k] = {"precision": prec, "recall": rec, "f1": f1}
    return {"accuracy": acc, "per_class": per_class}


# -----------------------------
# Pruning (reduced-error on a validation set)
# -----------------------------

def make_leaf_from_train(y_train: np.ndarray) -> Leaf:
    maj, cc = majority_class(y_train)
    return Leaf(prediction=maj, class_counts=cc)


def evaluate_accuracy(X_val: np.ndarray, y_val: np.ndarray, tree: Tree) -> float:
    preds = predict(X_val, tree)
    return float((preds == y_val).mean())


def is_leaf(t: Tree) -> bool:
    return isinstance(t, Leaf)


def prune_once(node: Tree, X_val: np.ndarray, y_val: np.ndarray, y_train_at_node: np.ndarray,
               path_filter: Optional[np.ndarray], X_train: np.ndarray) -> Tuple[Tree, bool]:
    """Attempt to prune any node whose children are leaves.
    Returns (possibly modified subtree, changed_flag).
    path_filter indicates which training rows reached this node (to build replacement leaf).
    """
    if is_leaf(node):
        return node, False

    # Recurse to children: update path filters
    left_mask_train = X_train[path_filter, node.attribute] <= node.threshold
    right_mask_train = ~left_mask_train

    # No training examples? keep as is
    left = node.left
    right = node.right
    changed_any = False

    new_left, chL = prune_once(left, X_val, y_val, y_train_at_node[left_mask_train], path_filter & (X_train[:, node.attribute] <= node.threshold), X_train)
    new_right, chR = prune_once(right, X_val, y_val, y_train_at_node[right_mask_train], path_filter & (X_train[:, node.attribute] > node.threshold), X_train)
    if chL or chR:
        node.left, node.right = new_left, new_right
        changed_any = True

    # If both children are leaves, try replacing with a majority leaf (using training labels that reached node)
    if is_leaf(node.left) and is_leaf(node.right):
        if y_train_at_node.size == 0:
            # No training evidence at this node – don’t attempt a replacement.
            return node, changed_any
        # Accuracy before
        acc_before = evaluate_accuracy(X_val, y_val, node)
        # Replacement leaf from y_train_at_node
        rep_leaf = make_leaf_from_train(y_train_at_node)
        acc_after = evaluate_accuracy(X_val, y_val, rep_leaf)
        if acc_after >= acc_before - 1e-12:
            return rep_leaf, True

    return node, changed_any


def reduced_error_prune(tree: Tree, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Tree:
    if X_val.size == 0:
        return tree
    # Global path filter starts as all True
    path_filter = np.ones(len(y_train), dtype=bool)
    changed = True
    current = tree
    while changed:
        current, changed = prune_once(current, X_val, y_val, y_train, path_filter, X_train)
    return current


# -----------------------------
# Cross-validation utilities (stratified)
# -----------------------------

def stratified_kfold_indices(y: np.ndarray, k: int, seed: int = 0) -> List[np.ndarray]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(y.tolist()):
        by_class.setdefault(label, []).append(idx)
    for lab in by_class:
        rng.shuffle(by_class[lab])
    folds = [list() for _ in range(k)]
    for lab, idxs in by_class.items():
        for i, idx in enumerate(idxs):
            folds[i % k].append(idx)
    return [np.array(sorted(fold), dtype=int) for fold in folds]


# -----------------------------
# Reporting helpers
# -----------------------------

def print_cm(cm: np.ndarray, labels: List[int]):
    header = "      " + " ".join(f"pred={l:>3}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row = "true=" + f"{l:>3} " + " ".join(f"{cm[i, j]:>6}" for j in range(len(labels)))
        print(row)


def tree_depth(t: Tree) -> int:
    if is_leaf(t):
        return 0
    return 1 + max(tree_depth(t.left), tree_depth(t.right))


# -----------------------------
# Visualization (bonus)
# -----------------------------
import os
import textwrap
import matplotlib.pyplot as plt



# -----------------------------
# Experiment pipeline per coursework spec
# -----------------------------

def run_crossval(path: str, seed: int = 0, k: int = 10, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, do_prune: bool = False, val_frac: float = 0.2) -> Dict[str, Any]:
    X, y = load_dataset(path)
    labels = sorted(np.unique(y).tolist())
    folds = stratified_kfold_indices(y, k=k, seed=seed)

    sum_cm_before = np.zeros((len(labels), len(labels)), dtype=int)
    depths_before: List[int] = []

    sum_cm_after = np.zeros_like(sum_cm_before)
    depths_after: List[int] = []

    for fold_i in range(k):
        test_idx = folds[fold_i]
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train base tree
        tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
        depths_before.append(tree_depth(tree))

        # Evaluate before pruning
        y_pred = predict(X_test, tree)
        cm = confusion_matrix(y_test, y_pred, labels)
        sum_cm_before += cm

        # Optionally prune via validation split inside the training set
        if do_prune:
            # simple stratified split of train -> train_sub / val
            folds2 = stratified_kfold_indices(y_train, k=int(1/val_frac), seed=seed + 1337 + fold_i)
            val_idx_local = folds2[0]
            tr_mask_local = np.ones(len(y_train), dtype=bool)
            tr_mask_local[val_idx_local] = False

            X_tr_sub, y_tr_sub = X_train[tr_mask_local], y_train[tr_mask_local]
            X_val, y_val = X_train[~tr_mask_local], y_train[~tr_mask_local]

            pruned = reduced_error_prune(tree, X_tr_sub, y_tr_sub, X_val, y_val)
            depths_after.append(tree_depth(pruned))
            y_pred_pruned = predict(X_test, pruned)
            cm_after = confusion_matrix(y_test, y_pred_pruned, labels)
            sum_cm_after += cm_after

    result = {
        "labels": labels,
        "cm_before": sum_cm_before,
        "depth_mean_before": float(np.mean(depths_before)) if depths_before else None,
    }
    if do_prune:
        result.update({
            "cm_after": sum_cm_after,
            "depth_mean_after": float(np.mean(depths_after)) if depths_after else None,
        })
    return result


def summarize_cm(cm: np.ndarray, labels: List[int], header: str):
    print("\n" + header)
    print_cm(cm, labels)
    m = metrics_from_cm(cm)
    print(f"Accuracy: {m['accuracy']:.4f}")
    print("Per-class metrics:")
    for i, lab in enumerate(labels):
        pc = m["per_class"][i]
        print(f"  class {lab}: precision={pc['precision']:.4f} recall={pc['recall']:.4f} F1={pc['f1']:.4f}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset txt file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--prune', action='store_true', help='Enable reduced-error pruning (nested CV)')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Validation fraction for pruning (inside train)')
    parser.add_argument('--visualize', action='store_true',
                    help='Render a Graphviz visualization of a tree trained on the FULL dataset')
    parser.add_argument('--viz_out', type=str, default='tree_clean',
                    help='Base filename (no extension) for visualization output (default: tree_clean)')
    parser.add_argument('--viz_format', type=str, default='svg', choices=['svg', 'png', 'pdf'],
                    help='Graphviz output format (default: svg)')
    parser.add_argument('--viz_rankdir', type=str, default='TB', choices=['TB', 'LR'],
                    help='Layout direction: TB=top-bottom, LR=left-right (default: TB)')
    args = parser.parse_args()

    # Run base 10-fold cross‑validation
    out = run_crossval(
        path=args.data,
        seed=args.seed,
        k=10,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        do_prune=args.prune,
        val_frac=args.val_frac,
    )

    labels = out['labels']
    summarize_cm(out['cm_before'], labels, header='10-fold results BEFORE pruning (summed confusion matrix)')
    print(f"Average depth BEFORE pruning: {out['depth_mean_before']:.2f}")

    if args.prune:
        summarize_cm(out['cm_after'], labels, header='10-fold results AFTER pruning (summed confusion matrix)')
        print(f"Average depth AFTER pruning: {out['depth_mean_after']:.2f}")

    # Optional visualization on full dataset (bonus requirement: train on entire clean dataset)
    if args.visualize:
        X, y = load_dataset(args.data)
        tree_full = build_tree(X, y)
        feature_names = [f"AP{i}" for i in range(X.shape[1])]
        visualize_tree(
            tree_full,
            feature_names=feature_names,
            title=f"Decision Tree trained on {args.data}",
            data_path=args.data,   # autosave next to dataset folder
            show_fig=False
        )
        


if __name__ == '__main__':
    # Allow importing this file as a module without running
    try:
        main()
    except SystemExit:
        # argparse will call sys.exit; ensure clean exit in notebooks
        pass
