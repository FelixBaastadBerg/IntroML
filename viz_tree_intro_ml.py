
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================
# Design Parameters (tweak me)
# ==========================
DESIGN = {
    # Geometry
    "x_spacing": 1,           # horizontal spacing between leaf positions
    "y_spacing": 1.9,           # vertical spacing between levels
    "equal_aspect": True,       # keep circles perfectly circular in output

    # Leaf nodes (circles)
    "leaf_radius": 0.45,        # circle radius (in data units)
    "leaf_fontsize": 12,        # number inside the circle
    "leaf_zorder": 4,           # draw leaves above edges

    # Internal decision nodes (rounded bbox text)
    "internal_fontsize": 12,    # text inside internal nodes
    "internal_box_lw": 1.4,     # border line width
    "internal_zorder": 5,       # draw internal node text above everything

    # Edge (lines and labels)
    "edge_linewidth": 1.2,
    "edge_zorder": 1,           # draw edges behind nodes
    "edge_label_true": "T",     # label for left/True branch (≤ threshold)
    "edge_label_false": "F",    # label for right/False branch (> threshold)
    "edge_fontsize": 10,
    # White background behind the edge label so it sits *in front* of the line
    "edge_label_bbox": dict(boxstyle="round,pad=0.40", fc="white", ec="black", lw=0.5),
    "edge_label_zorder": 3,     # above edges, below leaves/internal nodes

    # Title
    "title_fontsize": 14,

    # Legend
    "legend_fontsize": 30,
    # Put legend down-right. "loc" anchors the legend box *relative to* bbox_to_anchor.
    # For "further down to the right", push bbox_to_anchor to (1.10, -0.02) with loc="lower right".
    "legend_loc": "lower left",
    "legend_bbox_to_anchor": (0.05, 0.05),

    # Labels
    "leaf_label_prefix": "Room",  # prefix for legend entries
}

TreeLike = Union[Tuple[int, float, "TreeLike", "TreeLike"], float, int, Tuple[Any, Any]]

def _unwrap_tree(tree: TreeLike) -> TreeLike:
    """
    intro_to_ml_1 stores trees either as:
      - leaf: a numeric class label (int/float)
      - internal node: (feature_index:int, threshold:float, left:TreeLike, right:TreeLike)
    and sometimes as a pair: (tree, depth).
    This helper strips the optional (tree, depth) wrapper.
    """
    if isinstance(tree, tuple) and len(tree) == 2 and isinstance(tree[1], (int, float)):
        return tree[0]
    return tree

def _is_leaf(node: TreeLike) -> bool:
    node = _unwrap_tree(node)
    return not (isinstance(node, tuple) and len(node) == 4)

def _leaf_color(label: int) -> str:
    """Map label -> facecolor for leaf nodes. Defaults to lightgray for unknown classes."""
    palette = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
    }
    try:
        return palette.get(int(label), "lightgray")
    except Exception:
        return "lightgray"

def _leaf_count(node: TreeLike) -> int:
    node = _unwrap_tree(node)
    if _is_leaf(node):
        return 1
    _, _, left, right = node
    return _leaf_count(left) + _leaf_count(right)

def _max_depth(node: TreeLike) -> int:
    node = _unwrap_tree(node)
    if _is_leaf(node):
        return 0
    _, _, left, right = node
    return 1 + max(_max_depth(left), _max_depth(right))

def _inorder_positions(node: TreeLike, x0: float=0.0, y0: float=0.0, x_spacing: float=1.0, y_spacing: float=1.0):
    """
    Assign (x,y) positions to each node using in-order traversal so that subtrees don't overlap.
    Returns: dict mapping id(node_ref) -> (x, y).
    """
    node = _unwrap_tree(node)
    positions: Dict[int, Tuple[float, float]] = {}

    def visit(n: TreeLike, depth: int, next_x: float) -> float:
        n = _unwrap_tree(n)
        if _is_leaf(n):
            x = next_x
            y = y0 - depth * y_spacing
            positions[id(n)] = (x, y)
            return next_x + x_spacing
        feat, thr, left, right = n
        next_x = visit(left, depth+1, next_x)  # lay out left subtree
        # place current node at midpoint between left & right subtree spans
        x_left = positions[id(_unwrap_tree(left))][0]
        next_x = visit(right, depth+1, next_x)  # lay out right subtree
        x_right = positions[id(_unwrap_tree(right))][0]
        x = (x_left + x_right) / 2.0
        y = y0 - depth * y_spacing
        positions[id(n)] = (x, y)
        return next_x

    visit(node, depth=0, next_x=x0)
    return positions

def _draw_node(ax, node: TreeLike, positions: Dict[int, Tuple[float,float]], feature_names: Optional[List[str]]):
    node = _unwrap_tree(node)
    x, y = positions[id(node)]
    if _is_leaf(node):
        label = node
        face = _leaf_color(label)
        circ = plt.Circle((x, y), DESIGN["leaf_radius"], facecolor=face, edgecolor="black",
                          linewidth=1.4, zorder=DESIGN["leaf_zorder"])
        ax.add_patch(circ)
        ax.text(x, y, f"{int(label)}", ha="center", va="center",
                fontsize=DESIGN["leaf_fontsize"],
                color="white" if face!="lightgray" else "black",
                zorder=DESIGN["leaf_zorder"]+0.1)
    else:
        feat, thr, left, right = node
        # Node box with feature name and threshold
        txt_feat = feature_names[feat] if feature_names and 0 <= feat < len(feature_names) else f"f{feat}"
        text = f"{txt_feat} ≤ {thr:.3f}"
        bbox = dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=DESIGN["internal_box_lw"])
        ax.text(x, y, text, ha="center", va="center", bbox=bbox,
                fontsize=DESIGN["internal_fontsize"], zorder=DESIGN["internal_zorder"])

        # Draw edges + labels
        for child, side in ((left, DESIGN["edge_label_true"]), (right, DESIGN["edge_label_false"])):
            c = _unwrap_tree(child)
            cx, cy = positions[id(c)]
            ax.plot([x, cx], [y-0.12, cy+0.12], color="black",
                    linewidth=DESIGN["edge_linewidth"], zorder=DESIGN["edge_zorder"])
            # small side annotation with white background on top of the line
            ax.text((x+cx)/2.0, (y+cy)/2.0, side,
                    fontsize=DESIGN["edge_fontsize"],
                    ha="center", va="center",
                    bbox=DESIGN["edge_label_bbox"],
                    zorder=DESIGN["edge_label_zorder"])

def _gather_nodes(node: TreeLike) -> List[TreeLike]:
    node = _unwrap_tree(node)
    if _is_leaf(node):
        return [node]
    _, _, left, right = node
    return [node] + _gather_nodes(left) + _gather_nodes(right)

def _max_x_range(positions: Dict[int, Tuple[float, float]], nodes: List[TreeLike]) -> Tuple[float, float]:
    xs = [positions[id(_unwrap_tree(n))][0] for n in nodes]
    return min(xs) - 1.2, max(xs) + 1.2

def visualize_tree(
    tree: TreeLike,
    feature_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    data_path: Optional[str] = None,
    dpi: int = 150,
    show_fig: bool = True,
) -> str:
    """
    Visualize a decision tree produced by intro_to_ml_1.py (tuple-based).
    Compatible signature with viz_tree.visualize_tree.
    Saves a PNG and returns the file path.
    """
    tree = _unwrap_tree(tree)

    # Layout parameters
    x_spacing = DESIGN["x_spacing"]
    y_spacing = DESIGN["y_spacing"]

    positions = _inorder_positions(tree, x0=0.0, y0=0.0, x_spacing=x_spacing, y_spacing=y_spacing)
    nodes = _gather_nodes(tree)

    # Figure limits
    min_x, max_x = _max_x_range(positions, nodes)
    depth = _max_depth(tree)
    y_top = DESIGN["leaf_radius"] + 0.6
    y_bottom = - (depth + 1) * y_spacing - DESIGN["leaf_radius"]

    fig_w = max(6.5, (max_x - min_x) * 0.9 + 3.0)
    fig_h = max(4.5, (depth + 1) * 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(y_bottom, y_top)
    if DESIGN["equal_aspect"]:
        ax.set_aspect('equal', adjustable='box')
    ax.axis("off")

    # Draw nodes (edges drawn within node draw)
    for n in nodes[::-1]:  # children first
        _draw_node(ax, n, positions, feature_names)

    # Legend for classes 1..4
    prefix = DESIGN["leaf_label_prefix"]
    patches = [mpatches.Patch(color=_leaf_color(c), label=f"{prefix} {c}") for c in [1,2,3,4]]
    ax.legend(handles=patches,
              loc=DESIGN["legend_loc"],
              bbox_to_anchor=DESIGN["legend_bbox_to_anchor"],
              frameon=True,
              fontsize=DESIGN["legend_fontsize"])

    #if title:
    #    ax.set_title(title, fontsize=DESIGN["title_fontsize"], pad=8)

    # Decide output path
    if data_path:
        out_dir = os.path.dirname(data_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(data_path))[0]
        out_path = os.path.join(out_dir, f"tree_{base}_intro_ml.png")
    else:
        out_path = "tree_noisy_pruned.png"

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz_intro_ml] Saved visualization to: {out_path}")
    return out_path
