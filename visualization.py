from typing import Any, Dict, List, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
General idea is find number of leaf nodes and height of tree.
Space the leaf nodes evenly out in x-direction.
Place the split tree deision labels of internal nodes in the middle of its children. 
Draw edges.
"""

# Design plot look. change vars to get different look. 
DESIGN = {
    # Layout
    "x_spacing": 1,
    "y_spacing": 1.9,
    "equal_aspect": True,

    # Leaves (class circles)
    "leaf_radius": 0.45,
    "leaf_fontsize": 12,
    "leaf_zorder": 4, # without this the lines are above the leafs.

    # Internal nodes (rounded text boxes)
    "internal_fontsize": 12,
    "internal_box_lw": 1.4,
    "internal_zorder": 5,

    # Edges and labels
    "edge_linewidth": 1.2,
    "edge_zorder": 1,
    "edge_label_true": "T",     # left branch (≤ threshold)
    "edge_label_false": "F",    # right branch (> threshold)
    "edge_fontsize": 10,
    "edge_label_bbox": dict(boxstyle="round,pad=0.40", fc="white", ec="black", lw=0.5),
    "edge_label_zorder": 3,

    # Title (unused right now)
    "title_fontsize": 14,

    # Legend
    "legend_fontsize": 30,
    "legend_loc": "lower left",
    "legend_bbox_to_anchor": (0.05, 0.05),

    # Can use "Label" here instead of "room" so that it is for general problems. 
    "leaf_label_prefix": "Room", 
}
# Object type alias to make it easier to handle / iterate through tree nodes
TreeLike = Union[Tuple[int, float, "TreeLike", "TreeLike"], float, int, Tuple[Any, Any]]

def _unwrap_tree(tree: TreeLike) -> TreeLike:
    # removes the depth from (tree, depth).
    # sometimes just need the tree 
    if isinstance(tree, tuple) and len(tree) == 2 and isinstance(tree[1], (int, float)):
        return tree[0]
    return tree



def _is_leaf(node: TreeLike) -> bool:
    node = _unwrap_tree(node)
    return not (isinstance(node, tuple) and len(node) == 4)

# map from label to color for the leafs
def _leaf_color(label: int) -> str:
    palette = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    return palette[label]


# Need leaf count to calculate the width of the figure as this determines the spacing. 
def _leaf_count(node: TreeLike) -> int:
    node = _unwrap_tree(node)
    if _is_leaf(node):
        return 1
    _, _, left, right = node
    return _leaf_count(left) + _leaf_count(right)

# Need max depth for computation of height of figure. 
def _max_depth(node: TreeLike) -> int:
    node = _unwrap_tree(node)
    if _is_leaf(node):
        return 0
    _, _, left, right = node
    return 1 + max(_max_depth(left), _max_depth(right))

def _inorder_positions(
    node: TreeLike,
    x0: float = 0.0,
    y0: float = 0.0,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0
) -> Dict[int, Tuple[float, float]]:
    # Walks through the tree in-order and assigns (x, y) 
    # positions to each node so branches don’t overlap.
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
        next_x = visit(left, depth + 1, next_x)

        x_left = positions[id(_unwrap_tree(left))][0]
        next_x = visit(right, depth + 1, next_x)
        x_right = positions[id(_unwrap_tree(right))][0]

        x =  (x_left + x_right) / 2.0
        y =  y0 - depth * y_spacing
        positions[id(n)] = (x, y)
        return next_x

    visit(node, depth=0, next_x=x0)
    return positions

def _draw_node(ax, node: TreeLike, positions: Dict[int, Tuple[float, float]], feature_names: Optional[List[str]]):
    node = _unwrap_tree(node)
    x, y = positions[id(node)]

    if _is_leaf(node):
        label = node
        face = _leaf_color(label)
        circ = plt.Circle(
            (x, y),
            DESIGN["leaf_radius"],
            facecolor=face,
            edgecolor="black",
            linewidth=1.4,
            zorder=DESIGN["leaf_zorder"],
        )
        ax.add_patch(circ)
        ax.text(
            x, y, f"{int(label)}",
            ha="center", va="center",
            fontsize=DESIGN["leaf_fontsize"],
            color="white" if face != "lightgray" else "black",
            zorder=DESIGN["leaf_zorder"] + 0.1,
        )
        return

    feat, thr, left, right = node
    # Node label: feature name if we have it, otherwise f{idx}
    name = feature_names[feat] if feature_names and 0 <= feat < len(feature_names) else f"f{feat}"
    text = f"{name} ≤ {thr:.3f}"

    ax.text(
        x, y, text,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=DESIGN["internal_box_lw"]),
        fontsize=DESIGN["internal_fontsize"],
        zorder=DESIGN["internal_zorder"],
    )

    # Draw edges and true false tags.
    for child, side in ((left, DESIGN["edge_label_true"]), (right, DESIGN["edge_label_false"])):
        c = _unwrap_tree(child)
        cx, cy = positions[id(c)]
        ax.plot(
            [x, cx], [y - 0.12, cy + 0.12],
            color="black", linewidth=DESIGN["edge_linewidth"], zorder=DESIGN["edge_zorder"]
        )
        ax.text(
            (x + cx) / 2.0, (y + cy) / 2.0, side,
            fontsize=DESIGN["edge_fontsize"], ha="center", va="center",
            bbox=DESIGN["edge_label_bbox"],
            zorder=DESIGN["edge_label_zorder"],
        )

# generates a list that can be iterated over to be plotted. 
def _gather_nodes(node: TreeLike) -> List[TreeLike]:
    node = _unwrap_tree(node)
    if _is_leaf(node):
        return [node]
    _, _, left, right = node
    return [node] + _gather_nodes(left) + _gather_nodes(right)

def _max_x_range(positions: Dict[int, Tuple[float, float]], nodes: List[TreeLike]) -> Tuple[float, float]:
    xs = [positions[id(_unwrap_tree(n))][0] for n in nodes]
    return min(xs) - 1.2, max(xs) + 1.2


# -------
# main function that is imported from the train file.
def visualize_tree(
    tree: TreeLike,
    feature_names: Optional[List[str]] = None,
    title: Optional[str] = None, # currently not using
    data_path: Optional[str] = None,
    dpi: int = 150, # makes plot zoomable in the report. 
    show_fig: bool = True,   # neither with this one
    pruned: bool = False
) -> str:
    """
    Visualize a tuple-based decision tree (same shape as intro_to_ml_1.py).
    Saves a PNG and returns its path.
    """
    tree = _unwrap_tree(tree)

    # Layout
    x_spacing = DESIGN["x_spacing"]
    y_spacing = DESIGN["y_spacing"]

    positions = _inorder_positions(tree, x0=0.0, y0=0.0, x_spacing=x_spacing, y_spacing=y_spacing)
    nodes = _gather_nodes(tree)

    # Canvas sizing
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
        ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Draw children first so parents sit above
    for n in nodes[::-1]:
        _draw_node(ax, n, positions, feature_names)

    # Legend for classes 1..4
    prefix = DESIGN["leaf_label_prefix"]
    patches = [mpatches.Patch(color=_leaf_color(c), label=f"{prefix} {c}") for c in [1, 2, 3, 4]]
    ax.legend(
        handles=patches,
        loc=DESIGN["legend_loc"],
        bbox_to_anchor=DESIGN["legend_bbox_to_anchor"],
        frameon=True,
        fontsize=DESIGN["legend_fontsize"],
    )

    # Uncomment this to get title
    # if title:
    #     ax.set_title(title, fontsize=DESIGN["title_fontsize"], pad=8)

    # Output path
    img_dir = "img"
    os.makedirs(img_dir, exist_ok=True)

    # Infer noisy/clean from filename; default "noisy"
    if data_path:
        fname = os.path.basename(data_path).lower()
        noisy_or_clean = "noisy" if "noisy" in fname else "clean"
    else:
        noisy_or_clean = "noisy"

    pruned_or_no = "prune" if pruned else "noprune"
    out_path = os.path.join(img_dir, f"tree_{pruned_or_no}_{noisy_or_clean}.png")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization to: {out_path}")
    return out_path # dont really need a  return value.
