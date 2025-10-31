from typing import Optional, List, Any
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _leaf_color(label: int) -> str:
    """Map label -> facecolor for leaf nodes."""
    palette = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
    }
    return palette.get(label, "lightgray")

def _is_leaf(node: Any) -> bool:
    return hasattr(node, "prediction") and not (hasattr(node, "left") and hasattr(node, "right"))

def _leaf_count(node: Any) -> int:
    if _is_leaf(node):
        return 1
    return _leaf_count(node.left) + _leaf_count(node.right)

def _node_label(node: Any, feature_names: Optional[List[str]]) -> str:
    # text inside the nodes
    if _is_leaf(node):
        counts = getattr(node, "class_counts", {})
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        return f"{{{counts_str}}}"
    j = getattr(node, "attribute", None)
    thr = getattr(node, "threshold", None)
    name = f"x[{j}]" if (feature_names is None or j is None) else feature_names[j]
    thr_txt = f"{thr:.2f}" if isinstance(thr, (int, float)) else str(thr)
    return f"{name} ≤ {thr_txt}"

def _collect_leaf_labels(node: Any, acc=None):

    if acc is None:
        acc = set()
    if _is_leaf(node):
        pred = getattr(node, "prediction", None)
        if pred is not None:
            acc.add(int(pred))
    else:
        _collect_leaf_labels(node.left, acc)
        _collect_leaf_labels(node.right, acc)
    return acc



def visualize_tree(# This is the function to call from main
    tree: Any,
    feature_names: Optional[List[str]] = None,
    title: str = "Decision Tree",
    data_path: Optional[str] = None, 
    show_fig: bool = False,
    dpi: int = 250,
    x_gap: float = 1.6, # Spacing between nodes         
    y_gap: float = 1.4, 
) -> str:

    
    pos = {}  # id(node) -> (x, y)

    def _assign_positions(node: Any, x_left: float, depth: int):
        # Assign the positions so that there is no overlap. Space using the leaf nodes. 
        n_leaves = _leaf_count(node)
        center_x = x_left + (n_leaves - 1) * x_gap / 2.0
        pos[id(node)] = (center_x, -depth * y_gap)
        if not _is_leaf(node):
            left_leaves = _leaf_count(node.left)
            _assign_positions(node.left, x_left, depth + 1)
            _assign_positions(node.right, x_left + left_leaves * x_gap, depth + 1)

    _assign_positions(tree, 0.0, 0)

    width_leaves = _leaf_count(tree)
    max_depth_levels = 0
    for _, y in pos.values():
        lvl = int(round(-y / y_gap))
        max_depth_levels = max(max_depth_levels, lvl)

    fig_w = max(10.0, width_leaves * 0.75)
    fig_h = max(5.0, (max_depth_levels + 1) * 1.1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Making the draw functions below here -----

    def _draw_edges(node: Any):
        if _is_leaf(node):
            return
        x0, y0 = pos[id(node)]
        xl, yl = pos[id(node.left)]
        xr, yr = pos[id(node.right)]
        ax.plot([x0, xl], [y0, yl], "k-", lw=0.7)
        ax.text((x0 + xl)/2, (y0 + yl)/2 + 0.08, "≤", ha="center", va="bottom", fontsize=8)
        ax.plot([x0, xr], [y0, yr], "k-", lw=0.7)
        ax.text((x0 + xr)/2, (y0 + yr)/2 + 0.08, ">", ha="center", va="bottom", fontsize=8)
        _draw_edges(node.left)
        _draw_edges(node.right)

    _draw_edges(tree)

    def _draw_nodes(node: Any):
        x, y = pos[id(node)]
        lbl = _node_label(node, feature_names)

        if _is_leaf(node):
            face = _leaf_color(int(node.prediction))
            size = 2000  # adjust this to make the size of leaf nodes bigger (radius)
            ax.scatter(
                [x], [y],
                s=size, marker='o',
                c=[face], edgecolors='black', linewidths=0.9,
                zorder=2
            )
            ax.text(
                x, y, _node_label(node, feature_names),
                ha="center", va="center", fontsize=9, weight="bold", # was 9
                zorder=3
            )
        else:
            ax.text(
                x, y, _node_label(node, feature_names),
                ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.35", ec="black", lw=0.8, fc="white"),
                zorder=2
            )

        if not _is_leaf(node):
            _draw_nodes(node.left)
            _draw_nodes(node.right)

    _draw_nodes(tree)

    # LEGENDS:)))))
    leaf_labels = sorted(_collect_leaf_labels(tree))
    if leaf_labels:
        legend_handles = [
            mpatches.Patch(facecolor=_leaf_color(lbl), edgecolor="black", label=f"Room {lbl}")
            for lbl in leaf_labels
        ]

        # Place legend in the bottom-left corner with bigger text
        ax.legend(
            handles=legend_handles,
            title="Leaf label colors",
            frameon=True,
            loc="lower left",            # ⬅️ bottom-left corner
            bbox_to_anchor=(0.02, 0.02), # ⬅️ slight margin from edges
            borderaxespad=0.5,
            fontsize=25,                 # ⬅️ larger legend text
            title_fontsize=25            # ⬅️ larger title
        )

    plt.tight_layout()

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    if xs and ys:
        ax.set_xlim(min(xs) - x_gap, max(xs) + x_gap)
        ax.set_ylim(min(ys) - y_gap * 0.8, max(ys) + y_gap * 0.8)

    if data_path:
        out_dir = os.path.dirname(data_path) or "."
        base = os.path.splitext(os.path.basename(data_path))[0] 
        out_path = os.path.join(out_dir, f"tree_{base}.png")
    else:
        out_path = "tree_visualization.png" # change output paths for new trees. 

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if show_fig:
        plt.show()

    print(f"[viz] Saved visualization to: {out_path}")
    return out_path
