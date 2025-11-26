import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt

def entropy(column):
    counts = Counter(column)
    total = len(column)

    return sum(
        -(count/total) * math.log2(count/total)
        for count in counts.values()
    )

def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()

    weighted_entropy = 0
    for v in values:
        subset = df[df[attribute] == v]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy

def id3(df, target, attributes):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    if len(attributes) == 0:
        return df[target].mode()[0]
    
    gains = {attr: information_gain(df, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}

    for value in df[best_attr].unique():
        subset = df[df[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = df[target].mode()[0]
        else:
            remaining = [a for a in attributes if a != best_attr]
            tree[best_attr][value] = id3(subset, target, remaining)

    return tree

def draw_node(ax, text, x, y, node_width=0.2, node_height=0.1):
    box = plt.Rectangle((x - node_width/2, y - node_height/2), node_width, node_height,
                        fill=True, edgecolor="black", facecolor="#e0f2ff")
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center")

def plot_id3_tree(tree, x=0.5, y=0.9, x_spacing=0.2, y_spacing=0.15, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_axis_off()

    if not isinstance(tree, dict):
        draw_node(ax, str(tree), x, y)
        return ax

    attribute = list(tree.keys())[0]
    draw_node(ax, attribute, x, y)

    branches = tree[attribute]
    n = len(branches)

    start_x = x - (n-1) * x_spacing / 2

    for i, (value, subtree) in enumerate(branches.items()):
        child_x = start_x + i * x_spacing
        child_y = y - y_spacing

        ax.annotate("", xy=(child_x, child_y + 0.05), xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5))

        ax.text((x + child_x)/2, (y + child_y)/2, str(value),
                ha="center", va="center")

        plot_id3_tree(subtree, child_x, child_y, x_spacing*0.7, y_spacing, ax=ax)

    return ax

# def print_tree(tree, indent=""):
#     if not isinstance(tree, dict):
#         print(indent + "→ " + str(tree))
#         return

#     for attribute, branches in tree.items():
#         print(indent + f"[Attribute: {attribute}]")
#         for value, subtree in branches.items():
#             print(indent + f"  Value = {value}:")
#             print_tree(subtree, indent + "    ")

def discretize_dataframe(df, target, bins=4):
    #ID3 does not work with numbers, it needs discrete elements
    df_discrete = df.copy()

    for col in df.columns: 
        if col == target:   # Skip the target column
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df_discrete[col] = pd.cut(
                df[col],
                bins=bins,
                labels=[f"bin_{i}" for i in range(bins)],
                include_lowest=True
            )
        else:
            # Leave categorical columns unchanged
            df_discrete[col] = df[col]

    return df_discrete

if __name__ == "__main__":
       
    target_attribute = "wins"         # <- change to your target column name

    df = pd.read_csv("cleanedStats.csv")
    df_discrete = discretize_dataframe(df, target_attribute, bins=4)
    # Save the discretized dataset
    df_discrete.to_csv("cleanedStats_discretized.csv", index=False)

    print("Discretization complete! Saved cleanedStats_discretized.csv")
    # Prepare attributes (all except target)

    attributes = [col for col in df.columns if col != target_attribute and col != "year" and col != "team"]

    # Build decision tree
    tree = id3(df_discrete, target_attribute, attributes)

    print("\n=== DECISION TREE (ID3) ===\n")
    plot_id3_tree(tree)
    plt.show()