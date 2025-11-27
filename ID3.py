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
       
    target_attribute = "wins"

    df = pd.read_csv("cleanedStats.csv")
    df_discrete = discretize_dataframe(df, target_attribute, bins=4)

    df_discrete.to_csv("cleanedStats_discretized.csv", index=False)     #save to file

    print("Discretization complete! Saved cleanedStats_discretized.csv")
    # Prepare attributes (all except target)

    attributes = [col for col in df.columns if col != target_attribute and col != "year" and col != "team"]

    # Build decision tree
    tree = id3(df_discrete, target_attribute, attributes)
   