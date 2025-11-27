import pandas as pd
output_file = "cleanedStats.csv"

# Columns you want to remove
columns_to_drop = ["mov", "g", "ties", "points", "points_opp"]

# Load CSV
df = pd.read_csv("originalStats.csv")

# Drop selected columns
df_cleaned = df.drop(columns=columns_to_drop, errors="ignore")

# Save cleaned CSV
df_cleaned.to_csv(output_file, index=False)

print("Cleaned CSV saved as:", output_file)