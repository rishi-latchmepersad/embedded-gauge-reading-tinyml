import pandas as pd
import numpy as np

# Load train split
df = pd.read_csv("data/splits/canonical_split_v1_train.csv")

print("Value range:", df["value"].min(), "to", df["value"].max())

# Create bins like the training script does
min_val = df["value"].min()
max_val = df["value"].max()
bin_size = 5.0
bin_edges = np.arange(
    np.floor(min_val / bin_size) * bin_size,
    np.ceil(max_val / bin_size) * bin_size + bin_size,
    bin_size,
)
print("Bin edges:", bin_edges)

df["value_bin"] = pd.cut(df["value"], bins=bin_edges, include_lowest=True)
print("\nBin counts:")
print(df["value_bin"].value_counts())
print("\nAny NaN bins:", df["value_bin"].isna().sum())

# Check for zero counts
bin_counts = df["value_bin"].value_counts()
print("\nAny zero counts?", (bin_counts == 0).any())
print("Total bins:", len(bin_counts))
print("Non-zero bins:", (bin_counts > 0).sum())
