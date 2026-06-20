import pandas as pd
import numpy as np

# 1. Load the realistic datasets
df_ops = pd.read_csv("compressor_operational_realistic.csv")
df_fails = pd.read_csv("compressor_failures_realistic.csv")

# 2. Clean up column headers (remove symbols like °, (, ), / and replace spaces with underscores)
df_ops.columns = df_ops.columns.str.replace(r'[()°/]', '', regex=True).str.replace(' ', '_').str.lower()
df_fails.columns = df_fails.columns.str.replace(r'[()°/]', '', regex=True).str.replace(' ', '_').str.lower()

# 3. Perform a left join so we keep all operational hours rows and map matching failure logs
df_merged = pd.merge(df_ops, df_fails, on=['year', 'compressor_id'], how='left')

# 4. Engineer our Target Label: 1 if a failure type is recorded, 0 if safe operation
df_merged['has_failure'] = df_merged['failure_type'].notna().astype(int)

# Fill empty failure types with a standard string for tracking
df_merged['failure_type'] = df_merged['failure_type'].fillna('None')

# 5. Export our high-dimensional master dataset
df_merged.to_csv("compressor_advanced_master.csv", index=False)
print("✅ Master Dataset Formed! Total Rows:", len(df_merged))
print("Available Parameters:\n", df_merged.columns.tolist())