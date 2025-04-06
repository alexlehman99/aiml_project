import pandas as pd
import glob
import os
import re

# 1) Gather all advanced-team files
advanced_files = sorted(glob.glob("data/*_advanced-team.csv"))
print("Found advanced-team files:", advanced_files)

all_data = []

# 2) Loop over each advanced-team file
for adv_file in advanced_files:
    # Extract the year from the filename (e.g., "2004" from "2004_advanced-team.csv")
    match = re.search(r"(\d{4})_advanced-team\.csv", adv_file)
    if not match:
        continue
    year = match.group(1)

    # 3) Build the corresponding per-poss filename
    per_file = f"data/{year}_per_poss-team.csv"

    # Check that the per-poss file exists
    if not os.path.exists(per_file):
        continue

    # 4) Read both CSV files into DataFrames
    adv_df = pd.read_csv(adv_file)
    per_df = pd.read_csv(per_file)

    # 5) Merge on "Team" (inner join keeps only matching teams in both files)
    merged_df = pd.merge(adv_df, per_df, on="Team", how="inner")

    # 6) Add the Year column
    merged_df["Year"] = year

    # Append to list
    all_data.append(merged_df)

# 7) Concatenate all years into a single DataFrame and save to CSV
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("combined_data.csv", index=False)
    print("Successfully created combined_data.csv with all merged years.")
else:
    print("No data was merged. Check your file paths or filenames.")
