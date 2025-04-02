import time
import pandas as pd
import re

START_YEAR = 2004  # set your desired range
END_YEAR = 2024

# Regex to capture something like:
#   "Eastern Conference Finals Detroit Pistons over Indiana Pacers (4-2) Series Stats"
# Group 1 -> Eastern Conference Finals
# Group 2 -> Detroit Pistons
# Group 3 -> Indiana Pacers
# Group 4 -> 4-2
playoff_pattern = re.compile(
    r"^(.*?)\s+(.+?)\s+over\s+(.+?)\s+\((4-\d)\)(?:.*)$",
    re.IGNORECASE
)

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Processing season ending in {year}...")

    url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"

    # -------------------------------
    # Per 100 Possessions Stats (table with id="per_poss-team")
    # -------------------------------
    try:
        per100_tables = pd.read_html(url, attrs={"id": "per_poss-team"})
    except Exception as e:
        print(f"  Could not load per 100 possessions table for {year}: {e}")
        continue

    if not per100_tables:
        print(f"  No per 100 possessions table found for {year}.")
        continue

    per100_df = per100_tables[0]
    if "Rk" in per100_df.columns:
        per100_df = per100_df[per100_df["Rk"] != "Rk"]

    # -------------------------------
    # Advanced Stats (table with id="advanced-team")
    # -------------------------------
    try:
        advanced_tables = pd.read_html(url, attrs={"id": "advanced-team"})
    except Exception as e:
        print(f"  Could not load advanced stats table for {year}: {e}")
        continue

    if not advanced_tables:
        print(f"  No advanced stats table found for {year}.")
        continue

    advanced_df = advanced_tables[0]
    advanced_df = advanced_df.iloc[1:]  # remove the first (extra header) row
    if "Rk" in advanced_df.columns:
        advanced_df = advanced_df[advanced_df["Rk"] != "Rk"]

    # -------------------------------
    # Playoff Series (table with id="all_playoffs")
    # -------------------------------
    try:
        playoffs_tables = pd.read_html(url, attrs={"id": "all_playoffs"})
    except Exception as e:
        print(f"  Could not load playoffs table for {year}: {e}")
        playoffs_tables = []

    # We'll parse the table if found
    playoffs_data = []
    if playoffs_tables:
        playoffs_df = playoffs_tables[0]
        # Each row might look like: "Eastern Conference Finals Detroit Pistons over Indiana Pacers (4-2) Series Stats"
        # We'll convert row to text and apply our regex
        for _, row in playoffs_df.iterrows():
            # Combine non‐NaN cells into a single string
            row_text = " ".join(str(x) for x in row if pd.notna(x)).strip()

            match = playoff_pattern.search(row_text)
            if match:
                series = match.group(1).strip()
                winner = match.group(2).strip()
                loser  = match.group(3).strip()
                score  = match.group(4).strip()  # e.g. "4-1", "4-3"
                playoffs_data.append([series, winner, loser, score])

    # Convert parsed playoffs into a DataFrame
    playoffs_df_clean = pd.DataFrame(
        playoffs_data,
        columns=["SERIES", "WINNER", "LOSER", "SCORE"]
    )

    # -------------------------------
    # Save CSV files
    # -------------------------------
    per100_filename    = f"data/{year}_per100.csv"
    advanced_filename  = f"data/{year}_advanced.csv"
    playoffs_filename  = f"data/{year}_playoffs.csv"

    per100_df.to_csv(per100_filename, index=False)
    advanced_df.to_csv(advanced_filename, index=False)
    playoffs_df_clean.to_csv(playoffs_filename, index=False)

    print(f"  Saved {per100_filename}, {advanced_filename}, and {playoffs_filename}")

    # Pause between requests
    time.sleep(3)
