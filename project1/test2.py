import pandas as pd
import re
import time
import os

# DO NOT SEND MORE THAN 20 REQUESTS IN A MINUTE

def clean_team_name(name):
    return re.sub(r'[^\w\s]', '', name).strip()

def fetch_table(url, table_id):
    # advanced-team has an extra header
    header = 1 if table_id == 'advanced-team' else 0
    df = pd.read_html(url, header=header, attrs={'id': table_id})[0]

    # Remove completely empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Remove unneeded columns
    for col in ["Rk", "Arena", "Attend.", "Attend./G"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Clean team names
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).apply(clean_team_name)

    # Drop last row for advanced-team
    if table_id == 'advanced-team':
        df = df[:-1].reset_index(drop=True)

        # Rename offensive and defensive duplicate column names
        rename_map = {
            "eFG%": "Off_eFG%",
            "TOV%": "Off_TOV%",
            "FT/FGA": "Off_FT/FGA",
            "eFG%.1": "Def_eFG%",
            "TOV%.1": "Def_TOV%",
            "FT/FGA.1": "Def_FT/FGA"
        }
        df.rename(columns=rename_map, inplace=True)

    return df


def main():
    base_url = "https://www.basketball-reference.com/leagues/NBA_{}.html"
    os.makedirs("data", exist_ok=True)

    for year in range(2025, 2026):
        print(f"📦 Processing {year} season...")
        url = base_url.format(year)

        # Fetch both tables
        df_advanced = fetch_table(url, "advanced-team")
        df_perposs = fetch_table(url, "per_poss-team")

        if df_advanced is not None and df_perposs is not None:
            # Merge on Team
            merged = pd.merge(df_advanced, df_perposs, on="Team", suffixes=("_adv", "_poss"))

            # Add Year column
            merged["Year"] = year

            # Save single merged file
            merged.to_csv(f"{year}_merged.csv", index=False)
            print(f"✅ Saved merged data to data/{year}_merged.csv")
        else:
            print(f"⚠️ Skipping {year} due to missing data.")

        time.sleep(3)  # Respect rate limit


if __name__ == "__main__":
    main()
