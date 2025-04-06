import pandas as pd
import re
import time

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
    table_ids = [
        "advanced-team",
        "per_poss-team"
    ]

    for year in range(2022, 2024):
        print(f"Processing {year} season...")
        url = base_url.format(year)

        for table_id in table_ids:
            df = fetch_table(url, table_id)
            if df is not None:
                filename = f"data/{year}_{table_id}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")

        time.sleep(3)


if __name__ == "__main__":
    main()
