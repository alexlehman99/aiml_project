import pandas as pd

# Load the combined CSV file
df = pd.read_csv("combined_data.csv")

# Define a dictionary mapping each year to the champion team.
# Update the team names to match those in your dataset.
champion_teams = {
    "2004": "Detroit Pistons",
    "2005": "San Antonio Spurs",
    "2006": "Miami Heat",
    "2007": "San Antonio Spurs",
    "2008": "Boston Celtics",
    "2009": "Los Angeles Lakers",
    "2010": "Los Angeles Lakers",
    "2011": "Dallas Mavericks",
    "2012": "Miami Heat",
    "2013": "Miami Heat",
    "2014": "San Antonio Spurs",
    "2015": "Golden State Warriors",
    "2016": "Cleveland Cavaliers",
    "2017": "Golden State Warriors",
    "2018": "Golden State Warriors",
    "2019": "Toronto Raptors",
    "2020": "Los Angeles Lakers",
    "2021": "Milwaukee Bucks",
    "2022": "Golden State Warriors",
    "2023": "Denver Nuggets",
}

# Create the Champion column: 1 if the team's name matches the champion for that year, 0 otherwise.
df["Champion"] = df.apply(lambda row: 1 if row["Team"].strip() == champion_teams.get(str(row["Year"]), None) else 0, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv("combined_data_with_champions.csv", index=False)
print("Updated CSV saved as combined_data_with_champions.csv")
