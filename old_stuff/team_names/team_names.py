import pandas as pd
import sys

output_file = sys.argv[1]

data = pd.read_csv('data/march_data.csv')

team_names = data['Team'].unique()
team_names = sorted(team_names)

for team in data['Team']:
    if team not in team_names:
        team_names.append(team)
        
with open(output_file, "w") as f:
        sys.stdout = f
        
        i=0
        for team in team_names:
            print(team_names[i])
            i+=1