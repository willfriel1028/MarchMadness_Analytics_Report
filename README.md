# March Madness Matchup Analytics Report

## Description

This program provides a full, analytically-driven, head-to-head scouting report for any two teams that have participated in the NCAA Division 1 Men's College Basketball Tournament in a given year. This program uses publicly available data dating back to the 2007-2008 season, and can generate a report for any two teams to qualify for the tournament in the same year since the 2012-13 season. 

It is designed to help users optimize their bracket selections by providing in-depth statistical analysis of head-to-head matchups.

### Matchup Insights Provided

- Strength of Schedule
- Tempo and Pace
- Experience
- Offensive Efficiency vs. Defensive Efficiency
- 2-Point, 3-Point, and Free Throw Analysis
- Turnover Matchups
- Rebounding Matchups

### Advanced Predictive Features

- A matchup-specific win probability algorithm
- Logistic regression models that estimate each team’s likelihood of reaching:
  - Round of 32
  - Sweet 16
  - Elite 8
  - Final Four
  - National Champion
- Poisson regression model that projects each team’s expected number of tournament wins

## Repository

```
march_analytics_report/
|- data/                                          # Folder containing any datasets for this program
|    |- data_official.csv                            # The dataset used for this program
|    |- ...
|
|- samples/                                       # Folder where you can find some sample matchup outputs
|      |- 2025/                                   # Each year's folder contains some matchups from that year
|      |    |- Florida-Houston.txt                # Output for Florida vs. Houston (2025)
|      |- 2024/
|      |- ...                                     # Other years
|
|- team_names/                                    # Folder containing team names specifics
|      |- team_names.py                           # Script to generate list of each team in the dataset alphabetically
|      |- team_names.txt                          # Alphabetical list of all valid team names
|
|- .gitignore                                     # Specifies files and folders to ignore in version control
|
|- README.md                                      # The README for this program
|
|- main.py                                        # The script for this program
|
|- matchup_scores_explained                       # Explains the [-1,1] matchup scores
|
|- model_creation.ipynb                           # Goes through the creating/tuning/testing of predictive models
|
|- model_tuning_walkthrough.md                    # Explains the tuning and testing process for the poisson regression models
|
|- output_walkthrough.md                          # Walks through an example output, explaining each section
|
|- requirements.txt                               # Required Python libraries

```
## Set Up and Installation

Run these lines in your terminal to install this program:
```
git clone https://github.com/willfriel1028/MarchMadness_Analytics_Report.git
```
```
cd marchmadness_analytics_report
```
```
python3 -m venv venv
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```
## Example Command Line Usage
```
python main.py "Florida" "Houston" 2025 FLAvHOU2025.txt
   |      |        |         |       |         |
   |      |        |         |       |         The .txt file you want the output to print to
   |      |        |         |       |
   |      |        |         |       Tournament year
   |      |        |         |
   |      |        |         The second team in the matchup (Entered as a string)
   |      |        |
   |      |        The first team in the matchup (Entered as a string)
   |      |
   |      Calls the main script to be ran
   |
   Indicates running this script in python
```   
**IMPORTANT:** When inputting team names, they have to match the exact same spelling/format that is in the dataset. All valid team name formats are listed in team_names/team_names.txt

## Author
```
William Friel
williamfriel2003@gmail.com
https://www.linkedin.com/in/william-friel/
```