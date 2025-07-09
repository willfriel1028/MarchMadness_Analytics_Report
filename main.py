#!/usr/bin/env python3

# PACKAGE IMPORTS
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
import sys

def main():
    '''
    This is the main function that gets called when the script runs.
    Uses the command line to:
        Read in the two teams being analyzed in a matchup scenario
        Read in the year 
        Read in the output file
    Converts raw stats into z-scores
    Assigns and edits data_train and data_test
    Calls team_matchup and prints results onto output file
    '''
    
    data = pd.read_csv("data/march_data.csv") # READ DATASET
    
    # Save teams' offensive/defensive efficiencies into variables (fixed tempo to be ~average - 66.5)
    data['FT_Off_Eff'] = data['FT%'] * data['FTR'] * 66.5 * (1 - data['TOV%'])
    data['FT_Def_Eff'] = 0.715 * data['FTRD'] * 66.5 * (1 - data['TOV%D'])
    data['2pt_Off_Eff'] = data['2PT%'] * data ['2PTR'] * 66.5 * (1 - data['TOV%'])
    data['2pt_Def_Eff'] = data['2PT%D'] * data ['2PTRD'] * 66.5 * (1 - data['TOV%D'])
    data['3pt_Off_Eff'] = data['3PT%'] * data ['3PTR'] * 66.5 * (1 - data['TOV%'])
    data['3pt_Def_Eff'] = data['3PT%D'] * data ['3PTRD'] * 66.5 * (1 - data['TOV%D'])                                                              

    features = ['AdjO', 'AdjD', 'BARTHAG', 'EFG%', 'EFG%D', 'FT_Off_Eff', 'FT_Def_Eff', 'TOV%', 'TOV%D', 'OREB%', 'DREB%', '2pt_Off_Eff', '2pt_Def_Eff', '3pt_Off_Eff', '3pt_Def_Eff', 'AST%', 'Elite_SOS', 'WAB'] # FEATURES TO BE USED IN LOGISTIC AND POISSON REGRESSIONS

    for col in features:
        mean_col = data.groupby('Year')[col].transform('mean')     # CONVERTING FEATURES FROM RAW STATS INTO Z-SCORES
        std_col = data.groupby('Year')[col].transform('std')       # Z-SCORES ALLOWS FOR MORE ACCURATE YEAR-TO-YEAR MODELS
        data[col + '_z'] = (data[col] - mean_col) / std_col  

    team1 = sys.argv[1] # READ IN TEAM1
    team2 = sys.argv[2] # READ IN TEAM2

    year = int(sys.argv[3]) # READ IN YEAR
    data_train = data[data['Year'] < year]        # ASSIGN TRAINING DATA (THE DATA FROM BEFORE THE CHOSEN YEAR)
    data_test0 = data[data['Year'] == year].copy() # ASSIGN TESTINNG DATA (THE PART OF THE DATASET WITH CHOSEN YEAR)

    data_test = get_ranks(data_train, data_test0)  # UPDATES TESTING DATA WITH RESULTS FROM LOGISTIC AND POISSON REGRESSIONS

    output_file = sys.argv[4] # READ IN OUTPUT FILE
    
    data.to_csv('data/data_clean.csv', index=False)
    
    with open(output_file, "w") as f:     # PRINTS OUTPUT ONTO OUTPUT FILE
        sys.stdout = f
    
        team_matchup(team1, team2, data_test)

def get_ranks(data_train, data_test):
    '''
    This function collects the results from calling: 
        logreg_ridge (The logistic regression model)
        get_safe_wins (The poisson regression model for the safer wins predictions)
        get_agg_wins (The poisson regression model for the aggressive wins predictions)
    Stores the results into a variable, and ranks each team in data_test based on the results
    Returns updated data_test
    '''
    
    x_train = data_train[['Won_ConfT', 'Experience',                                 # Training features
                      'AdjO_z', 'AdjD_z', 'BARTHAG_z', 'EFG%_z', 'EFG%D_z',
                      'FT_Off_Eff_z', 'FT_Def_Eff_z', 'TOV%_z', 'TOV%D_z',
                      'OREB%_z', 'DREB%_z', '2pt_Off_Eff_z', '2pt_Def_Eff_z',
                      '3pt_Off_Eff_z', '3pt_Def_Eff_z', 'AST%_z',
                      'Elite_SOS_z', 'WAB_z']]

    x_test = data_test[['Won_ConfT', 'Experience',                                   # Testing features
                    'AdjO_z', 'AdjD_z', 'BARTHAG_z', 'EFG%_z', 'EFG%D_z',
                    'FT_Off_Eff_z', 'FT_Def_Eff_z', 'TOV%_z', 'TOV%D_z',
                    'OREB%_z', 'DREB%_z', '2pt_Off_Eff_z', '2pt_Def_Eff_z',
                    '3pt_Off_Eff_z', '3pt_Def_Eff_z', 'AST%_z',
                    'Elite_SOS_z', 'WAB_z']]

    data_test.loc[:, 'Win_Prob'] = logreg_ridge(x_train, x_test, 'Won_Natty', data_train)                    
                # Stores each team's chance to win Championship into 'Win_Prob'
    data_test.loc[:, 'Natty_Rank'] = data_test['Win_Prob'].rank(ascending=False, method='min').astype(int)   
                # Ranks each team's chance to win Championnship and stores it as 'Natty_Rank'
    
    data_test.loc[:, 'CG_Prob'] = logreg_ridge(x_train, x_test, 'Champ_Game', data_train)                    
                # Stores each team's chance to reach championship game into 'CG_Prob'
    data_test.loc[:, 'CG_Rank'] = data_test['CG_Prob'].rank(ascending=False, method='min').astype(int)       
                # Ranks each team's chance to reach championship game and stores it as 'CG_Rank'
    
    data_test.loc[:, 'F4_Prob'] = logreg_ridge(x_train, x_test, 'Final_4', data_train)                       
                # Stores each team's chance to reach Final 4 into 'F4_Prob'
    data_test.loc[:, 'F4_Rank'] = data_test['F4_Prob'].rank(ascending=False, method='min').astype(int)       
                # Ranks each team's chance to reach Final 4 and stores it as 'F4_Rank'
    
    data_test.loc[:, 'E8_Prob'] = logreg_ridge(x_train, x_test, 'Elite_8', data_train)                       
                # Stores each team's chance to reach Elite 8 into 'E8_Prob'
    data_test.loc[:, 'E8_Rank'] = data_test['E8_Prob'].rank(ascending=False, method='min').astype(int)       
                # Ranks each team's chance to reach Elite 8 and stores it as 'E8_Rank'
    
    data_test.loc[:, 'S16_Prob'] = logreg_ridge(x_train, x_test, 'Sweet_16', data_train)                     
                # Stores each team's chance to reach Sweet 16 into 'S16_Prob'
    data_test.loc[:, 'S16_Rank'] = data_test['S16_Prob'].rank(ascending=False, method='min').astype(int)     
                # Ranks each team's chance to reach Sweet 16 and stores it as 'S16_Rank'
    
    data_test.loc[:, 'R32_Prob'] = logreg_ridge(x_train, x_test, 'Round_32', data_train)                     
                # Stores each team's chance to reach the Round of 32 into 'R32_Prob'
    data_test.loc[:, 'R32_Rank'] = data_test['R32_Prob'].rank(ascending=False, method='min').astype(int)     
                # Ranks each team's chance to reach the Round of 32 and stores it as 'R32_Rank'
    
    data_test.loc[:, 'Proj_Wins_s'] = get_safe_wins(data_train, data_test, data_train['Wins'])
                # Stores each team's Safer Projected wins as 'Proj_Wins_s'
    data_test.loc[:, 'Proj_Wins_a'] = get_agg_wins(data_train, data_test, data_train['Wins'])
                # Stores each team's Aggressive Projected wins as 'Proj_Wins_a'
    
    data_test.loc[:, "Proj_Wins_Safer"] = 0.67 * data_test["Proj_Wins_s"] + 0.33 * data_test["Proj_Wins_a"]
                # Blends the two models together with emphasis on the safer model
    data_test.loc[:, 'Safer_Wins_Rank'] = data_test['Proj_Wins_Safer'].rank(ascending=False, method='min').astype(int)
                # Ranks each team's safer projected wins as 'Safer_Wins_Rank'  
    
    data_test.loc[:, "Proj_Wins_Agg"] = 0.25 * data_test["Proj_Wins_s"] + 0.75 * data_test["Proj_Wins_a"]
                # Blends the two models together with emphasis on the aggressive model
    data_test.loc[:, 'Agg_Wins_Rank'] = data_test['Proj_Wins_Agg'].rank(ascending=False, method='min').astype(int)
                # Ranks each team's aggressive projected wins as 'Safer_Wins_Rank' 
    
    return data_test    # Returns the updated data_test

def logreg_ridge(x_train, x_test, y_train, data_train):
    '''
    This function runs a logistic regression with ridge penalty to calculate the probability that each team in x_test will reach a certain round of the tournament (y) given their statistical profile.
    The function does the following:
        Reads in: 
            x_train (the training data), 
            x_test (the testing data), 
            y_train (a string, indicating which round of tournament is being used), 
            data_train (the dataset being used for training)
        Creates a ridge logistic regression model
        Fits said model on the training data
        Predicts and returns the probability for each team in the testing data based on the training model
    '''
    
    y_train = data_train[y_train] # Saves the training outcome variable
    
    scaler = StandardScaler()                 # Creates a standardization tool that will scale features to have mean = 0 and standard deviation = 1
    x_scaled = scaler.fit_transform(x_train)  # Computes the mean and standard deviation of x_train, then scales all its features accordingly

    ridge = LogisticRegressionCV(         # Creates model, saves it as ridge
        cv=5,                             # 5-fold cross-validation
        penalty='l2',                     # Ridge penalty - shrinks insignificant variables to near-zero
        solver='lbfgs',                   # LBFGS optimization algorithm
        scoring='average_precision',      # "Average precision" scoring method          
        class_weight = 'balanced',        # Automatically adjusts class weights to handle imbalanced classes
        max_iter=10000,                   # Sets the maximum number of iterations to 10,000 to ensure the model has enough time to converge
        Cs = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1, 10, 100] # Assigns multiple C values for the model to loop through and see which performs best
    )
    ridge.fit(x_scaled, y_train)          # Fits the model on the training data
    
    x_test_scaled = scaler.transform(x_test)                   # Uses the same scaling parameters from the training data to standardize the test data
    predicted_probs = ridge.predict_proba(x_test_scaled)[:, 1] # Runs the logistic regression model on the scaled test data and gets the probability that each sample belongs to class 1
    
    return predicted_probs    # Returns the probability that each sample belongs to class 1 for test variables

def get_safe_wins(data_train, data_test, y_train_series):
    '''
    This function runs a poisson regression to safely calculate the number of tournament wins each team in x_test will have based on their statistical profile
    The function does the following:
        Reads in:
            data_train (the training data)
            data_test (the testing data)
            y_train_series (a series, the 'Wins' column in the training data)
        Creates a Poisson regression model
        Fits the model on the training data
        Predicts the value for each team in the testing data
        Scales the outputs to make them as realistic as possible
        Returns scaled outputs
    '''

    x_train = data_train[['WAB_z', 'AdjEM', 'Talent', 'Elite_SOS_z', 'W', 'Won_ConfT', 'OREB%_z', 'Height', '2pt_Off_Eff_z', 'TOV%_z', '2PT%D', 'FT_Def_Eff_z', 'Experience', '3PT%D', '3PT%']]
    x_test = data_test[['WAB_z', 'AdjEM', 'Talent', 'Elite_SOS_z', 'W', 'Won_ConfT', 'OREB%_z', 'Height', '2pt_Off_Eff_z', 'TOV%_z', '2PT%D', 'FT_Def_Eff_z', 'Experience', '3PT%D', '3PT%']]
    
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.38, max_iter=10000)) # Creates poisson regression model as model
    model.fit(x_train, y_train_series) # Fits model to training data
    y_pred = model.predict(x_test)     # Predicts outcome of testing data based on model
    
    scale = 63 / y_pred.sum()         
    
    y_pred_scaled = y_pred * scale    # Scales the output so the sum is 63 (the number of wins in every tournament)
    
    return y_pred_scaled # Returns the predicted wins for each team in test data

def get_agg_wins(data_train, data_test, y_train_series):
    '''
    This function runs a poisson regression to aggressively caluclate the number of tournament wins each team in x_test will have based on their statistical profile
    The function does the following:
        Reads in:
            data_train (the training data)
            data_test (the testing data)
            y_train_series (a series, the 'Wins' column in the training data)
        Creates a Poisson regression model
        Fits the model on the training data
        Predicts the value for each team in the testing data
        Scales the outputs to make them as realistic as possible
        Returns scaled outputs
    '''

    x_train = data_train[['Won_ConfT', 'AdjO',
       'AdjD', 'AdjEM', 'BARTHAG', 'EFG%', 'EFG%D', 'FT%', 'FTR', 'FTRD',
       'TOV%', 'TOV%D', 'TOV%_Diff', 'OREB%', 'DREB%', '2PT%', '2PTR', '2PT%D',
       '2PTRD', '3PT%', '3PTR', '3PT%D', '3PTRD', 'AST%', 'Height',
       'Experience', 'Talent', 'AdjT', 'W', 'Elite_SOS', 'WAB', 'FT_Off_Eff',
       'FT_Def_Eff', '2pt_Off_Eff', '2pt_Def_Eff', '3pt_Off_Eff',
       '3pt_Def_Eff', 'AdjO_z', 'AdjD_z', 'BARTHAG_z', 'EFG%_z', 'EFG%D_z',
       'FT_Off_Eff_z', 'FT_Def_Eff_z', 'TOV%_z', 'TOV%D_z', 'OREB%_z',
       'DREB%_z', '2pt_Off_Eff_z', '2pt_Def_Eff_z', '3pt_Off_Eff_z',
       '3pt_Def_Eff_z', 'AST%_z', 'Elite_SOS_z', 'WAB_z']]
    x_test = data_test[['Won_ConfT', 'AdjO',
       'AdjD', 'AdjEM', 'BARTHAG', 'EFG%', 'EFG%D', 'FT%', 'FTR', 'FTRD',
       'TOV%', 'TOV%D', 'TOV%_Diff', 'OREB%', 'DREB%', '2PT%', '2PTR', '2PT%D',
       '2PTRD', '3PT%', '3PTR', '3PT%D', '3PTRD', 'AST%', 'Height',
       'Experience', 'Talent', 'AdjT', 'W', 'Elite_SOS', 'WAB', 'FT_Off_Eff',
       'FT_Def_Eff', '2pt_Off_Eff', '2pt_Def_Eff', '3pt_Off_Eff',
       '3pt_Def_Eff', 'AdjO_z', 'AdjD_z', 'BARTHAG_z', 'EFG%_z', 'EFG%D_z',
       'FT_Off_Eff_z', 'FT_Def_Eff_z', 'TOV%_z', 'TOV%D_z', 'OREB%_z',
       'DREB%_z', '2pt_Off_Eff_z', '2pt_Def_Eff_z', '3pt_Off_Eff_z',
       '3pt_Def_Eff_z', 'AST%_z', 'Elite_SOS_z', 'WAB_z']]
    
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.001, max_iter=10000)) # Creates poisson regression model as model
    model.fit(x_train, y_train_series) # Fits model to training data
    y_pred = model.predict(x_test)     # Predicts outcome of testing data based on model
    
    scale = 63 / y_pred.sum()         
    
    y_pred_scaled = y_pred * scale    # Scales the output so the sum is 63 (the number of wins in every tournament)
    
    return y_pred_scaled # Returns the predicted wins for each team in test data

def team_matchup(team1, team2, data_test):
    '''
    This function calls all of the other matchup-related functions in this program
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    The function calls:
        sos_matchup - the function that analyzes the strenghth of schedule comparison between team1 and team2
        tempo_matchup - the function that analyzes the tempo comparison between team1 and team2
        off_def_matchup - the function that analyzes the offense/defense comparison between team1 and team2
        twopt_matchup - the function that analyzes the 2pt comparison between team1 and team2
        threept_matchup - the function that analyzes the 3pt comparison between team1 and team2
        ft_matchup - the function that analyzes the free throw comparison between team1 and team2
        to_matchup - the function that analyzes the turnover comparison between team1 and team2
        reb_matchup - the function that analyzes the rebounding comparison between team1 and team2
        get_win_percentages - the function that calculates win percentage for each team
    '''
        
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    seedA = teamA['Seed'].iloc[0] # Saves team1's seed as seedA
    seedB = teamB['Seed'].iloc[0] # Saves team2's seed as seedB
    
    print("    " * 4 + f"{team1}({seedA}) vs. {team2}({seedB})\n") # Prints the two teams and their respective seeds
    
    wab1, wab2 = sos_matchup(team1, team2, data_test)    
        # Calls sos_matchup and saves the returns into variables
    tempo1, tempo2, tempo = tempo_matchup(team1, team2, data_test)   
        # Calls tempo_matchup and saves the returns into variables
    odnet1, odnet2 = off_def_matchup(team1, team2, data_test)   
        # Calls off_def_matchup and saves the returns into variables
    p2net1, p2net2 = twopt_matchup(team1, team2, tempo, data_test)
        # Calls twopt_matchup and saves the returns into variables
    p3net1, p3net2 = threept_matchup(team1, team2, tempo, data_test)
        # Calls threept_matchup and saves the returns into variables
    ftnet1, ftnet2 = ft_matchup(team1, team2, tempo, data_test)
        # Calls ft_matchup and saves the returns into variables
    to1net, to2net = to_matchup(team1, team2, data_test)
        # Calls to_matchup and saves the returns into variables
    rb1net, rb2net = reb_matchup(team1, team2, data_test)
        # Calls reb_matchup and saves the returns into variables
    
    get_win_percentages(team1, team2, wab1, wab2, odnet1, odnet2, p2net1, p2net2, p3net1, p3net2, ftnet1, ftnet2, to1net, to2net, rb1net, rb2net, seedA, seedB, data_test) # Calls get_win_percetages
        
def get_seed_bounds(team, seed, data_test):
    '''
    This function calculates the range each seed falls into if each team were to be ranked by seed
    The function reads:
        team (the team whose seed is evaluated)
        seed (the seed of team)
        data_test (the dataset for the year the user input)
    Loops through each seed getting a total for each (most are 4)
    Finds the lower and upper bound for each seed
        Ex: 1 seeds will always be 1-4, 16 seed will always be 63-68, 11 seeds can be a range, depending on how the bubble played out that year
    Returns a statement stating the teams expected rank based on results
    '''
    
    seed_counts = data_test['Seed'].value_counts().sort_index() # Counts how many times each seed appears in the 'Seed' column, sorts it by seed
    
    lower = 1
    for s in range(1, 17):  # seeds 1 through 16
        count = seed_counts.get(s, 0) # Checks how many of each seed there are. If none, it defaults to 0
        upper = lower + count - 1 # Calculates the upper bound of the ranking for this seed
        
        if seed == s:                                                             # If seed matches the current loop value s:
            return f"{team}({seed}) is expected to rank between {lower}-{upper}"  # Return string which says where the team is expected to rank
        
        lower = upper + 1 # Change the lower bound to the next seed group
        
def get_win_percentages(team1, team2, wab1, wab2, odnet1, odnet2, p2net1, p2net2, p3net1, p3net2, ftnet1, ftnet2, to1net, to2net, rb1net, rb2net, seedA, seedB, data_test):
    '''
    This function calculates the % chance of winning each of 2 teams have in a head-to-head game
    It displays these percentages as well as the ranks calculated by the logistic and poisson regressions
    This function reads in the following for BOTH teams involved:
        team (the team involved)
        wab (the team's wins above bubble)
        odnet (the [-1,1] matchup score for the team's offense vs the opponent's defense)
        p2net (the [-1,1] matchup score for the team's 2pt scoring vs the opponent's 2pt defense)
        p3net (the [-1,1] matchup score for the team's 3pt scoring vs the opponent's 3pt defense)
        ftnet (the [-1,1] matchup score for the team's free throw scoring vs the opponent's free throw defense)
        tonet (the [-1,1] matchup score for the team's turnover offense vs the opponent's turnover defense)
        rbnet (the [-1,1] matchup score for the team's offensive rebounding vs the opponent's defensive rebounding)
        seed (the team's seed)
        data_test (the dataset for the year the user input)
    Saves all of the team's ranks and projected wins into variables
    Normalizes the matchup scores to be [0,1] instead of [-1,1]
    Calculates the best and worst possible outcome for a team
    Calculates win percentage for team1:
        Creates a score for team1 based on my own algorithm which weighs each matchup score differently based on how important they are
        Normalizes this score based on the best and worst possible outcomes
        Scales this score using a sigmoid function to further differentiate the good and bad teams
            This produces team1's win percentage on a [0,1] scale
            This is multiplied by 100 to get the win percentage on a [0,100] scale
            team2's win percentage is simply 100 - team1's percentage
    Displays each team's win percentage, as well as their ranks to reach each round in the tournament
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    count = len(data_test) # Get the number of teams in tournament
    
    # Saving both team's ranks/proj wins into variables
    teamA_nat = teamA['Natty_Rank'].iloc[0]
    teamB_nat = teamB['Natty_Rank'].iloc[0]
    teamA_cg = teamA['CG_Rank'].iloc[0]
    teamB_cg = teamB['CG_Rank'].iloc[0]
    teamA_f4 = teamA['F4_Rank'].iloc[0]
    teamB_f4 = teamB['F4_Rank'].iloc[0]
    teamA_e8 = teamA['E8_Rank'].iloc[0]
    teamB_e8 = teamB['E8_Rank'].iloc[0]
    teamA_s16 = teamA['S16_Rank'].iloc[0]
    teamB_s16 = teamB['S16_Rank'].iloc[0]
    teamA_r32 = teamA['R32_Rank'].iloc[0]
    teamB_r32 = teamB['R32_Rank'].iloc[0]
    teamA_s_pwins = teamA['Proj_Wins_Safer'].iloc[0]
    teamB_s_pwins = teamB['Proj_Wins_Safer'].iloc[0]
    teamA_s_winsrank = teamA['Safer_Wins_Rank'].iloc[0]
    teamB_s_winsrank = teamB['Safer_Wins_Rank'].iloc[0]
    teamA_a_pwins = teamA['Proj_Wins_Agg'].iloc[0]
    teamB_a_pwins = teamB['Proj_Wins_Agg'].iloc[0]
    teamA_a_winsrank = teamA['Agg_Wins_Rank'].iloc[0]
    teamB_a_winsrank = teamB['Agg_Wins_Rank'].iloc[0]
    
    # Convertinng all matchup scores from a [-1,1] scale to a [0,1] scale
    odnet1 = (odnet1+1)/2
    odnet2 = (odnet2+1)/2
    p2net1 = (p2net1+1)/2
    p2net2 = (p2net2+1)/2
    p3net1 = (p3net1+1)/2
    p3net2 = (p3net2+1)/2
    ftnet1 = (ftnet1+1)/2
    ftnet2 = (ftnet2+1)/2
    to1net = (to1net+1)/2
    to2net = (to2net+1)/2
    rb1net = (rb1net+1)/2
    rb2net = (rb2net+1)/2
    
    wab_max = data_test['WAB'].max()
    wab_min = data_test['WAB'].min()
    
    wab_net1 = (wab1 - wab_min) / (wab_max - wab_min)
    wab_net2 = (wab2 - wab_min) / (wab_max - wab_min)
    
    swins_max = data_test["Proj_Wins_Safer"].max()
    swins_min = data_test["Proj_Wins_Safer"].min()
    awins_max = data_test["Proj_Wins_Agg"].max()
    awins_min = data_test["Proj_Wins_Agg"].min()
    
    swins_net1 = (teamA_s_pwins - swins_min) / (swins_max - swins_min)
    awins_net1 = (teamA_a_pwins - awins_min) / (awins_max - awins_min)
    swins_net2 = (teamB_s_pwins - swins_min) / (swins_max - swins_min)
    awins_net2 = (teamB_a_pwins - awins_min) / (awins_max - awins_min)
    
    
    best = 30*1 + 5 + 6*1 + 5*1 + 6*1 + 4 + 10*(1) + 15*1 + 15*1                      # Calculates best possible outcome for a team
    worst = 30*(-1) + (-5) + 6*(-1) + 5*(-1) + 6*(-1) + (-4) + 10*(-1) + 15*(-1) + 15*(-1)  # Calculates worst possible outcome for a team
    
    score = 30*(odnet1 - odnet2) + 5*(p2net1 - p2net2) + 6*(p3net1 - p3net2) + 5*(ftnet1 - ftnet2) + 6*(to1net - to2net) + 4*(rb1net - rb2net) + 10*(wab_net1 - wab_net2) + 15*(swins_net1 - swins_net2) + 15*(awins_net1 - awins_net2)
        # Calculates team1's score according to the formula
    
    normalized_score = (score - worst) / (best - worst)  # Normalizes team1's score to a [0,1] scale
    
    scaled_score = 12 * (normalized_score - 0.5)  # Centers the normalized score around 0        

    percentage1 = 100 / (1 + math.exp(-scaled_score)) # Applies sigmoid function to scaled score and multiplying by 100, giving team1's win percentage
    percentage2 = 100 - percentage1                   # Calculates team2's win percentage
    
    # DISPLAYS OF WIN PERCENTAGE, RANKS, AND PROJ. WINS
    print("\n\n\nWIN PERCENTAGE")
    print(f"{team1 + ':':<12} {percentage1:.1f} {'%'}    {team2 + ':':<12} {percentage2:.1f} {'%'}") # Displays both teams' win percentages
    print()
    print("==============================================================")
    print()
    print("KEEP IN MIND: ", get_seed_bounds(team1, seedA, data_test), '. ', get_seed_bounds(team2, seedB, data_test)) 
        # Calls get_seed_bounds to give context to ranks
    print()
    print("ROUND OF 32 RANK")
    print(f"{team1 + ':':<12} {teamA_r32}/{count}    {team2 + ':':<12} {teamB_r32}/{count}") # Displays both teams' R32 ranks
    print()
    print("SWEET 16 RANK")
    print(f"{team1 + ':':<12} {teamA_s16}/{count}    {team2 + ':':<12} {teamB_s16}/{count}") # Displays both teams' Sweet 16 ranks
    print()
    print("ELITE 8 RANK")
    print(f"{team1 + ':':<12} {teamA_e8}/{count}    {team2 + ':':<12} {teamB_e8}/{count}") # Displays both teams' Elite 8 ranks
    print()
    print("FINAL 4 RANK")
    print(f"{team1 + ':':<12} {teamA_f4}/{count}    {team2 + ':':<12} {teamB_f4}/{count}") # Displays both teams' Final 4 ranks
    print()
    #print("CHAMPIONSHIP GAME RANK")
    #print(f"{team1 + ':':<12} {teamA_cg}/{count}\t{team2 + ':':<12} {teamB_cg}/{count}") # Displays both teams' championship game ranks
    #print()
        # As of right now, championship rank is left out to prevent this section from being too crowded
    print("NATIONAL CHAMPION RANK")
    print(f"{team1 + ':':<12} {teamA_nat}/{count}    {team2 + ':':<12} {teamB_nat}/{count}") # Displays both teams' National Champion ranks
    print()
    print("==============================================================")
    print()
    print("PROJECTED TOURNAMENT WINS (SAFER MODEL)")
    print(f"{team1 + ':':<12} {teamA_s_pwins:.2f} (#{teamA_s_winsrank})    {team2 + ':':<12} {teamB_s_pwins:.2f} (#{teamB_s_winsrank})") 
    print()
    print("PROJECTED TOURNAMENT WINS (MORE AGGRESSIVE MODEL)")
    print(f"{team1 + ':':<12} {teamA_a_pwins:.2f} (#{teamA_a_winsrank})    {team2 + ':':<12} {teamB_a_pwins:.2f} (#{teamB_a_winsrank})")
        # Displays both teams' projected tournament wins
    
def sos_matchup(team1, team2, data_test):
    '''
    This function displays both team's respective strength of schedule analyses
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    Displays and returns each team's Elite SOS and WAB
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    teamA_wab = teamA['WAB'].iloc[0] # Stores team1's WAB
    teamB_wab = teamB['WAB'].iloc[0] # Stores team2's WAB
    teamA_esos = teamA['Elite_SOS'].iloc[0] # Stores team1's Elite SOS
    teamB_esos = teamB['Elite_SOS'].iloc[0] # Stores team2's Elite SOS
    
    # DISPLAYS OF ELITE SOS AND WAB
    print("STRENGTH OF SCHEDULE COMPARISON")
    print()
    print("ELITE SOS")
    print(f"{team1 + ':':<12} {teamA_esos}    {team2 + ':':<12} {teamB_esos}") # Displays each team's Elite SOS
    print()
    print("WINS ABOVE BUBBLE")
    print(f"{team1 + ':':<12} {teamA_wab}    {team2 + ':':<12} {teamB_wab}") # Displays each team's WAB
    
    return teamA_wab, teamB_wab # Returns each team's WAB
    
def tempo_matchup(team1, team2, data_test):
    '''
    This function displays the Tempo matchup between both teams
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    Calculates mean tempo
    Returns each team's tempo and the mean tempo
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    teamA_tempo = teamA['AdjT'].iloc[0] # Stores team1's tempo
    teamB_tempo = teamB['AdjT'].iloc[0] # Stores team2's tempo
    
    # Calculates mean tempo (expected # of possessions in specific matchup)
    tempo = (teamA_tempo + teamB_tempo) / 2
    
    # DISPLAYS EACH TEAMS' TEMPO
    print("\n\n\nTEMPO COMPARISON")
    print()
    print("ADJUSTED TEMPO")
    print(f"{team1 + ':':<12} {teamA_tempo:.2f}    {team2 + ':':<12} {teamB_tempo:.2f}") # Displays each teams' tempo
        
    return teamA_tempo, teamB_tempo, tempo # Returns tempos and mean tempo
    
    
def off_def_matchup(team1, team2, data_test):
    '''
    This function analyzes and displays how both team's offenses fare against the other's defense
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    Calculates best and worst possible matchups
    Calculates how both offenses matchup up against the opposing defense and scales it to be [-1,1]
    Displays and returns each team's offense and defensive efficiency, offensive and defensive efg%, and how they match up on a [-1,1] scale
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    min_OE = data_test['AdjO'].min() # Min Adjusted Offense efficiency
    max_OE = data_test['AdjO'].max() # Max Adjusted Offense efficiency
    min_DE = data_test['AdjD'].min() # Min Adjusted Defense efficiency
    max_DE = data_test['AdjD'].max() # Max Adjusted Defense efficiency
    
    min_matchup = np.log(min_OE * min_DE) # Worst possible matchup for an offense (Worst offense vs best defense)
    max_matchup = np.log(max_OE * max_DE) # Best possible matchup for an offense (Best offense vs worst defense)
    
    # Store each team's statistics into variables
    teamA_OE = teamA['AdjO'].iloc[0]
    teamA_DE = teamA['AdjD'].iloc[0]
    teamA_efgO = teamA['EFG%'].iloc[0]
    teamA_efgD = teamA['EFG%D'].iloc[0]   
    teamB_OE = teamB['AdjO'].iloc[0]
    teamB_DE = teamB['AdjD'].iloc[0]
    teamB_efgO = teamB['EFG%'].iloc[0]
    teamB_efgD = teamB['EFG%D'].iloc[0]
    
    AB = np.log(teamA_OE * teamB_DE) # Calculate log-scaled matchup strength of team1's offense vs. team2's defense
    AB_net = 2 * ((AB - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalizes to [-1,1] scale based on min and max
    BA = np.log(teamB_OE * teamA_DE) # Calculate log-scaled matchup strength of team2's offense vs. team2's defense
    BA_net = 2 * ((BA - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalizes to [-1,1] scale based on min and max
    
    # DISPLAYS EACH TEAMS OFF/DEF EFFICIENCIES, OFF/DEF EFG%, AND MATCHUP SCORES
    print("\n\n\nOFFENSE VS DEFENSE COMPARISON")
    print()
    print("ADJUSTED OFFENSIVE EFFICIENCY")
    print(f"{team1 + ':':<12} {teamA_OE:.2f}    {team2 + ':':<12} {teamB_OE:.2f}") # Displays each team's Adj. Off Efficiency
    print()
    print("ADJUSTED DEFENSIVE EFFICIENCY")
    print(f"{team1 + ':':<12} {teamA_DE:.2f}    {team2 + ':':<12} {teamB_DE:.2f}") # Displays each team's Adj. Def Efficiency
    print()
    print("OFFENSIVE EFFECTIVE FG%")
    print(f"{team1 + ':':<12} {teamA_efgO}    {team2 + ':':<12} {teamB_efgO}") # Displays each team's Off Effective FG%
    print()
    print("DEFENSIVE EFFECTIVE FG%")
    print(f"{team1 + ':':<12} {teamA_efgD}    {team2 + ':':<12} {teamB_efgD}") # Displays each team's Def Effective FG%
    print()
    print("HEAD TO HEAD MATCHUP")
    print("OFFENSE VS DEFENSE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)") # Explains what matchup score represents
    print(f"{team1 + ':':<12} {AB_net:.2f}    {team2 + ':':<12} {BA_net:.2f}") # Displays each team's OFF vs DEF matchup score
    
    return AB_net, BA_net # Returns everything
    
def twopt_matchup(team1, team2, tempo, data_test):
    '''
    This function analyzes and displays how both teams perform in 2-point scoring.
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        tempo (the expected number of possessions in this game)
        data_test (the dataset for the year the user input)
    Calculates each team's offensive and defensive 2pt metrics
    Calculates the best and worst possible 2pt matchups using log-scaled values.
    Calculates how each team's 2pt offense matches up against the other's defense.
    Normalizes the matchup scores to a [-1,1] scale.
    Displays and returns each team's net 2pt scores and the head-to-head matchup results.
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    # Calculate each team's expected off, def, and net 2pt given the tempo
    teamA_o2 = (teamA['2PT%'].iloc[0] * teamA['2PTR'].iloc[0] * tempo * (1 - teamA['TOV%'].iloc[0]))
    teamB_o2 = (teamB['2PT%'].iloc[0] * teamB['2PTR'].iloc[0] * tempo * (1 - teamB['TOV%'].iloc[0]))
    teamA_d2 = (teamA['2PT%D'].iloc[0] * teamA['2PTRD'].iloc[0] * tempo * (1 - teamA['TOV%D'].iloc[0]))
    teamB_d2 = (teamB['2PT%D'].iloc[0] * teamB['2PTRD'].iloc[0] * tempo * (1 - teamB['TOV%D'].iloc[0]))
    teamA_net2 = teamA_o2 - teamA_d2
    teamB_net2 = teamB_o2 - teamB_d2
    
    # Find the min and max values for offense and defense
    min_o2 = (data_test['2PT%'] * data_test['2PTR'] * (1 - data_test['TOV%'])).min() * tempo
    max_o2 = (data_test['2PT%'] * data_test['2PTR'] * (1 - data_test['TOV%'])).max() * tempo
    min_d2 = (data_test['2PT%D'] * data_test['2PTRD'] * (1 - data_test['TOV%D'])).min() * tempo
    max_d2 = (data_test['2PT%D'] * data_test['2PTRD'] * (1 - data_test['TOV%D'])).max() * tempo
    
    # Calculate best and worst possible matchups using log(offense × defense)
    min_matchup = np.log(min_o2 * min_d2) # Worst matchup (worst offense vs best defense)
    max_matchup = np.log(max_o2 * max_d2) # Best matchup (best offense vs worst defense)
    
    # Calculate log-scaled matchup strength of each team's offense vs the other team's defense
    AB_net = np.log(teamA_o2 * teamB_d2) # team1’s offense vs team2’s defense
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA_net = np.log(teamB_o2 * teamA_d2) # team2’s offense vs team1’s defense
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    
    # DISPLAY ALL STATS AND MATCHUPS
    print("\n\n\nTWO POINT COMPARISON")
    print()
    print("NET 2-POINTERS (Shows expected 2pt margin for each team at this game's expected tempo)")
    print(f"{team1 + ':':<12} {teamA_net2:.2f}    {team2 + ':':<12} {teamB_net2:.2f}")                 
    print()
    print("HEAD TO HEAD MATCHUP")
    print("OFFENSE VS DEFENSE 2PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)")
    print(f"{team1 + ':':<12} {AB_scaled:.2f}    {team2 + ':':<12} {BA_scaled:.2f}") 
    
    # RETURN EVERYTHING
    return AB_scaled, BA_scaled
    
def threept_matchup(team1, team2, tempo, data_test):
    '''
    This function analyzes and displays how both teams perform in 3-point scoring.
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        tempo (the expected number of possessions in this game)
        data_test (the dataset for the year the user input)
    Calculates each team's offensive and defensive 3pt metrics
    Calculates the best and worst possible 3pt matchups using log-scaled values.
    Calculates how each team's 3pt offense matches up against the other's defense.
    Normalizes the matchup scores to a [-1,1] scale.
    Displays and returns each team's net 3pt scores and the head-to-head matchup results.
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB
    
    # Calculate each team's expected off, def, and net 3pt given the tempo
    teamA_o3 = (teamA['3PT%'].iloc[0] * teamA['3PTR'].iloc[0] * tempo * (1 - teamA['TOV%'].iloc[0]))
    teamB_o3 = (teamB['3PT%'].iloc[0] * teamB['3PTR'].iloc[0] * tempo * (1 - teamB['TOV%'].iloc[0]))
    teamA_d3 = (teamA['3PT%D'].iloc[0] * teamA['3PTRD'].iloc[0] * tempo * (1 - teamA['TOV%D'].iloc[0]))
    teamB_d3 = (teamB['3PT%D'].iloc[0] * teamB['3PTRD'].iloc[0] * tempo * (1 - teamB['TOV%D'].iloc[0]))
    teamA_net3 = teamA_o3 - teamA_d3
    teamB_net3 = teamB_o3 - teamB_d3
    
    # Find the min and max values for offense and defense
    min_o3 = (data_test['3PT%'] * data_test['3PTR'] * (1 - data_test['TOV%'])).min() * tempo
    max_o3 = (data_test['3PT%'] * data_test['3PTR'] * (1 - data_test['TOV%'])).max() * tempo
    min_d3 = (data_test['3PT%D'] * data_test['3PTRD'] * (1 - data_test['TOV%D'])).min() * tempo
    max_d3 = (data_test['3PT%D'] * data_test['3PTRD'] * (1 - data_test['TOV%D'])).max() * tempo
    
    # Calculate best and worst possible matchups using log(offense × defense)
    min_matchup = np.log(min_o3 * min_d3) # Worst matchup (worst offense vs best defense)
    max_matchup = np.log(max_o3 * max_d3) # Best matchup (best offense vs worst defense)
    
    # Calculate log-scaled matchup strength of each team's offense vs the other team's defense
    AB_net = np.log(teamA_o3 * teamB_d3) # team1’s offense vs team2’s defense
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA_net = np.log(teamB_o3 * teamA_d3) # team2’s offense vs team1’s defense
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    
    # DISPLAY ALL STATS AND MATCHUPS
    print("\n\n\nTHREE POINT COMPARISON")
    print()
    print("NET 3-POINTERS (Shows expected 3pt margin for each team at this game's expected tempo)")
    print(f"{team1 + ':':<12} {teamA_net3:.2f}    {team2 + ':':<12} {teamB_net3:.2f}")                 
    print()
    print("HEAD TO HEAD MATCHUP")
    print("OFFENSE VS DEFENSE 3PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)")
    print(f"{team1 + ':':<12} {AB_scaled:.2f}    {team2 + ':':<12} {BA_scaled:.2f}") 
    
    # RETURN EVERYTHING
    return AB_scaled, BA_scaled

def ft_matchup(team1, team2, tempo, data_test):
    '''
    This function analyzes and displays how both teams perform in free throw scoring.
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        tempo (the expected number of possessions in this game)
        data_test (the dataset for the year the user input)
    Calculates each team's offensive and defensive FT metrics
    Calculates the best and worst possible FT matchups using log-scaled values.
    Calculates how each team's FT offense matches up against the other's defense.
    Normalizes the matchup scores to a [-1,1] scale.
    Displays and returns each team's net FT scores and the head-to-head matchup results.
    '''
    
    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB

    # Calculate each team's expected off, def, and net 3pt given the tempo
    # For defense, it is calculated using the opponent's FT%
    teamA_of = (teamA['FT%'].iloc[0] * teamA['FTR'].iloc[0] * tempo * (1 - teamA['TOV%'].iloc[0]))
    teamB_of = (teamB['FT%'].iloc[0] * teamB['FTR'].iloc[0] * tempo * (1 - teamB['TOV%'].iloc[0]))
    teamA_df = (teamB['FT%'].iloc[0] * teamA['FTRD'].iloc[0] * tempo * (1 - teamA['TOV%D'].iloc[0]))
    teamB_df = (teamA['FT%'].iloc[0] * teamB['FTRD'].iloc[0] * tempo * (1 - teamB['TOV%D'].iloc[0]))
    teamA_netf = teamA_of - teamA_df
    teamB_netf = teamB_of - teamB_df
    
    # Find the min and max values for offense and defense
    min_of = (data_test['FT%'] * data_test['FTR'] * (1 - data_test['TOV%'])).min() * tempo
    max_of = (data_test['FT%'] * data_test['FTR'] * (1 - data_test['TOV%'])).max() * tempo
    min_df = (0.715 * data_test['FTRD'] * (1 - data_test['TOV%D'])).min() * tempo
    max_df = (0.715 * data_test['FTRD'] * (1 - data_test['TOV%D'])).max() * tempo
    
    # Calculate best and worst possible matchups using log(offense × defense)
    min_matchup = np.log(min_of * min_df) # Worst matchup (worst offense vs best defense)
    max_matchup = np.log(max_of * max_df) # Best matchup (best offense vs worst defense)
    
    # Calculate log-scaled matchup strength of each team's offense vs the other team's defense
    AB_net = np.log(teamA_of * teamB_df) # team1’s offense vs team2’s defense
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA_net = np.log(teamB_of * teamA_df) # team2’s offense vs team1’s defense
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    
    # DISPLAY ALL STATS AND MATCHUPS
    print("\n\n\nFREE THROW COMPARISON")
    print()
    print("NET FREE THROWS (Shows expected FT margin for each team at this game's expected tempo)")
    print(f"{team1 + ':':<12} {teamA_netf:.2f}    {team2 + ':':<12} {teamB_netf:.2f}")                 
    print()
    print("HEAD TO HEAD MATCHUP")
    print("OFFENSE VS DEFENSE FT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)")
    print(f"{team1 + ':':<12} {AB_scaled:.2f}    {team2 + ':':<12} {BA_scaled:.2f}") 
    
    # RETURN EVERYTHING
    return AB_scaled, BA_scaled
    
def to_matchup(team1, team2, data_test):
    '''
    This function analyzes and displays how each team's offense handles turnovers and how effective each team's defense is at forcing turnovers.
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    It calculates the best and worst possible turnover matchups using log-scaled values.
    It retrieves each team's offensive and defensive turnover percentages and the difference between the two.
    It compares how each offense matches up against the opposing defense and normalizes the scores to [-1,1].
    The results are displayed in a clearly formatted head-to-head comparison.
    '''

    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB 

    # Get the minimum and maximum values for offensive and defensive turnover percentages in the dataset
    min_tov = data_test['TOV%'].min()   # Lowest offensive turnover rate
    max_tov = data_test['TOV%'].max()   # Highest offensive turnover rate
    min_tovd = data_test['TOV%D'].min() # Lowest defensive turnover rate (least turnovers forced)
    max_tovd = data_test['TOV%D'].max() # Highest defensive turnover rate (most turnovers forced)

    # Calculate worst (high TOV × high TOVD) and best (low TOV × low TOVD) matchups using logs
    worst = np.log(max_tov * max_tovd) # Worst-case turnover matchup (bad offense vs great defense)
    best = np.log(min_tov * min_tovd)  # Best-case turnover matchup (great offense vs weak defense)

    # Retrieve each team’s turnover stats
    teamA_tov = teamA['TOV%'].iloc[0]        # team1's offensive turnover percentage
    teamA_tovd = teamA['TOV%D'].iloc[0]      # team1's defensive turnover percentage
    teamA_diff = teamA['TOV%_Diff'].iloc[0]  # team1's net turnover margin
    teamB_tov = teamB['TOV%'].iloc[0]        # team2's offensive turnover percentage
    teamB_tovd = teamB['TOV%D'].iloc[0]      # team2's defensive turnover percentage
    teamB_diff = teamB['TOV%_Diff'].iloc[0]  # team2's net turnover margin

    # Compute head-to-head log-scaled matchup strength
    AB = np.log(teamA_tov * teamB_tovd) # team1’s offense vs team2’s defense
    AB_net = 2 * ((AB - worst) / (best - worst)) - 1 # Normalize to [-1,1] scale
    BA = np.log(teamB_tov * teamA_tovd) # team2’s offense vs team1’s defense
    BA_net = 2 * ((BA - worst) / (best - worst)) - 1 # Normalize to [-1,1] scale

    # DISPLAY TURNOVER COMPARISON
    print("\n\n\nTURNOVER COMPARISON")
    print()
    print("OFFENSIVE TURNOVER PERCENTAGE") # Each team’s average rate of committing turnovers
    print(f"{team1 + ':':<12} {teamA_tov}    {team2 + ':':<12} {teamB_tov}")
    print()
    print("DEFENSIVE TURNOVER PERCENTAGE") # Each team’s ability to force turnovers
    print(f"{team1 + ':':<12} {teamA_tovd}    {team2 + ':':<12} {teamB_tovd}")
    print()
    print("HEAD TO HEAD MATCHUP")
    print("OFFENSE VS DEFENSE TURNOVER PERCENTAGE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)") # Interpret scale
    print(f"{team1 + ':':<12} {AB_net:.2f}    {team2 + ':':<12} {BA_net:.2f}") # Print normalized results

    # RETURN key turnover metrics and head-to-head scores
    return AB_net, BA_net

def reb_matchup(team1, team2, data_test):
    '''
    This function analyzes and displays how well each team rebounds on both ends of the floor.
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    It calculates the best and worst possible rebounding matchups using log-scaled values.
    It retrieves each team’s offensive and defensive rebounding percentages and effective height.
    It compares how each team's offensive rebounding matches up against the opposing team's defensive rebounding,
    and normalizes the matchup score to a [-1,1] scale.
    The results are then displayed in a clear side-by-side comparison.
    '''

    teamA = data_test[data_test['Team'] == team1] # Saves the data for team1 as teamA
    teamB = data_test[data_test['Team'] == team2] # Saves the data for team2 as teamB

    # Get min and max rebounding stats in the dataset
    min_dreb = data_test['DREB%'].min() # Lowest defensive rebound %
    max_dreb = data_test['DREB%'].max() # Highest defensive rebound %
    min_oreb = data_test['OREB%'].min() # Lowest offensive rebound %
    max_oreb = data_test['OREB%'].max() # Highest offensive rebound %

    # Calculate matchup bounds using logs
    min_matchup = np.log(min_oreb / max_dreb) # Worst matchup: weak offensive rebounding vs strong defensive rebounding
    max_matchup = np.log(max_oreb / min_dreb) # Best matchup: strong offensive rebounding vs weak defensive rebounding

    # Get each team’s rebounding stats and height
    teamA_dreb = teamA['DREB%'].iloc[0]     # team1's defensive rebound %
    teamA_oreb = teamA['OREB%'].iloc[0]     # team1's offensive rebound %
    teamB_dreb = teamB['DREB%'].iloc[0]     # team2's defensive rebound %
    teamB_oreb = teamB['OREB%'].iloc[0]     # team2's offensive rebound %
    teamA_height = teamA['Height'].iloc[0]  # team1's average height
    teamB_height = teamB['Height'].iloc[0]  # team2's average height

    # Calculate head-to-head rebounding matchup scores
    AB = np.log(teamA_oreb / teamB_dreb) # team1's offense vs team2's defense
    AB_net = 2 * ((AB - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA = np.log(teamB_oreb / teamA_dreb) # team2's offense vs team1's defense
    BA_net = 2 * ((BA - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]

    # DISPLAY REBOUNDING COMPARISON
    print("\n\n\nREBOUNDING COMPARISON")
    print()
    print("EFFECTIVE HEIGHT") # Each team's height
    print(f"{team1 + ':':<12} {teamA_height:.2f}    {team2 + ':':<12} {teamB_height:.2f}")
    print()
    print("OFFENSIVE REBOUNDING PERCENTAGE") # Each team’s offensive rebounding %
    print(f"{team1 + ':':<12} {teamA_oreb}    {team2 + ':':<12} {teamB_oreb}")
    print()
    print("DEFENSIVE REBOUND PERCENTAGE") # Each team’s defensive rebounding %
    print(f"{team1 + ':':<12} {teamA_dreb}    {team2 + ':':<12} {teamB_dreb}")
    print()
    print("HEAD TO HEAD MATCHUP")
    print("OFFENSIVE VS DEFENSIVE REBOUNDING (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)") # Explanation of matchup scores
    print(f"{team1 + ':':<12} {AB_net:.2f}    {team2 + ':':<12} {BA_net:.2f}") # Print matchup scores

    # RETURN key rebounding stats and matchup scores
    return AB_net, BA_net
    
# This line runs the main() function only if this file is being run directly by the user
# If this script is imported by another Python file, main() will not execute automatically
if __name__ == "__main__":  
    main()  # Calls the main function to execute the full program
