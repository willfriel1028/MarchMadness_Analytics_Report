#!/usr/bin/env python3

# PACKAGE IMPORTS
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
import sys
import streamlit as st

def get_ranks(data_train, data_test):
    '''
    This function collects the results from calling: 
        logreg_ridge (The logistic regression model)
        get_safe_wins (The poisson regression model for the safer wins predictions)
        get_agg_wins (The poisson regression model for the aggressive wins predictions)
    Stores the results into a variable, and ranks each team in data_test based on the results
    Returns updated data_test
    '''
    
    # BEST FEATURES FOR EACH ROUND
    r32 = ['WAB', 'KADJ D', 'KADJ O', 'EFF HGT', 'OREB%', '2PT%D', 'TOV%', '2PT%', 'EXP', 'FTRD', '3PT%D', 'AST%', '3PT%']
    s16 = ['WAB', 'KADJ O', 'KADJ D', 'EFF HGT', 'OREB%', '2PT%', '2PT%D', 'TOV%', 'FTRD', 'EXP', '3PT%D', '3PT%']
    e8 = ['WAB', 'KADJ O', 'KADJ D', 'OREB%', 'EFF HGT', '2PT%', '2PT%D', 'TOV%', 'EXP', 'FTRD', '3PT%D']
    f4 = ['WAB', 'KADJ O', 'KADJ D', 'OREB%', 'EFG%D', 'EFG%', 'EFF HGT', 'TOV%', 'FTRD']    
    natty = ['WAB', 'KADJ O', 'KADJ D', 'OREB%', 'EFG%D', 'EFG%']

    data_test.loc[:, 'Win_Prob'] = logreg_ridge(data_train[natty], data_test[natty], 'CHAMP', data_train)                    
                # Stores each team's chance to win Championship into 'Win_Prob'
    data_test.loc[:, 'Natty_Rank'] = data_test['Win_Prob'].rank(ascending=False, method='min').astype(int)   
                # Ranks each team's chance to win Championship and stores it as 'Natty_Rank'
    
    #data_test.loc[:, 'CG_Prob'] = logreg_ridge(x_train, x_test, 'Champ_Game', data_train)                    
                # Stores each team's chance to reach championship game into 'CG_Prob'
    #data_test.loc[:, 'CG_Rank'] = data_test['CG_Prob'].rank(ascending=False, method='min').astype(int)       
                # Ranks each team's chance to reach championship game and stores it as 'CG_Rank'
    
    data_test.loc[:, 'F4_Prob'] = logreg_ridge(data_train[f4], data_test[f4], 'F4', data_train)                       
                # Stores each team's chance to reach Final 4 into 'F4_Prob'
    data_test.loc[:, 'F4_Rank'] = data_test['F4_Prob'].rank(ascending=False, method='min').astype(int)       
                # Ranks each team's chance to reach Final 4 and stores it as 'F4_Rank'
    
    data_test.loc[:, 'E8_Prob'] = logreg_ridge(data_train[e8], data_test[e8], 'E8', data_train)                       
                # Stores each team's chance to reach Elite 8 into 'E8_Prob'
    data_test.loc[:, 'E8_Rank'] = data_test['E8_Prob'].rank(ascending=False, method='min').astype(int)       
                # Ranks each team's chance to reach Elite 8 and stores it as 'E8_Rank'
    
    data_test.loc[:, 'S16_Prob'] = logreg_ridge(data_train[s16], data_test[s16], 'S16', data_train)                     
                # Stores each team's chance to reach Sweet 16 into 'S16_Prob'
    data_test.loc[:, 'S16_Rank'] = data_test['S16_Prob'].rank(ascending=False, method='min').astype(int)     
                # Ranks each team's chance to reach Sweet 16 and stores it as 'S16_Rank'
    
    data_test.loc[:, 'R32_Prob'] = logreg_ridge(data_train[r32], data_test[r32], 'R32', data_train)                     
                # Stores each team's chance to reach the Round of 32 into 'R32_Prob'
    data_test.loc[:, 'R32_Rank'] = data_test['R32_Prob'].rank(ascending=False, method='min').astype(int)     
                # Ranks each team's chance to reach the Round of 32 and stores it as 'R32_Rank'
    
    data_test.loc[:, 'Proj_Wins'] = get_proj_wins(data_train, data_test, data_train['Wins'])
                # Stores each team's projected wins as 'Proj_Wins'
    data_test.loc[:, 'Proj_Wins_Rank'] = data_test['Proj_Wins'].rank(ascending=False, method='min').astype(int)
                # Ranks each team's projected wins as 'Proj_Wins_Rank'  
    
    return data_test    # Returns the updated data_test

def logreg_ridge(x_train, x_test, y_tr, data_train):
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
    
    y_train = data_train[y_tr] # Saves the training outcome variable
    
    scaler = StandardScaler()                 # Creates a standardization tool that will scale features to have mean = 0 and standard deviation = 1
    x_scaled = scaler.fit_transform(x_train)  # Computes the mean and standard deviation of x_train, then scales all its features accordingly
    
    x_test_scaled = scaler.transform(x_test)

    ridge = LogisticRegression(         # Creates model, saves it as ridge
        penalty='l2',                     # Ridge penalty - shrinks insignificant variables to near-zero
        solver='lbfgs',                   # LBFGS optimization algorithm
        class_weight = None,        # Automatically adjusts class weights to handle imbalanced classes
        max_iter=10000,                   # Sets the maximum number of iterations to 10,000 to ensure the model has enough time to converge
        C = 0.25 # Assigns multiple C values for the model to loop through and see which performs best
    )
    calibrated = CalibratedClassifierCV(
        estimator=ridge,
        method='sigmoid',
        cv=5                   
    )

    calibrated.fit(x_scaled, y_train)

    predicted_probs = calibrated.predict_proba(x_test_scaled)[:, 1]
    
    return predicted_probs    # Returns the probability that each sample belongs to class 1 for test variables

def get_proj_wins(data_train, data_test, y_train_series):
    '''
    This function runs a poisson regression to calculate the number of tournament wins each team in x_test will have based on their statistical profile
    The function does the following:
        Reads in:
            data_train (the training data)
            data_test (the testing data)
            y_train_series (a series, the 'Wins' column in the training data)
        Creates a Poisson regression model
        Fits the model on the training data
        Predicts the value for each team in the testing data
        Returns output
    '''
    
    optimal = ['WAB','KADJ O', 'KADJ D', 'EFF HGT', 'OREB%', '2PT%D', 'TOV%', 'FTRD', 'EXP', '3PT%D', '3PT%']
    
    x_train = data_train[optimal]
    x_test = data_test[optimal]
    
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.3, max_iter=10000)) # Creates poisson regression model as model
    model.fit(x_train, y_train_series) # Fits model to training data
    y_pred = model.predict(x_test)     # Predicts outcome of testing data based on model
    
    return y_pred # Returns the predicted wins for each team in test data

def team_matchup(team1, team2, data, year):
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
        
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
        
    seedA = teamA['SEED'].iloc[0] # Saves team1's seed as seedA
    seedB = teamB['SEED'].iloc[0] # Saves team2's seed as seedB
    
    st.subheader("    " * 4 + f"{team1}({seedA}) vs. {team2}({seedB})\n") # Prints the two teams and their respective seeds
            
    wab1, wab2 = sos_matchup(team1, team2, data, year)    
        # Calls sos_matchup and saves the returns into variables
    tempo1, tempo2, tempo = tempo_matchup(team1, team2, data, year)   
        # Calls tempo_matchup and saves the returns into variables
    experience_matchup(team1, team2, data, year)
        # Calls experience_matchup
    odnet1, odnet2 = off_def_matchup(team1, team2, data, year)   
        # Calls off_def_matchup and saves the returns into variables
    p2net1, p2net2 = twopt_matchup(team1, team2, tempo, data, year)
        # Calls twopt_matchup and saves the returns into variables
    p3net1, p3net2 = threept_matchup(team1, team2, tempo, data, year)
        # Calls threept_matchup and saves the returns into variables
    ftnet1, ftnet2 = ft_matchup(team1, team2, tempo, data, year)
        # Calls ft_matchup and saves the returns into variables
    to1net, to2net = to_matchup(team1, team2, data, year)
        # Calls to_matchup and saves the returns into variables
    rb1net, rb2net = reb_matchup(team1, team2, data, year)
        # Calls reb_matchup and saves the returns into variables
    
    get_win_percentages(team1, team2, wab1, wab2, odnet1, odnet2, p2net1, p2net2, p3net1, p3net2, ftnet1, ftnet2, to1net, to2net, rb1net, rb2net, seedA, seedB, data, year) # Calls get_win_percetages
        
def get_seed_bounds(team, seed, data, year):
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
    
    df = data[data['YEAR'] == year]
    
    seed_counts = df['SEED'].value_counts().sort_index() # Counts how many times each seed appears in the 'Seed' column, sorts it by seed
    
    lower = 1
    for s in range(1, 17):  # seeds 1 through 16
        count = seed_counts.get(s, 0) # Checks how many of each seed there are. If none, it defaults to 0
        upper = lower + count - 1 # Calculates the upper bound of the ranking for this seed
        
        if seed == s:                                                             # If seed matches the current loop value s:
            return f"{team}({seed}) is expected to rank between {lower}-{upper}"  # Return string which says where the team is expected to rank
        
        lower = upper + 1 # Change the lower bound to the next seed group
        
def get_win_percentages(team1, team2, wab1, wab2, odnet1, odnet2, p2net1, p2net2, p3net1, p3net2, ftnet1, ftnet2, to1net, to2net, rb1net, rb2net, seedA, seedB, data, year):
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
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]
    projdf = data[(data['YEAR'] > 2012) & (data['YEAR'] <= year)]
    
    count = len(data[data['YEAR'] == year]) # Get the number of teams in tournament
    
    # Saving both team's ranks/proj wins into variables
    teamA_nat = int(teamA['Natty_Rank'].iloc[0])
    teamB_nat = int(teamB['Natty_Rank'].iloc[0])
    #teamA_cg = teamA['CG_Rank'].iloc[0]
    #teamB_cg = teamB['CG_Rank'].iloc[0]
    teamA_f4 = int(teamA['F4_Rank'].iloc[0])
    teamB_f4 = int(teamB['F4_Rank'].iloc[0])
    teamA_e8 = int(teamA['E8_Rank'].iloc[0])
    teamB_e8 = int(teamB['E8_Rank'].iloc[0])
    teamA_s16 = int(teamA['S16_Rank'].iloc[0])
    teamB_s16 = int(teamB['S16_Rank'].iloc[0])
    teamA_r32 = int(teamA['R32_Rank'].iloc[0])
    teamB_r32 = int(teamB['R32_Rank'].iloc[0])
    teamA_pwins = (teamA['Proj_Wins'].iloc[0])
    teamB_pwins = (teamB['Proj_Wins'].iloc[0])
    teamA_winsrank = int(teamA['Proj_Wins_Rank'].iloc[0])
    teamB_winsrank = int(teamB['Proj_Wins_Rank'].iloc[0])
    
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
    
    wab_max = df['WAB'].max()
    wab_min = df['WAB'].min()
    
    wab_net1 = (wab1 - wab_min) / (wab_max - wab_min)
    wab_net2 = (wab2 - wab_min) / (wab_max - wab_min)
    
    pwins_max = projdf["Proj_Wins"].max()
    pwins_min = projdf["Proj_Wins"].min()
    
    pwins_net1 = (teamA_pwins - pwins_min) / (pwins_max - pwins_min)
    pwins_net2 = (teamB_pwins - pwins_min) / (pwins_max - pwins_min)
    
    advancement = find_round(team1, team2, data, year)
        
    teamA_adv = teamA[advancement].iloc[0]
    teamB_adv = teamB[advancement].iloc[0]
    
    adv_max = projdf[advancement].max()
    adv_min = projdf[advancement].min()
    
    advnet1 = (teamA_adv - adv_min) / (adv_max - adv_min)
    advnet2 = (teamB_adv - adv_min) / (adv_max - adv_min)
    
    
    best = 30*1 + 5 + 6*1 + 5*1 + 6*1 + 4 + 10*(1) + 15*1 + 15*1                      # Calculates best possible outcome for a team
    worst = 30*(-1) + (-5) + 6*(-1) + 5*(-1) + 6*(-1) + (-4) + 10*(-1) + 15*(-1) + 15*(-1)  # Calculates worst possible outcome for a team
    
    score = 30*(odnet1 - odnet2) + 5*(p2net1 - p2net2) + 6*(p3net1 - p3net2) + 5*(ftnet1 - ftnet2) + 6*(to1net - to2net) + 4*(rb1net - rb2net) + 10*(wab_net1 - wab_net2) + 15*(pwins_net1 - pwins_net2) + 15*(advnet1 - advnet2)
        # Calculates team1's score according to the formula
    
    normalized_score = (score - worst) / (best - worst)  # Normalizes team1's score to a [0,1] scale
    
    scaled_score = 12.5 * (normalized_score - 0.5)  # Centers the normalized score around 0        

    percentage1 = 100 / (1 + math.exp(-scaled_score)) # Applies sigmoid function to scaled score and multiplying by 100, giving team1's win percentage
    percentage2 = 100 - percentage1                   # Calculates team2's win percentage
    
    # DISPLAYS OF WIN PERCENTAGE, RANKS, AND PROJ. WINS
    st.write("\n============================================================================================================================")
    st.write('''  ##### WIN PERCENTAGE''')
    st.text(f"{team1 + ':':<12} {percentage1:.1f} {'%'}\t\t\t{team2 + ':':<12} {percentage2:.1f} {'%'}") # Displays both teams' win percentages
    st.write()
    st.write("============================================================================================================================")
    st.write()
    st.write(''' ##### ROUND ADVANCEMENT RANKS''')
    st.write("KEEP IN MIND: ", get_seed_bounds(team1, seedA, data, year), '. ', get_seed_bounds(team2, seedB, data, year)) 
        # Calls get_seed_bounds to give context to ranks
    st.write()
    st.text(f"ROUND OF 32 RANK\n{team1 + ':':<12} {teamA_r32}/{count}\t\t\t{team2 + ':':<12} {teamB_r32}/{count}")
    #st.text(f"{team1 + ':':<12} {teamA_r32}/{count}\t\t\t{team2 + ':':<12} {teamB_r32}/{count}") # Displays both teams' R32 ranks
    st.write()
    st.text(f"SWEET 16 RANK\n{team1 + ':':<12} {teamA_s16}/{count}\t\t\t{team2 + ':':<12} {teamB_s16}/{count}")
    #st.text(f"{team1 + ':':<12} {teamA_s16}/{count}\t\t\t{team2 + ':':<12} {teamB_s16}/{count}") # Displays both teams' Sweet 16 ranks
    st.write()
    st.text(f"ELITE 8 RANK\n{team1 + ':':<12} {teamA_e8}/{count}\t\t\t{team2 + ':':<12} {teamB_e8}/{count}")
    #st.text(f"{team1 + ':':<12} {teamA_e8}/{count}\t\t\t{team2 + ':':<12} {teamB_e8}/{count}") # Displays both teams' Elite 8 ranks
    st.write()
    st.text(f"FINAL 4 RANK\n{team1 + ':':<12} {teamA_f4}/{count}\t\t\t{team2 + ':':<12} {teamB_f4}/{count}")
    #st.text(f"{team1 + ':':<12} {teamA_f4}/{count}\t\t\t{team2 + ':':<12} {teamB_f4}/{count}") # Displays both teams' Final 4 ranks
    st.write()
    #print("CHAMPIONSHIP GAME RANK")
    #print(f"{team1 + ':':<12} {teamA_cg}/{count}\t{team2 + ':':<12} {teamB_cg}/{count}") # Displays both teams' championship game ranks
    #print()
        # As of right now, championship rank is left out to prevent this section from being too crowded
    st.text(f"NATIONAL CHAMPION RANK\n{team1 + ':':<12} {teamA_nat}/{count}\t\t\t{team2 + ':':<12} {teamB_nat}/{count}")
    #st.text(f"{team1 + ':':<12} {teamA_nat}/{count}\t\t\t{team2 + ':':<12} {teamB_nat}/{count}") # Displays both teams' National Champion ranks
    st.write()
    st.write("============================================================================================================================")
    st.write()
    st.write('''  ##### PROJECTED TOURNAMENT WINS''')
    st.text(f"{team1 + ':':<12} {teamA_pwins:.2f} (#{teamA_winsrank})\t\t\t{team2 + ':':<12} {teamB_pwins:.2f} (#{teamB_winsrank})") 
        # Displays both teams' projected tournament wins
        
def find_round(team1, team2, data, year):
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]
    
    seedA = teamA['SEED'].iloc[0]
    seedB = teamB['SEED'].iloc[0]
    
    quadA = teamA['QUAD ID'].iloc[0]
    quadB = teamB['QUAD ID'].iloc[0]
    
    if quadA == quadB:
        if (
            (seedA in [1,4,5,8,9,12,13,16] and seedB in [2,3,6,7,10,11,14,15]) or 
            (seedA in [2,3,6,7,10,11,14,15] and seedB in [1,4,5,8,9,12,13,16])
        ):
            use = "F4_Prob"
            
        elif (
            (seedA in [1,8,9,16] and seedB in [4,5,12,13]) or 
            (seedB in [1,8,9,16] and seedA in [4,5,12,13]) or 
            (seedA in [2,7,10,15] and seedB in [3,6,11,14]) or 
            (seedB in [2,7,10,15] and seedA in [3,6,11,14])
        ):
            use = "E8_Prob"
            
        elif (
            (seedA in [1,16] and seedB in [8,9]) or
            (seedB in [1,16] and seedA in [8,9]) or
            (seedA in [2,15] and seedB in [7,10]) or
            (seedB in [2,15] and seedA in [7,10]) or
            (seedA in [3,14] and seedB in [6,11]) or
            (seedB in [3,14] and seedA in [6,11]) or
            (seedA in [4,13] and seedB in [5,12]) or
            (seedB in [4,13] and seedA in [5,12])
        ):
            use = "S16_Prob"
            
        elif (
            (seedA == 1 and seedB == 16) or
            (seedB == 1 and seedA == 16) or
            (seedA == 2 and seedB == 15) or
            (seedB == 2 and seedA == 15) or
            (seedA == 3 and seedB == 14) or
            (seedB == 3 and seedA == 14) or
            (seedA == 4 and seedB == 13) or
            (seedB == 4 and seedA == 13) or
            (seedA == 5 and seedB == 12) or
            (seedB == 5 and seedA == 12) or
            (seedA == 6 and seedB == 11) or
            (seedB == 6 and seedA == 11) or
            (seedA == 7 and seedB == 10) or
            (seedB == 7 and seedA == 10) or
            (seedA == 8 and seedB == 9) or
            (seedB == 8 and seedA == 9)
        ):
            use = "R32_Prob"
            
        else:
            use = "R32_Prob"                  # Only possible if they are the same seed - First Four
             
    else:
        use = "Win_Prob"
        
    return use
    
def sos_matchup(team1, team2, data, year):
    '''
    This function displays both team's respective strength of schedule analyses
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    Displays and returns each team's Elite SOS and WAB
    '''
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    teamA_wab = teamA['WAB'].iloc[0] # Stores team1's WAB
    teamB_wab = teamB['WAB'].iloc[0] # Stores team2's WAB
    teamA_esos = teamA['ELITE SOS'].iloc[0] # Stores team1's Elite SOS
    teamB_esos = teamB['ELITE SOS'].iloc[0] # Stores team2's Elite SOS
    
    # DISPLAYS OF ELITE SOS AND WAB
    st.write("\n============================================================================================================================")
    st.write(''' ##### STRENGTH OF SCHEDULE COMPARISON''')
    st.write()
    st.text(f"ELITE SOS\n{team1 + ':':<12} {teamA_esos}\t\t\t{team2 + ':':<12} {teamB_esos}")
    #st.text(f"{team1 + ':':<12} {teamA_esos}\t\t\t{team2 + ':':<12} {teamB_esos}")     # Displays each team's Elite SOS
    st.write()
    st.text(f"WINS ABOVE BUBBLE\n{team1 + ':':<12} {teamA_wab}\t\t\t{team2 + ':':<12} {teamB_wab}")
    #st.text(f"{team1 + ':':<12} {teamA_wab}\t\t\t{team2 + ':':<12} {teamB_wab}")     # Displays each team's WAB
    
    return teamA_wab, teamB_wab # Returns each team's WAB
    
def tempo_matchup(team1, team2, data, year):
    '''
    This function displays the Tempo matchup between both teams
    The function reads in:
        team1 (the first team the user input)
        team2 (the second team the user input)
        data_test (the dataset for the year the user input)
    Calculates mean tempo
    Returns each team's tempo and the mean tempo
    '''
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    teamA_tempo = teamA['KADJ T'].iloc[0] # Stores team1's tempo
    teamB_tempo = teamB['KADJ T'].iloc[0] # Stores team2's tempo
    
    # Calculates mean tempo (expected # of possessions in specific matchup)
    tempo = (teamA_tempo + teamB_tempo) / 2
    
    # DISPLAYS EACH TEAMS' TEMPO
    st.write("\n============================================================================================================================")
    st.write(''' ##### TEMPO COMPARISON''')
    st.write()
    st.text(f"ADJUSTED TEMPO\n{team1 + ':':<12} {teamA_tempo:.2f}\t\t\t{team2 + ':':<12} {teamB_tempo:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_tempo:.2f}\t\t\t{team2 + ':':<12} {teamB_tempo:.2f}") # Displays each teams' tempo
        
    return teamA_tempo, teamB_tempo, tempo # Returns tempos and mean tempo

def experience_matchup(team1, team2, data, year):
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    teamA_exp = teamA['EXP'].iloc[0] # Stores team1's experience
    teamB_exp = teamB['EXP'].iloc[0] # Stores team2's experience
    
    # DISPLAYS EACH TEAMS' EXPERIENCE
    st.write("\n============================================================================================================================")
    st.write('''  ##### EXPERIENCE COMPARISON''')
    st.write()
    st.text(f"EXPERIENCE\n{team1 + ':':<12} {teamA_exp:.2f}\t\t\t{team2 + ':':<12} {teamB_exp:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_exp:.2f}\t\t\t{team2 + ':':<12} {teamB_exp:.2f}") # Displays each teams' experience
    
def off_def_matchup(team1, team2, data, year):
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
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]
    
    min_OE = df['KADJ O'].min() # Min Adjusted Offense efficiency
    max_OE = df['KADJ O'].max() # Max Adjusted Offense efficiency
    min_DE = df['KADJ D'].min() # Min Adjusted Defense efficiency
    max_DE = df['KADJ D'].max() # Max Adjusted Defense efficiency
    
    min_matchup = np.log(min_OE * min_DE) # Worst possible matchup for an offense (Worst offense vs best defense)
    max_matchup = np.log(max_OE * max_DE) # Best possible matchup for an offense (Best offense vs worst defense)
    
    # Store each team's statistics into variables
    teamA_OE = teamA['KADJ O'].iloc[0]
    teamA_DE = teamA['KADJ D'].iloc[0]
    teamA_efgO = teamA['EFG%'].iloc[0]
    teamA_efgD = teamA['EFG%D'].iloc[0]   
    teamB_OE = teamB['KADJ O'].iloc[0]
    teamB_DE = teamB['KADJ D'].iloc[0]
    teamB_efgO = teamB['EFG%'].iloc[0]
    teamB_efgD = teamB['EFG%D'].iloc[0]
    
    AB = np.log(teamA_OE * teamB_DE) # Calculate log-scaled matchup strength of team1's offense vs. team2's defense
    AB_net = 2 * ((AB - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalizes to [-1,1] scale based on min and max
    BA = np.log(teamB_OE * teamA_DE) # Calculate log-scaled matchup strength of team2's offense vs. team2's defense
    BA_net = 2 * ((BA - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalizes to [-1,1] scale based on min and max
    
    # DISPLAYS EACH TEAMS OFF/DEF EFFICIENCIES, OFF/DEF EFG%, AND MATCHUP SCORES
    st.write("\n============================================================================================================================")
    st.write('''  ##### OFFENSE VS DEFENSE COMPARISON''')
    st.write()
    st.text(f"ADJUSTED OFFENSIVE EFFICIENCY\n{team1 + ':':<12} {teamA_OE:.2f}\t\t\t{team2 + ':':<12} {teamB_OE:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_OE:.2f}\t\t\t{team2 + ':':<12} {teamB_OE:.2f}") # Displays each team's Adj. Off Efficiency
    st.write()
    st.text(f"ADJUSTED DEFENSIVE EFFICIENCY\n{team1 + ':':<12} {teamA_DE:.2f}\t\t\t{team2 + ':':<12} {teamB_DE:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_DE:.2f}\t\t\t{team2 + ':':<12} {teamB_DE:.2f}") # Displays each team's Adj. Def Efficiency
    st.write()
    st.text(f"OFFENSIVE EFFECTIVE FG%\n{team1 + ':':<12} {teamA_efgO}\t\t\t{team2 + ':':<12} {teamB_efgO}")
    #st.text(f"{team1 + ':':<12} {teamA_efgO}\t\t\t{team2 + ':':<12} {teamB_efgO}") # Displays each team's Off Effective FG%
    st.write()
    st.text(f"DEFENSIVE EFFECTIVE FG%\n{team1 + ':':<12} {teamA_efgD}\t\t\t{team2 + ':':<12} {teamB_efgD}")
    #st.text(f"{team1 + ':':<12} {teamA_efgD}\t\t\t{team2 + ':':<12} {teamB_efgD}") # Displays each team's Def Effective FG%
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}")
    #st.text(f"OFFENSE VS DEFENSE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}") # Explains what matchup score represents
    #st.text(f"{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}") # Displays each team's OFF vs DEF matchup score
    
    return AB_net, BA_net # Returns everything
    
def twopt_matchup(team1, team2, tempo, data, year):
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
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]
    
    # Calculate each team's expected off, def, and net 2pt given the tempo
    teamA_o2 = (teamA['2PT%'].iloc[0] * teamA['2PTR'].iloc[0] * tempo * (1 - teamA['TOV%'].iloc[0]))
    teamB_o2 = (teamB['2PT%'].iloc[0] * teamB['2PTR'].iloc[0] * tempo * (1 - teamB['TOV%'].iloc[0]))
    teamA_d2 = (teamA['2PT%D'].iloc[0] * teamA['2PTRD'].iloc[0] * tempo * (1 - teamA['TOV%D'].iloc[0]))
    teamB_d2 = (teamB['2PT%D'].iloc[0] * teamB['2PTRD'].iloc[0] * tempo * (1 - teamB['TOV%D'].iloc[0]))
    teamA_net2 = teamA_o2 - teamA_d2
    teamB_net2 = teamB_o2 - teamB_d2
    
    # Find the min and max values for offense and defense
    min_o2 = (df['2PT%'] * df['2PTR'] * (1 - df['TOV%'])).min() * tempo
    max_o2 = (df['2PT%'] * df['2PTR'] * (1 - df['TOV%'])).max() * tempo
    min_d2 = (df['2PT%D'] * df['2PTRD'] * (1 - df['TOV%D'])).min() * tempo
    max_d2 = (df['2PT%D'] * df['2PTRD'] * (1 - df['TOV%D'])).max() * tempo
    
    # Calculate best and worst possible matchups using log(offense × defense)
    min_matchup = np.log(min_o2 * min_d2) # Worst matchup (worst offense vs best defense)
    max_matchup = np.log(max_o2 * max_d2) # Best matchup (best offense vs worst defense)
    
    # Calculate log-scaled matchup strength of each team's offense vs the other team's defense
    AB_net = np.log(teamA_o2 * teamB_d2) # team1’s offense vs team2’s defense
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA_net = np.log(teamB_o2 * teamA_d2) # team2’s offense vs team1’s defense
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    
    # DISPLAY ALL STATS AND MATCHUPS
    st.write("\n============================================================================================================================")
    st.write('''  ##### TWO POINT COMPARISON''')
    st.write()
    st.text(f"NET 2-POINTERS (Shows expected 2pt margin for each team at this game's expected tempo)\n{team1 + ':':<12} {teamA_net2:.2f}\t\t\t{team2 + ':':<12} {teamB_net2:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_net2:.2f}\t\t\t{team2 + ':':<12} {teamB_net2:.2f}")                 
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE 2PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}")
    #st.write("OFFENSE VS DEFENSE 2PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)")
    #st.text(f"{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}") 
    
    # RETURN EVERYTHING
    return AB_scaled, BA_scaled
    
def threept_matchup(team1, team2, tempo, data, year):
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
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]
    
    # Calculate each team's expected off, def, and net 3pt given the tempo
    teamA_o3 = (teamA['3PT%'].iloc[0] * teamA['3PTR'].iloc[0] * tempo * (1 - teamA['TOV%'].iloc[0]))
    teamB_o3 = (teamB['3PT%'].iloc[0] * teamB['3PTR'].iloc[0] * tempo * (1 - teamB['TOV%'].iloc[0]))
    teamA_d3 = (teamA['3PT%D'].iloc[0] * teamA['3PTRD'].iloc[0] * tempo * (1 - teamA['TOV%D'].iloc[0]))
    teamB_d3 = (teamB['3PT%D'].iloc[0] * teamB['3PTRD'].iloc[0] * tempo * (1 - teamB['TOV%D'].iloc[0]))
    teamA_net3 = teamA_o3 - teamA_d3
    teamB_net3 = teamB_o3 - teamB_d3
    
    # Find the min and max values for offense and defense
    min_o3 = (df['3PT%'] * df['3PTR'] * (1 - df['TOV%'])).min() * tempo
    max_o3 = (df['3PT%'] * df['3PTR'] * (1 - df['TOV%'])).max() * tempo
    min_d3 = (df['3PT%D'] * df['3PTRD'] * (1 - df['TOV%D'])).min() * tempo
    max_d3 = (df['3PT%D'] * df['3PTRD'] * (1 - df['TOV%D'])).max() * tempo
    
    # Calculate best and worst possible matchups using log(offense × defense)
    min_matchup = np.log(min_o3 * min_d3) # Worst matchup (worst offense vs best defense)
    max_matchup = np.log(max_o3 * max_d3) # Best matchup (best offense vs worst defense)
    
    # Calculate log-scaled matchup strength of each team's offense vs the other team's defense
    AB_net = np.log(teamA_o3 * teamB_d3) # team1’s offense vs team2’s defense
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA_net = np.log(teamB_o3 * teamA_d3) # team2’s offense vs team1’s defense
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    
    # DISPLAY ALL STATS AND MATCHUPS
    st.write("\n============================================================================================================================")
    st.write('''  ##### THREE POINT COMPARISON''')
    st.write()
    st.text(f"NET 3-POINTERS (Shows expected 3pt margin for each team at this game's expected tempo)\n{team1 + ':':<12} {teamA_net3:.2f}\t\t\t{team2 + ':':<12} {teamB_net3:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_net3:.2f}\t\t\t{team2 + ':':<12} {teamB_net3:.2f}")                 
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE 3PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}")
    #st.write("OFFENSE VS DEFENSE 3PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)")
    #st.text(f"{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}") 
    
    # RETURN EVERYTHING
    return AB_scaled, BA_scaled

def ft_matchup(team1, team2, tempo, data, year):
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
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]
    
    # Calculate each team's expected off, def, and net 3pt given the tempo
    # For defense, it is calculated using the opponent's FT%
    teamA_of = (teamA['FT%'].iloc[0] * teamA['FTR'].iloc[0] * tempo * (1 - teamA['TOV%'].iloc[0]))
    teamB_of = (teamB['FT%'].iloc[0] * teamB['FTR'].iloc[0] * tempo * (1 - teamB['TOV%'].iloc[0]))
    teamA_df = (teamB['FT%'].iloc[0] * teamA['FTRD'].iloc[0] * tempo * (1 - teamA['TOV%D'].iloc[0]))
    teamB_df = (teamA['FT%'].iloc[0] * teamB['FTRD'].iloc[0] * tempo * (1 - teamB['TOV%D'].iloc[0]))
    teamA_netf = teamA_of - teamA_df
    teamB_netf = teamB_of - teamB_df
    
    # Find the min and max values for offense and defense
    min_of = (df['FT%'] * df['FTR'] * (1 - df['TOV%'])).min() * tempo
    max_of = (df['FT%'] * df['FTR'] * (1 - df['TOV%'])).max() * tempo
    min_df = (0.715 * df['FTRD'] * (1 - df['TOV%D'])).min() * tempo
    max_df = (0.715 * df['FTRD'] * (1 - df['TOV%D'])).max() * tempo
    
    # Calculate best and worst possible matchups using log(offense × defense)
    min_matchup = np.log(min_of * min_df) # Worst matchup (worst offense vs best defense)
    max_matchup = np.log(max_of * max_df) # Best matchup (best offense vs worst defense)
    
    # Calculate log-scaled matchup strength of each team's offense vs the other team's defense
    AB_net = np.log(teamA_of * teamB_df) # team1’s offense vs team2’s defense
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA_net = np.log(teamB_of * teamA_df) # team2’s offense vs team1’s defense
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    
    # DISPLAY ALL STATS AND MATCHUPS
    st.write("\n============================================================================================================================")
    st.write('''  ##### FREE THROW COMPARISON''')
    st.write()
    st.text(f"NET FREE THROWS (Shows expected FT margin for each team at this game's expected tempo)\n{team1 + ':':<12} {teamA_netf:.2f}\t\t\t{team2 + ':':<12} {teamB_netf:.2f}")
    #st.text(f"{team1 + ':':<12} {teamA_netf:.2f}\t\t\t{team2 + ':':<12} {teamB_netf:.2f}")                 
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE FT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}")
    #st.write("OFFENSE VS DEFENSE FT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)")
    #st.text(f"{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}") 
    
    # RETURN EVERYTHING
    return AB_scaled, BA_scaled
    
def to_matchup(team1, team2, data, year):
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

    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]

    # Get the minimum and maximum values for offensive and defensive turnover percentages in the dataset
    min_tov = df['TOV%'].min()   # Lowest offensive turnover rate
    max_tov = df['TOV%'].max()   # Highest offensive turnover rate
    min_tovd = df['TOV%D'].min() # Lowest defensive turnover rate (least turnovers forced)
    max_tovd = df['TOV%D'].max() # Highest defensive turnover rate (most turnovers forced)

    # Calculate worst (high TOV × high TOVD) and best (low TOV × low TOVD) matchups using logs
    worst = np.log(max_tov * max_tovd) # Worst-case turnover matchup (bad offense vs great defense)
    best = np.log(min_tov * min_tovd)  # Best-case turnover matchup (great offense vs weak defense)

    # Retrieve each team’s turnover stats
    teamA_tov = teamA['TOV%'].iloc[0]        # team1's offensive turnover percentage
    teamA_tovd = teamA['TOV%D'].iloc[0]      # team1's defensive turnover percentage
    teamB_tov = teamB['TOV%'].iloc[0]        # team2's offensive turnover percentage
    teamB_tovd = teamB['TOV%D'].iloc[0]      # team2's defensive turnover percentage

    # Compute head-to-head log-scaled matchup strength
    AB = np.log(teamA_tov * teamB_tovd) # team1’s offense vs team2’s defense
    AB_net = 2 * ((AB - worst) / (best - worst)) - 1 # Normalize to [-1,1] scale
    BA = np.log(teamB_tov * teamA_tovd) # team2’s offense vs team1’s defense
    BA_net = 2 * ((BA - worst) / (best - worst)) - 1 # Normalize to [-1,1] scale

    # DISPLAY TURNOVER COMPARISON
    st.write("\n============================================================================================================================")
    st.write('''  ##### TURNOVER COMPARISON''')
    st.write()
    st.text(f"OFFENSIVE TURNOVER PERCENTAGE\n{team1 + ':':<12} {teamA_tov}\t\t\t{team2 + ':':<12} {teamB_tov}") # Each team’s average rate of committing turnovers
    #st.text(f"{team1 + ':':<12} {teamA_tov}\t\t\t{team2 + ':':<12} {teamB_tov}")
    st.write()
    st.text(f"DEFENSIVE TURNOVER PERCENTAGE\n{team1 + ':':<12} {teamA_tovd}\t\t\t{team2 + ':':<12} {teamB_tovd}") # Each team’s ability to force turnovers
    #st.text(f"{team1 + ':':<12} {teamA_tovd}\t\t\t{team2 + ':':<12} {teamB_tovd}")
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE TURNOVER PERCENTAGE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}")
    #st.write("OFFENSE VS DEFENSE TURNOVER PERCENTAGE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)") # Interpret scale
    #st.text(f"{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}") # Print normalized results

    # RETURN key turnover metrics and head-to-head scores
    return AB_net, BA_net

def reb_matchup(team1, team2, data, year):
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

    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] # Saves the data for team1 as teamA
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] # Saves the data for team1 as teamA
    
    df = data[data['YEAR'] <= year]

    # Get min and max rebounding stats in the dataset
    min_dreb = df['DREB%'].min() # Lowest defensive rebound %
    max_dreb = df['DREB%'].max() # Highest defensive rebound %
    min_oreb = df['OREB%'].min() # Lowest offensive rebound %
    max_oreb = df['OREB%'].max() # Highest offensive rebound %

    # Calculate matchup bounds using logs
    min_matchup = np.log(min_oreb / max_dreb) # Worst matchup: weak offensive rebounding vs strong defensive rebounding
    max_matchup = np.log(max_oreb / min_dreb) # Best matchup: strong offensive rebounding vs weak defensive rebounding

    # Get each team’s rebounding stats and height
    teamA_dreb = teamA['DREB%'].iloc[0]     # team1's defensive rebound %
    teamA_oreb = teamA['OREB%'].iloc[0]     # team1's offensive rebound %
    teamB_dreb = teamB['DREB%'].iloc[0]     # team2's defensive rebound %
    teamB_oreb = teamB['OREB%'].iloc[0]     # team2's offensive rebound %
    teamA_height = teamA['EFF HGT'].iloc[0]  # team1's average height
    teamB_height = teamB['EFF HGT'].iloc[0]  # team2's average height

    # Calculate head-to-head rebounding matchup scores
    AB = np.log(teamA_oreb / teamB_dreb) # team1's offense vs team2's defense
    AB_net = 2 * ((AB - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]
    BA = np.log(teamB_oreb / teamA_dreb) # team2's offense vs team1's defense
    BA_net = 2 * ((BA - min_matchup) / (max_matchup - min_matchup)) - 1 # Normalize to [-1,1]

    # DISPLAY REBOUNDING COMPARISON
    st.write("\n============================================================================================================================")
    st.write('''  ##### REBOUNDING COMPARISON''')
    st.write()
    st.text(f"EFFECTIVE HEIGHT\n{team1 + ':':<12} {teamA_height:.2f}\t\t\t{team2 + ':':<12} {teamB_height:.2f}") # Each team's height
    #st.text(f"{team1 + ':':<12} {teamA_height:.2f}\t\t\t{team2 + ':':<12} {teamB_height:.2f}")
    st.write()
    st.text(f"OFFENSIVE REBOUNDING PERCENTAGE\n{team1 + ':':<12} {teamA_oreb}\t\t\t{team2 + ':':<12} {teamB_oreb}") # Each team’s offensive rebounding %
    #st.text(f"{team1 + ':':<12} {teamA_oreb}\t\t\t{team2 + ':':<12} {teamB_oreb}")
    st.write()
    st.text(f"DEFENSIVE REBOUND PERCENTAGE\n{team1 + ':':<12} {teamA_dreb}\t\t\t{team2 + ':':<12} {teamB_dreb}") # Each team’s defensive rebounding %
    #st.text(f"{team1 + ':':<12} {teamA_dreb}\t\t\t{team2 + ':':<12} {teamB_dreb}")
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSIVE VS DEFENSIVE REBOUNDING (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}")
    #st.write("OFFENSIVE VS DEFENSIVE REBOUNDING (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)") # Explanation of matchup scores
    #st.text(f"{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}") # Print matchup scores

    # RETURN key rebounding stats and matchup scores
    return AB_net, BA_net

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    
data = pd.read_csv("data/data_official.csv") # READ DATASET

data_sorted_year = data.sort_values("YEAR", ascending=False)
years1 = list(data_sorted_year["YEAR"].unique())
dropyears = [2008, 2009, 2010, 2011, 2012]
years = [y for y in years1 if y not in dropyears]
col1, col2, col3 = st.columns([1,1,1])
with col1:
    year = st.selectbox("Pick a year", options=years)

chosen_df = data[data["YEAR"] == year]
chosen_sorted = chosen_df.sort_values("TEAM")
chosen_teams = list(chosen_sorted["TEAM"].unique())

col1, col2, col3 = st.columns([1,1,2])
with col1:
    team1 = st.selectbox("Team 1", options=chosen_teams)
with col2:
    team2 = st.selectbox("Team 2", options=chosen_teams)
    
data['R32'] = [0 for _ in range(len(data))]
data['S16'] = [0 for _ in range(len(data))]
data['E8'] = [0 for _ in range(len(data))]
data['F4'] = [0 for _ in range(len(data))]
data['CHAMP'] = [0 for _ in range(len(data))]

data.loc[data['Wins'] > 0, 'R32'] = 1
data.loc[data['Wins'] > 1, 'S16'] = 1
data.loc[data['Wins'] > 2, 'E8'] = 1
data.loc[data['Wins'] > 3, 'F4'] = 1
data.loc[data['Wins'] > 5, 'CHAMP'] = 1

data['2PT%']  = data['2PT%'].div(100).round(3)
data['2PT%D'] = data['2PT%D'].div(100).round(3)
data['2PTR']  = data['2PTR'].div(100).round(3)
data['2PTRD'] = data['2PTRD'].div(100).round(3)
data['3PT%']  = data['3PT%'].div(100).round(3)
data['3PT%D'] = data['3PT%D'].div(100).round(3)
data['3PTR']  = data['3PTR'].div(100).round(3)
data['3PTRD'] = data['3PTRD'].div(100).round(3)
data['FT%']   = data['FT%'].div(100).round(3)
data['FTR']   = data['FTR'].div(100).round(3)
data['FTRD']  = data['FTRD'].div(100).round(3)
data['TOV%']  = data['TOV%'].div(100).round(3)
data['TOV%D'] = data['TOV%D'].div(100).round(3)
data['OREB%'] = data['OREB%'].div(100).round(3)
data['DREB%'] = data['DREB%'].div(100).round(3)
data['EFG%']  = data['EFG%'].div(100).round(3)
data['EFG%D'] = data['EFG%D'].div(100).round(3)
data['AST%']  = data['AST%'].div(100).round(3)

years = list(range(2013, year+1))
if 2020 in years:
    years.remove(2020)

dats = []

pre = data[data["YEAR"] < 2013]

for y in years:

    data_train = data[data['YEAR'] < y]        
    data_test0 = data[data['YEAR'] == y].copy() 

    data_test = get_ranks(data_train, data_test0)  

    dats.append(data_test)

post_model = pd.concat(dats)
full_data = pd.concat([pre, post_model])
full_data = full_data.sort_values(["YEAR"], ascending=False)

team_matchup(team1, team2, full_data, year)