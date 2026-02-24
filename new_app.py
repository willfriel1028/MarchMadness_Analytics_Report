import pandas as pd
import numpy as np
import streamlit as st
import math
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
MODELS_CACHE = {}

def team_matchup(team1, team2, data, year):

    df_simmed = pd.read_csv("data/" + str(year) + "_10000sims0.csv")
       
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
        
    seedA = teamA['SEED'].iloc[0] 
    seedB = teamB['SEED'].iloc[0] 
    
    st.subheader("    " * 4 + f"{team1}({seedA}) vs. {team2}({seedB})\n") 

    win_percentages(team1, team2, data, year)
    round_advancement(team1, team2, df_simmed, year)        
    sos_matchup(team1, team2, data, year)    
    tempo_matchup(team1, team2, data, year)   
    experience_matchup(team1, team2, data, year)
    off_def_matchup(team1, team2, data, year)   
    twopt_matchup(team1, team2, data, year)
    threept_matchup(team1, team2, data, year)
    ft_matchup(team1, team2, data, year)
    to_matchup(team1, team2, data, year)
    reb_matchup(team1, team2, data, year)

def win_percentages(team1, team2, df, year):

    if year not in MODELS_CACHE:
        try:
            with open(f'bracket_sims/RFmodels/rf_model_{year}.pkl', 'rb') as f:
                MODELS_CACHE[year] = pickle.load(f)
        except FileNotFoundError:
            raise ValueError(f"Model for year {year} not found. Please train it first.")
            
    model = MODELS_CACHE[year]

    perc1 = round(100 * find_percentages(df, team1, team2, year, model), 2)
    perc2 = round((100 - perc1), 2)

    st.write("\n============================================================================================================================")
    st.write(''' ##### WIN PERCENTAGES''')
    st.write()
    st.text(f"{team1 + ':':<12} {perc1}%\t\t\t{team2 + ':':<12} {perc2}%")

def game_style(df, team0, team1, year):
    
    cols = ['KADJ T',
       'KADJ O', 'KADJ D', 'EFG%', 'EFG%D', 'FTR', 'FTRD', 'TOV%',
       'TOV%D', 'OREB%', 'DREB%', '2PT%', '2PT%D', '3PT%', '3PT%D', 'AST%',
       '2PTR', '3PTR', '2PTRD', '3PTRD', 'EFF HGT', 'EXP', 'FT%',
       'ELITE SOS']
    
    t0 = df[(df["TEAM"] == team0) & (df["YEAR"] == year)].reset_index(drop=True)
    t1 = df[(df["TEAM"] == team1) & (df["YEAR"] == year)].reset_index(drop=True)
    
    game = pd.DataFrame()
    
    for col in cols:
        game[col] = t0[col] - t1[col]
    game.insert(0, "YEAR", year)
    game.insert(1, "Team0", t0["TEAM"])
    game.insert(2, "Team1", t1["TEAM"])
    
    return game

def find_percentages(df, team0, team1, year, model):
    
    features = ['KADJ T', 'KADJ O', 'KADJ D', 'EFG%', 'EFG%D', 'FTR', 'FTRD', 
                'TOV%', 'TOV%D', 'OREB%', 'DREB%', '2PT%', '2PT%D', '3PT%', 
                '3PT%D', 'AST%', '2PTR', '3PTR', '2PTRD', '3PTRD', 'EFF HGT', 
                'EXP', 'FT%', 'ELITE SOS']
    
    game_01 = game_style(df, team0, team1, year)
    game_10 = game_style(df, team1, team0, year)
    
    X_test_01 = game_01[features].values
    X_test_10 = game_10[features].values
    
    probs_01 = model.predict_proba(X_test_01)[0]
    probs_10 = model.predict_proba(X_test_10)[0]
    
    team0_win_prob = (probs_01[0] + probs_10[1]) / 2
    team1_win_prob = (probs_01[1] + probs_10[0]) / 2
    
    return team0_win_prob

def round_advancement(team1, team2, data, year):

    ranks = get_ranks(data)

    t1 = data[data["TEAM"] == team1]
    r32perc1 = round(100 * (len(t1[t1["Sim_Wins"] >= 1]) / 10000), 3)
    r32rank1 = ranks[ranks["TEAM"] == team1]["R32_rank"].iloc[0]
    s16perc1 = round(100 * (len(t1[t1["Sim_Wins"] >= 2]) / 10000), 3)
    s16rank1 = ranks[ranks["TEAM"] == team1]["S16_rank"].iloc[0]
    e8perc1 = round(100 * (len(t1[t1["Sim_Wins"] >= 3]) / 10000), 3)
    e8rank1 = ranks[ranks["TEAM"] == team1]["E8_rank"].iloc[0]
    f4perc1 = round(100 * (len(t1[t1["Sim_Wins"] >= 4]) / 10000), 3)
    f4rank1 = ranks[ranks["TEAM"] == team1]["F4_rank"].iloc[0]
    cgperc1 = round(100 * (len(t1[t1["Sim_Wins"] >= 5]) / 10000), 3)
    cgrank1 = ranks[ranks["TEAM"] == team1]["CG_rank"].iloc[0]
    winperc1 = round(100 * (len(t1[t1["Sim_Wins"] == 6]) / 10000), 3)
    winrank1 = ranks[ranks["TEAM"] == team1]["WIN_rank"].iloc[0]

    t2 = data[data["TEAM"] == team2]
    r32perc2 = round(100 * (len(t2[t2["Sim_Wins"] >= 1]) / 10000), 3)
    r32rank2 = ranks[ranks["TEAM"] == team2]["R32_rank"].iloc[0]
    s16perc2 = round(100 * (len(t2[t2["Sim_Wins"] >= 2]) / 10000), 3)
    s16rank2 = ranks[ranks["TEAM"] == team2]["S16_rank"].iloc[0]
    e8perc2 = round(100 * (len(t2[t2["Sim_Wins"] >= 3]) / 10000), 3)
    e8rank2 = ranks[ranks["TEAM"] == team2]["E8_rank"].iloc[0]
    f4perc2 = round(100 * (len(t2[t2["Sim_Wins"] >= 4]) / 10000), 3)
    f4rank2 = ranks[ranks["TEAM"] == team2]["F4_rank"].iloc[0]
    cgperc2 = round(100 * (len(t2[t2["Sim_Wins"] >= 5]) / 10000), 3)
    cgrank2 = ranks[ranks["TEAM"] == team2]["CG_rank"].iloc[0]
    winperc2 = round(100 * (len(t2[t2["Sim_Wins"] == 6]) / 10000), 3)
    winrank2 = ranks[ranks["TEAM"] == team2]["WIN_rank"].iloc[0]

    st.write("\n============================================================================================================================")
    st.write(''' ##### ROUND ADVANCEMENT (Chance of reaching each round)''')
    st.write()
    st.text(f"ROUND OF 32\n{team1 + ':':<12} {r32perc1}% (#{r32rank1})\t\t\t{team2 + ':':<12} {r32perc2}% (#{r32rank2})")
    st.write()
    st.text(f"SWEET 16\n{team1 + ':':<12} {s16perc1}% (#{s16rank1})\t\t\t{team2 + ':':<12} {s16perc2}% (#{s16rank2})")
    st.write()
    st.text(f"ELITE 8\n{team1 + ':':<12} {e8perc1}% (#{e8rank1})\t\t\t{team2 + ':':<12} {e8perc2}% (#{e8rank2})")
    st.write()
    st.text(f"FINAL 4\n{team1 + ':':<12} {f4perc1}% (#{f4rank1})\t\t\t{team2 + ':':<12} {f4perc2}% (#{f4rank2})")
    st.write()
    st.text(f"CHAMPIONSHIP GAME\n{team1 + ':':<12} {cgperc1}% (#{cgrank1})\t\t\t{team2 + ':':<12} {cgperc2}% (#{cgrank2})")
    st.write()
    st.text(f"NATIONAL CHAMPION\n{team1 + ':':<12} {winperc1}% (#{winrank1})\t\t\t{team2 + ':':<12} {winperc2}% (#{winrank2})")

def get_ranks(df):

    teams = list(df["TEAM"].unique())

    dfs = []
    for team in teams:
        dfx = {}
        dfx["TEAM"] = team
        dfx["R32"] = len(df[(df["TEAM"] == team) & (df["Sim_Wins"] >= 1)])
        dfx["S16"] = len(df[(df["TEAM"] == team) & (df["Sim_Wins"] >= 2)])
        dfx["E8"] = len(df[(df["TEAM"] == team) & (df["Sim_Wins"] >= 3)])
        dfx["F4"] = len(df[(df["TEAM"] == team) & (df["Sim_Wins"] >= 4)])
        dfx["CG"] = len(df[(df["TEAM"] == team) & (df["Sim_Wins"] >= 5)])
        dfx["WIN"] = len(df[(df["TEAM"] == team) & (df["Sim_Wins"] == 6)])
        dfs.append(dfx)
        
    dfc = pd.DataFrame(dfs).reset_index(drop=True)
    
    for team in teams:
        dfc['R32_rank'] = dfc['R32'].rank(ascending=False, method='min').astype(int)
        dfc['S16_rank'] = dfc['S16'].rank(ascending=False, method='min').astype(int)
        dfc['E8_rank'] = dfc['E8'].rank(ascending=False, method='min').astype(int)
        dfc['F4_rank'] = dfc['F4'].rank(ascending=False, method='min').astype(int)
        dfc['CG_rank'] = dfc['CG'].rank(ascending=False, method='min').astype(int)
        dfc['WIN_rank'] = dfc['WIN'].rank(ascending=False, method='min').astype(int)

    return dfc       
    
def sos_matchup(team1, team2, data, year):
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    teamA_wab = teamA['WAB'].iloc[0] 
    teamB_wab = teamB['WAB'].iloc[0] 
    teamA_esos = teamA['ELITE SOS'].iloc[0] 
    teamB_esos = teamB['ELITE SOS'].iloc[0] 
    
    st.write("\n============================================================================================================================")
    st.write(''' ##### STRENGTH OF SCHEDULE COMPARISON''')
    st.write()
    st.text(f"ELITE SOS\n{team1 + ':':<12} {teamA_esos}\t\t\t{team2 + ':':<12} {teamB_esos}")
    st.write()
    st.text(f"WINS ABOVE BUBBLE\n{team1 + ':':<12} {teamA_wab}\t\t\t{team2 + ':':<12} {teamB_wab}")

def tempo_matchup(team1, team2, data, year):
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    teamA_tempo = teamA['KADJ T'].iloc[0] 
    teamB_tempo = teamB['KADJ T'].iloc[0] 
        
    st.write("\n============================================================================================================================")
    st.write(''' ##### TEMPO COMPARISON''')
    st.write()
    st.text(f"ADJUSTED TEMPO\n{team1 + ':':<12} {teamA_tempo:.2f}\t\t\t{team2 + ':':<12} {teamB_tempo:.2f}")

def experience_matchup(team1, team2, data, year):
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    teamA_exp = teamA['EXP'].iloc[0] 
    teamB_exp = teamB['EXP'].iloc[0] 
    
    st.write("\n============================================================================================================================")
    st.write('''  ##### EXPERIENCE COMPARISON''')
    st.write()
    st.text(f"EXPERIENCE\n{team1 + ':':<12} {teamA_exp:.2f}\t\t\t{team2 + ':':<12} {teamB_exp:.2f}")

def off_def_matchup(team1, team2, data, year):

    data['EFG%']  = data['EFG%'].div(100).round(3)
    data['EFG%D'] = data['EFG%D'].div(100).round(3)
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[data['YEAR'] <= year]
    
    min_OE = df['KADJ O'].min() 
    max_OE = df['KADJ O'].max() 
    min_DE = df['KADJ D'].min() 
    max_DE = df['KADJ D'].max() 
    
    min_matchup = np.log(min_OE * min_DE) 
    max_matchup = np.log(max_OE * max_DE) 
    
    teamA_OE = teamA['KADJ O'].iloc[0]
    teamA_DE = teamA['KADJ D'].iloc[0]
    teamA_efgO = teamA['EFG%'].iloc[0]
    teamA_efgD = teamA['EFG%D'].iloc[0]   
    teamB_OE = teamB['KADJ O'].iloc[0]
    teamB_DE = teamB['KADJ D'].iloc[0]
    teamB_efgO = teamB['EFG%'].iloc[0]
    teamB_efgD = teamB['EFG%D'].iloc[0]
    
    AB = np.log(teamA_OE * teamB_DE) 
    AB_net = 2 * ((AB - min_matchup) / (max_matchup - min_matchup)) - 1 
    BA = np.log(teamB_OE * teamA_DE) 
    BA_net = 2 * ((BA - min_matchup) / (max_matchup - min_matchup)) - 1 
    
    st.write("\n============================================================================================================================")
    st.write('''  ##### OFFENSE VS DEFENSE COMPARISON''')
    st.write()
    st.text(f"ADJUSTED OFFENSIVE EFFICIENCY\n{team1 + ':':<12} {teamA_OE:.2f}\t\t\t{team2 + ':':<12} {teamB_OE:.2f}")
    st.write()
    st.text(f"ADJUSTED DEFENSIVE EFFICIENCY\n{team1 + ':':<12} {teamA_DE:.2f}\t\t\t{team2 + ':':<12} {teamB_DE:.2f}")
    st.write()
    st.text(f"OFFENSIVE EFFECTIVE FG%\n{team1 + ':':<12} {teamA_efgO}\t\t\t{team2 + ':':<12} {teamB_efgO}")
    st.write()
    st.text(f"DEFENSIVE EFFECTIVE FG%\n{team1 + ':':<12} {teamA_efgD}\t\t\t{team2 + ':':<12} {teamB_efgD}")
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}")

def twopt_matchup(team1, team2, data, year):

    data['2PT%']  = data['2PT%'].div(100).round(3)
    data['2PT%D'] = data['2PT%D'].div(100).round(3)
    data['2PTR']  = data['2PTR'].div(100).round(3)
    data['2PTRD'] = data['2PTRD'].div(100).round(3)
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[data['YEAR'] <= year]
    
    teamA_o2 = teamA['2PT%'].iloc[0] * teamA['2PTR'].iloc[0]
    teamB_o2 = teamB['2PT%'].iloc[0] * teamB['2PTR'].iloc[0]
    teamA_d2 = teamA['2PT%D'].iloc[0] * teamA['2PTRD'].iloc[0]
    teamB_d2 = teamB['2PT%D'].iloc[0] * teamB['2PTRD'].iloc[0]
    teamA_net2 = teamA_o2 - teamA_d2
    teamB_net2 = teamB_o2 - teamB_d2
    
    min_o2 = (df['2PT%'] * df['2PTR']).min()
    max_o2 = (df['2PT%'] * df['2PTR']).max()
    min_d2 = (df['2PT%D'] * df['2PTRD']).min()
    max_d2 = (df['2PT%D'] * df['2PTRD']).max()
    
    min_matchup = np.log(min_o2 * min_d2) 
    max_matchup = np.log(max_o2 * max_d2) 
    
    AB_net = np.log(teamA_o2 * teamB_d2) 
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    BA_net = np.log(teamB_o2 * teamA_d2) 
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    
    st.write("\n============================================================================================================================")
    st.write('''  ##### TWO POINT COMPARISON''')
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE 2PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}")

def threept_matchup(team1, team2, data, year):

    data['3PT%']  = data['3PT%'].div(100).round(3)
    data['3PT%D'] = data['3PT%D'].div(100).round(3)
    data['3PTR']  = data['3PTR'].div(100).round(3)
    data['3PTRD'] = data['3PTRD'].div(100).round(3)
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[data['YEAR'] <= year]
    
    teamA_o3 = teamA['3PT%'].iloc[0] * teamA['3PTR'].iloc[0]
    teamB_o3 = teamB['3PT%'].iloc[0] * teamB['3PTR'].iloc[0]
    teamA_d3 = teamA['3PT%D'].iloc[0] * teamA['3PTRD'].iloc[0]
    teamB_d3 = teamB['3PT%D'].iloc[0] * teamB['3PTRD'].iloc[0]
    teamA_net3 = teamA_o3 - teamA_d3
    teamB_net3 = teamB_o3 - teamB_d3
    
    min_o3 = (df['3PT%'] * df['3PTR']).min()
    max_o3 = (df['3PT%'] * df['3PTR']).max()
    min_d3 = (df['3PT%D'] * df['3PTRD']).min()
    max_d3 = (df['3PT%D'] * df['3PTRD']).max()
    
    min_matchup = np.log(min_o3 * min_d3) 
    max_matchup = np.log(max_o3 * max_d3) 
    
    AB_net = np.log(teamA_o3 * teamB_d3) 
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    BA_net = np.log(teamB_o3 * teamA_d3) 
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    
    st.write("\n============================================================================================================================")
    st.write('''  ##### THREE POINT COMPARISON''')
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE 3PT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}")

def ft_matchup(team1, team2, data, year):

    data['FT%']   = data['FT%'].div(100).round(3)
    data['FTR']   = data['FTR'].div(100).round(3)
    data['FTRD']  = data['FTRD'].div(100).round(3)
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[data['YEAR'] <= year]
    
    teamA_of = teamA['FT%'].iloc[0] * teamA['FTR'].iloc[0]
    teamB_of = teamB['FT%'].iloc[0] * teamB['FTR'].iloc[0]
    teamA_df = teamB['FT%'].iloc[0] * teamA['FTRD'].iloc[0]
    teamB_df = teamA['FT%'].iloc[0] * teamB['FTRD'].iloc[0]
    teamA_netf = teamA_of - teamA_df
    teamB_netf = teamB_of - teamB_df
    
    min_of = (df['FT%'] * df['FTR']).min()
    max_of = (df['FT%'] * df['FTR']).max()
    min_df = (0.715 * df['FTRD']).min()
    max_df = (0.715 * df['FTRD']).max()
    
    min_matchup = np.log(min_of * min_df) 
    max_matchup = np.log(max_of * max_df) 
    
    AB_net = np.log(teamA_of * teamB_df) 
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    BA_net = np.log(teamB_of * teamA_df) 
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    
    st.write("\n============================================================================================================================")
    st.write('''  ##### FREE THROW COMPARISON''')
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE FT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_scaled:.2f}\t\t\t{team2 + ':':<12} {BA_scaled:.2f}")

def to_matchup(team1, team2, data, year):

    data['TOV%']  = data['TOV%'].div(100).round(3)
    data['TOV%D'] = data['TOV%D'].div(100).round(3)
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[data['YEAR'] <= year]

    min_tov = df['TOV%'].min()   
    max_tov = df['TOV%'].max()   
    min_tovd = df['TOV%D'].min() 
    max_tovd = df['TOV%D'].max() 

    worst = np.log(max_tov * max_tovd) 
    best = np.log(min_tov * min_tovd)  

    teamA_tov = teamA['TOV%'].iloc[0]        
    teamA_tovd = teamA['TOV%D'].iloc[0]      
    teamB_tov = teamB['TOV%'].iloc[0]        
    teamB_tovd = teamB['TOV%D'].iloc[0]      

    AB = np.log(teamA_tov * teamB_tovd) 
    AB_net = 2 * ((AB - worst) / (best - worst)) - 1 
    BA = np.log(teamB_tov * teamA_tovd) 
    BA_net = 2 * ((BA - worst) / (best - worst)) - 1 

    st.write("\n============================================================================================================================")
    st.write('''  ##### TURNOVER COMPARISON''')
    st.write()
    st.text(f"OFFENSIVE TURNOVER PERCENTAGE\n{team1 + ':':<12} {teamA_tov}\t\t\t{team2 + ':':<12} {teamB_tov}") 
    st.write()
    st.text(f"DEFENSIVE TURNOVER PERCENTAGE\n{team1 + ':':<12} {teamA_tovd}\t\t\t{team2 + ':':<12} {teamB_tovd}") 
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSE VS DEFENSE TURNOVER PERCENTAGE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}")

def reb_matchup(team1, team2, data, year):

    data['OREB%'] = data['OREB%'].div(100).round(3)
    data['DREB%'] = data['DREB%'].div(100).round(3)
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[data['YEAR'] <= year]

    min_dreb = df['DREB%'].min() 
    max_dreb = df['DREB%'].max() 
    min_oreb = df['OREB%'].min() 
    max_oreb = df['OREB%'].max() 

    min_matchup = np.log(min_oreb / max_dreb) 
    max_matchup = np.log(max_oreb / min_dreb) 

    teamA_dreb = teamA['DREB%'].iloc[0]     
    teamA_oreb = teamA['OREB%'].iloc[0]     
    teamB_dreb = teamB['DREB%'].iloc[0]     
    teamB_oreb = teamB['OREB%'].iloc[0]     
    teamA_height = teamA['EFF HGT'].iloc[0]  
    teamB_height = teamB['EFF HGT'].iloc[0]  

    AB = np.log(teamA_oreb / teamB_dreb) 
    AB_net = 2 * ((AB - min_matchup) / (max_matchup - min_matchup)) - 1 
    BA = np.log(teamB_oreb / teamA_dreb) 
    BA_net = 2 * ((BA - min_matchup) / (max_matchup - min_matchup)) - 1 

    st.write("\n============================================================================================================================")
    st.write('''  ##### REBOUNDING COMPARISON''')
    st.write()
    st.text(f"EFFECTIVE HEIGHT\n{team1 + ':':<12} {teamA_height:.2f}\t\t\t{team2 + ':':<12} {teamB_height:.2f}") 
    st.write()
    st.text(f"OFFENSIVE REBOUNDING PERCENTAGE\n{team1 + ':':<12} {teamA_oreb}\t\t\t{team2 + ':':<12} {teamB_oreb}") 
    st.write()
    st.text(f"DEFENSIVE REBOUND PERCENTAGE\n{team1 + ':':<12} {teamA_dreb}\t\t\t{team2 + ':':<12} {teamB_dreb}") 
    st.write()
    st.text(f"HEAD TO HEAD MATCHUP\nOFFENSIVE VS DEFENSIVE REBOUNDING (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)\n{team1 + ':':<12} {AB_net:.2f}\t\t\t{team2 + ':':<12} {BA_net:.2f}")
                
data = pd.read_csv("data/data_official.csv")

years = [2025, 2024, 2023, 2022, 2021, 2019]
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

team_matchup(team1, team2, data, year)