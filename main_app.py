import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from itertools import zip_longest
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
            with open(f'bracket_sims/RFModels/rf_model_{year}.pkl', 'rb') as f:
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
    grouped = df.groupby("TEAM")["Sim_Wins"].apply(list).reset_index()
    
    dfc = pd.DataFrame()
    dfc["TEAM"] = grouped["TEAM"]
    dfc["R32"] = grouped["Sim_Wins"].apply(lambda x: sum(1 for w in x if w >= 1))
    dfc["S16"] = grouped["Sim_Wins"].apply(lambda x: sum(1 for w in x if w >= 2))
    dfc["E8"]  = grouped["Sim_Wins"].apply(lambda x: sum(1 for w in x if w >= 3))
    dfc["F4"]  = grouped["Sim_Wins"].apply(lambda x: sum(1 for w in x if w >= 4))
    dfc["CG"]  = grouped["Sim_Wins"].apply(lambda x: sum(1 for w in x if w >= 5))
    dfc["WIN"] = grouped["Sim_Wins"].apply(lambda x: sum(1 for w in x if w == 6))

    dfc['R32_rank'] = dfc['R32'].rank(ascending=False, method='min').astype(int)
    dfc['S16_rank'] = dfc['S16'].rank(ascending=False, method='min').astype(int)
    dfc['E8_rank']  = dfc['E8'].rank(ascending=False, method='min').astype(int)
    dfc['F4_rank']  = dfc['F4'].rank(ascending=False, method='min').astype(int)
    dfc['CG_rank']  = dfc['CG'].rank(ascending=False, method='min').astype(int)
    dfc['WIN_rank'] = dfc['WIN'].rank(ascending=False, method='min').astype(int)

    return dfc.reset_index(drop=True)       
    
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
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[(data['YEAR'] >= (year - 5)) & (data['YEAR'] <= year)]
    
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
    st.text(f"OFFENSIVE EFFECTIVE FG%\n{team1 + ':':<12} {teamA_efgO}%\t\t\t{team2 + ':':<12} {teamB_efgO}%")
    st.write()
    st.text(f"DEFENSIVE EFFECTIVE FG%\n{team1 + ':':<12} {teamA_efgD}%\t\t\t{team2 + ':':<12} {teamB_efgD}%")
    st.write()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[float(BA_net)],
        y=[team2],
        orientation='h',
        name=team2,
        marker_color='green' if float(BA_net) > 0 else 'red'
    ))
    
    fig.add_trace(go.Bar(
        x=[float(AB_net)],
        y=[team1],
        orientation='h',
        name=team1,
        marker_color='green' if float(AB_net) > 0 else 'red'
    ))
    
    fig.update_layout(
        title="Predicted Offensive Scoring Advantage In This Game",
        xaxis=dict(
            range=[-1.1, 1.1], 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='black'
        ),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14, color='black'))
    )

    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig)

def twopt_matchup(team1, team2, data, year):
    
    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[(data['YEAR'] >= (year - 5)) & (data['YEAR'] <= year)]

    teamA_perc = teamA['2PT%'].iloc[0]
    teamA_rate = teamA['2PTR'].iloc[0]
    teamB_perc = teamB['2PT%'].iloc[0]
    teamB_rate = teamB['2PTR'].iloc[0]
    
    teamA_o2 = teamA_perc * teamA_rate
    teamB_o2 = teamB_perc * teamB_rate
    
    teamA_dperc = teamA['2PT%D'].iloc[0]
    teamA_drate = teamA['2PTRD'].iloc[0]
    teamB_dperc = teamB['2PT%D'].iloc[0]
    teamB_drate = teamB['2PTRD'].iloc[0]
    
    teamA_d2 = teamA_dperc * teamA_drate
    teamB_d2 = teamB_dperc * teamB_drate
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[float(BA_scaled)],
        y=[team2],
        orientation='h',
        name=team2,
        marker_color='green' if float(BA_scaled) > 0 else 'red'
    ))
    
    fig.add_trace(go.Bar(
        x=[float(AB_scaled)],
        y=[team1],
        orientation='h',
        name=team1,
        marker_color='green' if float(AB_scaled) > 0 else 'red'
    ))
    
    fig.update_layout(
        title="Predicted Offensive 2PT Scoring Advantage In This Game",
        xaxis=dict(
            range=[-1.1, 1.1], 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='black'
        ),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14, color='black'))
    )

    st.write("\n============================================================================================================================")
    st.write('''  ##### TWO POINT COMPARISON''')
    st.write()
    st.text(f"OFFENSIVE STATS")
    st.text(f"2 POINT PERCENTAGE\n{team1 + ':':<12} {teamA_perc:.1f}%\t\t\t{team2 + ':':<12} {teamB_perc:.1f}%")
    st.text(f"2 POINT ATTEMPT RATE\n{team1 + ':':<12} {teamA_rate:.1f}%\t\t\t{team2 + ':':<12} {teamB_rate:.1f}%")
    st.write("")
    st.write("")
    st.text(f"DEFENSIVE STATS")
    st.text(f"2 POINT PERCENTAGE\n{team1 + ':':<12} {teamA_dperc:.1f}%\t\t\t{team2 + ':':<12} {teamB_dperc:.1f}%")
    st.text(f"2 POINT ATTEMPT RATE\n{team1 + ':':<12} {teamA_drate:.1f}%\t\t\t{team2 + ':':<12} {teamB_drate:.1f}%")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig)

def threept_matchup(team1, team2, data, year):

    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[(data['YEAR'] >= (year - 5)) & (data['YEAR'] <= year)]
    
    teamA_perc = teamA['3PT%'].iloc[0]
    teamA_rate = teamA['3PTR'].iloc[0]
    teamB_perc = teamB['3PT%'].iloc[0]
    teamB_rate = teamB['3PTR'].iloc[0]
    
    teamA_o3 = teamA_perc * teamA_rate
    teamB_o3 = teamB_perc * teamB_rate
    
    teamA_dperc = teamA['3PT%D'].iloc[0]
    teamA_drate = teamA['3PTRD'].iloc[0]
    teamB_dperc = teamB['3PT%D'].iloc[0]
    teamB_drate = teamB['3PTRD'].iloc[0]
    
    teamA_d3 = teamA_dperc * teamA_drate
    teamB_d3 = teamB_dperc * teamB_drate
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[float(BA_scaled)],
        y=[team2],
        orientation='h',
        name=team2,
        marker_color='green' if float(BA_scaled) > 0 else 'red'
    ))
    
    fig.add_trace(go.Bar(
        x=[float(AB_scaled)],
        y=[team1],
        orientation='h',
        name=team1,
        marker_color='green' if float(AB_scaled) > 0 else 'red'
    ))
    
    fig.update_layout(
        title="Predicted Offensive 3PT Scoring Advantage In This Game",
        xaxis=dict(
            range=[-1.1, 1.1], 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='black'
        ),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14, color='black'))
    )

    st.write("\n============================================================================================================================")
    st.write('''  ##### THREE POINT COMPARISON''')
    st.write()
    st.text(f"OFFENSIVE STATS")
    st.text(f"3 POINT PERCENTAGE\n{team1 + ':':<12} {teamA_perc:.1f}%\t\t\t{team2 + ':':<12} {teamB_perc:.1f}%")
    st.text(f"3 POINT ATTEMPT RATE\n{team1 + ':':<12} {teamA_rate:.1f}%\t\t\t{team2 + ':':<12} {teamB_rate:.1f}%")
    st.write("")
    st.write("")
    st.text(f"DEFENSIVE STATS")
    st.text(f"3 POINT PERCENTAGE\n{team1 + ':':<12} {teamA_dperc:.1f}%\t\t\t{team2 + ':':<12} {teamB_dperc:.1f}%")
    st.text(f"3 POINT ATTEMPT RATE\n{team1 + ':':<12} {teamA_drate:.1f}%\t\t\t{team2 + ':':<12} {teamB_drate:.1f}%")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig)

def ft_matchup(team1, team2, data, year):

    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[(data['YEAR'] >= (year - 5)) & (data['YEAR'] <= year)]
    
    teamA_perc = teamA['FT%'].iloc[0]
    teamA_rate = teamA['FTR'].iloc[0]
    teamB_perc = teamB['FT%'].iloc[0]
    teamB_rate = teamB['FTR'].iloc[0]
    
    teamA_of = teamA_perc * teamA_rate
    teamB_of = teamB_perc * teamB_rate
    
    teamA_dperc = 71.5
    teamA_drate = teamA['FTRD'].iloc[0]
    teamB_dperc = 71.5
    teamB_drate = teamB['FTRD'].iloc[0]
    
    teamA_df = teamA_dperc * teamA_drate
    teamB_df = teamB_dperc * teamB_drate
    teamA_netf = teamA_of - teamA_df
    teamB_netf = teamB_of - teamB_df
    
    min_of = (df['FT%'] * df['FTR']).min()
    max_of = (df['FT%'] * df['FTR']).max()
    min_df = (71.5 * df['FTRD']).min()
    max_df = (71.5 * df['FTRD']).max()
    
    min_matchup = np.log(min_of * min_df) 
    max_matchup = np.log(max_of * max_df) 
    
    AB_net = np.log(teamA_of * teamB_df) 
    AB_scaled = 2 * ((AB_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    BA_net = np.log(teamB_of * teamA_df) 
    BA_scaled = 2 * ((BA_net - min_matchup) / (max_matchup - min_matchup)) - 1 
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[float(BA_scaled)],
        y=[team2],
        orientation='h',
        name=team2,
        marker_color='green' if float(BA_scaled) > 0 else 'red'
    ))
    
    fig.add_trace(go.Bar(
        x=[float(AB_scaled)],
        y=[team1],
        orientation='h',
        name=team1,
        marker_color='green' if float(AB_scaled) > 0 else 'red'
    ))
    
    fig.update_layout(
        title="Predicted Offensive FT Scoring Advantage In This Game",
        xaxis=dict(
            range=[-1.1, 1.1], 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='black'
        ),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14, color='black'))
    )

    st.write("\n============================================================================================================================")
    st.write('''  ##### FREE THROW COMPARISON''')
    st.write()
    st.text(f"OFFENSIVE STATS")
    st.text(f"FREE THROW PERCENTAGE\n{team1 + ':':<12} {teamA_perc:.1f}%\t\t\t{team2 + ':':<12} {teamB_perc:.1f}%")
    st.text(f"FREE THROW ATTEMPT RATE\n{team1 + ':':<12} {teamA_rate:.1f}%\t\t\t{team2 + ':':<12} {teamB_rate:.1f}%")
    st.write("")
    st.write("")
    st.text(f"DEFENSIVE STATS")
    st.text(f"FREE THROW ATTEMPT RATE\n{team1 + ':':<12} {teamA_drate:.1f}%\t\t\t{team2 + ':':<12} {teamB_drate:.1f}%")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig)

def to_matchup(team1, team2, data, year):

    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[(data['YEAR'] >= (year - 5)) & (data['YEAR'] <= year)]

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

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[float(BA_net)],
        y=[team2],
        orientation='h',
        name=team2,
        marker_color='green' if float(BA_net) > 0 else 'red'
    ))
    
    fig.add_trace(go.Bar(
        x=[float(AB_net)],
        y=[team1],
        orientation='h',
        name=team1,
        marker_color='green' if float(AB_net) > 0 else 'red'
    ))
    
    fig.update_layout(
        title="Predicted Offensive Turnover Advantage In This Game",
        xaxis=dict(
            range=[-1.1, 1.1], 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='black'
        ),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14, color='black'))
    )

    st.write("\n============================================================================================================================")
    st.write('''  ##### TURNOVER COMPARISON''')
    st.write()
    st.text(f"OFFENSIVE TURNOVER PERCENTAGE\n{team1 + ':':<12} {teamA_tov}%\t\t\t{team2 + ':':<12} {teamB_tov}%") 
    st.write()
    st.text(f"DEFENSIVE TURNOVER PERCENTAGE\n{team1 + ':':<12} {teamA_tovd}%\t\t\t{team2 + ':':<12} {teamB_tovd}%") 

    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig)

def reb_matchup(team1, team2, data, year):

    teamA = data[(data['YEAR'] == year) & (data['TEAM'] == team1)] 
    teamB = data[(data['YEAR'] == year) & (data['TEAM'] == team2)] 
    
    df = data[(data['YEAR'] >= (year - 5)) & (data['YEAR'] <= year)]

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

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[float(BA_net)],
        y=[team2],
        orientation='h',
        name=team2,
        marker_color='green' if float(BA_net) > 0 else 'red'
    ))
    
    fig.add_trace(go.Bar(
        x=[float(AB_net)],
        y=[team1],
        orientation='h',
        name=team1,
        marker_color='green' if float(AB_net) > 0 else 'red'
    ))
    
    fig.update_layout(
        title="Predicted Offensive Rebounding Advantage In This Game",
        xaxis=dict(
            range=[-1.1, 1.1], 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='black'
        ),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14, color='black'))
    )

    st.write("\n============================================================================================================================")
    st.write('''  ##### REBOUNDING COMPARISON''')
    st.write()
    st.text(f"EFFECTIVE HEIGHT\n{team1 + ':':<12} {teamA_height:.2f}\t\t\t{team2 + ':':<12} {teamB_height:.2f}") 
    st.write()
    st.text(f"OFFENSIVE REBOUNDING PERCENTAGE\n{team1 + ':':<12} {teamA_oreb}%\t\t\t{team2 + ':':<12} {teamB_oreb}%") 
    st.write()
    st.text(f"DEFENSIVE REBOUND PERCENTAGE\n{team1 + ':':<12} {teamA_dreb}%\t\t\t{team2 + ':':<12} {teamB_dreb}%") 
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig)

def show_round(df, rd):

    if rd == "Round of 32":
        graph_round(df, rd, 1)
        print_round(df, rd, 1)
    elif rd == "Sweet 16":
        graph_round(df, rd, 2)
        print_round(df, rd, 2)
    elif rd == "Elite 8":
        graph_round(df, rd, 3)
        print_round(df, rd, 3)
    elif rd == "Final 4":
        graph_round(df, rd, 4)
        print_round(df, rd, 4)
    elif rd == "Championship Game":
        graph_round(df, rd, 5)
        print_round(df, rd, 5)
    elif rd == "National Champion":
        graph_round(df, rd, 6)
        print_round(df, rd, 6)

def graph_round(df, rd, x):
    champ_counts = df.loc[df['Sim_Wins'] >= x, 'TEAM'].value_counts()
    champ_pct = (champ_counts / 10000 * 100).round(2).loc[lambda x: x > 0]

    fig, ax = plt.subplots(figsize=(12, min(len(champ_pct) * 0.2, 12)))
    champ_pct[::-1].plot(kind='barh', ax=ax)
    ax.set_title(rd + " Chance Per Team")
    ax.set_ylabel("Team")
    ax.set_xlabel(rd + " %")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for i, v in enumerate(champ_pct[::-1]):
        ax.text(v + 0.05, i, f"{v}%", va='center')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def print_round(df, rd, q):
    teams = list(df["TEAM"].unique())
    cc = []
    for team in teams:
        x = df[(df["TEAM"] == team) & (df["Sim_Wins"] >= q)]
        dfx = {}
        dfx["TEAM"] = team
        dfx["Seed"] = np.mean(x["SEED"])
        if len(x) == 0:
            dfx["Seed"] = np.mean(df[(df["TEAM"] == team)]["SEED"])
        dfx["count"] = len(x)
        cc.append(dfx)

    counts = pd.DataFrame(cc).sort_values(["count", "Seed"], ascending=[False, True]).reset_index(drop=True)
    counts["Pct"] = (counts["count"] / 10000 * 100).round(2)

    ranks = []
    rank = 1
    for i, row in counts.iterrows():
        if i > 0 and row["Pct"] < counts.iloc[i-1]["Pct"]:
            rank = i + 1
        ranks.append(rank)
    counts["Rank"] = ranks

    st.subheader(rd + " Chance Per Team")

    n = len(counts)
    chunk = n // 3 + (1 if n % 3 else 0)
    left   = counts.iloc[:chunk]
    middle = counts.iloc[chunk:chunk*2]
    right  = counts.iloc[chunk*2:]

    col1, col2, col3 = st.columns(3)
    for col, chunk_df in zip([col1, col2, col3], [left, middle, right]):
        with col:
            for _, row in chunk_df.iterrows():
                #st.text(f"{int(row['Rank']):>2}. {row['TEAM']}({int(row['Seed'])}){'':<18}{row['Pct']:>5}%")
                st.text(f"{int(row['Rank']):>2}. {f'{row[chr(84)+(chr(69)+chr(65)+chr(77))]}({int(row[chr(83)+chr(101)+chr(101)+chr(100)])})':<30}{row['Pct']:>5}%")
                
                
data = pd.read_csv("data/data_official_26.csv")

years = [2026, 2025, 2024, 2023, 2022, 2021, 2019, 2018, 2017, 2016, 2015]

view = st.radio("View", ["Matchup Analysis", "Field Analysis"], horizontal=True)

if view == "Matchup Analysis":

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
    with col3:
        st.write("") 
        st.write("")
        button = st.button("Go")
    
    if button:
        team_matchup(team1, team2, data, year)

else:
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        year = st.selectbox("Pick a year", options=years)
        
    rds = ["Round of 32", "Sweet 16", "Elite 8", "Final 4", "Championship Game", "National Champion"]
    co1, co2, co3 = st.columns([2,1,2])
    with co2:
        rd = st.selectbox("Pick a round", options=rds)
    with co3:
        st.write("") 
        st.write("")
        button = st.button("Go")

    if button:
        dat = pd.read_csv("data/" + str(year) + "_10000sims0.csv")
        x1, x2, x3 = st.columns([1,3,1])
        with x2:
            show_round(dat, rd)

    