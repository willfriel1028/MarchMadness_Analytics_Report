## Impact vs Efficiency Scores

### Factoring Tempo into Predictive Basketball Analytics

Factoring tempo into predictive basketball analytics is very difficult. The main reason for this is because it is very difficult to predict who will control the tempo in a head-to-head matchup. 

For example, if a very fast-paced team plays against a very slow-paced team:

Would the fast-paced team speed up the slow-paced team and control the pace?
Would the slow-paced team slow down the fast-paced team and control the pace?
Would they meet somewhere in the middle, causing a neutral paced game?

Unfortunately these are not questions that I am capable of solving at this moment.

*However*,
     I knew that I wanted pace to play somewhat of a factor when I was designing this program, so I came up with the idea of implementing Impact and Efficiency Scores.
     
### Impact Scores

Impact scores is what I use when tempo **is** predictable.

I think that tempo is predictable when teams are within 5 possessions of each other on average, so when that is the case my program uses impact scores for its win percentage algorithm and it suggests that the user primarily analyzes those scores. 

If teams are within 5 possesions with each other on average, their impact scores should accurately relay how effective they can be at that aspect of the game

Impact scores involve a given team's tempo.

**Calculating Impact Scores**
For this example, we are calculating a team's offensive 3pt impact score:

*3PT% `*` 3PR `*` Adjusted Tempo `*` (1 - TOV%)*

For example, let's calculate 2025 Florida's offensive 3pt impact score:

*0.355 `*` 0.436 `*` 69.5912 `*` (1 - 0.15) = 9.1556*

**What Does It Mean?**

Offensive Impact Scores calculate how much of a certain aspect of basketball (2pt, 3pt, FT) a team is expected to have in a game played at their average tempo.

Defensive Impact Scores calculate how much of a certain aspect of basketball (2pt, 3pt, FT) a team is expected to allow in a game played at their average tempo.

Looking at the previous example, we would expect Florida to score ~9.16 threes in a game played at their tempo.

### Efficiency Scores

Efficiency scores is what I use when tempo **is not** predictable

I think that tempo is not predictable when teams are more than 5 possessions apart on average, so when that is the case my program uses efficiency scores for its win percentage algorithm and it suggests that the user primarily analyzes those scores.

If teams are more than 5 possessions apart on average, I find that most fast-paced teams have a huge advantage in comparison with their impact scores. Since we do not know which team exactly will control the tempo in a lot of games, I decided to create efficiency scores.

Efficiency scores involve a fixed tempo of 66, which is about average.

To ensure fairness among all teams, efficiency scores are used as features in the logistic and poisson regression models.

**Calculating Efficiency Scores**

For this example, we are calculating a team's offensive 3pt efficiency score:

*3PT% `*` 3PR `*` 66 `*` (1 - TOV%)*

For example, let's calculate 2025 Florida's offensive 3pt efficiency score:

*0.355 `*` 0.436 `*` 66 `*` (1-0.15) = 8.6832*

**What Does It Mean?**

Offensive Efficiency Scores calculate how much of a certain aspect of basketball (2pt, 3pt, FT) a team is expected to have in a game played at an average tempo.

Defensive Efficiency Scores calculate how much of a certain aspect of basketball (2pt, 3pt, FT) a team is expected to allow in a game played at an average tempo.

Looking at the previous example, we would expect Florida to score ~8.68 threes in a game played at an average tempo.

### How could this be improved?

Something I will definitely be looking to do going forward is to improve this system.

I think a system I will look to implement into my program in the future would probably be to display:

Team A's expected (2pt, 3pt, FT) at Team A's tempo:
    
*3PT% `*` 3PR `*` TeamA Tempo `*` (1 - TOV%)*
    
Team A's expected (2pt, 3pt, FT) at Team B's tempo:
    
*3PT% `*` 3PR `*` TeamB Tempo `*` (1 - TOV%)*
    
Team A's expected (2pt, 3pt, FT) at the average of Team A and Team B's tempo:
    
*3PT% `*` 3PR `*` TeamA Tempo `*` (1 - TOV%)*
    
And vice versa.

This would also allow for the user to possibly make a guess on who they think will dominate the tempo. They could see who has an advantage in these categories if either team were to dominate tempo, or if they met in the middle.

In this scenario, I would probably use the average tempo for the win percentage algorithm.