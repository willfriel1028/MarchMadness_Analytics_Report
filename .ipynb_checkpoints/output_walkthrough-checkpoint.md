This file contains a walkthrough/explanation of what an output could look like. For this walkthrough, I decided to take a closer look at this year's (2025) second round matchup between Arizona (4) and Oregon (5). I chose this matchup because from what I saw across a lot of brackets was that people were pretty split on this matchup. Thus, I decided to take a closer look to examine it.

## Arizona(4) vs. Oregon(5)
```
_______________________________________________

STRENGTH OF SCHEDULE COMPARISON

ELITE SOS
Arizona:     44.259    Oregon:      35.999

WINS ABOVE BUBBLE
Arizona:     5.4    Oregon:      5.3
_______________________________________________
```
This section highlights how challenging each team’s schedule was (Elite SOS) and how well each team performed relative to a typical bubble team (WAB).

**Matchup-specific insights**

Arizona faced tougher competition this season.
Both teams performed very similarly when compared to how a bubble team would perform against their respective schedules.

Since both teams played relatively difficult competition and performed similarly, neither team gains a significant edge here.

**Overall insights**

Elite SOS is mainly important to keep in mind when analyzing statistics further down in the output. If a team has a high Elite SOS and great raw statistics, they are considered the best of the best in that category. However, if a team has a low Elite SOS and great raw statistics, keep in mind that their weaker schedule could be inflating their performance.

In a situation where a team has a significantly higher WAB, it shows that they performed much better throughout the season compared to the other team, even if their statistics seem close.


```
_______________________________________________

TEMPO COMPARISON

ADJUSTED TEMPO
Arizona:     69.93    Oregon:      67.67
Suggestion: Look at the teams' respective Impact Scores
_______________________________________________
```
This section displays each team's adjusted tempo, as well as whether these teams' impact or efficiency scores should be analyzed.

*For more information about impact/efficiency scores and factoring tempo into matchups, check impact_vs_efficiency.md*

**Matchup-specific insights**

Both teams play at a similar pace - tempo will likely not be a major factor in this game.

We should prioritize looking at impact scores for the rest of the output.

**Overall insights**

Tempo alone will not tell you too much generally unless there is a significant gap between tempos (think >10 difference). It is then up to the user to determine which team, if any, will dominate the tempo in that matchup.


```
_______________________________________________

OFFENSE VS DEFENSE COMPARISON

ADJUSTED OFFENSIVE EFFICIENCY
Arizona:     122.57    Oregon:      116.64

ADJUSTED DEFENSIVE EFFICIENCY
Arizona:     96.82    Oregon:      96.49

OFFENSIVE EFFECTIVE FG%
Arizona:     0.529    Oregon:      0.522

DEFENSIVE EFFECTIVE FG%
Arizona:     0.486    Oregon:      0.491

HEAD TO HEAD MATCHUP
OFFENSE VS DEFENSE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     0.20    Oregon:      0.02
_______________________________________________
```
This is one of the most telling sections of the output. It displays each teams Adjusted Offensive/Defensive Efficiency, their Offensive/Defensive Effective FG%, and their Offense vs Defense head-to-head matchup score.

This head-to-head matchup score, gives you an idea of how well each teams' offense matches up with the opposing defense. 

**Matchup-specific insights**

Arizona has an advantage in terms of offensive efficiency
Both teams are very similar in terms of defensive efficiency

Both offensive/defensive EFG% are very similar as well.

The head-to-head matchup scores tell us that Arizona's offense has a slight advantage compared to Oregon's offense. 
It is important to keep in mind that the .18 difference is not as big as it may seem, considering this is a [-1,1] scale.

Arizona’s score of 0.20 suggests they hold a slight offensive advantage when matched up against Oregon’s defense
Oregon’s score of 0.02 is very close to 0, which suggests that their offense and Arizona’s defense are roughly evenly matched

We can conclude that Arizona's offense has a VERY slight advantage over Oregon's offense from this.

**Overall insights**

Both adjusted offensive and defensive efficiency come from KenPom, so it is okay to analyze both at face-value, because strength of opposing defenses is calculated in the score.

Effective offensive/defensive FG% is useful to see how effective a team's offense/defense is, but it does not take strength of opponents into consideration.

From what I have found, off/def EFG% can be useful to help detect upsets, especially if it is a mid-major team playing a Power 5 team. For example, it is common for most Power 5 teams to have higher Adj. Off. Efficiency than most mid-major teams, because they play much tougher opponents. With that being said, if you notice a mid-major team has a slightly lower Adj. Off. Efficiency than a Power 5 team, and they have a significantly higher EFG%, it could be an indicator they actually do have a better offense, but they have not had the opportunity to show it off against top tier defenses. The same logic applies for defense, as well.

It is important to note that not all mid-major teams with a higher off/def EFG% than a Power 5 team will not automatically be better. It is always a guessing game as we know, and it is important to use these stats more as context clues than end-all-be-all.

Each team's Offense vs Defense matchup scores are very telling. It takes the into account the strength of one team's offense against the strength of the other team's defense, and returns a score on a [-1,1] scale. This score represents how well that team's offense should perform against the other team's defense. 

The matchup score uses Adj Off/Def Efficiencyies, so they do take strength of opponents into account.

For these reasons, Offense vs Defense matchup score is the most heavily weighted factor in the Win Percentage algorithm.


```
_______________________________________________

TWO POINT COMPARISON

NET 2PT IMPACT SCORE (Shows estimated 2pt margin for team given adjusted tempo)
Arizona:     5.12    Oregon:      1.00

NET 2PT EFFICIENCY SCORE (Shows estimated 2pt margin for team given a fixed tempo)
Arizona:     4.84    Oregon:      0.98

HEAD TO HEAD MATCHUP
OFFENSE VS DEFENSE IMPACT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     0.46    Oregon:      0.01

OFFENSE VS DEFENSE EFFICIENCY SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     0.46    Oregon:      -0.09
_______________________________________________
```
This section displays each team's net 2pt impact/efficiency score, as well as their 2pt Offense vs Defense impact/efficiency scores.

The net 2pt scores show how many more 2-pointers a team will generally score compared to their opponents.

This head-to-head matchup score gives you an idea of how well each team should be able to score 2-pointers against the opposing team's 2-point defense.

**Matchup-specific insights**

Arizona generally scores 5 more 2-pointers than their opponent, a significant margin
Oregon only will score 1 more 2-pointer than their opponent, a very slight edge

The head-to-head matchup scores show us that Arizona's offense should have a significant advantage in generating 2pt shots compared to Oregon's offense.
The 0.45 difference in score is pretty significant on a [-1,1] scale.

Arizona's score of 0.46 suggests that they have an advantage in scoring 2-pointers against Oregon's 2-point defense.
Oregon's score of 0.01 means that their offense does not particularly have an advantage or disadvantage scoring 2-pointers against Arizona's 2-point defense.

We can conclude that Arizona is expected to have an advantage at scoring 2-pointers over Oregon.

**Overall insights**

The statistics this section is calculated with do not particularly take strength of opponent's defense into consideration, so make sure to keep each team's Elite SOS in the back of your mind when analyzing this. 

If one team appears to be significantly better than the other AND they have a better Elite SOS, then that is probably a solid indication that they will outperform the other team in terms of 2 pointers.

If one team appears to be significantly better than the other and has a worse Elite SOS, they might still be the better 2pt team, but their stats could be inflated by a weaker schedule.

This section largely represents each team's offensive playstyle, which could possibly still translate for a worse team playing a better team, it is up for the user to interpret.


```
_______________________________________________

THREE POINT COMPARISON

NET 3PT IMPACT SCORE (Shows estimated 3pt margin for team given adjusted tempo)
Arizona:     -1.64    Oregon:      0.83

NET 3PT EFFICIENCY SCORE (Shows estimated 3pt margin for team given a fixed tempo)
Arizona:     -1.55    Oregon:      0.81

HEAD TO HEAD MATCHUP
OFFENSE VS DEFENSE IMPACT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     -0.13    Oregon:      0.36

OFFENSE VS DEFENSE EFFICIENCY SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     -0.28    Oregon:      0.23
_______________________________________________
```
This section displays each team's net 3pt impact/efficiency score, as well as their 3pt Offense vs Defense impact/efficiency scores.

The net 3pt scores show how many more 3-pointers a team will generally score compared to their opponents.

This head-to-head matchup score gives you an idea of how well each team should be able to score 3-pointers against the opposing team's 3-point defense.

**Matchup-specific insights**

Arizona generally scores 1.6 less 3-pointers than their opponent, a relatively slight deficit.
Oregon generally scores 0.83 more 3-pointers than their opponent, which is pretty even.

The head-to-head matchup scores shows us that Oregon should have a significant advantage in scoring 3pt shots compared to Arizona's offense.
The 0.49 difference in score is pretty significant.

Arizona's score of -0.13 suggests that their offense has a very slight disadvantage at generating 3-pointers against Oregon's defense.
Oregon's score of 0.36 suggests their offense has a decent advantage at generating 3-pointers against Arizona.

We can conclude that Oregon is expected to have an advantage at scoring 3-pointers over Arizona.

**Overall insights**

The statistics this section is calculated with do not particularly take strength of opponent's defense into consideration, so make sure to keep each team's Elite SOS in the back of your mind when analyzing this. 

If one team appears to be significantly better than the other AND they have a better Elite SOS, then that is probably a solid indication that they will outperform the other team in terms of 3 pointers.

If one team appears to be significantly better than the other and has a worse Elite SOS, they might still be the better 3pt team, but their stats could be inflated by a weaker schedule.

This section largely represents each team's offensive playstyle, which could possibly still translate for a worse team playing a better team, it is up for the user to interpret.


```
_______________________________________________

FREE THROW COMPARISON

NET FT IMPACT SCORE (Shows estimated FT margin for team given adjusted tempo)
Arizona:     4.64    Oregon:      3.41

NET FT EFFICIENCY SCORE (Shows estimated FT margin for team given a fixed tempo)
Arizona:     4.38    Oregon:      3.32

HEAD TO HEAD MATCHUP
OFFENSE VS DEFENSE IMPACT SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     0.33    Oregon:      0.26

OFFENSE VS DEFENSE EFFICIENCY SCORE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     0.26    Oregon:      0.19
_______________________________________________
```
This section displays each team's net free throw impact/efficiency score, as well as their Free Throw Offense vs Defense impact/efficiency scores.

The net FT scores show how many more free throws a team will generally score compared to their opponents.

This head-to-head matchup score gives you an idea of how well each team should be able to score free throws against the opposing team's free throw defense.

**Matchup-specific insights**

Arizona generally scores 4.64 more free throws than their opponent, which is a solid margin
Oregon will score 3.41 more free throws than their opponent, which is also pretty solid

The head-to-head matchup scores show us that both teams are pretty even in terms of scoring free throws aginst one another.
The 0.07 difference in matchup scores is pretty insignificant on a [-1,1] scale. 

Arizona's score of 0.33 suggests their offense has a slight advantage at generating free throws against Oregon's defense
Oregon's score of 0.26 suggests their offense has a slight advantage at generating free throws against Arizona's defense

We can not confidently conclude who will score more free throws from this.

**Overall insights**

The statistics this section is calculated with do not particularly take strength of opponent's defense into consideration, so make sure to keep each team's Elite SOS in the back of your mind when analyzing this. 

If one team appears to be significantly better than the other AND they have a better Elite SOS, then that is probably a solid indication that they will outperform the other team in terms of free throws.

If one team appears to be significantly better than the other and has a worse Elite SOS, they might still be the better FT team, but their stats could be inflated by a weaker schedule.

This section largely represents how well each team gets to the free throw line while not allowing their opponent to get to the free throw line. This is something that could translate for a weaker team playing a stronger team, it is up for the user to interpret this.


```
_______________________________________________

TURNOVER COMPARISON

OFFENSIVE TURNOVER PERCENTAGE
Arizona:     0.162    Oregon:      0.162

DEFENSIVE TURNOVER PERCENTAGE
Arizona:     0.166    Oregon:      0.176

HEAD TO HEAD MATCHUP
OFFENSE VS DEFENSE TURNOVER PERCENTAGE (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     -0.06    Oregon:      0.04
_______________________________________________
```
This section displays each teams offensive/defensive turnover percentage, as well as their TO Offense vs Defense score.

This head-to-head matchup score gives you an idea of how each offense should be able to prevent turning the ball over against the opposing defense.

**Matchup-specific insights**

Both teams have the same offensive turnover percentage
Oregon has the slightly better defensive turnover percentage

The head-to-head matchup scores suggests that both teams are relatively even in terms of turning the ball over.
Even though Arizona is in the negative and Oregon is positive, the 0.1 difference is relatively insignificant on a [-1,1] scale. 

Arizona's score of -0.06 suggests their offense does not have an advantage or disadvantage in terms of turnovers against Oregon's defense.
Oregon's score of 0.04 suggests their offense does not have an advantage or disadvantage in terms of turnovers against Arizona's defense.

We can not confidently conclude who will turn the ball over less from this.

**Overall insights**

The statistics this section is calculated with do not particularly take strength of opponent's defense into consideration, so make sure to keep each team's Elite SOS in the back of your mind when analyzing this. 

If one team appears to be significantly better than the other AND they have a better Elite SOS, then that is probably a solid indication that they will outperform the other team in terms of turnovers.

If one team appears to be significantly better than the other and has a worse Elite SOS, they might still be the better TO team, but their stats could be inflated by a weaker schedule.

This section largely shows off which teams tend to take care of the ball and force turnovers. If a weaker team is elite in either of those categories, there is a pretty good chance it could translate when playing a stronger team.


```
_______________________________________________

REBOUNDING COMPARISON

EFFECTIVE HEIGHT
Arizona:     80.50    Oregon:      81.55

OFFENSIVE REBOUNDING PERCENTAGE
Arizona:     0.359    Oregon:      0.306

DEFENSIVE REBOUND PERCENTAGE
Arizona:     0.715    Oregon:      0.708

HEAD TO HEAD MATCHUP
OFFENSIVE VS DEFENSIVE REBOUNDING (Closer to -1: Disadvantage, Closer to 0: Even, Closer to 1: Advantage)
Arizona:     0.52    Oregon:      0.15
_______________________________________________
```
This section displays each team's effective height, offensive/defensive rebounding percentage, and their Rebounding Offense vs Defense score.

Effective Height is the average height of players that play significant minutes for each team.

This head-to-head matchup score gives an idea of how well each team should be able to grab offensive rebounds against the other team.

**Matchup-specific insights**

Oregon is a slightly taller team overall.

Arizona is a significantly better team at offensive rebounding than Oregon.
Both teams are relatively similar in terms of defensive rebounding.

The head-to-head matchup score suggests that Arizona should have an easier time grabbing offensive rebounds than Oregon.
The score difference of 0.37 is pretty significant.

Arizona's score of 0.52 suggests they have an advantage at grabbing offensive rebounds against Oregon's defense.
Oregon's score of 0.15  suggests they have a very slight advantage at grabbing offensive rebounds against Arizona's defense.

We can conclude that Arizona is likely to grab more offensive rebounds from this.

**Overall insights**

The statistics this section is calculated with do not particularly take strength of opponent's defense into consideration, so make sure to keep each team's Elite SOS in the back of your mind when analyzing this. 

If one team appears to be significantly better than the other AND they have a better Elite SOS, then that is probably a solid indication that they will outperform the other team in terms of rebounds.

If one team appears to be significantly better than the other and has a worse Elite SOS, they might still be the better rebounding team, but their stats could be inflated by a weaker schedule.

This section largely shows off which teams hustle more for offensive and defensive rebounding. If a weaker team shows that they go for rebounds aggressively it could translate going against a stronger team.


```
_______________________________________________

WIN PERCENTAGE
Arizona:     56.9 %    Oregon:      43.1 %
_______________________________________________
```
This section displays each team's win percentage. (Their chance to win the game)

**Matchup specific insights**

According to the algorithm,
     
   Arizona has a 56.9% chance to win this game.
   Oregon has a 43.1% chance to win this game.
   
This matchup appears to be very close, which we would expect to see in a 4 vs 5 matchup. Overall, Arizona has a slight advantage, but according to the algorithm this appears to be a toss-up game.

**Overall insights**

This is a statistically-driven algorithm which takes all previously mentioned factors into account. It uses all of the head-to-head scores as well as strength of schedule and tempo to calculate this win percentage.

Strength of schedule is adequately measured in this algorithm, with its two biggest factors being Offense vs Defense matchup score and Wins above Bubble. Both of these have a lot to do with strength of opponents.

I personally interpret two teams' win percentages like this (generally):

   Team A: 50-60%     Team B: 40-50%
   A true toss-up game.
   
   Team A: 60-68%     Team B: 32-40%
   Anything could happen, but heavily lean towards Team A
   
   Team A: 68-85%     Team B: 15-32%
   More than likely to pick Team A. Would need to see a significant statistical edge somewhere for Team B to pick them.
   
   Team A: 85%+       Team B: 15% or less
   Almost certainly would pick Team A to win.

Seed is important to keep in mind here, however. 
For example, a 5 vs 12 game would usually have the 5 seed having a 65-70% chance of winning. If you run this program and see that a 5 seed only has a 55% chance of winning, that could help you decide if you want to pick a 12 over 5 upset in your bracket.


```
_______________________________________________

KEEP IN MIND:  Arizona(4) is expected to rank between 13-16 .  Oregon(5) is expected to rank between 17-20

ROUND OF 32 RANK
Arizona:     11/68    Oregon:      31/68

SWEET 16 RANK
Arizona:     11/68    Oregon:      29/68

ELITE 8 RANK
Arizona:     12/68    Oregon:      30/68

FINAL 4 RANK
Arizona:     13/68    Oregon:      31/68

NATIONAL CHAMPION RANK
Arizona:     12/68    Oregon:      32/68

PROJECTED TOURNAMENT WINS
Arizona:     1.38 (#11)    Oregon:      0.81 (#32)
_______________________________________________
```
This section has a lot to unpack.

It displays where we would expect each team to rank based on their seed, their overall rank to advance to each round of the tournament, and their projected tournament wins (and rank).

Each rank was callculated using a logistic regression with ridge penalty. It uses historic data to analyze each team's chance of advancing to a certain round of the tournament and ranks them based on how they statistically compare with other teams that have reached that round.

The projected wins was calculated using a poisson regression. It uses historical data to estimate how many wins we expect each team to have in the tournament (so the theoretical min and max should be 0 and 6 respectively).

**Matchup-specific insights**

Since this output analyzes a second round matchup, we should analyze the *Sweet 16 Rank*, since we are trying to decide what team will advance to that round.

Arizona is the 11th ranked team to reach the Sweet 16. This is slightly better than we would expect for them as a 4 seed (13-16)
Oregon is the 29th ranked team to reach the Sweet 16. This is much worse than we would expect for them as a 5 seed (17-20)

Arizona is projected to win 1.38 games in the tournament. This ranks them at 11.
Oregon is projected to win 0.81 games in the tournament. This ranks them at 32.

Since Arizona consistently ranks in the 11-13 range, and they are expected to rank 13-16, we can assume that they are very good, and perhaps even slightly underseeded, for a 4 seed.
Since Oregon consistently ranks in the 29-32 range, and they are expected to rank 17-20, we can assume that they are not very good, and probably overseeded, for a 5 seed.

This is one of my favorite parts about this program. Even though these teams seemed evenly matched according to the statistics, it is clear now that Arizona much better matches the mold of a team that would advance in the tournament according to historical data.

**Overall insights**

It is important to keep in mind that these ranks base teams based on how well they fit the mold of teams that advance in the tournament according to historical data.

Seeds were not included in the features, so each team is ranked purely based on their performance, and not enhanced by what a committee may think of them.

Ranks and projected win models use statistical z-scores for their features, not raw statistics, to ensure that the model works similarly on a year-to-year basis. This is useful because, for example, teams will not be given an unfair advantage in more recent years where every team makes more threes than they were back in 2010.

While looking where team rank across all different rounds can give you a better overall picture of where teams rank, I like to hone in on what round the winner would be advancing to. 
This is because for each round the models rank teams based on how well they fit previous teams to advance to that specific round, so there could be an instance, for example, that a team fits an archetype where they should be a lock to win 1 or 2 tournament games and not make it any further. 
A scenario like this would have said team ranking very highly to advance in the first 2 ranks, then they would have a steep drop-off from there on out.

I think that looking at projected wins is the best way to get an overall feel for how each team should perform. 
This is because there is much more wins each year than teams that advance to certain rounds of the tournament
   For example, each season in the dataset there are 63 allotted wins. There is only 1 team that wins the championship, 4 that make the Final 4, etc.
More data allows for more accurate modelling.

If teams' projected wins seem low, remember that the model has to allocate 63 wins across 68 teams, so usually the best teams will be projected 4-5.5 wins.

If there is a year where the best 4-5 teams are projected, for example, in the 4+ range, this would mean the tournament is very top-heavy. (2025)

If there is a year where the best 4-5 teams are projected more towards the 3-4 range, this would mean the tournament is very balanced. Perhaps pick more unexpected teams to go far (2023)

If there is a year where there is a clear cut 1-3 teams that are projected to win much more than the rest of the field, I would for sure pick one of those to win the tournament. (2019)

## CONCLUSION

Given the fact that Arizona has a slight statistical advantage according to the win percentage AND they are much better in terms of Sweet 16 Rank and Projected Wins, this program strongly suggests that Arizona should be chosen to win this game.

