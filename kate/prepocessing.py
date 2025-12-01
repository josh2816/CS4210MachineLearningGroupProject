import pandas as pd

keep = ["wins","losses","points","points_opp","total_yards","plays_offense","yds_per_play_offense","turnovers","fumbles_lost","first_down","pass_cmp","pass_att", "win_superbowl"]


df = pd.read_csv('Preprocessed_Data copy.csv', usecols=keep)
# df = df.drop('year', axis=1) 
# df = df.drop('team', axis=1)
#df = df.drop('g', axis=1)
# df = df.drop('ties', axis=1)
# df = df.drop('mov', axis=1)
# df = df.drop('points_diff', axis=1)
# df = df.drop('win_loss_perc', axis=1)
# df = df.drop('exp_pts_tot', axis=1)



df['win_superbowl'] = df['win_superbowl'].replace('No', 0)
df['win_superbowl'] = df['win_superbowl'].replace('Yes', 1)
df.fillna(0, inplace=True) # Replaces all NaN with 0



df.to_csv('knntest.csv', index=False)
