import pandas as pd
import statsmodels.api as sm
import numpy as np

shot_logs_file_path = 'shot_logs.csv'
df = pd.read_csv(shot_logs_file_path)
df['IS_POST_HOT'] = 0

# Drop rows with missing values in key columns
df = df.dropna(subset=['SHOT_RESULT', 'IS_POST_HOT', 'SHOT_DIST', 'CLOSE_DEF_DIST'])

# Encode SHOT_RESULT as binary
df['SHOT_MADE'] = df['SHOT_RESULT'].map({'made': 1, 'missed': 0})

def check_for_hot_streak(player_shot_results):
    n = len(player_shot_results)

    if n < 4:
        return False, None  # not enough shots

    for start in range(n):
        for end in range(start + 4, n + 1):  # ← allows windows up to full list
            window = player_shot_results[start:end]
            makes = [x[1] for x in window]
            make_count = makes.count(1)
            length = end - start

            # Skip the 3/4 special case
            if length == 4 and make_count == 3:
                continue

            if make_count / length >= 0.7:
                # If a next shot exists, tag it
                if end < n:
                    return True, player_shot_results[end][0]  # index of next shot
    return False, None

        
for GAME_ID in df['GAME_ID'].unique():
    game_df = df[df['GAME_ID'] == GAME_ID]
    for player_id in game_df['player_id'].unique():
        player_shots_df = game_df[game_df['player_id'] == player_id].sort_values(by='SHOT_NUMBER')
        player_shot_results = []
        for index, row in player_shots_df.iterrows():
            if row['SHOT_RESULT'] == 'made':
                player_shot_results.append((row.name, 1))
            elif row['SHOT_RESULT'] == 'missed':
                player_shot_results.append((row.name, 0))
        is_hot, next_shot_index = check_for_hot_streak(player_shot_results)
        if is_hot and next_shot_index is not None:
            df.loc[next_shot_index, 'IS_POST_HOT'] = 1

# Define predictors and response
X = df[['IS_POST_HOT', 'SHOT_DIST', 'CLOSE_DEF_DIST']]
X = sm.add_constant(X)  # adds intercept term
y = df['SHOT_MADE']

# Fit model
model = sm.Logit(y, X).fit()
print(model.summary())

odds_ratios = np.exp(model.params)
print(odds_ratios)