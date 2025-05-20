import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load shot log data
shot_logs_file_path = 'shot_logs.csv'
df = pd.read_csv(shot_logs_file_path)

# Initialize a new column to flag shots that come after a hot streak
df['IS_POST_HOT'] = 0

# Remove any rows missing data
df = df.dropna(subset=['SHOT_RESULT', 'IS_POST_HOT', 'SHOT_DIST', 'CLOSE_DEF_DIST'])

# Convert shot results to binary: 1 = made, 0 = missed
df['SHOT_MADE'] = df['SHOT_RESULT'].map({'made': 1, 'missed': 0})

# Helper function to detect hot streaks in a player’s shot sequence
def check_for_hot_streak(player_shot_results):
    n = len(player_shot_results)
    if n < 4:
        return False, None  # Not enough shots to qualify

    # Try all windows of length ≥ 4 within the shot sequence
    for start in range(n):
        for end in range(start + 4, n + 1):
            window = player_shot_results[start:end]
            makes = [x[1] for x in window]
            make_count = makes.count(1)
            length = end - start

            # Ignore 3/4 makes to avoid borderline cases
            if length == 4 and make_count == 3:
                continue

            # If shooting percentage in window ≥ 70%, flag next shot
            if make_count / length >= 0.7 and end < n:
                return True, player_shot_results[end][0]  # Return index of next shot
    return False, None

# Loop through each game and player and flag shots that come immediately after a hot streak
for GAME_ID in df['GAME_ID'].unique():
    game_df = df[df['GAME_ID'] == GAME_ID]
    for player_id in game_df['player_id'].unique():
        # Sort shots in chronological order for each player within the game
        player_shots_df = game_df[game_df['player_id'] == player_id].sort_values(by='SHOT_NUMBER')
        player_shot_results = []

        # Convert shot sequence to a list of (index, make/miss) tuples
        for index, row in player_shots_df.iterrows():
            if row['SHOT_RESULT'] == 'made':
                player_shot_results.append((row.name, 1))
            elif row['SHOT_RESULT'] == 'missed':
                player_shot_results.append((row.name, 0))

        # Check for hot streak and flag the next shot if found
        is_hot, next_shot_index = check_for_hot_streak(player_shot_results)
        if is_hot and next_shot_index is not None:
            df.loc[next_shot_index, 'IS_POST_HOT'] = 1

# Define independent variables and response variable
X = df[['IS_POST_HOT', 'SHOT_DIST', 'CLOSE_DEF_DIST']]  # Predictors
X = sm.add_constant(X)  # Adds an intercept term to the regression
y = df['SHOT_MADE']  # Binary outcome: 1 = made, 0 = missed

# Fit logistic regression model to estimate impact of each factor on shot success
model = sm.Logit(y, X).fit()

# Output model summary
print(model.summary())

# Convert log-odds coefficients to odds ratios for clearer interpretation
odds_ratios = np.exp(model.params)
print(odds_ratios)