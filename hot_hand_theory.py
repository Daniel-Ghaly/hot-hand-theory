import pandas as pd

shot_logs_file_path = 'shot_logs.csv'
df = pd.read_csv(shot_logs_file_path)

def check_for_hot_streak(player_shot_results):
    player_shot_results = [x[1] for x in player_shot_results]
    if(len(player_shot_results) <= 3):
        return
    else:
        for number_of_shots_to_check in range(4, len(player_shot_results)):
            shots_to_check = player_shot_results[-number_of_shots_to_check:]
            make_count = shots_to_check.count(1)
            if number_of_shots_to_check == 4 and make_count == 3:
                    continue
            if make_count/number_of_shots_to_check >= 0.7:  
                return True
        return False
        
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
        check_for_hot_streak(player_shot_results)
        
