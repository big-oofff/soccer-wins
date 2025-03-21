import pandas as pd
import xml.etree.ElementTree as ET
import math
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
file_path = 'Match.csv'  # Update with the actual path
df = pd.read_csv(file_path)

team_file_path = 'Team.csv'  # Update with the actual path
teams_df = pd.read_csv(team_file_path)
team_id_to_name = dict(zip(teams_df['team_api_id'], teams_df['team_long_name']))

# Function to count shots on target from XML
def count_shots_on_target(xml_data):
    if pd.isna(xml_data) or not xml_data.strip():
        return 0
    try:
        root = ET.fromstring(xml_data)
        shoton_count = 0
        for value in root.findall(".//value"):
            shoton = value.find(".//shoton")
            if shoton is not None and shoton.text == '1':
                shoton_count += 1
        return shoton_count
    except ET.ParseError:
        return 0

# Apply the function and filter out rows with no shoton XML data
df['shots_on_target'] = df['shoton'].apply(count_shots_on_target)
df = df[df['shots_on_target'] > 0]  # Keep only rows with shoton counts greater than 0

# Filter rows starting from index 1730
df_filtered = df.iloc[1730:].reset_index(drop=True)

# Calculate wins
df_filtered['home_win'] = df_filtered['home_team_goal'] > df_filtered['away_team_goal']
df_filtered['away_win'] = df_filtered['away_team_goal'] > df_filtered['home_team_goal']

# Create separate dataframes for home and away teams
df_home = df_filtered.copy()
df_home['team'] = df_home['home_team_api_id']
df_home['goals'] = df_home['home_team_goal']
df_home['shots_on_target'] = df_home['shots_on_target']
df_home['total_wins'] = df_home['home_win']

df_away = df_filtered.copy()
df_away['team'] = df_away['away_team_api_id']
df_away['goals'] = df_away['away_team_goal']
df_away['shots_on_target'] = df_away['shots_on_target']
df_away['total_wins'] = df_away['away_win']

# Aggregate data by season and team
home_stats = df_home.groupby(['season', 'team']).agg({
    'total_wins': 'sum',
    'goals': 'sum',
    'shots_on_target': 'sum'
}).reset_index()

away_stats = df_away.groupby(['season', 'team']).agg({
    'total_wins': 'sum',
    'goals': 'sum',
    'shots_on_target': 'sum'
}).reset_index()

# Combine home and away stats
combined_stats = pd.concat([home_stats, away_stats], ignore_index=True)
combined_stats = combined_stats.groupby(['season', 'team']).agg({
    'total_wins': 'sum',
    'goals': 'sum',
    'shots_on_target': 'sum'
}).reset_index()

# Convert team IDs to team names
combined_stats['team'] = combined_stats['team'].map(team_id_to_name)

# Ensure all numerical values are integers
combined_stats['total_wins'] = combined_stats['total_wins'].astype(int)
combined_stats['goals'] = combined_stats['goals'].astype(int)
combined_stats['shots_on_target'] = combined_stats['shots_on_target'].astype(int)

# Convert to a list of tuples
result_tuples = [tuple(x) for x in combined_stats.to_records(index=False)]

# Calculate x and y points, and P values
points = []
predicted_P = []
actual_wins = []
for record in result_tuples:
    season, team, wins, goals, shotons = record
    if goals == 0 or shotons == 0 or wins == 0:
        continue
    x = math.log(goals / shotons)
    y = math.log(wins / shotons)
    points.append((x, y))
    P = 0.345 * (goals ** 1.1755) * (shotons ** -0.1755)
    predicted_P.append(P)
    actual_wins.append(wins)
    print(f"Team: {team}, Season: {season}, Calculated P: {P:.2f}, Actual Wins: {wins}")

# Calculate least squares regression line for log points
x_vals = np.array([point[0] for point in points])
y_vals = np.array([point[1] for point in points])

A = np.vstack([x_vals, np.ones(len(x_vals))]).T
slope, intercept = np.linalg.lstsq(A, y_vals, rcond=None)[0]
print(f"Least Squares Regression Line (log points): y = {slope:.4f}x + {intercept:.4f}")

# Calculate least squares regression line for actual vs predicted P points
actual_P_vals = np.array(actual_wins)
predicted_P_vals = np.array(predicted_P)

B = np.vstack([predicted_P_vals, np.ones(len(predicted_P_vals))]).T
slope_P, intercept_P = np.linalg.lstsq(B, actual_P_vals, rcond=None)[0]
print(f"Least Squares Regression Line (actual vs predicted P): P = {slope_P:.4f}*P_pred + {intercept_P:.4f}")

# Calculate mean absolute error
mae = np.mean(np.abs(predicted_P_vals - actual_P_vals))
print(f"Mean Absolute Error: {mae:.2f}")

# Plot the actual vs predicted P values
plt.scatter(actual_P_vals, predicted_P_vals, color='blue', label='Predicted vs Actual P')

# Plot the ideal fit line (y = x)
plt.plot([min(actual_P_vals), max(actual_P_vals)], [min(actual_P_vals), max(actual_P_vals)], color='red', linestyle='--', label='Ideal Fit y = x')

# Plot the least squares regression line for actual vs predicted P points
fit_x_vals = np.array([min(predicted_P_vals), max(predicted_P_vals)])
fit_y_vals = slope_P * fit_x_vals + intercept_P
plt.plot(fit_x_vals, fit_y_vals, color='green', linestyle='-', label=f'Best Fit y = {slope_P:.4f}x+{intercept_P:.4f}')

plt.xlabel('Actual Wins (X)')
plt.ylabel('Predicted Wins (Y)')
plt.title('Actual vs Predicted Wins')
plt.legend()
plt.text(0.05, 0.95, f'Regression Line: y = {slope_P:.4f}x+ {intercept_P:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()
