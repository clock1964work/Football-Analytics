import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler  # For scaling the data
from sklearn.decomposition import PCA  # For performing PCA (dimensionality reduction)
from sklearn.cluster import KMeans  # For KMeans clustering
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt  # For visualization
import numpy as np
import plotly.graph_objects as go



# Load the DataFrame
path = r"C:\Users\User\PycharmProjects\soccermatics\test2.xlsx"
df = pd.read_excel(path)
df.info()
# Display column names to check for missing or mismatched names
pd.set_option('display.max_rows', None)  # Show all rows
print(df.dtypes)  # Display all columns and their data types
pd.reset_option('display.max_rows')

# Function to categorize players based on their position
def position_grouping(x):
    if x in ['GK']:  # Goalkeeper
        return "GK"
    elif x in ["DF", 'DF,MF']:  # Defender
        return "Defender"
    elif x in ['FW,DF', 'DF,FW']:  # Wing-Back
        return "Wing-Back"
    elif x in ['MF,DF']:  # Defensive Midfielders
        return "Defensive-Midfielders"
    elif x in ['MF']:  # Central Midfielders
        return "Central Midfielders"
    elif x in ['MF,FW', 'FW,MF']:  # Attacking Midfielders
        return "Attacking Midfielders"
    elif x in ['FW']:  # Forwards
        return "Forwards"
    else:  # In case there's an unidentified position
        return "Unidentified Position"

# Apply the position_grouping function to create a 'position_group' column
df['position_group'] = df['Pos'].apply(position_grouping)
df.info()
print(df)
position_counts = df.groupby('position_group').size().reset_index(name='Count')
print(position_counts)


def per_90fi(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns
    for col in numeric_cols:
        if col != '90s':  # Avoid dividing '90s' by itself
            df[col] = df[col] / df['90s']
    return df

def key_stats_db(df, position):
    non_numeric_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'position_group']
    core_stats = ['90s', 'Passes_Total.Cmp%', 'Passes.KP',
                  'Passes.PPA', 'Passes.PrgP', 'PassesType.Crs','Passes.1/3','Defense_Challenges.Tkl%',
                  'Defense_Blocks.Blocks', 'Defense.Tkl+Int', 'Defense.Clr',
                  'Possession_carries.Carries', 'Possession_carries.PrgDist','Possession_takesOn.Succ%',
                  'SCA90', 'GCA90', 'Passes.CrsPA', 'Passes.xA', 'Possession_receiving.Rec',
                  'Possession_receiving.PrgR', 'Possession_Touches.Att 3rd','Possession.Total_Touches',
                  'Shooting_Xg.npxG', 'Shooting_Standard.Sh', 'Shooting_Standard.SoT','Misc_AerialDuels.Won%']



    # Drop rows with any missing values
    df.dropna(axis=0, how='any', inplace=True)



    # Apply the position grouping function
    df['position_group'] = df['Pos'].apply(position_grouping)



    # Check if the core stats columns exist in the DataFrame
    missing_cols = [col for col in core_stats if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing: {missing_cols}")
        core_stats = [col for col in core_stats if col not in missing_cols]

    # Filter the data to get only the players of the specific position group
    key_stats_df = df[df['position_group'] == position]



    # Select relevant columns for the analysis
    key_stats_df = key_stats_df[non_numeric_cols + core_stats]



    key_stats_df = key_stats_df[key_stats_df['90s'] > 4]



    # Normalize stats per 90 minutes
    key_stats_df = per_90fi(key_stats_df)



    return key_stats_df


def create_metrics_scores(key_stats_df):
    core_stats = ['90s', 'Passes_Total.Cmp%', 'Passes.KP', 'Passes.PPA', 'Passes.PrgP', 'Defense_Challenges.Tkl%',
                  'Defense_Blocks.Blocks',
                  'Defense.Tkl+Int', 'Defense.Clr',
                  'Possession_carries.Carries', 'Possession_carries.PrgDist','Possession_takesOn.Succ%',
                  'Possession_Touches.Att 3rd','SCA90', 'GCA90', 'Passes.CrsPA', 'Passes.xA',
                  'PassesType.Crs','Passes.1/3','Possession_receiving.Rec',
                  'Possession_receiving.PrgR','Shooting_Xg.npxG' , 'Shooting_Standard.Sh',
                  'Shooting_Standard.SoT','Misc_AerialDuels.Won%']
    passing_metrics = ['Passes_Total.Cmp%', 'Passes.KP', 'Passes.PPA', 'Passes.PrgP','PassesType.Crs','Passes.1/3']
    defending_metrics = ['Defense_Challenges.Tkl%', 'Defense_Blocks.Blocks', 'Defense.Tkl+Int', 'Defense.Clr','Misc_AerialDuels.Won%']
    creation_metrics = ['Possession_carries.Carries', 'Possession_carries.PrgDist','Possession_takesOn.Succ%','Possession.Total_Touches', 'SCA90', 'GCA90', 'Passes.CrsPA', 'Passes.xA', 'Possession_receiving.Rec', 'Possession_receiving.PrgR','Possession_Touches.Att 3rd']
    shooting_metrics = ['Shooting_Standard.Sh', 'Shooting_Standard.SoT','Shooting_Xg.npxG']
    print("Data for creation metrics before scaling:")
    print(key_stats_df[creation_metrics].head())  # Check the raw data in the creation metrics columns

    scaler = MinMaxScaler()

    stats_normalized = key_stats_df.copy()
    stats_normalized[core_stats] = scaler.fit_transform(stats_normalized[core_stats])

    stats_normalized['Passing_Score'] = stats_normalized[passing_metrics].mean(axis=1) * 10
    stats_normalized['Defending_Score'] = stats_normalized[defending_metrics].mean(axis=1) * 10
    stats_normalized['Creation_Score'] = stats_normalized[creation_metrics].mean(axis=1) * 10
    stats_normalized['Shooting_Score'] = stats_normalized[shooting_metrics].mean(axis=1) * 10

    stats_normalized['Passing_Score'] += stats_normalized.index * 0.001
    stats_normalized['Defending_Score'] += stats_normalized.index * 0.001
    stats_normalized['Creation_Score'] += stats_normalized.index * 0.001
    stats_normalized['Shooting_Score'] += stats_normalized.index * 0.001

    stats_normalized[['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']] = stats_normalized[['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']].clip(lower=0, upper=10)
    # Check scaled data for creation metrics
    print("Scaled data for creation metrics:")
    print(stats_normalized[creation_metrics].head())

    return stats_normalized



def adjust_player_rating_range(dataframe):
    # Select the relevant columns
    player_ratings = dataframe[['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']]

    # Define min and max ratings for the new 0-1 scale
    min_rating = 0
    max_rating = 1

    # Loop through each column (Passing_Score, Defending_Score, etc.)
    for col in player_ratings.columns:
        min_val = player_ratings[col].min()  # Get the minimum score in the column
        max_val = player_ratings[col].max()  # Get the maximum score in the column

        # If there is variation in the values (min_val != max_val)
        if max_val != min_val:
            # Perform min-max normalization
            normalized_ratings = min_rating + (max_rating - min_rating) * (
                        (player_ratings[col] - min_val) / (max_val - min_val))
        else:
            # If all values are the same, set the score to the middle value (0.5)
            normalized_ratings = (min_rating + max_rating) / 2

        # Update the original DataFrame with the normalized ratings
        dataframe[col] = normalized_ratings

    return dataframe




# Add this before calling the function
position = "Forwards"  # Change this to the position you want to analyze
key_stats_df = key_stats_db(df,   position)  # Store the returned DataFrame


scoring = create_metrics_scores(key_stats_df)
pitch_iq_scoring = adjust_player_rating_range(scoring)
pitch_iq_scoring = scoring[['Player', 'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']]
scores = pd.merge(key_stats_df, scoring, on='Player', how='left')

def create_kmeans_df(df):
    KMeans_cols = ['Player', 'Squad', 'Passes_Total.Cmp%', 'Passes.KP',
                   'Passes.PPA', 'Passes.PrgP', 'Defense_Challenges.Tkl%', 'Defense_Blocks.Blocks', 'Defense.Tkl+Int',
                   'Defense.Clr', 'Possession_carries.Carries', 'Possession_carries.PrgDist',
                   'Possession_takesOn.Succ%', 'Possession_Touches.Att 3rd', 'SCA90', 'GCA90',
                   'Passes.CrsPA', 'Passes.xA', 'PassesType.Crs', 'Passes.1/3', 'Possession_receiving.Rec',
                   'Possession_receiving.PrgR',
                   'Shooting_Xg.npxG',
                   'Shooting_Standard.Sh', 'Shooting_Standard.SoT', 'Misc_AerialDuels.Won%', 'Possession.Total_Touches']

    df = df[KMeans_cols]

    # Store player names and squads before dropping them
    player_names = df['Player'].tolist()
    squad_names = df['Squad'].tolist()

    df = df.drop(['Player', 'Squad'], axis=1)

    # Normalize the feature data
    x = df.values
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)

    # Apply PCA for dimensionality reduction (keeping the original number of components you intend)
    pca = PCA(n_components=5)  # Adjust this to whatever number of components you want
    reduced = pd.DataFrame(pca.fit_transform(X_norm))

    # Add cluster, name, and squad back to the reduced DataFrame
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    reduced['cluster'] = clusters
    reduced['name'] = player_names
    reduced['squad'] = squad_names

    return reduced


df.to_excel("df.xlsx")
# Run the function
kmeans_df = create_kmeans_df(key_stats_df)
kmeans_df.to_excel("Kmeand.xlsx")
kmeans_df.info()

# Function to find similar players and include squad info
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances



def find_similar_players(player_name, df, top_n=10):
    # Get player's cluster
    player = df[df['name'] == player_name].iloc[0]
    player_cluster = player['cluster']

    # Filter players from the same cluster
    same_cluster_df = df[df['cluster'] == player_cluster]

    # Get the index of the player in the dataset
    player_index = same_cluster_df[same_cluster_df['name'] == player_name].index[0]

    # Calculate pairwise Euclidean distances
    distances = euclidean_distances(same_cluster_df.drop(['name', 'squad', 'cluster'], axis=1),
                                    same_cluster_df.loc[player_index].drop(['name', 'squad', 'cluster']).values.reshape(1, -1))

    # Store the computed distances in the DataFrame
    same_cluster_df['distance'] = distances

    # Normalize the similarity score (inverse of distance)
    same_cluster_df['perc_similarity'] = 100 / (1 + same_cluster_df['distance'])  # Simple inverse distance formula

    # Sort and return top N similar players (excluding the selected player)
    similar_players = same_cluster_df.sort_values('distance').head(top_n + 1)  # +1 to include the player themselves
    similar_players = similar_players[similar_players['name'] != player_name]  # Exclude the player themselves

    # Return a DataFrame with 'name', 'squad', and 'perc_similarity' for the most similar players
    return similar_players[['name', 'squad', 'perc_similarity']]



# Example usage
player_name = 'Che Adams'  # Change this to any player
similarity_table = find_similar_players(player_name, kmeans_df)

print(similarity_table)
similarity_table.info()

# Format the perc_similarity column to 2 decimal places
similarity_table['perc_similarity'] = similarity_table['perc_similarity'].round(2)

# Add a horizontal bar for perc_similarity column using Pandas Styler
styled_table = similarity_table.style \
    .set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', 'darkblue'), ('color', 'white'), ('font-size', '14px')]},  # Header style
        {'selector': 'tbody td', 'props': [('background-color', 'lightgray'), ('text-align', 'center'), ('font-size', '12px')]},  # Cell style
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'whitesmoke')]},  # Zebra striping
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
    ]) \
    .highlight_max(axis=0, color='yellow')  # Highlight max in each column

# Create horizontal bars for the perc_similarity column (using the bar method)
styled_table = styled_table.bar(subset=['perc_similarity'], color='lightblue', width=100)

# Export the styled table to HTML with UTF-8 encoding
html_table = styled_table.to_html()

# Save the HTML to a file with UTF-8 encoding
html_file = "similarity_table_with_bars.html"
with open(html_file, "w", encoding="utf-8") as f:  # Use UTF-8 encoding to avoid encoding errors
    f.write(html_table)

# Automatically open the HTML file in your browser
import webbrowser
webbrowser.open(html_file)

# Save the table to an Excel file with the correct extension
similarity_table.to_excel("similarity_table.xlsx", index=False)  # Save as .xlsx

print("Table saved as 'similarity_table.xlsx' and opened in browser")