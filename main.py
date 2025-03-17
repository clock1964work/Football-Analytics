import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer.pitch import VerticalPitch,Pitch
import io  # Add this line
import numpy as np

import os
import streamlit as st


import psycopg2
from psycopg2 import pool

# Create a database connection pool
db_pool = pool.SimpleConnectionPool(
    minconn=1,  # Minimum number of connections
    maxconn=10, # Maximum number of connections
    host="aws-0-eu-central-1.pooler.supabase.com",
    database="postgres",
    user="postgres.dukvqeeuktlsotlwgtjb",
    password="8FQUT1zEGep0numj"
)

# Function to get a connection from the pool
def get_db_connection():
    return db_pool.getconn()

# Function to release a connection back to the pool
def release_db_connection(conn):
    db_pool.putconn(conn)

# Fetch match data
def fetch_match_data(conn, match_id):
    cur = conn.cursor()
    cur.execute("""
        select m.team_id,m.x,m.y,m.end_x,m.end_y,m.is_goal,m.is_shot,m.type_display_name,m.outcome_type_display_name,m.match_id,p.name,t.team_name
        from match_events m
        join teams t on m.team_id = t.team_id
        join players p on m.player_id = p.player_id

        WHERE m.match_id = %s
    """, (match_id,))
    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()
    return df

# Fetch available dates (optionally filtered by team)
def fetch_available_dates(conn, selected_team=None):
    cur = conn.cursor()

    if selected_team:
        cur.execute("""
            SELECT DISTINCT start_time::date
            FROM additional_info
            WHERE home_team_name = %s OR away_team_name = %s
            ORDER BY start_time ASC
        """, (selected_team, selected_team))
    else:
        cur.execute("""
            SELECT DISTINCT start_time::date
            FROM additional_info
            ORDER BY start_time ASC
        """)

    available_dates = [row[0] for row in cur.fetchall()]
    cur.close()
    return available_dates


# Fetch games based on selected filters (team, date, or both)
def fetch_games(conn, selected_team=None, selected_date=None):
    cur = conn.cursor()

    query = """
        SELECT match_id, home_team_name, away_team_name, venue_name, attendance, referee_name
        FROM additional_info
    """

    conditions = []
    params = []

    if selected_team:
        conditions.append("(home_team_name = %s OR away_team_name = %s)")
        params.extend([selected_team, selected_team])

    if selected_date:
        conditions.append("start_time::date = %s")
        params.append(selected_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    cur.execute(query, tuple(params))

    games = {}
    game_details = {}

    for row in cur.fetchall():
        match_id = row[0]
        games[match_id] = f"{row[1]} vs {row[2]}"
        game_details[match_id] = {
            "home_team": row[1],
            "away_team": row[2],
            "venue_name": row[3],
            "attendance": row[4],
            "referee_name": row[5]
        }

    cur.close()
    return games, game_details


# Plot Passing Network
def plot_passing_network(team_name, passes, ax, pitch, color):
    # Filter passes based on the team_name
    team_passes = passes[passes['team_name'] == team_name]

    team_passes["receiver"] = team_passes["name"].shift(-1)
    team_passes = team_passes.dropna(subset=["end_x", "end_y", "receiver"])
    team_passes["pair_key"] = team_passes.apply(lambda x: "_".join(sorted([x["name"], x["receiver"]])), axis=1)
    lines_df = team_passes.groupby("pair_key").x.count().reset_index()
    lines_df.rename(columns={'x': 'pass_count'}, inplace=True)
    lines_df = lines_df[lines_df['pass_count'] > 2]

    scatter_df = pd.DataFrame()
    for i, name in enumerate(team_passes["name"].unique()):
        passx = team_passes[team_passes["name"] == name]["x"].to_numpy()
        recx = team_passes[team_passes["receiver"] == name]["end_x"].to_numpy()
        passy = team_passes[team_passes["name"] == name]["y"].to_numpy()
        recy = team_passes[team_passes["receiver"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "name"] = name
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        scatter_df.at[i, "no"] = team_passes[team_passes["name"] == name].shape[0]

    scatter_df["marker_size"] = scatter_df["no"] / scatter_df["no"].max() * 70
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color=color, edgecolors='grey', linewidth=1,
                  alpha=1, ax=ax)
    for i, row in lines_df.iterrows():
        player1, player2 = row["pair_key"].split("_")
        if player1 in scatter_df["name"].values and player2 in scatter_df["name"].values:
            player1_x, player1_y = scatter_df[scatter_df["name"] == player1][["x", "y"]].values[0]
            player2_x, player2_y = scatter_df[scatter_df["name"] == player2][["x", "y"]].values[0]
            line_width = row["pass_count"] / lines_df["pass_count"].max() * 1
            pitch.lines(player1_x, player1_y, player2_x, player2_y, lw=line_width, color=color, ax=ax)

# Create Table
def create_table(ax, df, teams):
    # Count Goals for each team (filtering by is_goal)
    team1_goals = df[(df["team_name"] == teams[0]) & (df["is_goal"] == True)].shape[0]
    team2_goals = df[(df["team_name"] == teams[1]) & (df["is_goal"] == True)].shape[0]

    # Count Total Shots for each team (filtering by is_shot)
    team1_shots = df[(df["team_name"] == teams[0]) & (df["is_shot"] == True)].shape[0]
    team2_shots = df[(df["team_name"] == teams[1]) & (df["is_shot"] == True)].shape[0]

    # Count Shots on Target for each team (filtering by type_display_name for 'SavedShot' and is_shot)
    team1_shots_on_target = df[(df["team_name"] == teams[0]) & (df["type_display_name"] == "SavedShot")].shape[0]
    team2_shots_on_target = df[(df["team_name"] == teams[1]) & (df["type_display_name"] == "SavedShot")].shape[0]

    # Shots on target should also include the goals (as goals are considered shots on target)
    team1_shots_on_target += team1_goals
    team2_shots_on_target += team2_goals

    # Count Total Passes for each team (filtering by type_display_name for "Pass")
    team1_passes = df[(df["team_name"] == teams[0]) & (df["type_display_name"] == "Pass")].shape[0]
    team2_passes = df[(df["team_name"] == teams[1]) & (df["type_display_name"] == "Pass")].shape[0]

    # Count Successful Passes for each team (filtering by outcome_type_display_name for "Successful")
    team1_successful_passes = df[(df["team_name"] == teams[0]) & (df["type_display_name"] == "Pass") & (df["outcome_type_display_name"] == "Successful")].shape[0]
    team2_successful_passes = df[(df["team_name"] == teams[1]) & (df["type_display_name"] == "Pass") & (df["outcome_type_display_name"] == "Successful")].shape[0]

    # Calculate Pass Completion Percentage (successful passes / total passes)
    team1_pass_completion = (team1_successful_passes / team1_passes) * 100 if team1_passes > 0 else 0
    team2_pass_completion = (team2_successful_passes / team2_passes) * 100 if team2_passes > 0 else 0

    # Prepare the column labels and the data for the table
    column_labels = [teams[0], "Stat", teams[1]]
    table_vals = [
        [str(team1_goals), "Goals", str(team2_goals)],
        [str(team1_shots), "Shots", str(team2_shots)],
        [str(team1_shots_on_target), "Shots on Target", str(team2_shots_on_target)],
        [str(team1_passes), "Passes", str(team2_passes)],
        [f"{team1_pass_completion:.2f}%", "Pass Completion", f"{team2_pass_completion:.2f}%"]
    ]

    # Create the table
    table = ax.table(
        cellText=table_vals,
        colLabels=column_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],  # Define table position and size
        edges='horizontal',  # Only horizontal edges
    )

    # Customize cell borders to remove the top line
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.visible_edges = 'B'  # Keep only the bottom edge for the header
        else:
            cell.visible_edges = 'horizontal'  # Only horizontal lines for other rows

    table.set_fontsize(12)
    table.auto_set_column_width([0, 1, 2])  # Adjust column width
    ax.axis('off')



import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from mplsoccer import Pitch

# Force Matplotlib to apply edge colors properly
mpl.rcParams["patch.force_edgecolor"] = True

def create_combined_shotmap(data, team1, team2, ax):
    shots_df = data[data["is_shot"] == True]
    pitch = Pitch(pitch_type='opta', goal_type='box', linewidth=.85, line_color='black')
    pitch.draw(ax=ax)

    # Iterate over each shot and determine the corresponding color and type
    for _, shot in shots_df.iterrows():
        x, y = shot["x"], shot["y"]
        goal = shot["type_display_name"] == "Goal"
        miss = shot["type_display_name"] == "MissedShots"  # Missed shot (off target)
        team_name = shot["team_name"]

        # Define colors based on shot type
        if goal:
            color = "green"  # Goal
            edgecolor = "black"  # Black edge for goals
        elif miss:
            color = "red"  # Missed shot (off target)
            edgecolor = "black"  # Black edge for missed shots
        else:
            color = "white"  # Shot on target (everything else)
            edgecolor = "red"  # Red edge for shot on target

        # If the shot is from team1, flip the coordinates
        if team_name == team1:
            x, y = 100 - x, 100 - y

        # Plot the shot with correct edge color
        ax.scatter([x], [y], color=color, edgecolors=edgecolor, s=100, linewidth=1.5, zorder=10)

    # Define the legend with updated colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,markeredgecolor='black', label='Goals'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,markeredgecolor='black', label='Shot off target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=9, markeredgewidth=1.5,
               markeredgecolor='red', label='Shot on target', linestyle='None')
    ]

    # Adjust legend description font size
    legend_font_size = 8

    # Adjust legend size and position
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=legend_font_size)

# Function to fetch team names from the database
def fetch_team_names(conn):
  cur = conn.cursor()
  cur.execute("SELECT team_name FROM teams;")
  team_names = [row[0] for row in cur.fetchall()]
  cur.close()
  return team_names

# Function to fetch players based on team
def fetch_players_by_team(conn, team):
  cur = conn.cursor()
  cur.execute("""
      SELECT players.name
      FROM players
      INNER JOIN teams ON players.team_id = teams.team_id
      WHERE teams.team_name = %s;
  """, (team,))
  players = [row[0] for row in cur.fetchall()]
  cur.close()
  return players

# Function to fetch shot data for a player
import pandas as pd

import pandas as pd

import pandas as pd

def fetch_shot_data(conn, player):
    cur = conn.cursor()

    # Optimized SQL query: Select only necessary columns & filter is_shot directly in SQL
    cur.execute("""
        SELECT match_events.event_id, match_events.minute, match_events.second, 
               match_events.team_id, match_events.player_id, match_events.x, match_events.y, 
               match_events.end_x, match_events.end_y, match_events.qualifiers, match_events.is_touch, 
               match_events.blocked_x, match_events.blocked_y, match_events.goal_mouth_z, 
               match_events.goal_mouth_y, match_events.is_goal, 
               match_events.type_display_name, match_events.outcome_type_display_name, 
               match_events.match_id, players.name, players.position, players.age
        FROM match_events
        INNER JOIN players ON match_events.player_id = players.player_id
        WHERE players.name = %s AND match_events.is_shot = TRUE;
    """, (player,))

    # Fetch the optimized dataset
    shot_data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    cur.close()
    return shot_data




# Function to fetch danger passes data for a player

def fetch_danger_passes_data(conn, player):
  cur = conn.cursor()
  cur.execute("""
      SELECT *
      FROM match_events
      WHERE
          player_id = (
              SELECT player_id
              FROM players
              WHERE name = %s
          )
          AND (is_shot = true OR type_display_name = 'Pass');
  """, (player,))

  # Fetch all data from the SQL query
  data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
  cur.close()

  # Filter out "ThrowIn" events
  data = data[~data['qualifiers'].astype(str).str.contains("ThrowIn")]

  return data

# Function to fetch defensive action data for a player
def fetch_defensive_actions_data(conn, player):
  cur = conn.cursor()
  cur.execute("""
      SELECT *
      FROM match_events
      INNER JOIN players ON match_events.player_id = players.player_id
      WHERE
          players.name = %s
          AND type_display_name IN ('BallRecovery', 'Clearance', 'Interception', 'Tackle');
  """, (player,))

  # Fetch all data from the SQL query
  defensive_actions_data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
  cur.close()

  return defensive_actions_data

# Function to fetch passes data for a selected player
def fetch_passes_data(conn, player):
   cur = conn.cursor()
   cur.execute("""
       SELECT *
       FROM match_events
       INNER JOIN players ON match_events.player_id = players.player_id
       WHERE players.name = %s
       AND type_display_name = 'Pass';
   """, (player,))

   # Fetch all data from the SQL query
   passes_data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
   cur.close()

   # Filter out passes that contain 'CornerTaken' in qualifiers
   passes_data = passes_data[~passes_data['qualifiers'].astype(str).str.contains("CornerTaken")]

   return passes_data

# Function to fetch key passes data for a selected player
def fetch_key_passes_data(conn, player):
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM match_events
        INNER JOIN players ON match_events.player_id = players.player_id
        WHERE players.name = %s
        AND type_display_name = 'Pass';
    """, (player,))

    # Fetch all data from the SQL query
    data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()

    # Filter out passes that are not key passes
    key_passes_data = data[data['qualifiers'].astype(str).str.contains('KeyPass')]
    # Filter out passes that contain 'CornerTaken' in qualifiers
    key_passes_data = key_passes_data[~key_passes_data['qualifiers'].astype(str).str.contains("CornerTaken")]

    return key_passes_data
def display_shot_map(shot_data, player_name):
  # Calculate number of goals, total shots, shots on target, and shots off target
  num_goals = shot_data[shot_data['is_goal'] == True].shape[0]
  num_total_shots = shot_data.shape[0]
  num_shots_on_target = shot_data[(shot_data['type_display_name'] == 'SavedShot') | (shot_data['type_display_name'] == 'Goal')].shape[0]
  num_shots_off_target = num_total_shots - num_shots_on_target

  fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
  pitch = VerticalPitch(half=True,
                        pitch_type='opta',
                        goal_type='box',
                        linewidth=.85,
                        line_color='black',
                        )

  pitch.draw(ax=ax)


  for i, row in shot_data.iterrows():
      if row["type_display_name"] == 'Goal':
          pitch.scatter(row["x"], row["y"], alpha=1, s=30, color="green", ax=ax)
      elif row["type_display_name"] == "MissedShots":
          pitch.scatter(row.x, row.y, alpha=1, s=30, color="red", edgecolor="red", ax=ax)
      else:
          pitch.scatter(row.x, row.y, alpha=1, s=30, color="white", edgecolor="red", ax=ax)

          plt.suptitle('', fontsize=25, weight='bold')

  # Scatter plots for legend
  legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Goals'),
                     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
                            label='Shot off target'),
                     Line2D([0], [0], marker='o', color='red', markerfacecolor='white', markersize=9,
                            markeredgewidth=1, markeredgecolor='red', label='Shot On Target', linestyle='None')]

  # Adjust legend description font size
  legend_font_size = 8

  # Adjust legend size and position
  ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=legend_font_size)
  plt.suptitle(f"Shot Map for {player_name}")  # Add player's name to the title

  # Save the plot as an image
  img_buffer = io.BytesIO()
  plt.savefig(img_buffer, format='png')
  img_buffer.seek(0)

  # Display the image using st.image
  st.image(img_buffer)

  # Add comments
  comments = f"Number of goals: {num_goals}\n"
  comments += f"Number of total shots: {num_total_shots}\n"
  comments += f"Number of shots on target: {num_shots_on_target}\n"
  comments += f"Number of shots off target: {num_shots_off_target}\n"

  # Display comments
  st.text(comments)
# Function to display danger passes heatmap
def display_heatmap(data, player_name):
    # Process danger passes for the selected player
    danger_passes = pd.DataFrame()
    for period in [1, 2]:
        shots = data[data["is_shot"] == True].set_index("id")
        passes = data[(data["type_display_name"] == "Pass") & (data["outcome_type_display_name"] == "Successful")]
        passes_no_throw_in = passes[~passes["qualifiers"].astype(str).str.contains("ThrowIn")]

        # Extract necessary columns
        x = passes_no_throw_in["x"]
        y = passes_no_throw_in["y"]
        second = passes_no_throw_in["second"]
        minute = passes_no_throw_in["minute"]
        player_id = passes_no_throw_in["player_id"]
        end_x = passes_no_throw_in["end_x"]
        end_y = passes_no_throw_in["end_y"]
        shotSecond = shots["second"]
        shotminute = shots["minute"]
        shot_times = shots['minute'] * 60 + shots['second']
        shot_window = 15
        shot_start = shot_times - shot_window
        shot_start = shot_start.apply(lambda i: i if i > 0 else (period - 1) * 45)
        pass_times = passes_no_throw_in['minute'] * 60 + passes_no_throw_in['second']
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())
        danger_passes_period = passes_no_throw_in.loc[pass_to_shot]
        danger_passes = pd.concat([danger_passes, danger_passes_period], ignore_index=True)

    # Plot pitch
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    pitch = Pitch(pitch_type='opta', goal_type='box', linewidth=.85, line_color='black')
    pitch.draw(ax=ax)

    # 2D histogram
    bin_statistic = pitch.bin_statistic(danger_passes.x, danger_passes.y, statistic='count', bins=(6, 5),
                                        normalize=False)

    if len(data) != 0:
        bin_statistic["statistic"] = bin_statistic["statistic"] / len(data)

    if not np.isnan(bin_statistic["statistic"]).any():
        bin_statistic["statistic"] = bin_statistic["statistic"] / len(data)

    heatmap = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax)

    # Add legend indicating color range and danger level below the plot
    cbar = plt.colorbar(heatmap, orientation='horizontal', aspect=20, shrink=0.5)
    cbar.set_ticks([bin_statistic['statistic'].min(), bin_statistic['statistic'].max()])
    cbar.ax.set_xticklabels(['Low', 'High'], fontsize=8)

    plt.title('Starting location of Danger passes heatmap by ' + player_name)

    # Add arrow below the heatmap (same size and position as in passing network)
    ax.annotate("", xy=(0.66, -0.05), xytext=(0.33, -0.05),  # Position adjusted for consistency
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    st.pyplot(fig)



# Function to display defensive actions plot using Matplotlib
def display_defensive_actions_plot(defensive_actions_data, player_name):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=.85,
        line_color='black',
    )
    pitch.draw(ax=ax)

    for i, defense in defensive_actions_data.iterrows():
        x = defense["x"]
        y = defense["y"]
        action_type = defense["type_display_name"]

        color = "green" if action_type == "BallRecovery" else \
            "red" if action_type == "Clearance" else \
                "blue" if action_type == "Interception" else \
                    "black"
        ax.scatter(x, y, color=color, label=action_type)

    # Define legend elements based on defensive action types
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Ball Recovery'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Clearance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Interception'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Tackle')
    ]
    legend_font_size = 8

    # Add legend below the pitch, closer to the line
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=4,
              fontsize=legend_font_size)

    plt.title(f'Defensive Actions for {player_name}')

    # Add One Arrow Below the Plot
    ax.annotate("", xy=(0.66, -0.05), xytext=(0.33, -0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    num_tackle = defensive_actions_data[defensive_actions_data['type_display_name'] == 'Tackle'].shape[0]
    num_clearance = defensive_actions_data[defensive_actions_data['type_display_name'] == 'Clearance'].shape[0]
    num_interception = defensive_actions_data[defensive_actions_data['type_display_name'] == 'Interception'].shape[0]
    num_ball_recovery = defensive_actions_data[defensive_actions_data['type_display_name'] == 'BallRecovery'].shape[0]

    comment = f"Tackle: {num_tackle}\nClearance: {num_clearance}\nInterception: {num_interception}\nBall Recovery: {num_ball_recovery}"

    st.pyplot(fig)
    st.text(comment)


# Function to display passes using Matplotlib
def display_passes(passes_data, player_name):
    num_successful_passes = passes_data[passes_data['outcome_type_display_name'] == 'Successful'].shape[0]
    num_total_passes = passes_data.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=.85,
        line_color='black',
    )
    pitch.draw(ax=ax)

    for i, thepass in passes_data.iterrows():
        x = thepass["x"]
        y = thepass["y"]
        successful = thepass["outcome_type_display_name"] == "Successful"

        if successful:
            circle = plt.Circle((x, y), .5, color="green", alpha=.3)
            ax.add_patch(circle)
        else:
            circle = plt.Circle((x, y), .5, color="red", alpha=.3)
            ax.add_patch(circle)

        dx = thepass["end_x"] - x
        dy = thepass["end_y"] - y
        arrow_color = "green" if successful else "red"
        arrow = plt.Arrow(x, y, dx, dy, linewidth=.5, color=arrow_color)
        ax.add_patch(arrow)

    # Legend for arrows
    legend_elements = [
        plt.Line2D([0], [0], color='green', linewidth=1, label='Successful'),
        plt.Line2D([0], [0], color='red', linewidth=1, label='Unsuccessful')
    ]
    legend_font_size = 8

    # Add legend below the pitch, closer to the line
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=4, fontsize=legend_font_size)

    plt.title(f'Passes for {player_name}')

    # Add One Arrow Below the Plot
    ax.annotate("", xy=(0.66, -0.05), xytext=(0.33, -0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    st.pyplot(fig)

    # Calculate percentage of successful passes
    percent_successful_passes = (num_successful_passes / num_total_passes) * 100

    comments = f"Total passes: {num_total_passes}\n"
    comments += f"Successful passes: {num_successful_passes}\n"
    comments += f"Percentage of successful passes: {percent_successful_passes:.2f}%\n"

    st.text(comments)


# Function to display key passes using Matplotlib
def display_key_passes(key_passes_data, player_name):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=.85,
        line_color='black',
    )
    pitch.draw(ax=ax)

    for i, thepass in key_passes_data.iterrows():
        x = thepass["x"]
        y = thepass["y"]
        successful = thepass["outcome_type_display_name"] == "Successful"

        if successful:
            circle = plt.Circle((x, y), .5, color="green", alpha=.3)
            ax.add_patch(circle)

            dx = thepass["end_x"] - x
            dy = thepass["end_y"] - y
            arrow = plt.Arrow(x, y, dx, dy, linewidth=.5, color="green")
            ax.add_patch(arrow)

    # Legend for arrows
    legend_elements = [
        plt.Line2D([0], [0], color='green', linewidth=1, label='Key Passes')
    ]
    legend_font_size = 8

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=4,
              fontsize=legend_font_size)

    plt.title(f'Key Passes for {player_name}')

    #  Add One Arrow Below the Plot
    ax.annotate("", xy=(0.66, -0.05), xytext=(0.33, -0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    st.pyplot(fig)

    # Calculate total key passes
    num_key_passes = len(key_passes_data)

    comment = f"Total key passes: {num_key_passes}"

    # Display the comment below the plot
    st.text(comment)


import streamlit as st


import streamlit as st

import streamlit as st

def main():
    # Connect to the PostgreSQL database
    conn = get_db_connection()


    # Streamlit UI
    st.title('Football Data Visualization App\nEnglish Premier League 2024-2025')

    # Page selection in sidebar
    page = st.sidebar.radio("Select Page", ["Player Report", "Match Report"])

    if page == "Player Report":
        st.header("Player Report")

        # Create columns to display filters in a single row
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_team = st.selectbox('Select Team', fetch_team_names(conn))

        with col2:
            selected_player = st.selectbox('Select Player', fetch_players_by_team(conn, selected_team))

        with col3:
            visualization_type = st.selectbox(  # Changed from st.radio to st.selectbox
                'Select Visualization',
                ['Shot Map', 'Heatmap', 'Defensive Actions', 'Passes', 'Key Passes']
            )

        # Fetch data based on the selected player
        shot_data = fetch_shot_data(conn, selected_player)
        heatmap_data = fetch_danger_passes_data(conn, selected_player)
        defensive_actions_data = fetch_defensive_actions_data(conn, selected_player)
        passes_data = fetch_passes_data(conn, selected_player)
        key_passes_data = fetch_key_passes_data(conn, selected_player)

        # Display visualization
        if visualization_type == "Shot Map":
            display_shot_map(shot_data, selected_player)
        elif visualization_type == "Heatmap":
            display_heatmap(heatmap_data, selected_player)
        elif visualization_type == "Defensive Actions":
            display_defensive_actions_plot(defensive_actions_data, selected_player)
        elif visualization_type == "Passes":
            display_passes(passes_data, selected_player)
        elif visualization_type == "Key Passes":
            display_key_passes(key_passes_data, selected_player)






    elif page == "Match Report":

        st.title("Match Report")

        # Create three columns to place the filters horizontally

        col1, col2, col3 = st.columns([1, 1, 1])

        # Fetch all teams

        all_teams = fetch_team_names(conn)

        # Team selection (Optional) in the first column

        with col1:

            selected_team = st.selectbox("Select Team (Optional)", ["All Teams"] + all_teams)

        # Fetch available dates based on selected team in the second column

        with col2:

            available_dates = fetch_available_dates(conn, None if selected_team == "All Teams" else selected_team)

            selected_date = st.selectbox("Select Match Date (Optional)", ["All Dates"] + available_dates)

        # Fetch games based on filters in the third column

        with col3:

            games, game_details = fetch_games(

                conn,

                None if selected_team == "All Teams" else selected_team,

                None if selected_date == "All Dates" else selected_date

            )

        if not games:
            st.warning("No games available for the selected filters.")

            st.stop()

        # Game selection (Optional) - Use a selectbox for selecting a specific game

        selected_game_id = st.selectbox("Select Game", list(games.keys()), format_func=lambda x: games[x])

        # Load match data and display visualizations

        st.subheader(f"Match Report for {games[selected_game_id]}")

        match_data = fetch_match_data(conn, selected_game_id)

        if match_data is not None:
            # Extract home and away team names
            home_team_name, away_team_name = games[selected_game_id].split(" vs ")

            # Fetch additional game details
            selected_game_info = game_details[selected_game_id]
            venue = selected_game_info["venue_name"]
            attendance = selected_game_info["attendance"]
            referee = selected_game_info["referee_name"]

            # Create figure for passing network and shot map visualization
            fig = plt.figure(figsize=(15, 11))

            # **Top Row - Passing Networks & Table**
            ax1 = fig.add_axes([0.1, 0.55, 0.25, 0.35])  # Home team passing network
            pitch = Pitch(pitch_type='opta', goal_type='box', linewidth=0.85, line_color='black')
            pitch.draw(ax1)
            plot_passing_network(home_team_name, match_data, ax1, pitch, color="blue")
            ax1.set_title(f"Passing Network - {home_team_name}")
            # Add arrow under ax1
            ax1.annotate("", xy=(0.66, -0.05), xytext=(0.33, -0.05),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="->", color="black", lw=2))

            ax3 = fig.add_axes([0.65, 0.55, 0.25, 0.35])  # Away team passing network
            pitch.draw(ax3)
            plot_passing_network(away_team_name, match_data, ax3, pitch, color="red")
            ax3.set_title(f"Passing Network - {away_team_name}")
            # Add arrow under ax3
            ax3.annotate("", xy=(0.66, -0.05), xytext=(0.33, -0.05),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="->", color="black", lw=2))

            ax2 = fig.add_axes([0.4, 0.55, 0.2, 0.35])  # Stats table
            create_table(ax2, match_data, [home_team_name, away_team_name])

            # **Bottom Row - Shot Map**
            ax4 = fig.add_axes([0.05, 0.05, 0.9, 0.45])  # Full width, bigger height
            ax4.clear()

            title_text = f"{venue} | Att - {attendance} | Ref - {referee}"
            ax4.text(0.5, 1.05, title_text, fontsize=14, ha='center', transform=ax4.transAxes, fontweight='bold')

            # Generate the shot map
            create_combined_shotmap(match_data, home_team_name, away_team_name, ax=ax4)

            # Show the plot
            st.pyplot(fig)


# ✅ This ensures Streamlit runs the app when executing the script
if __name__ == "__main__":
    main()
    