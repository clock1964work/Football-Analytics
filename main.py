import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer.pitch import VerticalPitch,Pitch
import io  # Add this line
import numpy as np
from flask import Flask
app = Flask(__name__)
# Function to establish a connection to the PostgreSQL database
def connect_to_database():
  conn = psycopg2.connect(
      host="aws-0-eu-central-1.pooler.supabase.com",
      database="postgres",
      user="postgres.dukvqeeuktlsotlwgtjb",
      password="8FQUT1zEGep0numj"
  )
  return conn

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
def fetch_shot_data(conn, player):
  cur = conn.cursor()
  cur.execute("""
      SELECT *
      FROM match_events
      INNER JOIN players ON match_events.player_id = players.player_id
      WHERE players.name = %s;
  """, (player,))

  # Fetch all data from the SQL query
  shot_data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
  cur.close()

  # Filter the DataFrame based on is_shot column
  shot_data = shot_data[shot_data['is_shot'] == True]

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

    # Handle division by zero when len(data) is zero
    if len(data) != 0:
        bin_statistic["statistic"] = bin_statistic["statistic"] / len(data)
    else:
        # Handle the case when len(data) is zero to avoid division by zero

        pass

    # Check for NaN values before dividing
    if not np.isnan(bin_statistic["statistic"]).any():
        bin_statistic["statistic"] = bin_statistic["statistic"] / len(data)
    else:
        # Handle the case when bin_statistic["statistic"] contains NaN values

        pass

    heatmap = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax)
    # Add legend indicating color range and danger level below the plot
    cbar = plt.colorbar(heatmap, orientation='horizontal', aspect=20, shrink=0.5)
    cbar.set_ticks([bin_statistic['statistic'].min(), bin_statistic['statistic'].max()])
    cbar.ax.set_xticklabels(['Low', 'High'], fontsize=8)

    plt.title('Starting location of Danger passes heatmap by ' + player_name)
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
    # Adjust legend size and position
    # Add legend below the pitch, closer to the line
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=4,
              fontsize=legend_font_size)

    plt.title(f'Defensive Actions for {player_name}')

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
    st.pyplot(fig)

    # Calculate total key passes
    num_key_passes = len(key_passes_data)


    comment = f"Total key passes: {num_key_passes}"

    # Display the comment below the plot
    st.text(comment)


# Modify the main function to include the passes display
def main():
   # Connect to the PostgreSQL database
   conn = connect_to_database()

   # Streamlit UI
   st.title('Football Data Visualization App\nEnglish Premier League 2023-2024')


   # Add filter to the sidebar for selecting team
   selected_team = st.sidebar.selectbox('Select Team', fetch_team_names(conn))

   # Fetch players based on selected team
   players = fetch_players_by_team(conn, selected_team)

   # Input widget for selecting player
   selected_player = st.sidebar.selectbox('Select Player', players)

   # Fetch shot, passes,danger passes ,defensive actions and key passes data for the selected player
   shot_data = fetch_shot_data(conn, selected_player)
   heatmap_data = fetch_danger_passes_data(conn, selected_player)
   defensive_actions_data = fetch_defensive_actions_data(conn, selected_player)
   passes_data = fetch_passes_data(conn,selected_player)
   key_passes_data = fetch_key_passes_data(conn,selected_player)


   # Display shot map, heatmap, defensive actions plot, and passes
   visualization_type = st.sidebar.radio('Select Visualization', ['Shot Map', 'Heatmap', 'Defensive Actions', 'Passes','Key Passes'])
   if visualization_type == "Shot Map":
       display_shot_map(shot_data, selected_player)
   elif visualization_type == "Heatmap":
       display_heatmap(heatmap_data, selected_player)
   elif visualization_type == "Defensive Actions":
       display_defensive_actions_plot(defensive_actions_data, selected_player)
   elif visualization_type == "Passes":
       display_passes(passes_data, selected_player)
   elif visualization_type == "Key Passes":
       display_key_passes(key_passes_data,selected_player)

# Run the app

if __name__ == '__main__':
    main()
