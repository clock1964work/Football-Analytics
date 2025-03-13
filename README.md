# Football Analytics Repository

This repository is dedicated to the analysis of football data, focusing on various aspects like player analytics, football modeling, and clustering analysis. It includes tools and models for data collection, processing, and visualization. The repository also offers interactive data visualizations using Streamlit, web scraping scripts for football event data, and clustering models for football-related data.

## Projects in this Repository

### 1. **Streamlit App for Football Data Visualization**

#### Overview
This app provides football data visualization for players in the English Premier League 2024-2025 season. It allows users to explore various aspects of a selected player’s performance, including shot maps, heatmaps for danger passes, defensive actions, passes, and key passes. All the data used in the app is scraped from football event data sources such as WhoScored and then inserted into a database to manage and query the data efficiently for visualization purposes.

#### Features
- **Shot Map**: Visualizes a player's shots on goal, color-coded by whether the shot resulted in a goal, was off-target, or on target.
- **Danger Passes Heatmap**: Displays a heatmap of passes that contributed to shooting opportunities for the player.
- **Defensive Actions**: Shows various defensive actions (ball recovery, clearance, interception, tackle) on the pitch.
- **Passes**: Visualizes the player's passes, differentiating successful and unsuccessful passes.
- **Key Passes**: Visualizes key passes made by the player that directly lead to shooting opportunities.

#### Technologies Used
- **Streamlit**: For building the interactive web app interface.
- **Pandas**: For data manipulation and cleaning.
- **psycopg2**: To connect to and interact with the PostgreSQL database.
- **Matplotlib**: For creating visualizations such as shot maps and heatmaps.
- **mplsoccer**: For generating football pitch visualizations.
- **Numpy**: For numerical operations and data handling.
- **io**: For handling data input and output, especially for image plotting and saving.
- **os**: For managing file paths and operating system interactions.

The app fetches football event data from a PostgreSQL database in Supabase. The data in this database is populated by a web scraper that gathers match and player statistics from WhoScored using Python's BeautifulSoup and requests libraries.

---

### 2. **Web Scraping Pipeline**

#### Overview
This section explains the web scraping process, where the data is fetched from WhoScored using BeautifulSoup and requests. The data is then inserted into a PostgreSQL database on Supabase for further analysis and visualization.

#### Key Libraries Used
- **json** and **time**: For managing JSON data formats and handling timing between requests.
- **numpy** and **pandas**: For data manipulation and efficient handling of numerical and tabular data.
- **BeautifulSoup (from bs4)**: For parsing and extracting specific HTML content from the web pages.
- **pydantic**: For defining structured data models and validating the data schema.
- **typing**: To specify types such as List and Optional for structured data handling.
- **selenium**: For automating web browser interactions to access dynamic content on WhoScored.
- **supabase (from supabase-py)**: To establish a direct connection to the Supabase database and insert data via `create_client`.

#### Data Flow for These Projects
- **Data Retrieval**: The app uses Supabase to fetch pre-scraped data from the PostgreSQL database, ensuring up-to-date and consistent information.
- **Data Processing**: Retrieved data is validated and structured using Pydantic models for easy handling and visualization.
- **Data Visualization**: The app displays player and match performance data using Streamlit and visualization libraries like Matplotlib and mplsoccer.

---

### 3. **Clustering_Passes.py**

#### Overview
This script performs clustering analysis on football passes, specifically focusing on progressive passes made during a match. It analyzes the progression and direction of passes made by a specific player (Liverpool's Trent Alexander-Arnold, in this case), using K-Means clustering. The script visualizes these progressive passes on a football pitch grid, showing the success or failure of each pass within the identified clusters.

#### Key Features
- **Progressive Passes Identification**: The script defines and identifies progressive passes based on their movement and direction on the field.
- **Clustering with KMeans**: Using the KMeans algorithm, it clusters the passes into different groups based on their start and end coordinates, pass angles, and other features.
- **Elbow Method & GAP Statistic**: To find the optimal number of clusters, the script uses the elbow method and GAP statistic.
- **Visualization on Football Pitch**: The final output displays the clustered passes on a football pitch using the mplsoccer library, color-coded for successful and unsuccessful passes.

#### Technologies Used
- **pandas**: For data manipulation and cleaning.
- **numpy**: For numerical operations.
- **sklearn**: For performing KMeans clustering.
- **mplsoccer**: For visualizing passes on a football pitch.
- **matplotlib**: For plotting and visualizing data.

---

### 4. **Kmean_Players_Similarity_model.py**

#### Overview
This script clusters football players based on their performance metrics and calculates player similarity using KMeans clustering. It uses player statistics across various metrics (passing, defending, creation, and shooting scores) and visualizes the similarity between players within the same cluster using Euclidean distances. The script helps identify players with similar playing styles and attributes, useful for scouting or comparison purposes.

#### Key Features
- **Position Grouping**: The script categorizes players into different position groups (e.g., forwards, midfielders, defenders) based on their position data.
- **Data Normalization**: The script normalizes player statistics (such as passes, defense actions, and shooting metrics) to make them comparable across players.
- **Player Metrics**: The script calculates scores for various categories such as passing, defending, creation, and shooting based on player performance data.
- **KMeans Clustering**: It performs clustering of players into groups using KMeans based on their overall performance metrics.
- **Player Similarity**: The script calculates the similarity between players within the same cluster using Euclidean distance and ranks them based on similarity scores.
- **Visualization**: The results are visualized, and the most similar players are displayed in an interactive table.

#### Technologies Used
- **pandas**: For data manipulation and cleaning.
- **sklearn**: For performing KMeans clustering and Euclidean distance calculations.
- **plotly**: For interactive visualizations.
- **matplotlib**: For plotting and visualizing data.
- **numpy**: For numerical operations.

---

## Installation and Requirements

To run the scripts and applications in this repository, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn plotly matplotlib mplsoccer streamlit psycopg2
