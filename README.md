# Football Data Visualization Streamlit App (English Premier League 2024-2025)
### main.py
## Overview
This app provides football data visualization for players in the English Premier League 2024-2025 season. It allows users to explore various aspects of a selected player’s performance, including shot maps, heatmaps for danger passes, defensive actions, passes, and key passes. All the data used in the app is scraped from football event data sources such as WhoScored and then inserted into a database to manage and query the data efficiently for visualization purposes.

## Features

- **Shot Map**: Visualizes a player's shots on goal, color-coded by whether the shot resulted in a goal, was off-target, or on target.
- **Danger Passes Heatmap**: Displays a heatmap of passes that contributed to shooting opportunities for the player.
- **Defensive Actions**: Shows various defensive actions (ball recovery, clearance, interception, tackle) on the pitch.
- **Passes**: Visualizes the player's passes, differentiating successful and unsuccessful passes.
- **Key Passes**: Visualizes key passes made by the player that directly lead to shooting opportunities.

## Technologies Used

- **Streamlit**: For building the interactive web app interface.
- **Pandas**: For data manipulation and cleaning.
- **psycopg2**: To connect to and interact with the PostgreSQL database.
- **Matplotlib**: For creating visualizations such as shot maps and heatmaps.
- **mplsoccer**: For generating football pitch visualizations.
- **Numpy**: For numerical operations and data handling.
- **io**: For handling data input and output, especially for image plotting and saving.
- **os**: For managing file paths and operating system interactions.


The app fetches football event data from a PostgreSQL database in Supabase. The data in this database is populated by a web scraper that gathers match and player statistics from WhoScored using Python's BeautifulSoup and requests libraries.
# Web Scrapper Pipeline 
### all_together.py
## Web Scraping Process
Request the Web Page: Use requests to fetch the HTML content of the WhoScored page (or another data source).
Parse HTML with BeautifulSoup: Use BeautifulSoup to parse the HTML content and extract the relevant data (such as shots, passes, tackles, etc.).
**Store Data in Database:** After extracting the data from WhoScored, the web scraper inserts it into a PostgreSQL database on Supabase, which allows for efficient querying and storage for future visualizations. The web scraper pipeline uses the following key libraries:

- **`json`** and **`time`**: For managing JSON data formats and handling timing between requests.
- **`numpy`** and **`pandas`**: For data manipulation and efficient handling of numerical and tabular data.
- **`BeautifulSoup`** (from `bs4`): For parsing and extracting specific HTML content from the web pages.
- **`pydantic`**: For defining structured data models and validating the data schema.
- **`typing`**: To specify types such as `List` and `Optional` for structured data handling.
- **`selenium`**: For automating web browser interactions to access dynamic content on WhoScored.
- **`supabase`** (from `supabase-py`): To establish a direct connection to the Supabase database and insert data via `create_client`.


## Database Structure
The database structure is organized to store player and team data  across various matches. For example, tables can include:

- **Players**: Stores player information (name, team, position, etc.).
- **Teams**: Stores teams information (name, country, etch.).
- **Event Data**: Stores player event data for each match (shots, coordinates, assists, passes, tackles, etc.).
- **Additional Info**: Stores additional match details such as date, stadium, attendance, referee, etc.

## Data Flow for Both projects

- **Data Retrieval**: The app uses Supabase to fetch pre-scraped data from the PostgreSQL database, ensuring up-to-date and consistent information.
- **Data Processing**: Retrieved data is validated and structured using Pydantic models for easy handling and visualization.
- **Data Visualization**: The app displays player and match performance data using Streamlit and visualization libraries like Matplotlib and mplsoccer.
