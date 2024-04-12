import json
import time

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from pydantic import BaseModel
from typing import List, Optional

from selenium import webdriver

from supabase import create_client, Client
from bs4 import BeautifulSoup as BS
import bs4
import soupsieve

driver = webdriver.Chrome()




class MatchEvent(BaseModel):
    id: int
    event_id: int
    minute: int
    second: Optional[float] = None
    team_id: int
    player_id: int
    x: float
    y: float
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    qualifiers: List[dict]
    is_touch: bool
    blocked_x: Optional[float] = None
    blocked_y: Optional[float] = None
    goal_mouth_z: Optional[float] = None
    goal_mouth_y: Optional[float] = None
    is_shot: bool
    card_type: bool
    is_goal: bool
    type_display_name: str
    outcome_type_display_name: str
    period_display_name: str
    match_id: Optional[int] = None

def insert_match_events(df, supabase,match_id):
    # Convert DataFrame to a list of dictionaries
    events = df.to_dict(orient='records')

    # Create MatchEvent instances with match_id set
    events_data = [MatchEvent(**{**event, 'match_id': match_id}) for event in events]

    # Convert MatchEvent instances to a list of dictionaries
    events_dict_list = [event.dict() for event in events_data]

    # Upsert data into the 'match_events' table
    execution = supabase.table('match_events').upsert(events_dict_list).execute()


class Player(BaseModel):
    player_id: int
    shirt_no: int
    name: str
    age: int
    position: str
    team_id: int

def insert_players(team_info, supabase):
    players = []
    for team in team_info:
        for player in team['players']:
            players.append({
                'player_id': player['playerId'],
                'team_id': team['team_id'],
                'shirt_no': player['shirtNo'],
                'name': player['name'],
                'position': player['position'],
                'age': player['age']
            })

    execution = supabase.table('players').upsert(players).execute()

class Team(BaseModel):
    team_id: int
    team_name: str
    manager_name: str
    country_name:str

def insert_teams(team_info, supabase):
    teams = []
    for team in team_info:
            teams.append({
                'team_id': team['team_id'],
                'team_name': team['name'],
                'manager_name': team['manager_name'],
                'country_name':team["country_name"]
            })

    execution = supabase.table('teams').upsert(teams).execute()

class AdditionalInfo(BaseModel):
    match_id:Optional[int] = None #primarykey
    start_time: str
    venue_name: str
    attendance: int
    referee_name: str
    home_team_name: str
    away_team_name: str
    home_team_id: int
    away_team_id: int
    ht_score: str
    ft_score: str
    firsthalf_addition_minutes: int
    secondhalf_addition_minutes: int

def insert_additionalInfo(df, supabase, match_id):
    # Convert DataFrame to a list of dictionaries
    additional_info = df.to_dict(orient='records')

    # Create AdditionalInfo instances with match_id set
    additional_info_data = [AdditionalInfo(**{**info, 'match_id': match_id}) for info in additional_info]

    # Convert AdditionalInfo instances to a list of dictionaries
    additional_info_dict_list = [info.dict() for info in additional_info_data]

    # Upsert data into the 'additional_info' table
    execution = supabase.table('additional_info').upsert(additional_info_dict_list).execute()




supabase_password ="8FQUT1zEGep0numj"
project_url = "https://dukvqeeuktlsotlwgtjb.supabase.co"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR1a3ZxZWV1a3Rsc290bHdndGpiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDIzNzkzMjYsImV4cCI6MjAxNzk1NTMyNn0.UlaIj_to62_TLGvR-7bgYPB3XQfbr_MPhbP7yZxUgUY"
supabase = create_client(project_url, api_key)

def scrape_match_events(whoscored_url, driver):
    driver.get(whoscored_url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    element = soup.select_one('script:-soup-contains("matchCentreData")')
    if element is None:
        print("Error: Could not find 'matchCentreData' element.")
        return  # or raise an exception, depending on your requirements

    matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
    match_events = matchdict['events']
    df = pd.DataFrame(match_events)
    match_id = json.loads(element.text.split("matchId:")[1].split(",")[0])
    df['match_id'] = match_id
    df.dropna(subset='playerId', inplace=True)

    df = df.where(pd.notnull(df), None)
    df = df.rename(
        {
            'eventId': 'event_id',
            'expandedMinute': 'expanded_minute',
            'outcomeType': 'outcome_type',
            'isTouch': 'is_touch',
            'playerId': 'player_id',
            'teamId': 'team_id',
            'endX': 'end_x',
            'endY': 'end_y',
            'blockedX': 'blocked_x',
            'blockedY': 'blocked_y',
            'goalMouthZ': 'goal_mouth_z',
            'goalMouthY': 'goal_mouth_y',
            'isShot': 'is_shot',
            'cardType': 'card_type',
            'isGoal': 'is_goal'
        },
        axis=1
    )
    df['period_display_name'] = df['period'].apply(lambda x: x['displayName'])
    df['type_display_name'] = df['type'].apply(lambda x: x['displayName'])
    df['outcome_type_display_name'] = df['outcome_type'].apply(lambda x: x['displayName'])
    df.drop(columns=["period", "type", "outcome_type"], inplace=True)
    if 'is_goal' not in df.columns:
        print('missing goals')
        df['is_goal'] = False
    if 'card_type' in df.columns:
        # Access the 'card_type' column
        # Your code to work with the 'card_type' column goes here
        pass  # Placeholder to indicate that no action is needed
    else:
        # Handle the case where the 'card_type' column is missing
        # For example, you could print a message or perform an alternative action
        # Since 'card_type' column is missing, we set it to None for all rows
        df['card_type'] = None

    df = df[~(df['type_display_name'] == "OffsideGiven")]
    df = df[[
        'id', 'event_id', 'minute', 'second', 'team_id', 'player_id', 'x', 'y', 'end_x', 'end_y',
        'qualifiers', 'is_touch', 'blocked_x', 'blocked_y', 'goal_mouth_z', 'goal_mouth_y', 'is_shot',
        'card_type', 'is_goal', 'type_display_name', 'outcome_type_display_name',
        'period_display_name'
    ]]
    df[['id', 'event_id', 'minute', 'team_id', 'player_id']] = df[
        ['id', 'event_id', 'minute', 'team_id', 'player_id']].astype(np.int64)
    df[['second', 'x', 'y', 'end_x', 'end_y']] = df[['second', 'x', 'y', 'end_x', 'end_y']].astype(float)
    df[['is_shot', 'is_goal', 'card_type']] = df[['is_shot', 'is_goal', 'card_type']].astype(bool)

    df['is_goal'] = df['is_goal'].fillna(False)
    df['is_shot'] = df['is_shot'].fillna(False)
    for column in df.columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.float32:
            df[column] = np.where(
                np.isnan(df[column]),
                None,
                df[column]
            )
    insert_match_events(df, supabase, match_id)
    team_info = []
    team_info.append({
        'team_id': matchdict['home']['teamId'],
        'name': matchdict['home']['name'],
        'country_name': matchdict['home']['countryName'],
        'manager_name': matchdict['home']['managerName'],
        'players': matchdict['home']['players'],
    })

    team_info.append({
        'team_id': matchdict['away']['teamId'],
        'name': matchdict['away']['name'],
        'country_name': matchdict['away']['countryName'],
        'manager_name': matchdict['away']['managerName'],
        'players': matchdict['away']['players'],
    })

    insert_players(team_info, supabase)
    insert_teams(team_info, supabase)
    venue_name = matchdict.get("venueName", None)
    attendance = matchdict.get("attendance", None)
    referee_first_name = matchdict["referee"].get("firstName", "")
    referee_last_name = matchdict["referee"].get("lastName", "")
    referee_name = f"{referee_first_name} {referee_last_name}"
    home_team_name = matchdict['home'].get('name', None)
    away_team_name = matchdict['away'].get('name', None)
    home_team_id = matchdict['home'].get('teamId', None)
    away_team_id = matchdict['away'].get('teamId', None)
    start_time = matchdict.get("startTime", None)
    ht_score = matchdict.get("htScore", None)
    ft_score = matchdict.get("ftScore", None)
    firsthalf_addition_minutes = matchdict["periodEndMinutes"].get("1", None)
    secondhalf_addition_minutes = matchdict["periodEndMinutes"].get("2", None)
    additional_info_data = {
        "venue_name": [venue_name],
        "attendance": [attendance],
        "referee_name": [referee_name],
        "home_team_name": [home_team_name],
        "away_team_name": [away_team_name],
        "home_team_id": [home_team_id],
        "away_team_id": [away_team_id],
        "start_time": [start_time],
        "ht_score": [ht_score],
        "ft_score": [ft_score],
        "firsthalf_addition_minutes": [firsthalf_addition_minutes],
        "secondhalf_addition_minutes": [secondhalf_addition_minutes]
    }

    additional_info_df = pd.DataFrame(additional_info_data)
    insert_additionalInfo(additional_info_df, supabase, match_id)
    return print('Success')


driver.get('https://www.whoscored.com/Teams/163/Fixtures/England-Sheffield-United')
time.sleep(3)
soup = BeautifulSoup(driver.page_source, 'html.parser')
all_urls = soup.select('a[href*="\/Live\/"]')
all_urls = list(set([
    'https://www.whoscored.com' + x.attrs['href']
    for x in all_urls
]))
for url in all_urls:
    print(url)
    scrape_match_events(
        whoscored_url=url,
        driver=driver
    )

    time.sleep(2)


