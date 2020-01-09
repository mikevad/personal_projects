import pandas as pd
from fuzzywuzzy import fuzz
import requests
import csv
import time
import json
import argparse
from bs4 import BeautifulSoup

def scrape_song_url(url):
	'''
	this function accepts a url and scrapes the lyrics off the webpage.
	used for genius lyric pages
	'''

    try:
        page = requests.get(url)
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics = html.find('div', class_='lyrics').get_text()

        return lyrics
    
    except:
        return None

def get_lyrics(song_title, artist_name, genius_token_string):
	'''
	searches genius.com for a song title and artist combination, using fuzzy string matching.
	if a match is found, it takes the url and applies scrape_song_url to it, and returns lyrics
	'''

    client_access_token = genius_token_string
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + client_access_token}
    search_url = base_url + '/search'
    data = {'q': song_title + ' ' + artist_name}
    
    while True:
        
        try:
            response = requests.get(search_url, data=data, headers=headers)
            json = response.json()
            remote_song_info = None
            remote_artist_info = None
            break
            
        except Exception as e:
            print(e)
            time.sleep(5)

#     Search for matches in the request response

    for hit in json['response']['hits']:

        artist_name = artist_name.lower()
        search_result_artist = hit['result']['primary_artist']['name'].lower()

        song_title = song_title.lower()
        search_result_song = hit['result']['title']

        artist_token_set_ratio = fuzz.token_sort_ratio(artist_name, search_result_artist)
        song_token_set_ratio = fuzz.token_sort_ratio(song_title, search_result_song)
        

        if (artist_token_set_ratio > 70) & (song_token_set_ratio > 70):
            remote_song_info = hit
            break

#     Extract lyrics from URL if the song was found
    if remote_song_info:
        song_url = remote_song_info['result']['url']
        lyrics = scrape_song_url(song_url)

        return lyrics

def save_lyrics(df, genius_token_string):
	'''
	this function accepts a pandas dataframe (apple music library playist) as an argument, and gets the lyrics. every 1000 songs, it saves a json file as a backup in case of crash.

	there's probably a better of doing this (splitting the dataframe into parts and parallellizing, but since the initial scrape is the hardest, might not be necessary for subsequent thing)
	'''
    
    print('scraping some lyrics...')
    
    lyrics = []
    order_list = []
    song_title_list = []
    artist_name_list = []
    
    search_terms = [(song, artist) for song, artist in zip(df['Name'], df['Artist'])]
    number_of_terms = len(search_terms)
    
    for n, search_tuple in zip(range(number_of_terms), search_terms):
        
        song_title = search_tuple[0]
        artist_name = search_tuple[1]
        epoch_time = int(time.time())
        song_lyrics = get_lyrics(song_title, artist_name, genius_token_string)
        
        try:
        
            if (n+1) % 1000 != 0:
                lyrics.append(song_lyrics)
                order_list.append(n)
                song_title_list.append(song_title)
                artist_name_list.append(artist_name)
                
                if (n+1) == number_of_terms:
                
                    checkpoint = {
                        'order':order_list,
                        'lyrics':lyrics,
                        'song_title':song_title_list,
                        'artist_name':artist_name_list
                        }
                    with open(f'data/lyrics_backup_{epoch_time}.json', 'w') as fp:
                        json.dump(checkpoint, fp)
                    break

            elif (n+1) % 1000 == 0:
                lyrics.append(song_lyrics)
                order_list.append(n+1)
                song_title_list.append(song_title)
                artist_name_list.append(artist_name)

                checkpoint = {
                    'order':order_list,
                    'lyrics':lyrics,
                    'song_title':song_title_list,
                    'artist_name':artist_name_list
                    }

                with open(f'data/lyrics_backup_{epoch_time}.json', 'w') as fp:
                    json.dump(checkpoint, fp)

                lyrics.clear()
                order_list.clear()
                song_title_list.clear()
                artist_name_list.clear()
                if (n+1) % 10000 == 0:
                    print(f'row {n+1} completed...')
                
        except:
            print(f'error found on row {n+1}')
    
    print('Done')
            
#     return number_of_terms

if __name__ == 'main':
	parser = argparse.ArgumentParser()
	parser.add_argument('playlist_filepath', type = str, help = 'filepath of the apple music file')
	# parser.add_argument('genius_access_token', type = str, help = 'genius site credential')
	args = parser.parse_args()

	json_file = json.load(open('cred.json'))
	client_access_token = json_file['client_access_token']

	music = pd.read_csv(open(args.playlist_filepath, 'r'), sep = '\t', encoding = 'utf-8')
	music.to_csv('data/all_music.csv', index = None)
	save_lyrics(music, client_access_token)