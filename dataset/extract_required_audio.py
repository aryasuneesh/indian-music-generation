import json
import csv
import os
import re
import yt_dlp as youtube_dl
from pydub import AudioSegment
from music21 import converter
import numpy as np
import torch


# Define path of CSV and JSON files
csv_path = os.path.join(os.getcwd(), 'balanced_train_segments.csv')
json_path = os.path.join(os.getcwd(), 'ontology.json')
audio_dir = os.path.join(os.getcwd(), 'audio')
midi_dir = os.path.join(os.getcwd(), 'midi')

# Load the label file
def load_label_data(json_path):
    with open(json_path, "r") as f:
        label_data = json.load(f)
    return label_data

# Get the IDs for the instruments of interest
def get_instrument_ids():
    label_data = load_label_data(json_path)
    instrument_ids = []
    # create a list to store the [id, name, description] for each label
    metadata = {}
    for label in label_data:
        if label['name'].lower() in ['flute', 'tabla', 'guitar', 'sitar', 'violin', 'carnatic music', 'music of bollywood', 'bowed string instrument', 'piano', 'acoustic guitar', 'electric piano', 'violin, fiddle', 'string section']:
            instrument_ids.append(label['id'])
            if label['id'] not in metadata.keys():
                metadata[label['id']] = [label['name'], label['description']]
    return instrument_ids, metadata

# Open the CSV file
def get_compiled(instrument_ids, metadata):
    compiled = []
    #convert metadata to csv file
    with open('metadata.csv', 'w') as m:
        writer = csv.writer(m)
        for key, value in metadata.items():
            writer.writerow([key, value])
    with open('balanced_train_segments.csv', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if row[0][0] == '#':
                continue
            video_id = row[0][0:-1]
            start_time = int(row[1][:-5])
            end_time = int(row[2][:-5])
            video_link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s_{end_time}s"
            instrument_id = row[3].split(",")
            if any(x in instrument_id for x in instrument_ids):
                compiled.append([video_link, instrument_id, start_time, end_time, video_id])
    return compiled

# Download audio from YouTube link and process into MIDI
def download_audio(video_link: str, start_time: int, end_time: int, audio_dir: str, audio_format: str) -> None:
    """
    Downloads audio from a provided YouTube link and saves it to a specified directory.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(audio_dir, f"%(id)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ]
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_link])
        except youtube_dl.utils.DownloadError:
            pass


# first obtain links of all necessary video links and then, download them using the above function
def download_all(compiled, audio_dir, audio_format):
    for i in range(len(compiled)):
        download_audio(compiled[i][0], compiled[i][2], compiled[i][3], audio_dir, audio_format)
        print(f"Downloaded audio for {compiled[i][4]}")

# download_all(get_compiled(get_instrument_ids()), audio_dir, 'wav')

# while downloading the audio, trim the audio to the required length
def download_and_trim(compiled, audio_dir, audio_format):
    for i in range(len(compiled)):
        download_audio(compiled[i][0], compiled[i][2], compiled[i][3], audio_dir, audio_format)
        print(f"Downloaded audio for {compiled[i][4]}")
        try:
            audio = AudioSegment.from_file(os.path.join(audio_dir, f"{compiled[i][4]}.wav"))
            # use compiled[i][2] and compiled[i][3] to trim the audio
            audio = audio[compiled[i][2]*1000:compiled[i][3]*1000]
            # replace the original audio with the trimmed audio
            audio.export(os.path.join(audio_dir, f"{compiled[i][4]}.wav"), format="wav")
        except:
            print(f"Error in trimming audio for {compiled[i][4]}")
            continue
            
instrument_ids, metadata = get_instrument_ids()
download_and_trim(get_compiled(instrument_ids, metadata), audio_dir, 'wav')


