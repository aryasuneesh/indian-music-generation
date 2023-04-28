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

# Get the IDs for the instruments of interest
def get_instrument_ids():
    label_data = load_label_data(json_path)
    instrument_ids = []
    for label in label_data:
        if label['name'].lower() in ['flute', 'tabla', 'guitar', 'sitar', 'violin', 'carnatic music', 'music of bollywood', 'bowed string instrument', 'piano', 'acoustic guitar', 'electric piano', 'violin, fiddle', 'string section']:
            instrument_ids.append(label['id'])
    return instrument_ids

# Open the CSV file
def get_compiled(instrument_ids):
    compiled = []
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
def download_and_process():
    compiled = get_compiled(get_instrument_ids())
    for link, instrument_id, start_time, end_time, video_id in compiled:
        print(f"Processing {link}...")
        # Download audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f"{audio_dir}/%(id)s.%(ext)s",
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        # Load audio and trim to desired length
        audio_file = f"{audio_dir}\\{video_id}.mp3"
        audio = AudioSegment.from_file(audio_file)
        audio = audio[start_time * 1000:end_time * 1000] # in milliseconds
        # Export to WAV for processing
        wav_file = f"{audio_dir}\\{video_id}.wav"
        audio.export(wav_file, format="wav")
        # Convert WAV to MIDI
        midi_file = f"{midi_dir}\\{video_id}.mid"
        try:
            converter.parse(wav_file).write('midi', midi_file)
        except:
            print(f"Could not convert {wav_file} to MIDI.")
            continue
