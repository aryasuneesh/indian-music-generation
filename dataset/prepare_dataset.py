import json
import csv
import os

from typing import Tuple
import yt_dlp as youtube_dl
from pydub import AudioSegment
from music21 import converter

import numpy as np

import torch
import torchaudio

# Define path of CSV and JSON files
csv_path = os.path.join(os.getcwd(), 'balanced_train_segments.csv')
json_path = os.path.join(os.getcwd(), 'ontology.json')
audio_dir = os.path.join(os.getcwd(), 'audio')


# Load the label file
def load_label_data(json_path):
    with open(json_path, "r") as f:
        label_data = json.load(f)
    return label_data

    
# Get the ID-description mapping for the instruments of interest
def get_instrument_dict():
    instrument_dict = {}
    for label in label_data:
        if label['name'].lower() in ['flute', 'tabla', 'guitar', 'sitar', 'violin', 'carnatic music', 'music of bollywood', 'bowed string instrument', 'piano', 'acoustic guitar', 'electric piano', 'violin, fiddle', 'string section']:
            instrument_dict[label['id']] = label['name']
    return instrument_dict

# Open the CSV file
def get_compiled():
    instrument_dict = get_instrument_dict()
    compiled = []
    with open('balanced_train_segments.csv', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if row[0][0] == '#':
                continue
            video_id = row[0][:-1]
            start_time = int(row[1][:-5])
            end_time = int(row[2][:-5])
            video_link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s_{end_time}s" 
            instrument_id = row[3].split(",")
            instrument_desc = [instrument_dict[id] for id in instrument_id if id in instrument_dict]
            if instrument_desc:
                compiled.append([video_link, instrument_desc, start_time, end_time, video_id])
    return compiled

# Download audio from YouTube and prepare dataset
sr = 22050
audio_length = 5  # in seconds

def download_audio(video_link: str, start_time: int, end_time: int, audio_dir: str, audio_format: str) -> None:
    """
    Downloads audio from a provided YouTube link and saves it to a specified directory.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(audio_dir, f"%(id)s.{audio_format}"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ]
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_link])

def preprocess_audio(audio_file: str) -> Tuple[np.ndarray, str]:
    """
    Preprocesses an audio file and returns a tuple containing the raw audio waveform and its corresponding instrument description.
    """
    waveform, _ = torchaudio.load(audio_file, sr=sr)
    num_segments = int(waveform.shape[1] // (sr * audio_length))
    text = np.random.randint(0, 20000, (num_segments, 256))
    
    return waveform, text


# Download audio and preprocess the dataset
def get_dataset():
    dataset = []
    compiled = get_compiled()
    for data in compiled:
        video_link, instrument_desc, start_time, end_time, video_id = data
        filename = f"{video_id}.webm"
        download_audio(video_link, start_time, end_time, audio_dir, "webm")
        input_file = os.path.join(audio_dir, filename)
        output_file = os.path.join(audio_dir, f"{video_id}.wav")
        AudioSegment.from_file(input_file).export(output_file, format="wav")
        os.remove(input_file)
        wav, text = preprocess_audio(output_file)
        for i in range(len(wav)):
            dataset.append((wav[i], instrument_desc[i]))
    return dataset


# Split dataset and create PyTorch dataloader
def create_dataloader():
    dataset = get_dataset()
    np.random.shuffle(dataset)
    dataset_size = len(dataset)
    split = int(dataset_size * 0.8)  # use 80% of the data for training
    train_set = dataset[:split]
    val_set = dataset[split:]

    train_data = [(torch.from_numpy(wav), text) for (wav, text) in train_set]
    val_data = [(torch.from_numpy(wav), text) for (wav, text) in val_set]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

    return train_loader, val_data, train_data