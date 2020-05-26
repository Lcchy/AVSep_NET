import sys
import os
import subprocess
import copy
import shutil
import numpy as np
import datetime
import random
import scipy.io.wavfile
from PIL import Image
import torch
import sys
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from parameters import PATH_TO_DATA_CSV, PATH_TO_CACHE, PATH_TO_VISUAL, PATH_TO_AUDIO, PATH_TO_MEDIA, \
    V_LENGTH, A_LENGTH, STFT_NORMALIZED, VERBOSE
from logger import Logger_custom

LOGGER = Logger_custom("Global Logger")


def custom_join(d, out="", leave_out=[]):
    """Join a dict of strings recursively using an arg accumulator.
    Leaves out part of dict if key is in given list.
    Terminates because I believe."""
    if type(d) == dict and len(d) == 0:
        return ""
    elif type(d) != dict:
        return (str(d) + " " + out).rstrip()
    else:
        first_key = list(d)[0]
        head = d[first_key] if first_key not in leave_out else {}
        tail = {key:value for (key,value) in d.items() if key != first_key and not key in leave_out}
        return (custom_join(head, out, leave_out) + " " + custom_join(tail, out, leave_out)).rstrip()


def install_yt_dl():
    """Manually download youtube-dl for latest working version (and install ffmpeg)"""
    assert sys.platform == "linux"
    cmd1 = "sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl"
    cmd2 = "sudo chmod a+rx /usr/local/bin/youtube-dl"
    cmd3 = "yes | sudo apt-get install ffmpeg"
    subprocess.run(cmd1.split(" "), shell=True)
    subprocess.run(cmd2.split(" "), shell=True)
    subprocess.run(cmd3.split(" "), shell=True)


class AVDataset(Dataset):

    def __init__(self):
        self.csv_data = pd.read_csv(PATH_TO_DATA_CSV, header=2, quotechar='"', skipinitialspace=True)
        self.av_parameters = {
            'a_length': 1,
            'v_length': 0,
            'a_codec': "copy",
            'a_channels': 1,
            'a_freq': 48000,
            'v_size': (640,480),
            'v_framerate': "25/1",
            'v_codec': "copy",
            'v_aspect': "4:3",
            'v_pixel': "+",
            'cache_path': PATH_TO_CACHE,
            'media_path': PATH_TO_MEDIA,
            'v_path': PATH_TO_VISUAL,
            'a_path': PATH_TO_AUDIO,
        }
        self.data_list = os.listdir(self.av_parameters['cache_path'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """Returns 3 tensors: audio, visual, match (bool_int)"""
        return torch.load(self.av_parameters['cache_path'] / self.data_list[index])


    def download_database(self):
        """Downoad entire Audio-Visual database by randomly sampling the Youtube videos to the format given in av_parameters.
        Middle sample is for positive label, random for negative."""
        p = self.av_parameters

        # Purge the media directory if necessary
        if len(os.listdir(p['v_path'])) + len(os.listdir(p['a_path'])) > 0:
            shutil.rmtree(p['v_path'])
            shutil.rmtree(p['a_path'])
            os.mkdir(p['v_path'])
            os.mkdir(p['a_path'])
        for index in range(len(self.csv_data)):
            # Progress bar
            progress = int(20 * (index + 1) / len(self.csv_data))
            LOGGER.print_log("[" + "=" * progress + " " * (20 - progress) + "]", end="\033[K\r")

            # Actual download calls
            yt_id = self.csv_data.iloc[index, 0]
            mid_pos = (self.csv_data.iloc[index, 1] + self.csv_data.iloc[index, 2]) / 2
            rand_pos = random.uniform(self.csv_data.iloc[index, 1], self.csv_data.iloc[index, 2] - 1)    # Taking a 1 sec margin
            self.yt_download(yt_id, mid_pos, p['a_length'], p['v_length'], match=1)
            self.yt_download(yt_id, rand_pos, p['a_length'], p['v_length'], match=0)


    def cache_data(self):
        """Store the downloaded media into torch tensor format in cache folder.
        Pair up visual and audio parts randomly to create false pairs.
        Assumes that the file lists in visual and audio correspond."""
        p = self.av_parameters
        v_file_list = os.listdir(p['v_path'])
        permute = np.random.permutation(len(v_file_list))

        for (index, v_filename) in enumerate(v_file_list):
            # Progress bar
            progress = int(20 * (index + 1) / len(v_file_list))
            LOGGER.print_log("[" + "=" * progress + " " * (20 - progress) + "]", end="\033[K\r")
            
            filename = v_filename.split(".")[0]
            match = int(filename[-1])            # Can't use .split("_") because it occurs in filename

            v_file = Image.open(p['v_path'] / v_filename)
            v_file = np.copy(v_file)
            v_tensor = torch.tensor(v_file)
            v_tensor = v_tensor.permute(2,0,1).float()  # Get the torch input dims right

            if not match:
                #select random audio to form negative pair by taking the yt_id of a random file
                rand_index = permute[index]
                a_filename = v_file_list[rand_index].split(".")[0][:-2] + "_0.wav"
            else:
                a_filename = filename + ".wav"

            freq, a_file = scipy.io.wavfile.read(p['a_path'] / a_filename)
            assert freq == p['a_freq']
            a_tensor = self.process_audio(a_file)
            a_tensor = torch.unsqueeze(a_tensor, 0).float()
            a_tensor = a_tensor.permute(0,2,1)

            match = torch.tensor(match)

            if VERBOSE: LOGGER.print_log("Caching {} and {} into {} with dims: {} | {} | {}".format(a_filename, v_filename, filename, a_tensor.size(), v_tensor.size(), match.size()), end="\033[K\n")
            torch.save((a_tensor, v_tensor, match), p['cache_path'] / (filename + ".pt"))

        # Update data list
        self.data_list = os.listdir(self.av_parameters['cache_path'])


    def process_audio(self, a_file):
        """Process audio before caching. STFT"""
        p = self.av_parameters
        a_array = np.copy(a_file)
        a_tensor = torch.tensor(a_array, dtype=torch.float64)
        # Values from the paper are not opt, n_fft > win_length ??!
        spectrogram = torch.stft(a_tensor, 512, hop_length=240, win_length=480, window=torch.hann_window(480),
             center=True, pad_mode='reflect', normalized=STFT_NORMALIZED, onesided=True)
        log_spectrogram = torch.log(spectrogram[:,:200,0] ** 2 + 1)
        # normalize
        return log_spectrogram
        

    def yt_download(self, yt_id, pos, a_length, v_length, match):
        """Download the images and sound of a given youtube video using youtube-dl and ffmpeg by running a bash cmd
        Todo: use a single ffmpeg call with stream mapping."""

        a_cmd_str = self.form_command(yt_id, pos, a_length, match, leave_out="visual")
        v_cmd_str = self.form_command(yt_id, pos, v_length, match, leave_out="audio")

        if VERBOSE: print("\n---- AUDIO COMMAND: " + a_cmd_str + "\n")

        subprocess.run(a_cmd_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # shell=True is a security issue
                
        if VERBOSE: print("\n---- VISUAL COMMAND: " + v_cmd_str + "\n")    

        subprocess.run(v_cmd_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def form_command(self, yt_id, pos, length, match, leave_out):
        """Parse arguments into command to form the bash execution string"""
        p = self.av_parameters

        cmd = {
            'stream handler': "ffmpeg", 
            'input flags': {
                'overwrite': "-y",
                'position': "-ss {}".format(str(datetime.timedelta(seconds=pos))),                            # format: 00:00:00
                'duration': "-t {}".format(str(datetime.timedelta(seconds=length))) if length > 0 else "",                  # 
                'input': "-i",
            },
            'youtube-dl': {
                'head': "$(",
                'main': "/usr/local/bin/youtube-dl",
                'quality': "-f best",
                'url flag': "--get-url",
                'url': "https://www.youtube.com/watch?v=" + yt_id,
                'tail': ")",
            },
            'output': {
                'visual': {
                    'nb visual frames': "-frames:v 1" if length == 0 else "",
                    'framerate': "-r {}".format(p['v_framerate']) if length > 0 else "",
                    'image size': "-s {}x{}".format(p['v_size'][0], p['v_size'][1]),     #INPUT_DIM    
                    # 'birate': '-b {}',
                    #'pixel format': "-pix_fmt {}".format(p['v_pixel']),
                    #'video codec': "-codec:v {}".format(p['v_codec']),
                    #'aspect': "-aspect {}".format(p['v_aspect']),
                    'output path': p['v_path'] / ("{}_{}".format(yt_id, match) + ".jpg" if length == 0 else ".mp4"),
                },
                'audio': {
                    #'audio codec': "-codec:a {}".format(p['a_codec']),
                    #'layout': 
                    # 'birate': '-b {}',
                    'frequency': "-ar {}".format(p['a_freq']),
                    'channels': "-ac {}".format(p['a_channels']),
                    'output path': p['a_path'] / ("{}_{}".format(yt_id, match) + ".wav"),
                },
            },
        }

        cmd_str = custom_join(cmd, leave_out=[leave_out])
        return cmd_str