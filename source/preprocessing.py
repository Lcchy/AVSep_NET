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
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchaudio
import pandas as pd
from parameters import PATH_TO_DATA_CSV, PATH_TO_CACHE, PATH_TO_VISUAL, PATH_TO_AUDIO, PATH_TO_MEDIA, \
    V_LENGTH, A_LENGTH, DOWNLOAD_YT_MEDIA, CACHE_DATA, STFT_NORMALIZED


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
    subprocess.run(cmd1.split(" "), shell=True, stdout=subprocess.DEVNULL)
    subprocess.run(cmd2.split(" "), shell=True, stdout=subprocess.DEVNULL)
    subprocess.run(cmd3.split(" "), shell=True, stdout=subprocess.DEVNULL)


class AVDataset(Dataset):

    def __init__(self, size):
        self.csv_data = pd.read_csv(PATH_TO_DATA_CSV, header=2, quotechar='"', skipinitialspace=True)
        self.csv_data = self.csv_data[:size]
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
        if DOWNLOAD_YT_MEDIA and sys.platform == "linux": self.download_database()
        if CACHE_DATA: self.cache_data()
        self.size = len(os.listdir(self.av_parameters['cache_path']))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        p = self.av_parameters
        yt_id = self.csv_data.iloc[index//2, 0]
        tensor = torch.load(p['cache_path'] / "{}_{}".format(yt_id, index % 2))
        a_tensor = tensor[0,257,200]
        v_tensor = tensor[1,:,:]
        label = tensor[2,0,0]
        return a_tensor, v_tensor, label


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
            yt_id = self.csv_data.iloc[index, 0]
            mid_pos = (self.csv_data.iloc[index, 1] + self.csv_data.iloc[index, 2]) / 2
            rand_pos = random.uniform(self.csv_data.iloc[index, 1], self.csv_data.iloc[index, 2] - 1)    # Taking a 1 sec margin
            self.yt_download(yt_id, mid_pos, p['a_length'], p['v_length'], match=1)
            self.yt_download(yt_id, rand_pos, p['a_length'], p['v_length'], match=0)


    def cache_data(self):
        """Store the downloaded media into torch tensor format in cache folder.
        Pair up visual and audio parts randomly to create false pairs."""
        p = self.av_parameters
        permute = np.random.permutation(len(self.csv_data))

        for v_filename in os.listdir(p['v_path']):
            filename = v_filename.split(".")[0]
            yt_id = v_filename.split(".")[0].split("_")[1]
            match = int(filename.split("_")[1])

            v_file = Image.open(p['v_path'] / v_filename)
            v_file = np.asarray(v_file)
            v_tensor = torch.tensor(v_file)

            if filename.split("_")[1] == "0":
                #select random audio to form negative pair
                rand_index = permute[self.csv_data.iloc[rand_index, 0].index(yt_id)]
                a_filename = self.csv_data.iloc[rand_index, 0] + "_0.wav"
            else:
                a_filename = filename + ".wav"

            freq, a_file = scipy.io.wavfile.read(p['a_path'] / a_filename)
            assert freq == p['a_freq']
            a_tensor = self.process_audio(a_file)

            final_tensor = torch.zeros([3,640,480])
            final_tensor[0,:257,:200], final_tensor[1,:,:], final_tensor[2,0,0] = a_tensor.unsqueeze(0), v_tensor.unsqueeze(0), torch.tensor(match).unsqueeze(0)
            # final_tensor = torch.tensor([a_tensor, v_tensor, match])
            print("Caching {} and {} into {}".format(a_filename, v_filename, filename))
            print("Dimension : {}".format(final_tensor.size))
            torch.save(final_tensor, p['cache_path'] / filename)


    def process_audio(self, a_file):
        """Process audio before caching. STFT"""
        p = self.av_parameters
        a_array = np.asarray(a_file)
        a_tensor = torch.tensor(a_file, dtype=torch.float64)
        # Values from the paper are not opt, n_fft > win_length ??!
        spectrogram = torch.stft(a_tensor, 512, hop_length=240, win_length=480, window=torch.hann_window(480),
             center=True, pad_mode='reflect', normalized=STFT_NORMALIZED, onesided=True)
        # spectrogram = torchaudio.transforms.Spectrogram(tensor, 512, 480)     
        log_spectrogram = torch.log(spectrogram[:,:200,0] ** 2 + 1)
        # normalize
        return log_spectrogram
        

    def yt_download(self, yt_id, pos, a_length, v_length, match):
        """Download the images and sound of a given youtube video using youtube-dl and ffmpeg by running a bash cmd
        Todo: use a single ffmpeg call with stream mapping."""

        a_cmd_str = self.form_command(yt_id, pos, a_length, match, leave_out="visual")
        v_cmd_str = self.form_command(yt_id, pos, v_length, match, leave_out="audio")

        print("\n---- AUDIO COMMAND: " + a_cmd_str + "\n")

        subprocess.run(a_cmd_str, shell=True, stdout=subprocess.DEVNULL) # shell=True is a security issue
                
        print("\n---- VISUAL COMMAND: " + v_cmd_str + "\n")    

        subprocess.run(v_cmd_str, shell=True, stdout=subprocess.DEVNULL)


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


#%% DEBUG CELL

a = AVDataset(10)
#a.yt_download("--PJHxphWEs", 35, 1, 0)


# %%
