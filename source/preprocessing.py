import sys
import os
import subprocess
import copy
import shutil
import datetime
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from parameters import PATH_TO_DATA_CSV, PATH_TO_VISUAL, PATH_TO_AUDIO, PATH_TO_MEDIA, V_LENGTH, A_LENGTH, DOWNLOAD_YT_MEDIA


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

    def __init__(self):

        self.csv_data = pd.read_csv(PATH_TO_DATA_CSV, header=2, quotechar='"', skipinitialspace=True)
        self.av_parameters = {
            'a_length': 1,
            'v_length': 0,
            'a_codec': "copy",
            'v_size': (640,480),
            'v_framerate': "24/1",
            'v_codec': "copy",
            'v_aspect': "4:3",
            'v_pixel': "+",
            'media_path': PATH_TO_MEDIA,
            'v_path': PATH_TO_VISUAL,
            'a_path': PATH_TO_AUDIO,
        }
        if DOWNLOAD_YT_MEDIA and sys.platform == "linux": self.download_database()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        return torch.load(index)


    def download_database(self):
        """Downoad entire Audio-Visual database to the format given in av_parameters"""
        p = self.av_parameters

        # Purge the media directory if necessary
        if len(os.listdir(p['v_path'])) + len(os.listdir(p['a_path'])) > 0:
            shutil.rmtree(p['v_path'])
            shutil.rmtree(p['a_path'])
            os.mkdir(p['v_path'])
            os.mkdir(p['a_path'])
        for index in range(1):
            yt_id, pos = self.csv_data.iloc[index, 0], int((self.csv_data.iloc[index, 1] + self.csv_data.iloc[index, 2]) / 2)
            self.yt_download(yt_id, pos, p['a_length'], p['v_length'])
            #load as torch tensor with right dims, save into dataset/cache


    def yt_download(self, yt_id, pos, a_length, v_length):
        """Download the images and sound of a given youtube video using youtube-dl and ffmpeg by running a bash cmd
            (It would be better to let ffmpeg seperate the audio and visual than running 2 cmds..)"""

        a_cmd_str = self.form_command(yt_id, pos, a_length, leave_out="visual")
        v_cmd_str = self.form_command(yt_id, pos, v_length, leave_out="audio")

        print(a_cmd_str)
        print(v_cmd_str)    

        subprocess.run(a_cmd_str, shell=True, stdout=subprocess.DEVNULL) # shell=True is a security issue
        subprocess.run(v_cmd_str, shell=True, stdout=subprocess.DEVNULL)


    def form_command(self, yt_id, pos, length, leave_out):
        """Parse arguments into command to form the bash execution string"""
        p = self.av_parameters

        cmd = {
            'stream handler': "ffmpeg", 
            'input flags': {
                'overwrite': "-y",
                'position': "-ss {}".format(str(datetime.timedelta(seconds=int(pos-length/2)))),                            # format: 00:00:00
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
                    #'pixel format': "-pix_fmt {}".format(p['v_pixel']),
                    #'video codec': "-codec:v {}".format(p['v_codec']),
                    #'aspect': "-aspect {}".format(p['v_aspect']),
                    'output path': p['v_path'] / ("{}_{}".format(yt_id, pos) + ".jpg" if length == 0 else ".mp4"),
                },
                'audio': {
                    #'audio codec': "-codec:a {}".format(p['a_codec']),
                    #'layout': 
                    'output path': p['a_path'] / ("{}_{}".format(yt_id, pos) + ".wav"),
                },
            },
        }

        cmd_str = custom_join(cmd, leave_out=[leave_out])
        return cmd_str


#%% DEBUG CELL

a = AVDataset()
#a.yt_download("--PJHxphWEs", 35, 1, 0)


# %%
