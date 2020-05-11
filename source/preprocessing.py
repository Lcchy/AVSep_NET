import sys
from source.parameters import *
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import subprocess


class AVDataset(Dataset):

    def __init__(self, path_to_data):
        self.csv_data = pd.read_csv(path_to_data, header=2, quotechar='"', skipinitialspace=True)

        # Download the media files that constitute the database from Youtube
        if DOWNLOAD_YT_MEDIA and sys.platform == "linux":
            if len(os.listdir(PATH_TO_VISUAL)) + len(os.listdir(PATH_TO_AUDIO)) > 0:
                # Purge the media directory if necessary
                for file in os.listdir(PATH_TO_DATA / "media"):
                    os.remove(PATH_TO_DATA / "media" / file)
                os.mkdir(PATH_TO_VISUAL)
                os.mkdir(PATH_TO_AUDIO)
            install_yt_dl()
            for index in range(len(self.csv_data)):
                yt_id, pos = self.csv_data.iloc[index, 0], self.csv_data.iloc[index, 1]
                yt_download(yt_id, pos, NB_V_FRAMES, NB_A_FRAMES)
                #load as torch tensor with right dims, save into dataset/cache

    def __len__(self):
        return len(self.av_data)

    def __getitem__(self, index):
        #return torch.load(index)

def yt_download(yt_id, pos, nb_v_frames, nb_a_frames):
    """Download the images and sound of a given youtube video using youtube-dl and ffmpeg (bash)"""
    url = "https://www.youtube.com/watch?v=" + yt_id + ")"
    output = "{}_{}".format(yt_id, pos.replace(":","-"))
    cmd = "ffmpeg -ss {} -i $(/usr/local/bin/youtube-dl -f 22 --get-url {}) -frames:v {} -frames:a {} -q:v 2 {}"
    a_cmd = cmd.format\
        (
            pos,
            url,
            0,
            nb_a_frames,
            output + ".wav"
        )
    v_cmd = cmd.format\
        (
            pos,
            url,
            nb_v_frames,
            0,
            output + ".mp4"
        )
    subprocess.run(a_cmd.split(" "), stdout=subprocess.DEVNULL)
    subprocess.run(v_cmd.split(" "), stdout=subprocess.DEVNULL)

def install_yt_dl():
    """Manually download youtube-dl for latest working version"""
    assert sys.platform == "linux"
    cmd1 = "sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl"
    cmd2 = "sudo chmod a+rx /usr/local/bin/youtube-dl"
    subprocess.run(cmd1.split(" "), stdout=subprocess.DEVNULL)
    subprocess.run(cmd2.split(" "), stdout=subprocess.DEVNULL)